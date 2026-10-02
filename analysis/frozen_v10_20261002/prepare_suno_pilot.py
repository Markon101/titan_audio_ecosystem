#!/usr/bin/env python3
"""Make a local, blinded manual Suno pilot; never contacts Suno."""
import argparse
import csv
import hashlib
import json
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 48_000
BLOCK = 8 * 4_096  # Preserve 0.683 s internal texture in the order control.


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for data in iter(lambda: source.read(4 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def read(path):
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth()) != (2, SAMPLE_RATE, 2):
            raise ValueError(f"expected stereo 48 kHz PCM16: {path}")
        values = np.frombuffer(wav.readframes(wav.getnframes()), "<i2").copy()
    return values.reshape(-1, 2).astype(np.float64) / 32768.0


def write(path, values):
    encoded = np.round(np.clip(values, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(encoded.tobytes())


def rms(values):
    return float(np.sqrt(np.mean(values * values)))


def phase_surrogate(values, rng):
    spectrum = np.fft.rfft(values, axis=0)
    phase = rng.uniform(-np.pi, np.pi, len(spectrum))
    phase[0] = 0.0
    if len(values) % 2 == 0:
        phase[-1] = 0.0
    randomized = spectrum * np.exp(1j * phase)[:, None]
    result = np.fft.irfft(randomized, n=len(values), axis=0)
    relative_power_error = float(np.max(np.abs(np.abs(np.fft.rfft(result, axis=0)) - np.abs(spectrum))) /
                                 max(1e-12, float(np.max(np.abs(spectrum)))))
    cross_original = spectrum[:, 0] * np.conj(spectrum[:, 1])
    cross_new = randomized[:, 0] * np.conj(randomized[:, 1])
    relative_cross_error = float(np.max(np.abs(cross_new - cross_original)) /
                                 max(1e-12, float(np.max(np.abs(cross_original)))))
    if relative_power_error > 1e-9 or relative_cross_error > 1e-9:
        raise RuntimeError("phase surrogate did not preserve stereo power/cross-spectrum")
    return result, {"pre_processing_relative_power_error": relative_power_error,
                    "pre_processing_relative_cross_spectrum_error": relative_cross_error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--causal", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--quiet-cuts", action="store_true",
                        help="choose low-energy boundaries within +/-4096 frames of nominal blocks")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"pilot already exists: {args.out}")
    baseline, causal = read(args.baseline), read(args.causal)
    frames = min(len(baseline), len(causal)) // BLOCK * BLOCK
    if frames < 8 * BLOCK:
        raise ValueError("pilot needs at least eight complete 0.683 s blocks")
    baseline, causal = baseline[:frames], causal[:frames]
    rng = np.random.default_rng(args.seed)
    if args.quiet_cuts:
        cuts = [0]
        for nominal in range(BLOCK, frames, BLOCK):
            candidates = np.arange(nominal - BLOCK // 8, nominal + BLOCK // 8)
            boundary_energy = np.sum(baseline[candidates - 1] ** 2 + baseline[candidates] ** 2,
                                     axis=1)
            cuts.append(int(candidates[np.argmin(boundary_energy)]))
        cuts.append(frames)
    else:
        cuts = list(range(0, frames + 1, BLOCK))
    segments = [baseline[cuts[index]:cuts[index + 1]] for index in range(len(cuts) - 1)]
    order = rng.permutation(len(segments))
    if np.array_equal(order, np.arange(len(order))):
        order = np.roll(order, 1)
    reordered = np.concatenate([segments[index] for index in order], axis=0).copy()
    assert np.array_equal(np.sort(order), np.arange(len(order)))
    surrogate, null_check = phase_surrogate(baseline, rng)
    conditions = {
        "intact_frozen_titan": baseline,
        "causal_arm": causal,
        "short_block_time_order": reordered,
        "shared_phase_spectral_surrogate": surrogate,
    }
    # The same mild edge fade and RMS target are applied to every uploaded file.
    edge = min(2_400, frames // 20)
    envelope = np.ones(frames)
    envelope[:edge] = np.linspace(0, 1, edge)
    envelope[-edge:] = np.linspace(1, 0, edge)
    target_rms = rms(baseline * envelope[:, None])
    prepared = {}
    measurements = {}
    for name, data in conditions.items():
        original_rms = rms(data)
        faded = data * envelope[:, None]
        gain = target_rms / max(1e-12, rms(faded))
        peak_cap = min(1.0, 0.95 / max(1e-12, float(np.max(np.abs(faded * gain)))))
        effective_gain = gain * peak_cap
        prepared[name] = faded * effective_gain
        measurements[name] = {"source_rms": original_rms, "post_rms": rms(prepared[name]),
                              "peak": float(np.max(np.abs(prepared[name]))),
                              "gain": effective_gain, "peak_cap": peak_cap}
    condition_order = list(prepared)
    rng.shuffle(condition_order)
    uploads = args.out / "uploads"
    private = args.out / "private"
    uploads.mkdir(parents=True)
    private.mkdir()
    key = {"schema": 1, "seed": args.seed, "frames": frames,
           "duration_seconds": frames / SAMPLE_RATE, "sample_rate": SAMPLE_RATE,
           "source_sha256": {"baseline": sha(args.baseline), "causal": sha(args.causal)},
           "time_block_frames": BLOCK, "time_cut_points": cuts,
           "time_cut_method": "minimum_stereo_boundary_energy_within_4096_frames" if args.quiet_cuts else "fixed",
           "time_permutation": order.tolist(),
           "surrogate_validation": null_check, "processing": measurements, "conditions": {}}
    for index, name in enumerate(condition_order, 1):
        code = f"{index:03}"
        path = uploads / f"{code}.wav"
        write(path, prepared[name])
        key["conditions"][code] = {"identity": name, "sha256": sha(path)}
    (private / "answer_key.json").write_text(json.dumps(key, indent=2, sort_keys=True) + "\n")
    with (args.out / "generation_receipts.csv").open("w", newline="") as target:
        csv.writer(target).writerow(["anonymous_id", "generation_index", "input_sha256",
                                     "suno_version", "mode", "prompt", "sliders",
                                     "upload_order", "output_file", "output_sha256",
                                     "kept_or_rejected", "notes"])
    with (args.out / "blind_ratings.csv").open("w", newline="") as target:
        csv.writer(target).writerow(["output_file", "rater", "coherence_1_5",
                                     "interesting_transitions_1_5", "rhythmic_structural_variety_1_5",
                                     "personal_preference_1_5", "notes"])
    (args.out / "README.md").write_text(
        "# Manual blinded Suno pilot\n\nUpload the four files in `uploads/` in randomized order. "
        "Use the same Suno version, mode, prompt, sliders, and two or more generations "
        "per ID. Add prompt-only generations as a separate fifth condition. Record every "
        "output including rejected rerolls in `generation_receipts.csv`. Enter blind "
        "ratings in `blind_ratings.csv`. Rate coherence, interesting transitions, "
        "rhythmic/structural variety, and personal preference "
        "before opening `private/answer_key.json`. The time-order control may "
        "have seam discontinuities; see `control_validation.json` after rating. "
        "No files were sent by this script.\n"
    )
    print(json.dumps({"anonymous_ids": sorted(key["conditions"]), "frames": frames,
                      "duration_seconds": frames / SAMPLE_RATE, "output": str(args.out)}))


if __name__ == "__main__":
    main()
