#!/usr/bin/env python3
"""Check blinded manual-pilot controls and expose their residual confounds."""
import argparse
import hashlib
import json
import wave
from pathlib import Path

import numpy as np


def read(path):
    with wave.open(str(path), "rb") as wav:
        assert (wav.getnchannels(), wav.getframerate(), wav.getsampwidth()) == (2, 48000, 2)
        frames = wav.getnframes()
        data = np.frombuffer(wav.readframes(frames), "<i2").copy()
    return data.reshape(frames, 2).astype(np.float64) / 32768


def rms(values):
    return float(np.sqrt(np.mean(values * values)))


def average_log_spectrum(values):
    mono = values.mean(axis=1)
    width = 4096
    window = np.hanning(width)
    power = np.zeros(width // 2 + 1)
    count = 0
    for start in range(0, len(mono) - width + 1, 2048):
        power += np.abs(np.fft.rfft(mono[start:start + width] * window)) ** 2
        count += 1
    return np.log1p(power / max(1, count))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", required=True, type=Path)
    args = parser.parse_args()
    root = args.pilot
    key = json.loads((root / "private/answer_key.json").read_text())
    audio = {}
    for code, item in key["conditions"].items():
        path = root / "uploads" / f"{code}.wav"
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"pilot WAV changed: {code}")
        audio[item["identity"]] = read(path)
    if any(len(wav) != key["frames"] for wav in audio.values()):
        raise RuntimeError("pilot durations differ")
    block = key["time_block_frames"]
    order = np.asarray(key["time_permutation"])
    cuts = key.get("time_cut_points", list(range(0, key["frames"] + 1, block)))
    if not np.array_equal(np.sort(order), np.arange(len(order))):
        raise RuntimeError("time control lost or duplicated blocks")
    intact = audio["intact_frozen_titan"]
    reordered = audio["short_block_time_order"]
    nominal_boundaries = np.arange(block, len(intact), block)
    source_segments = [intact[cuts[index]:cuts[index + 1]] for index in range(len(order))]
    destination_cuts = np.cumsum([len(source_segments[index]) for index in order])[:-1]
    def jump(values, boundaries):
        return float(np.abs(values[boundaries] - values[boundaries - 1]).mean())
    # Edge fades and gain differ slightly; interior block waveform agreement
    # after mapping and RMS normalization is still a useful manipulation check.
    local_correlations = []
    start = 0
    for index, source in enumerate(order):
        length = len(source_segments[source])
        interior = slice(2400, length - 2400)
        a = source_segments[source][interior].ravel()
        b = reordered[start:start + length][interior].ravel()
        local_correlations.append(float(np.dot(a, b) / max(1e-12, np.linalg.norm(a) * np.linalg.norm(b))))
        start += length
    if start != key["frames"]:
        raise RuntimeError("time control changed total duration")
    intact_jump = jump(intact, nominal_boundaries)
    shuffled_jump = jump(reordered, destination_cuts)
    intact_spectrum = average_log_spectrum(intact)
    validation = {
        "schema": 1, "frames": key["frames"], "duration_seconds": key["duration_seconds"],
        "all_hashes_and_durations_match": True,
        "time_blocks_are_a_permutation": True,
        "time_fraction_moved": float(np.mean(order != np.arange(len(order)))),
        "time_median_interior_block_correlation": float(np.median(local_correlations)),
        "time_cut_method": key.get("time_cut_method", "fixed"),
        "time_segment_length_min": min(map(len, source_segments)),
        "time_segment_length_max": max(map(len, source_segments)),
        "intact_boundary_mean_jump": intact_jump,
        "intact_selected_cut_mean_jump": jump(intact, np.asarray(cuts[1:-1])),
        "time_control_boundary_mean_jump": shuffled_jump,
        "time_boundary_jump_ratio": shuffled_jump / max(1e-12, intact_jump),
        "time_average_log_spectral_rms": rms(average_log_spectrum(reordered) - intact_spectrum),
        "phase_average_log_spectral_rms": rms(average_log_spectrum(
            audio["shared_phase_spectral_surrogate"]) - intact_spectrum),
        "phase_surrogate_preprocessing": key["surrogate_validation"],
        "rms": {name: rms(wav) for name, wav in audio.items()},
        "limitation": "time-block reordering introduces boundary discontinuities; ratings cannot isolate global order from seams",
    }
    (root / "control_validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"fraction_moved": validation["time_fraction_moved"],
                      "block_correlation": validation["time_median_interior_block_correlation"],
                      "boundary_jump_ratio": validation["time_boundary_jump_ratio"]}))


if __name__ == "__main__":
    main()
