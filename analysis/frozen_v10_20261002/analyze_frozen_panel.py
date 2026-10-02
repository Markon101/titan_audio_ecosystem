#!/usr/bin/env python3
"""Compare paired frozen WAVs and latent traces without selecting interventions."""
import argparse
import json
import wave
from pathlib import Path

import numpy as np

CHUNK = 4096
RATE = 48000


def read_wav(path):
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth()) != (2, RATE, 2):
            raise ValueError(f"invalid analysis WAV: {path}")
        data = np.frombuffer(wav.readframes(wav.getnframes()), "<i2").copy()
    return data.reshape(-1, 2).astype(np.float64) / 32768.0


def rms(x):
    return float(np.sqrt(np.mean(x * x)))


def correlation(x, y):
    a, b = x.ravel(), y.ravel()
    a, b = a - a.mean(), b - b.mean()
    return float(np.dot(a, b) / max(1e-12, np.linalg.norm(a) * np.linalg.norm(b)))


def spectral_distribution(audio):
    mono = audio.mean(axis=1)
    width = 4096
    spectrum = np.zeros(width // 2 + 1, dtype=np.float64)
    window = np.hanning(width)
    count = 0
    for start in range(0, len(mono) - width + 1, 2048):
        spectrum += np.abs(np.fft.rfft(mono[start:start + width] * window)) ** 2
        count += 1
    return np.log1p(spectrum / max(count, 1))


def envelope_modulation(audio):
    mono = audio.mean(axis=1)
    blocks = len(mono) // 4800
    envelope = np.sqrt(np.mean(mono[:blocks * 4800].reshape(blocks, 4800) ** 2, axis=1))
    power = np.abs(np.fft.rfft(envelope - envelope.mean())) ** 2
    hz = np.fft.rfftfreq(blocks, 0.1)
    total = max(1e-12, float(power[hz >= 0.1].sum()))
    return envelope, {f"{low:g}_{high:g}_hz": float(power[(hz >= low) & (hz < high)].sum() / total)
                      for low, high in [(0.1, 1), (1, 4), (4, 10)]}


def geometry(audio):
    left, right = audio.T
    mid, side = (left + right) * 0.5, (left - right) * 0.5
    return {"rms": rms(audio), "peak": float(np.max(np.abs(audio))),
            "stereo_correlation": correlation(left, right),
            "side_mid_ratio": rms(side) / max(1e-12, rms(mid))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    args = parser.parse_args()
    root = args.panel / "ablations"
    suite = json.loads((root / "summary.json").read_text())
    report = json.loads((args.panel / "analysis_report.json").read_text())
    if not suite["no_op_clone_exact"] or not report["non_mutation"]["unchanged"]:
        raise RuntimeError("no-op or parent non-mutation gate failed")
    baseline = read_wav(root / "full/post_dsp.wav")
    base_spectrum = spectral_distribution(baseline)
    base_envelope, base_modulation = envelope_modulation(baseline)
    base_summary = json.loads((root / "full/summary.json").read_text())
    base_steps = base_summary["sampled_steps"]
    result = {"schema": 1, "source": str(args.panel), "frames": len(baseline),
              "duration_seconds": len(baseline) / RATE, "no_op_clone_exact": True,
              "parent_non_mutation": True, "baseline_geometry": geometry(baseline),
              "baseline_modulation": base_modulation, "conditions": {}}
    for item in suite["conditions"]:
        name = item["name"]
        audio = read_wav(root / name / "post_dsp.wav")
        if audio.shape != baseline.shape:
            raise ValueError(f"unmatched WAV shape: {name}")
        summary = json.loads((root / name / "summary.json").read_text())
        steps = summary["sampled_steps"]
        points = item["comparison"]["points"]
        width = min(64 * CHUNK, len(audio) // 2)
        spectrum = spectral_distribution(audio)
        envelope, modulation = envelope_modulation(audio)
        result["conditions"][name] = {
            "waveform_l2_full": rms(audio - baseline),
            "waveform_l2_first_64_chunks": rms(audio[:width] - baseline[:width]),
            "waveform_l2_last_64_chunks": rms(audio[-width:] - baseline[-width:]),
            "waveform_correlation_full": correlation(audio, baseline),
            "spectral_distribution_log_rms": rms(spectrum - base_spectrum),
            "envelope_correlation": correlation(envelope, base_envelope),
            "modulation": modulation, "geometry": geometry(audio),
            "sampled_action_changes": sum(a["action"] != b["action"]
                                           for a, b in zip(steps, base_steps)),
            "full_world_fingerprint_equal": summary["final_world_fingerprint"] ==
                                            base_summary["final_world_fingerprint"],
            "final_state_distance": points[-1]["state_distance"],
            "final_coarse_state_distance": points[-1]["coarse_state_distance"],
            "final_coarse_rms": steps[-1].get("coarse_rms"),
            "baseline_final_coarse_rms": base_steps[-1].get("coarse_rms"),
        }
    (args.panel / "audio_readout.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for name, values in result["conditions"].items():
        print(name, "full_l2", round(values["waveform_l2_full"], 5),
              "early", round(values["waveform_l2_first_64_chunks"], 5),
              "late", round(values["waveform_l2_last_64_chunks"], 5),
              "actions", values["sampled_action_changes"])


if __name__ == "__main__":
    main()
