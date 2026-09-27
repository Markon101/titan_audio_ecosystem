#!/usr/bin/env python3
"""Offline parametric EQ comparison for existing Titan Audio WAVs.

Applies a calibrated biquad bell de-emphasis filter targeting the 2–6 kHz sensitivity zone
(default: center 3500 Hz, Q 1.0, -3.0 dB). Attempts matched stereo RMS,
then applies a shared peak safety limit. Perceptual quality still needs listening.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import wave

import numpy as np


RATE = 48_000
BANDS = (
    (20, 200),
    (200, 2000),
    (2000, 6000),
    (6000, 12000),
    (12000, 20000),
    (20000, 24000),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def biquad_peaking_eq_coeffs(f0: float, q: float, gain_db: float, fs: float = RATE) -> tuple[np.ndarray, np.ndarray]:
    """Calculate RBJ Audio EQ Cookbook peaking filter coefficients."""
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / fs
    alpha = math.sin(w0) / (2.0 * q)
    cos_w0 = math.cos(w0)

    b0 = 1.0 + alpha * a
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * a
    a0 = 1.0 + alpha / a
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / a

    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
    a_norm = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return b, a_norm


def apply_biquad(samples: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Apply IIR biquad filter across stereo channels using Direct Form I."""
    out = np.zeros_like(samples, dtype=np.float64)
    for ch in range(samples.shape[1]):
        x = samples[:, ch]
        y = np.zeros_like(x)
        # Biquad difference equation: y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
        x1, x2 = 0.0, 0.0
        y1, y2 = 0.0, 0.0
        b0, b1, b2 = b[0], b[1], b[2]
        a1, a2 = a[1], a[2]
        for n in range(len(x)):
            xn = x[n]
            yn = b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            y[n] = yn
            x2 = x1
            x1 = xn
            y2 = y1
            y1 = yn
        out[:, ch] = y
    return out


def compute_spectral_metrics(samples: np.ndarray) -> dict:
    power = np.mean(samples ** 2, axis=0)
    rms = float(np.sqrt(np.mean(power)))
    peak = float(np.max(np.abs(samples)))
    centered = samples - np.mean(samples, axis=0)
    var = np.mean(centered ** 2, axis=0)
    correlation = (
        float(np.mean(centered[:, 0] * centered[:, 1]) / np.sqrt(var[0] * var[1]))
        if np.all(var > 0)
        else None
    )
    mid = (samples[:, 0] + samples[:, 1]) * 0.5
    side = (samples[:, 0] - samples[:, 1]) * 0.5
    mid_power = float(np.mean(mid ** 2))
    side_power = float(np.mean(side ** 2))
    side_over_mid_db = 10 * math.log10(side_power / mid_power) if side_power > 0 and mid_power > 0 else None

    size = min(4096, len(samples))
    window = np.hanning(size)
    starts = np.unique(np.linspace(0, len(samples) - size, min(128, 1 + len(samples) // size), dtype=int))
    powers = np.zeros((size // 2 + 1, 3))
    norm = size * float(np.sum(window ** 2))
    for start in starts:
        block = samples[start : start + size]
        trans = np.fft.rfft(block * window[:, None], axis=0)
        l, r = trans[:, 0], trans[:, 1]
        p = np.column_stack((
            np.abs(l) ** 2 + np.abs(r) ** 2,
            np.abs((l + r) * 0.5) ** 2,
            np.abs((l - r) * 0.5) ** 2,
        ))
        p[1:-1 if size % 2 == 0 else None] *= 2
        powers += p / norm
    powers /= len(starts)
    freqs = np.fft.rfftfreq(size, 1 / RATE)
    total_power = float(np.sum(powers[:, 0]))

    bands_out = []
    for low, high in BANDS:
        mask = (freqs >= low) & ((freqs <= high) if high == RATE // 2 else (freqs < high))
        band = np.sum(powers[mask], axis=0)
        bands_out.append({
            "hz": [low, high],
            "lr_power": float(band[0]),
            "lr_fraction": float(band[0] / total_power) if total_power > 0 else 0.0,
            "mid_power": float(band[1]),
            "side_power": float(band[2]),
        })

    return {
        "rms": rms,
        "peak": peak,
        "peak_dbfs": 20 * math.log10(peak) if peak > 0 else -120.0,
        "crest_db": 20 * math.log10(peak / rms) if rms > 0 and peak > 0 else 0.0,
        "lr_correlation": float(np.clip(correlation, -1, 1)) if correlation is not None else None,
        "side_over_mid_db": side_over_mid_db,
        "bands": {f"{b['hz'][0]}-{b['hz'][1]}Hz": b["lr_fraction"] for b in bands_out},
    }


def process_wav(
    input_wav: Path,
    output_wav: Path,
    center_hz: float = 3500.0,
    q: float = 1.0,
    gain_db: float = -3.0,
    target_peak_dbfs: float = -1.0,
) -> dict:
    input_wav = input_wav.resolve(strict=True)
    output_wav = output_wav.resolve(strict=False)
    receipt_path = output_wav.with_suffix(".receipt.json")
    if output_wav == input_wav or output_wav.exists() or receipt_path.exists():
        raise ValueError("output and receipt must be new paths distinct from the input")
    if not (math.isfinite(center_hz) and 20 <= center_hz < RATE / 2):
        raise ValueError("center frequency must be finite and between 20 Hz and Nyquist")
    if not math.isfinite(q) or q <= 0:
        raise ValueError("Q must be finite and positive")
    if not math.isfinite(gain_db) or not -12 <= gain_db <= 0:
        raise ValueError("de-emphasis gain must be finite and in [-12, 0] dB")
    if not math.isfinite(target_peak_dbfs) or target_peak_dbfs > 0:
        raise ValueError("target peak must be finite and at most 0 dBFS")
    source_hash_before = sha256_file(input_wav)
    with wave.open(str(input_wav), "rb") as reader:
        channels, rate, width = reader.getnchannels(), reader.getframerate(), reader.getsampwidth()
        if (channels, rate, width) != (2, RATE, 2):
            raise ValueError(f"Expected 48kHz 16-bit stereo WAV, got ch={channels}, rate={rate}, width={width}")
        nframes = reader.getnframes()
        if nframes < 4096:
            raise ValueError("source needs at least 4096 stereo frames")
        data = reader.readframes(nframes)
        if len(data) != nframes * 4:
            raise ValueError("truncated source WAV")
        raw_samples = np.frombuffer(data, dtype="<i2").reshape(-1, 2).astype(np.float64) / 32768.0

    before_metrics = compute_spectral_metrics(raw_samples)

    # Design and apply peaking de-emphasis filter
    b, a = biquad_peaking_eq_coeffs(center_hz, q, gain_db, RATE)
    filtered = apply_biquad(raw_samples, b, a)

    # Match total stereo RMS first. A peak guard can limit the achieved match;
    # record both gains so a listening comparison does not hide this tradeoff.
    raw_rms = before_metrics["rms"]
    filtered_rms = float(np.sqrt(np.mean(filtered ** 2)))
    rms_match_gain = raw_rms / filtered_rms if filtered_rms > 0 else 1.0
    matched_peak = float(np.max(np.abs(filtered))) * rms_match_gain
    target_linear_peak = 10.0 ** (target_peak_dbfs / 20.0)
    peak_guard_gain = min(1.0, target_linear_peak / matched_peak) if matched_peak > 0 else 1.0
    norm_gain = rms_match_gain * peak_guard_gain
    mastered = filtered * norm_gain

    # Quantize to 16-bit PCM
    quantized = np.rint(mastered * 32768.0).clip(-32768.0, 32767.0).astype("<i2")

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_wav), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(RATE)
        writer.writeframes(quantized.tobytes())

    after_metrics = compute_spectral_metrics(quantized.astype(np.float64) / 32768.0)

    source_hash_after = sha256_file(input_wav)
    if source_hash_after != source_hash_before:
        raise RuntimeError("input changed while processing; discard the new output")
    achieved_rms_delta_db = (
        20 * math.log10(after_metrics["rms"] / raw_rms)
        if raw_rms > 0 and after_metrics["rms"] > 0 else None
    )
    receipt = {
        "input_wav": str(input_wav),
        "input_sha256": source_hash_before,
        "input_sha256_after": source_hash_after,
        "output_wav": str(output_wav),
        "output_sha256": sha256_file(output_wav),
        "filter_parameters": {
            "type": "parametric_peaking_bell",
            "center_frequency_hz": center_hz,
            "q_factor": q,
            "gain_db": gain_db,
            "target_peak_dbfs": target_peak_dbfs,
            "rms_match_gain": rms_match_gain,
            "peak_guard_gain": peak_guard_gain,
            "post_gain_applied": norm_gain,
            "achieved_rms_delta_db": achieved_rms_delta_db,
            "rms_match_limited_by_peak": peak_guard_gain < 1.0,
        },
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "impact_summary": {
            "band_2k_6k_before": before_metrics["bands"]["2000-6000Hz"],
            "band_2k_6k_after": after_metrics["bands"]["2000-6000Hz"],
            "band_2k_6k_delta_percent": (after_metrics["bands"]["2000-6000Hz"] - before_metrics["bands"]["2000-6000Hz"]) * 100.0,
            "peak_dbfs_before": before_metrics["peak_dbfs"],
            "peak_dbfs_after": after_metrics["peak_dbfs"],
            "correlation_before": before_metrics["lr_correlation"],
            "correlation_after": after_metrics["lr_correlation"],
            "side_over_mid_db_before": before_metrics["side_over_mid_db"],
            "side_over_mid_db_after": after_metrics["side_over_mid_db"],
        },
    }

    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input stereo WAV file")
    parser.add_argument("--output", type=Path, required=True, help="Output de-emphasized WAV file")
    parser.add_argument("--center-hz", type=float, default=3500.0, help="Center frequency in Hz (default: 3500)")
    parser.add_argument("--q", type=float, default=1.0, help="Q factor (default: 1.0)")
    parser.add_argument("--gain-db", type=float, default=-3.0, help="De-emphasis gain in dB (default: -3.0)")
    parser.add_argument("--target-peak-dbfs", type=float, default=-1.0, help="Target peak in dBFS (default: -1.0)")
    args = parser.parse_args()

    receipt = process_wav(args.input, args.output, args.center_hz, args.q, args.gain_db, args.target_peak_dbfs)
    print(f"Mastering de-emphasis complete:")
    print(f"  Input:  {receipt['input_wav']}")
    print(f"  Output: {receipt['output_wav']}")
    print(f"  2-6 kHz Energy: {receipt['impact_summary']['band_2k_6k_before']*100:.1f}% -> {receipt['impact_summary']['band_2k_6k_after']*100:.1f}%")
    print(f"  Peak:           {receipt['impact_summary']['peak_dbfs_before']:.2f} dBFS -> {receipt['impact_summary']['peak_dbfs_after']:.2f} dBFS")
    print(f"  Receipt:        {args.output.with_suffix('.receipt.json')}")


if __name__ == "__main__":
    main()
