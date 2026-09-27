#!/usr/bin/env python3
"""Offline waveform low-pass and stereo listening probe; no learned diffusion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import wave

import numpy as np


RATE = 48_000
LISTENING_PEAK_LIMIT = 10 ** (-1 / 20)
BANDS = ((20, 200), (200, 2000), (2000, 6000), (6000, 12000),
         (12000, 20000), (20000, 24000))
CONDITIONS = ("original", "waveform_heat", "side_gain", "waveform_heat_and_side_gain")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def heat_diffusion(samples: np.ndarray, nu: float, steps: int) -> np.ndarray:
    """Explicit waveform heat equation with reflected endpoint neighbours."""
    if not math.isfinite(nu) or not 0 <= nu <= 0.25 or not 0 <= steps <= 64:
        raise ValueError("nu must be in [0, .25] and steps in [0, 64]")
    result = samples.copy()
    if nu == 0 or steps == 0:
        return result
    for _ in range(steps):
        padded = np.pad(result, ((1, 1), (0, 0)), mode="reflect")
        result += nu * (padded[:-2] - 2 * result + padded[2:])
    return result


def side_gain(samples: np.ndarray, gain: float) -> np.ndarray:
    if not math.isfinite(gain) or not 0 <= gain <= 4:
        raise ValueError("side gain must be in [0, 4]")
    mid = (samples[:, 0] + samples[:, 1]) * 0.5
    side = (samples[:, 0] - samples[:, 1]) * (0.5 * gain)
    return np.column_stack((mid + side, mid - side))


def variant(source: np.ndarray, condition: str, nu: float, steps: int,
            gain: float) -> np.ndarray:
    result = heat_diffusion(source, nu, steps) if "waveform_heat" in condition else source.copy()
    return side_gain(result, gain) if "side_gain" in condition else result


def db_ratio(numerator: float, denominator: float) -> float | None:
    return 10 * math.log10(numerator / denominator) if numerator > 0 and denominator > 0 else None


def envelope(samples: np.ndarray) -> np.ndarray:
    """Stereo RMS in complete 10 ms blocks, normalized to whole signal RMS."""
    count = len(samples) // 480
    if count == 0:
        return np.empty(0)
    values = np.sqrt(np.mean(samples[:count * 480].reshape(count, 480, 2) ** 2, axis=(1, 2)))
    rms = float(np.sqrt(np.mean(samples ** 2)))
    return values / rms if rms > 0 else np.empty(0)


def spectrum(samples: np.ndarray) -> dict:
    size = min(4096, len(samples))
    window = np.hanning(size)
    starts = np.unique(np.linspace(0, len(samples) - size, min(128, 1 + len(samples) // size), dtype=int))
    powers = np.zeros((size // 2 + 1, 3))
    normalization = size * float(np.sum(window ** 2))
    for start in starts:
        block = samples[start:start + size]
        transformed = np.fft.rfft(block * window[:, None], axis=0)
        left, right = transformed[:, 0], transformed[:, 1]
        power = np.column_stack((np.abs(left) ** 2 + np.abs(right) ** 2,
                                 np.abs((left + right) * 0.5) ** 2,
                                 np.abs((left - right) * 0.5) ** 2))
        power[1:-1 if size % 2 == 0 else None] *= 2
        powers += power / normalization
    powers /= len(starts)
    frequencies = np.fft.rfftfreq(size, 1 / RATE)
    total = float(np.sum(powers[:, 0]))
    bands = []
    for low, high in BANDS:
        mask = (frequencies >= low) & ((frequencies <= high) if high == RATE // 2 else (frequencies < high))
        band = np.sum(powers[mask], axis=0)
        bands.append({"hz": [low, high], "lr_power": float(band[0]),
                      "lr_total_fraction": float(band[0] / total) if total > 0 else None,
                      "mid_power": float(band[1]), "side_power": float(band[2])})
    return {"method": "sampled Hann FFT; one-sided power in full-scale squared units",
            "fft_frames": size, "sampled_windows": len(starts),
            "lr_total_power_including_dc": total, "bands": bands}


def metrics(samples: np.ndarray, source_envelope: np.ndarray) -> dict:
    power = np.mean(samples ** 2, axis=0)
    rms = float(np.sqrt(np.mean(power)))
    peak = float(np.max(np.abs(samples)))
    centered = samples - np.mean(samples, axis=0)
    variance = np.mean(centered ** 2, axis=0)
    correlation = (float(np.mean(centered[:, 0] * centered[:, 1]) / np.sqrt(variance[0] * variance[1]))
                   if np.all(variance > 0) else None)
    mid = (samples[:, 0] + samples[:, 1]) * 0.5
    side = (samples[:, 0] - samples[:, 1]) * 0.5
    env = envelope(samples)
    onset_delta = (float(np.mean(np.abs(np.diff(env) - np.diff(source_envelope))))
                   if len(env) > 1 and len(env) == len(source_envelope) else None)
    return {"stereo_rms": rms, "peak": peak,
            "at_or_over_full_scale_samples": int(np.count_nonzero(np.abs(samples) >= 1)),
            "pcm16_endpoint_samples": int(np.count_nonzero((samples <= -1) | (samples >= 32767 / 32768))),
            "crest_db": db_ratio(peak * peak, rms * rms),
            "lr_centered_correlation": float(np.clip(correlation, -1, 1)) if correlation is not None else None,
            "left_over_right_power_db": db_ratio(float(power[0]), float(power[1])),
            "side_over_mid_power_db": db_ratio(float(np.mean(side ** 2)), float(np.mean(mid ** 2))),
            "onset_envelope_change_proxy": onset_delta,
            "onset_proxy_definition": "mean absolute change in adjacent normalized 10ms RMS envelope differences from source; not musical quality",
            "spectrum": spectrum(samples)}


def export(source_wav: Path, output_dir: Path, seconds: float = 60,
           seed: int = 123, nu: float = 0.1, steps: int = 5,
           side: float = 2.0) -> dict:
    if not math.isfinite(seconds) or not 0 < seconds <= 60:
        raise ValueError("seconds must be greater than zero and at most 60")
    # Validate controls even if the input is silent.
    heat_diffusion(np.zeros((2, 2)), nu, steps)
    side_gain(np.zeros((2, 2)), side)
    source_wav = source_wav.resolve(strict=True)
    if source_wav.stat().st_size > 700_000_000:
        raise ValueError("source must be below 700 MB")
    before = sha256(source_wav)
    with wave.open(str(source_wav), "rb") as reader:
        if (reader.getnchannels(), reader.getframerate(), reader.getsampwidth(), reader.getcomptype()) != (2, RATE, 2, "NONE"):
            raise ValueError("expected 48 kHz stereo PCM16 WAV")
        if not 4 <= reader.getnframes() <= RATE * 3600 or source_wav.stat().st_size > 700_000_000:
            raise ValueError("source must contain 4 frames to 3600 seconds, and be below 700 MB")
        count = min(reader.getnframes(), int(seconds * RATE))
        if count < 4:
            raise ValueError("selected interval must contain at least 4 frames")
        data = reader.readframes(count)
        if len(data) != count * 4:
            raise ValueError("truncated source WAV")
        source = np.frombuffer(data, dtype="<i2").reshape(-1, 2).astype(np.float64) / 32768
    output_dir.mkdir(parents=True, exist_ok=False)
    original_copy = output_dir / "source_original.wav"
    shutil.copyfile(source_wav, original_copy)
    if sha256(original_copy) != before:
        raise RuntimeError("source changed while copying; discard incomplete package")
    source_env = envelope(source)
    raw = {}
    for condition in CONDITIONS:
        samples = variant(source, condition, nu, steps, side)
        raw[condition] = metrics(samples, source_env)
        del samples
    target = min(item["stereo_rms"] for item in raw.values())
    rms_gains = {key: target / item["stereo_rms"] if item["stereo_rms"] > 0 else 1.0
                 for key, item in raw.items()}
    largest_peak = max(raw[key]["peak"] * gain for key, gain in rms_gains.items())
    headroom = min(1.0, LISTENING_PEAK_LIMIT / largest_peak) if largest_peak > 0 else 1.0
    order = list(CONDITIONS)
    random.Random(seed).shuffle(order)
    listening = {}
    for index, condition in enumerate(order):
        label = chr(ord("A") + index)
        samples = variant(source, condition, nu, steps, side)
        gain = rms_gains[condition] * headroom
        quantized = np.rint(samples * gain * 32768).astype("<i2")
        path = output_dir / f"listen_{label}.wav"
        with wave.open(str(path), "wb") as writer:
            writer.setnchannels(2)
            writer.setsampwidth(2)
            writer.setframerate(RATE)
            writer.writeframes(quantized.tobytes())
        listening[label] = {"condition": condition, "file": path.name, "sha256": sha256(path),
                            "rms_match_gain": rms_gains[condition], "shared_headroom_gain": headroom,
                            "combined_gain": gain,
                            "metrics": metrics(quantized.astype(np.float64) / 32768, source_env)}
        del samples, quantized
    after = sha256(source_wav)
    if after != before:
        raise RuntimeError("source changed during probe; discard incomplete package")
    receipt = {"schema_version": 1, "description": "waveform low-pass and mid/side probe; no latent or learned diffusion",
               "tool": {"script_sha256": sha256(Path(__file__)), "numpy_version": np.__version__},
               "source": {"path": str(source_wav), "sha256_before": before, "sha256_after": after,
                          "exact_copy": original_copy.name, "copy_sha256": sha256(original_copy)},
               "interval": {"start_frame": 0, "frames": count, "sample_rate": RATE},
               "controls": {"nu": nu, "steps": steps, "side_gain": side, "seed": seed},
               "common_stereo_rms_target_before_headroom": target,
               "listening_peak_limit_dbfs": -1,
               "raw_float_metrics_before_gain": raw, "listening_pcm16": listening,
               "limitations": ["RMS matching is not perceptual loudness matching.",
                               "Inspect listening labels before the receipt to avoid revealing conditions.",
                               "FFT windows are sampled; band fractions cannot establish bass recovery.",
                               "Side gain preserves raw mono fold-down before level matching, cannot create side information from mono, and can emphasize existing side artifacts.",
                               "Waveform heat diffusion is low-pass filtering, not latent or learned diffusion.",
                               "No acoustic quality or downstream generation claim; source prefix only.",
                               "A silent condition makes the common target zero; all listening conditions then become silent."]}
    (output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    (output_dir / "LISTEN.txt").write_text("Compare listen_A.wav through listen_D.wav in any order.\n"
                                         "Note harshness, stereo placement, transients and retained detail.\n"
                                         "The receipt reveals conditions; these probes do not establish musical quality.\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-wav", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="must not already exist")
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--nu", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--side-gain", type=float, default=2)
    args = parser.parse_args()
    export(args.source_wav, args.output_dir, args.seconds, args.seed, args.nu, args.steps, args.side_gain)
    print(args.output_dir / "LISTEN.txt")


if __name__ == "__main__":
    main()
