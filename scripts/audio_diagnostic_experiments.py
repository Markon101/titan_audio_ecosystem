#!/usr/bin/env python3
"""Descriptive audio diagnostics with explicit run-lineage checks.

The tagged trace and run metadata are overwritten by later invocations. They
must be checked against the immutable hashed WAV before telemetry is used.
"""

from __future__ import annotations

import argparse
import csv
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


def db_ratio(num: float, den: float) -> float | None:
    return 10 * math.log10(num / den) if num > 0 and den > 0 else None


def compute_audio_metrics(samples: np.ndarray) -> dict:
    """Compute exact stereo, power, and spectral band metrics."""
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
    side_over_mid_db = db_ratio(side_power, mid_power)

    # Spectral analysis
    size = min(4096, len(samples))
    window = np.hanning(size)
    starts = np.unique(
        np.linspace(0, len(samples) - size, min(128, 1 + len(samples) // size), dtype=int)
    )
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
        "crest_db": db_ratio(peak * peak, rms * rms),
        "lr_correlation": float(np.clip(correlation, -1, 1)) if correlation is not None else None,
        "side_over_mid_db": side_over_mid_db,
        "lr_total_power": total_power,
        "bands": bands_out,
    }


def load_wav_pcm16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as reader:
        channels = reader.getnchannels()
        rate = reader.getframerate()
        sampwidth = reader.getsampwidth()
        if (channels, rate, sampwidth) != (2, RATE, 2):
            raise ValueError(f"Expected 48kHz 16-bit stereo WAV, got ch={channels}, rate={rate}, width={sampwidth}")
        nframes = reader.getnframes()
        data = reader.readframes(nframes)
        return np.frombuffer(data, dtype="<i2").reshape(-1, 2)


def load_wav_samples(path: Path) -> np.ndarray:
    return load_wav_pcm16(path).astype(np.float64) / 32768.0


def find_exact_prime_alignment(full_pcm: np.ndarray, prime_pcm: np.ndarray, fade_frames: int = 2048) -> dict:
    """Locate a renderer prime by its exact, unfaded PCM16 interior."""
    if len(prime_pcm) <= 2 * fade_frames + 32 or len(prime_pcm) > len(full_pcm):
        raise ValueError("Prime has no checkable unfaded interior")
    anchor = fade_frames + 64
    candidates = np.flatnonzero(np.all(full_pcm == prime_pcm[anchor], axis=1))
    matches = []
    for position in candidates:
        start = int(position) - anchor
        if start < 0 or start + len(prime_pcm) > len(full_pcm):
            continue
        if not np.array_equal(full_pcm[start + anchor : start + anchor + 32], prime_pcm[anchor : anchor + 32]):
            continue
        if np.array_equal(full_pcm[start + fade_frames : start + len(prime_pcm) - fade_frames],
                          prime_pcm[fade_frames : -fade_frames]):
            matches.append(start)
    if len(matches) != 1:
        raise ValueError(f"Expected one exact PCM interior alignment, found {len(matches)}")
    start = matches[0]
    return {
        "method": "unique exact PCM16 stereo match of all samples except 2048 faded frames per edge",
        "start_frame": start,
        "end_frame_exclusive": start + len(prime_pcm),
        "start_sec": start / RATE,
        "end_sec": (start + len(prime_pcm)) / RATE,
    }


def run_experiment_e1(full_wav_path: Path, prime_wav_path: Path) -> dict:
    """E1: Slice full 600s audio into non-overlapping 60s windows and compare to prime."""
    print("Running Experiment E1: Full-Run Per-Interval Audio Audit...")
    full_pcm = load_wav_pcm16(full_wav_path)
    prime_pcm = load_wav_pcm16(prime_wav_path)

    total_duration_sec = len(full_pcm) / RATE
    window_sec = 60.0
    window_frames = int(window_sec * RATE)
    num_windows = math.ceil(len(full_pcm) / window_frames)

    prime_metrics = compute_audio_metrics(prime_pcm.astype(np.float64) / 32768.0)
    prime_alignment = find_exact_prime_alignment(full_pcm, prime_pcm)

    window_results = []
    for w in range(num_windows):
        start_frame = w * window_frames
        end_frame = min(start_frame + window_frames, len(full_pcm))
        sub_samples = full_pcm[start_frame:end_frame].astype(np.float64) / 32768.0
        m = compute_audio_metrics(sub_samples)
        m["window_index"] = w
        m["start_sec"] = w * window_sec
        m["end_sec"] = end_frame / RATE
        m["frame_count"] = end_frame - start_frame
        m["partial_window"] = end_frame - start_frame < window_frames
        window_results.append(m)

    correlations = [w["lr_correlation"] for w in window_results if w["lr_correlation"] is not None]
    side_mids = [w["side_over_mid_db"] for w in window_results if w["side_over_mid_db"] is not None]
    mid_2k_6k_fracs = [w["bands"][2]["lr_fraction"] for w in window_results]

    summary = {
        "full_wav": str(full_wav_path),
        "full_wav_sha256": sha256_file(full_wav_path),
        "prime_wav": str(prime_wav_path),
        "prime_wav_sha256": sha256_file(prime_wav_path),
        "total_duration_sec": total_duration_sec,
        "total_frames": len(full_pcm),
        "num_windows": num_windows,
        "prime_alignment": prime_alignment,
        "prime_metrics": {
            "rms": prime_metrics["rms"],
            "peak_dbfs": prime_metrics["peak_dbfs"],
            "lr_correlation": prime_metrics["lr_correlation"],
            "side_over_mid_db": prime_metrics["side_over_mid_db"],
            "bands_fraction": {f"{b['hz'][0]}-{b['hz'][1]}Hz": b["lr_fraction"] for b in prime_metrics["bands"]},
        },
        "windows_distribution": {
            "correlation_min": float(np.min(correlations)),
            "correlation_max": float(np.max(correlations)),
            "correlation_mean": float(np.mean(correlations)),
            "correlation_std": float(np.std(correlations)),
            "correlation_quartiles": [float(np.percentile(correlations, p)) for p in [25, 50, 75]],
            "side_over_mid_min_db": float(np.min(side_mids)),
            "side_over_mid_max_db": float(np.max(side_mids)),
            "side_over_mid_mean_db": float(np.mean(side_mids)),
            "band_2k_6k_fraction_min": float(np.min(mid_2k_6k_fracs)),
            "band_2k_6k_fraction_max": float(np.max(mid_2k_6k_fracs)),
            "band_2k_6k_fraction_mean": float(np.mean(mid_2k_6k_fracs)),
        },
        "windows": window_results,
    }
    return summary


def validate_run_lineage(full_wav_path: Path, metadata_path: Path, trace_path: Path) -> dict:
    """Fail closed when overwrite-prone metadata/trace cannot identify this WAV."""
    with wave.open(str(full_wav_path), "rb") as reader:
        audio_frames = reader.getnframes()
    reasons = []
    metadata = None
    rows = []
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, ValueError) as exc:
        reasons.append(f"Run metadata unavailable: {exc}")
    try:
        with trace_path.open("r", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        reasons.append(f"Trace unavailable: {exc}")

    if metadata is not None:
        run = metadata.get("run", {})
        expected_frames = run.get("completed_chunks", -1) * 4096
        if expected_frames != audio_frames:
            reasons.append(
                f"metadata completed_chunks imply {expected_frames} frames, but hashed WAV has {audio_frames} frames"
            )
        if abs(run.get("rendered_seconds", -1) - audio_frames / RATE) > 4096 / RATE:
            reasons.append("metadata rendered_seconds differs from hashed WAV duration")
        if metadata.get("audio_file_hash") not in full_wav_path.stem:
            reasons.append("metadata audio_file_hash does not occur in hashed WAV filename")
        output_audio = metadata.get("outputs", {}).get("audio")
        if not output_audio or Path(output_audio).name != full_wav_path.name:
            reasons.append("metadata output audio basename differs from hashed WAV")
        if not rows:
            reasons.append("trace has no rows")
        else:
            run_ids = {row.get("run_id") for row in rows}
            if run_ids != {metadata.get("run_id")}:
                reasons.append("trace run_id differs from metadata run_id")
            steps = [int(row["step"]) for row in rows]
            if steps[0] != run.get("start_global_step") or steps[-1] >= run.get("end_global_step", -1):
                reasons.append("trace step range differs from metadata run step range")
            stride = metadata.get("telemetry", {}).get("trace_stride_chunks")
            if not isinstance(stride, int) or stride < 1 or len(rows) != math.ceil(run.get("completed_chunks", 0) / stride):
                reasons.append("trace row count differs from metadata completed chunks and stride")

    return {
        "status": "matched" if not reasons else "unavailable",
        "reasons": reasons,
        "audio_frames": audio_frames,
        "audio_seconds": audio_frames / RATE,
        "metadata_path": str(metadata_path),
        "metadata_run_id": metadata.get("run_id") if metadata else None,
        "metadata_completed_chunks": metadata.get("run", {}).get("completed_chunks") if metadata else None,
        "trace_path": str(trace_path),
        "trace_sha256": sha256_file(trace_path) if trace_path.exists() else None,
        "trace_rows": len(rows),
        "trace_run_ids": sorted({row.get("run_id", "") for row in rows}),
        "trace_step_range": [int(rows[0]["step"]), int(rows[-1]["step"])] if rows else None,
    }


def run_experiment_e2(trace_path: Path, lineage: dict) -> dict:
    """Report telemetry only if it belongs to the hashed audio run."""
    if lineage["status"] != "matched":
        return {"status": "unavailable", "reason": "trace/run metadata do not match the hashed WAV", "lineage_reasons": lineage["reasons"]}
    with trace_path.open("r", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    target_counts = {}
    for row in rows:
        name = row["target_file"]
        target_counts[name] = target_counts.get(name, 0) + 1
    def stats(key: str) -> dict:
        values = np.array([float(row[key]) for row in rows])
        return {"min": float(values.min()), "max": float(values.max()), "mean": float(values.mean())}
    return {
        "status": "descriptive_telemetry_only",
        "trace_file": str(trace_path),
        "sampled_trace_rows": len(rows),
        "target_file_distribution_at_sampled_steps": target_counts,
        "width_control": stats("decoder_width_control"),
        "decoder_correlation": stats("decoder_stereo_corr"),
        "target_correlation": stats("target_stereo_corr"),
        "limitation": "Sampled register values do not identify the cause of a waveform feature.",
    }


def run_experiment_e3(lineage: dict) -> dict:
    """A sampled trace cannot replay exact per-chunk prime selection scores."""
    reasons = list(lineage["reasons"])
    reasons.append("exact per-chunk score stream is unavailable; trace is sampled every 10 chunks")
    reasons.append("prior reconstruction omitted the renderer score's observation-width term")
    return {"status": "unavailable", "reasons": reasons}


def run_experiment_e4(prime_wav_path: Path) -> dict:
    """Simulate THD on pure test tones; do not attribute real audio harshness."""
    print("Running Experiment E4: 1-kHz Test-Tone Saturation Simulation...")
    samples = load_wav_samples(prime_wav_path)
    # Simulate tanh(x * 0.92) vs linear
    # What harmonic distortion does tanh(x * 0.92) introduce on test sinusoids?
    test_freq = 1000.0  # 1 kHz
    t = np.arange(RATE * 1) / RATE  # 1 sec
    amplitudes = [0.1, 0.3, 0.5, 0.7, 0.9]
    thd_results = []
    for amp in amplitudes:
        sine = amp * np.sin(2 * np.pi * test_freq * t)
        distorted = np.tanh(sine * 0.92)
        # FFT to measure THD (total harmonic distortion)
        spec = np.abs(np.fft.rfft(distorted * np.hanning(len(distorted))))
        freqs = np.fft.rfftfreq(len(distorted), 1 / RATE)
        fund_idx = np.argmin(np.abs(freqs - test_freq))
        fund_power = spec[fund_idx] ** 2
        # Harmonics at 2k, 3k, 4k, 5k...
        harm_power = 0.0
        for h in range(2, 10):
            h_idx = np.argmin(np.abs(freqs - h * test_freq))
            harm_power += spec[h_idx] ** 2
        thd = float(np.sqrt(harm_power / fund_power)) if fund_power > 0 else 0.0
        thd_db = 20 * math.log10(thd) if thd > 0 else -120.0
        thd_results.append({
            "amplitude": amp,
            "thd_ratio": thd,
            "thd_percent": thd * 100.0,
            "thd_db": thd_db,
        })

    # Crest factor on actual prime
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(samples ** 2)))
    crest_db = 20 * math.log10(peak / rms)

    return {
        "status": "simulation_only",
        "test_signal": "one second, 1 kHz pure sine at each listed pre-tanh amplitude",
        "nonlinearity": "tanh(0.92 * test_signal)",
        "limitation": "The pre-tanh renderer waveform was not saved. These test-tone THD values cannot measure actual render distortion or identify the cause of perceived harshness.",
        "prime_actual_peak": peak,
        "prime_actual_rms": rms,
        "prime_actual_crest_db": crest_db,
        "simulated_tanh_0_92_thd": thd_results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-wav", type=Path, default=Path("/sdcard/Download/rust_ecosystem_out_v9-long-01_9bcf582d4b13.wav"))
    parser.add_argument("--prime-wav", type=Path, default=Path("/sdcard/Download/titan_prime_60s_v9-long-01_9bcf582d4b13.wav"))
    parser.add_argument("--trace", type=Path, default=Path("/sdcard/Download/uncertainty_trace_rust_v9-long-01.csv"))
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "analysis/audio_center_harshness_20260926/diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    parser.add_argument("--run-metadata", type=Path, default=root / "analysis/audio_center_harshness_20260926/metadata_v9_9bcf582d4b13.json")
    parser.add_argument("--output", type=Path, default=out_dir / "diagnostic_report.corrected.json")
    args = parser.parse_args()

    e1_res = run_experiment_e1(args.full_wav, args.prime_wav)
    lineage = validate_run_lineage(args.full_wav, args.run_metadata, args.trace)
    e2_res = run_experiment_e2(args.trace, lineage)
    e3_res = run_experiment_e3(lineage)
    e4_res = run_experiment_e4(args.prime_wav)

    report = {
        "schema_version": 2,
        "supersedes": "diagnostic_report.json and RESEARCH_DIAGNOSTIC_FINDINGS.md",
        "run_lineage": lineage,
        "experiment_e1_full_run_intervals": e1_res,
        "experiment_e2_telemetry_trace": e2_res,
        "experiment_e3_prime_selection_sensitivity": e3_res,
        "experiment_e4_post_mastering_distortion": e4_res,
    }

    out_file = args.output
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2))
    print(f"\nAll diagnostic experiments completed successfully! Results written to: {out_file}")


if __name__ == "__main__":
    main()
