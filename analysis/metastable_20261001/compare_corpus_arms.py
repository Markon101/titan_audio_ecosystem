#!/usr/bin/env python3
"""Summarize preregistered A/B/C corpus arms with exposure and probe gates."""

import csv
import gzip
import json
from pathlib import Path
import statistics
import wave

import numpy as np

HERE = Path(__file__).resolve().parent
TAG = "v10-msfield-fresh-20260930-02"
RUNS = {
    "a": HERE / "runs/observer_step_58107",
    "b": HERE / "runs/corpus_new_family_run",
    "c": HERE / "runs/corpus_existing_variant_run",
}
TRACE = {key: HERE / f"corpus_{key}_openended_trace.jsonl" for key in RUNS}
TRACE["a"] = HERE / "combined_openended_trace.jsonl"


def load_csv(path):
    with gzip.open(path, "rt") as stream:
        return list(csv.DictReader(stream))


def median_field(rows, key):
    values = []
    for row in rows:
        try:
            value = float(row[key])
            if np.isfinite(value):
                values.append(value)
        except (KeyError, TypeError, ValueError):
            pass
    return float(np.median(values)) if values else None


def trace_summary(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) < 500:
        raise ValueError(f"arm trace too short: {path}")
    second = [row for row in rows if row["global_step"] >= 59513]
    summarize = lambda subset: {
        "samples": len(subset),
        "confirmed_new_regimes": sum(bool(row["persistent_new_regime"]) for row in subset),
        "candidate_regions": sum(bool(row["candidate_regime_created"]) for row in subset),
        "median_trajectory_participation": median_field(subset, "trajectory_participation_ratio"),
        "max_dwell_chunks": max(row["regime_dwell_chunks"] for row in subset),
        "over_resident_samples": sum(row["residence_state"] == "over_resident" for row in subset),
        "median_health": median_field(subset, "health"),
        "median_stagnation": median_field(subset, "stagnation"),
        "last_transition_count": subset[-1]["transition_count"],
        "last_revisit_count": subset[-1]["revisit_count"],
    }
    return {"all": summarize(rows), "second_block": summarize(second)}


def uncertainty_summary(rows):
    keys = ("development_mean_spectral", "development_mean_chroma",
            "validation_mean_spectral", "validation_mean_chroma", "activity_health",
            "stagnation", "grad_norm", "clip_scale", "field_entropy", "stereo_corr",
            "width", "motifs", "motif_stored_total", "motif_rejected_similarity")
    summary = {f"median_{key}": median_field(rows, key) for key in keys}
    summary["trace_rows"] = len(rows)
    summary["clipped_optimizer_trace_fraction"] = sum(
        float(row["clip_scale"]) < 0.999 for row in rows) / max(1, len(rows))
    return summary


def exposure(rows, names):
    matches = [Path(row["target_file"]).name for row in rows
               if Path(row["target_file"]).name in names]
    episodes = 0
    previous = None
    for row in rows:
        current = Path(row["target_file"]).name
        if current != previous and current in names:
            episodes += 1
        previous = current
    return {"unique_target_files": sorted(set(matches)), "unique_count": len(set(matches)),
            "sampled_trace_rows": len(matches), "sampled_episodes": episodes}


def audio_summary(path):
    with wave.open(str(path), "rb") as stream:
        if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) != (48000, 2, 2):
            raise ValueError(f"unexpected prime WAV format: {path}")
        samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").reshape(-1, 2).astype(np.float64) / 32768.0
    left, right = samples[:, 0], samples[:, 1]
    mid, side = 0.5 * (left + right), 0.5 * (left - right)
    sample = mid[:min(len(mid), 48000 * 30)]
    spectrum = np.abs(np.fft.rfft(sample * np.hanning(len(sample))))
    hz = np.fft.rfftfreq(len(sample), 1 / 48000)
    return {"frames": len(samples), "peak": float(np.max(np.abs(samples))),
            "clipped_sample_fraction": float(np.mean(np.abs(samples) >= 32767 / 32768)),
            "mid_rms": float(np.sqrt(np.mean(mid * mid))),
            "side_rms": float(np.sqrt(np.mean(side * side))),
            "side_mid_rms_ratio": float(np.sqrt(np.mean(side * side)) / max(1e-9, np.sqrt(np.mean(mid * mid)))),
            "stereo_correlation": float(np.corrcoef(left, right)[0, 1]),
            "spectral_centroid_hz_first_30s": float(np.sum(hz * spectrum) / max(1e-9, np.sum(spectrum)))}


def main():
    receipt = json.loads((HERE / "corpus_forks_receipt.json").read_text())
    new_names = {item["file"] for item in receipt["new_families"]}
    variant_names = {item["variant_file"] for item in receipt["existing_variants"]}
    arms = {}
    for label, root in RUNS.items():
        metadata = json.loads((root / f"titan_run_metadata_v10_msfield_{TAG}.json").read_text())
        if metadata["run"]["start_global_step"] != 59513 or metadata["run"]["end_global_step"] != 60919:
            raise ValueError(f"arm {label} did not finish matched second block")
        second_rows = load_csv(HERE / f"corpus_{label}_segment2_uncertainty_trace.csv.gz")
        data = {"metadata_run_id": metadata["run_id"],
                "training_families": metadata["corpus"]["training_families"],
                "training_files": metadata["corpus"]["training_files"],
                "validation_is_strict": metadata["corpus"]["validation_is_strict"],
                "optimizer_updates_cumulative": metadata["run"]["optimizer_updates_cumulative"],
                "motifs_active_final": metadata["final_state"]["motifs_active"],
                "regime": trace_summary(TRACE[label]),
                "fixed_probe_and_health": uncertainty_summary(second_rows),
                "prime_audio": audio_summary(Path(metadata["outputs"]["prime"]))}
        if label in ("b", "c"):
            first_rows = load_csv(HERE / f"corpus_{label}_segment1_uncertainty_trace.csv.gz")
            names = new_names if label == "b" else variant_names
            data["added_source_exposure"] = exposure(first_rows + second_rows, names)
        arms[label] = data
    qualified = arms["b"]["added_source_exposure"]["unique_count"] >= 2
    c_qualified = arms["c"]["added_source_exposure"]["unique_count"] >= 2
    result = {"schema": 1, "arms": arms, "b_exposure_qualified": qualified,
              "c_exposure_qualified": c_qualified,
              "bc_comparison_qualified": qualified and c_qualified,
              "scope": "single seed/checkpoint; target schedules differ with manifest; objective audio metrics do not establish beauty or Suno transfer"}
    (HERE / "corpus_contrast_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"b_exposure_qualified": qualified,
                      "c_exposure_qualified": c_qualified,
                      "new_families_sampled": arms["b"]["added_source_exposure"]["unique_count"],
                      "confirmed_regimes": {k: v["regime"]["all"]["confirmed_new_regimes"] for k, v in arms.items()}}))


if __name__ == "__main__":
    main()
