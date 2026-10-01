#!/usr/bin/env python3
"""Summarize the separately preregistered six-family acute exposure arms."""

import json
from pathlib import Path

from compare_corpus_arms import audio_summary, exposure, load_csv, median_field, uncertainty_summary

HERE = Path(__file__).resolve().parent
TAG = "v10-msfield-fresh-20260930-02"


def regime_summary(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not 170 <= len(rows) <= 180:
        raise ValueError(f"acute observer sample count is wrong: {path}: {len(rows)}")
    half = len(rows) // 2
    return {"samples": len(rows),
            "confirmed_regimes": sum(row["persistent_new_regime"] for row in rows),
            "candidate_regions": sum(row["candidate_regime_created"] for row in rows),
            "max_dwell_chunks": max(row["regime_dwell_chunks"] for row in rows),
            "median_trajectory_participation": median_field(rows, "trajectory_participation_ratio"),
            "second_half_median_participation": median_field(rows[half:], "trajectory_participation_ratio"),
            "over_resident_samples": sum(row["residence_state"] == "over_resident" for row in rows),
            "transition_count": rows[-1]["transition_count"],
            "revisit_count": rows[-1]["revisit_count"]}


def main():
    manifest = json.loads((HERE / "acute_exposure_forks_receipt.json").read_text())
    arms = {}
    for label in ("a", "b", "c"):
        run = HERE / f"runs/acute_{label}_run"
        metadata = json.loads((run / f"titan_run_metadata_v10_msfield_{TAG}.json").read_text())
        if metadata["run"]["start_global_step"] != 58107 or metadata["run"]["end_global_step"] != 58810:
            raise ValueError(f"acute arm {label} did not complete 703 chunks")
        sampled = load_csv(HERE / f"acute_{label}_uncertainty_trace.csv.gz")
        train_names = {item["file"] for item in manifest["arms"][label]["train_files"]}
        arms[label] = {
            "run_id": metadata["run_id"],
            "training_families": metadata["corpus"]["training_families"],
            "validation_is_strict": metadata["corpus"]["validation_is_strict"],
            "target_exposure": exposure(sampled, train_names),
            "regime": regime_summary(HERE / f"acute_{label}_openended_trace.jsonl"),
            "fixed_probe_and_health": uncertainty_summary(sampled),
            "motifs_active_final": metadata["final_state"]["motifs_active"],
            "optimizer_updates_cumulative": metadata["run"]["optimizer_updates_cumulative"],
            "prime_audio": audio_summary(Path(metadata["outputs"]["prime"])),
        }
    qualified = all(item["target_exposure"]["unique_count"] >= 2 for item in arms.values())
    result = {"schema": 1, "arms": arms, "exposure_qualified": qualified,
              "scope": "short single-seed acute ecology replacement; distinct source audio and target schedules remain confounds"}
    (HERE / "acute_exposure_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"qualified": qualified,
                      "families_sampled": {label: item["target_exposure"]["unique_count"] for label, item in arms.items()},
                      "confirmed_regimes": {label: item["regime"]["confirmed_regimes"] for label, item in arms.items()}}))


if __name__ == "__main__":
    main()
