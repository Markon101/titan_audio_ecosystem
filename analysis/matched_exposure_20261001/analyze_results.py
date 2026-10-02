#!/usr/bin/env python3
"""Analyze six fixed-schedule forks using the frozen regime calibration."""

import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIOR = ROOT / "analysis/metastable_20261001"
sys.path.insert(0, str(PRIOR))
from checkpoint_geometry import SafeTensorStore  # noqa: E402
from compare_capture_views import summarize as view_summary  # noqa: E402
from compare_corpus_arms import audio_summary, median_field, uncertainty_summary  # noqa: E402
from compare_field_checkpoints import compare as field_compare, load as field_load  # noqa: E402
from compare_weight_checkpoints import compare as weight_compare  # noqa: E402

TAG = "v10-msfield-fresh-20260930-02"


def target_rows(path):
    with gzip.open(path, "rt") as source:
        rows = list(csv.DictReader(source))
    sequence = [(int(row["step"]), row["target_file"],
                 int(row["target_frame"]), int(row["target_chunks_left"]))
                for row in rows]
    return rows, sequence


def trace_summary(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != 256:
        raise RuntimeError(f"expected 256 regime samples: {path}: {len(rows)}")
    second = rows[128:]
    return {
        "confirmed_regimes": sum(row["persistent_new_regime"] for row in rows),
        "confirmed_second_half": sum(row["persistent_new_regime"] for row in second),
        "candidate_regions": sum(row["candidate_regime_created"] for row in rows),
        "max_dwell_chunks": max(row["regime_dwell_chunks"] for row in rows),
        "transitions": rows[-1]["transition_count"],
        "revisits": rows[-1]["revisit_count"],
        "over_resident_samples": sum(row["residence_state"] == "over_resident" for row in rows),
        "median_participation": median_field(rows, "trajectory_participation_ratio"),
        "second_half_participation": median_field(second, "trajectory_participation_ratio"),
        "median_health": median_field(rows, "health"),
    }


def sha_text(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def main():
    campaign = json.loads((HERE / "matched_run_receipts.json").read_text())
    packaged = json.loads((HERE / "package_receipt.json").read_text())
    if len(campaign["runs"]) != 6 or set(campaign["runs"]) != set(packaged["runs"]):
        raise RuntimeError("six completed and packaged runs are required")
    calibration = PRIOR / "regime_calibration.json"
    source_model = SafeTensorStore(PRIOR / "frozen_step_58107/titan_model_v10_msfield_v10-msfield-fresh-20260930-02.safetensors")
    source_step, source_field = field_load(PRIOR / "runs/world_export_step_58107")
    if source_step != 58107:
        raise RuntimeError("source field step changed")
    report = {"schema": 1, "source_step": source_step, "schedule_seeds": {},
              "scope": "two exogenous schedule seeds on one mature model/world, not independent model seeds"}
    for seed in (20261002, 20261003):
        arms = {}
        sequences = {}
        for arm in "acb":
            key = f"{seed}_{arm}"
            run = HERE / f"runs/seed_{key}"
            metadata = json.loads((HERE / f"{key}_metadata.json").read_text())
            capture = HERE / f"{key}_capture.jsonl.gz"
            state = HERE / f"{key}_archive_state.json"
            trace = HERE / f"{key}_openended_trace.jsonl"
            graph = HERE / f"{key}_regime_graph.json"
            if state.exists() or trace.exists() or graph.exists():
                raise RuntimeError(f"analysis output exists: {key}")
            subprocess.run(["python3", str(PRIOR / "regime_archive.py"), "analyze",
                            str(capture), "--calibration", str(calibration),
                            "--state-out", str(state), "--trace-out", str(trace),
                            "--graph-out", str(graph)], cwd=ROOT, check=True, capture_output=True)
            subprocess.run(["dot", "-Tsvg", str(graph.with_suffix(".dot")),
                            "-o", str(graph.with_suffix(".svg"))], check=True)
            uncertainty, sequence = target_rows(HERE / f"{key}_uncertainty_trace.csv.gz")
            sequences[arm] = sequence
            export = run / "field_export"
            subprocess.run([str(ROOT / "target/release/titan"), "--export-msfield-world",
                            str(run / f"titan_world_v10_msfield_{TAG}.bin"), str(export)],
                           cwd=ROOT, check=True, capture_output=True)
            final_step, final_field = field_load(export)
            if final_step != 59131:
                raise RuntimeError(f"wrong final field step: {key}")
            model = SafeTensorStore(run / f"titan_model_v10_msfield_{TAG}.safetensors")
            arms[arm] = {
                "run_id": metadata["run_id"],
                "regimes": trace_summary(trace),
                "view_geometry": view_summary([capture], seed + ord(arm)),
                "fixed_probe_and_health": uncertainty_summary(uncertainty),
                "weight_change": weight_compare(source_model, model),
                "field_change": field_compare(source_field, final_field),
                "motifs_active_final": metadata["final_state"]["motifs_active"],
                "prime_audio": audio_summary(Path(metadata["outputs"]["prime"])),
                "resource_guard": json.loads((HERE / f"{key}_resource.json").read_text()),
            }
        if not (sequences["a"] == sequences["c"] == sequences["b"]):
            raise RuntimeError(f"target schedules diverged for seed {seed}")
        report["schedule_seeds"][str(seed)] = {
            "target_trace_rows": len(sequences["a"]),
            "target_trace_sha256": sha_text(sequences["a"]),
            "matched_target_schedule": True,
            "arms": arms,
        }
        print(f"analyzed schedule seed {seed}", flush=True)
    report["directional_checks"] = {}
    for seed, panel in report["schedule_seeds"].items():
        arms = panel["arms"]
        field_pr = lambda item: sum(item["view_geometry"]["views"][f"field_{scale}"]["first_difference_participation"]
                                    for scale in ("fine", "meso", "coarse")) / 3
        b, controls = arms["b"], (arms["a"], arms["c"])
        report["directional_checks"][seed] = {
            "b_more_persistent_than_both": all(b["regimes"]["confirmed_regimes"] > c["regimes"]["confirmed_regimes"] for c in controls),
            "b_more_field_dimension_than_both": all(field_pr(b) > field_pr(c) for c in controls),
            "b_field_pr": field_pr(b),
            "control_field_pr": [field_pr(c) for c in controls],
            "b_clipped_trace_fraction": b["fixed_probe_and_health"]["clipped_optimizer_trace_fraction"],
        }
    report["both_seeds_show_organized_expansion"] = all(
        item["b_more_persistent_than_both"] and item["b_more_field_dimension_than_both"]
        for item in report["directional_checks"].values())
    (HERE / "matched_exposure_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"both_seeds_show_organized_expansion": report["both_seeds_show_organized_expansion"],
        "confirmed": {seed: {arm: item["regimes"]["confirmed_regimes"] for arm, item in panel["arms"].items()}
                      for seed, panel in report["schedule_seeds"].items()}}))


if __name__ == "__main__":
    main()
