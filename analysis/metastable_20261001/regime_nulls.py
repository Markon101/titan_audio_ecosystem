#!/usr/bin/env python3
"""Test whether regime counts can be inflated by time shuffling or noise."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from regime_archive import Archive, read_capture


def summarize(rows, calibration):
    archive = Archive(calibration)
    outputs = [archive.process(row) for row in rows]
    dwells = [row["regime_dwell_samples"] for row in outputs if row["regime_id"] is not None]
    dims = [row["trajectory_participation_ratio"] for row in outputs
            if row["trajectory_participation_ratio"] is not None]
    persistent = sum(row["persistent_new_regime"] for row in outputs)
    return {"persistent_regimes": persistent, "archive_size": len(archive.state["regimes"]),
            "transitions": archive.state["transition_count"],
            "median_dwell_samples": float(np.median(dwells)) if dwells else None,
            "median_trajectory_participation": float(np.median(dims)) if dims else None,
            "over_resident_samples": sum(row["residence_state"] == "over_resident" for row in outputs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=69117)
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 16:
        parser.error("--repeats must be 1..16")
    rows = list(read_capture([args.capture]))
    if len(rows) < 32:
        parser.error("null comparison needs at least 32 captured samples")
    cal = json.loads(args.calibration.read_text())
    rng = np.random.default_rng(args.seed)
    result = {"schema": 1, "seed": args.seed, "samples": len(rows),
              "repeats_per_null": args.repeats,
              "observed": summarize(rows, cal), "nulls": {
                  "independent_view_time_shuffle": [],
                  "four_sample_block_shuffle": [],
                  "matched_gaussian_white_noise": []}}
    # Each view is independently permuted in time, preserving its marginal
    # distribution while breaking cross-view temporal coherence.
    blocks = [rows[i:i + 4] for i in range(0, len(rows), 4)]
    for _ in range(args.repeats):
        independent = copy.deepcopy(rows)
        for name in rows[0]["views"]:
            order = rng.permutation(len(rows))
            for i, row in enumerate(independent):
                row["views"][name] = rows[int(order[i])]["views"][name]
        result["nulls"]["independent_view_time_shuffle"].append(summarize(independent, cal))
        # Preserve four-sample local structure but destroy long-range order.
        order = rng.permutation(len(blocks))
        shuffled = copy.deepcopy(rows)
        source = [item for i in order for item in blocks[int(i)]]
        for target, origin in zip(shuffled, source):
            target["views"] = origin["views"]
        result["nulls"]["four_sample_block_shuffle"].append(summarize(shuffled, cal))
        gaussian = copy.deepcopy(rows)
        for name in rows[0]["views"]:
            values = np.asarray([r["views"][name] for r in rows])
            surrogate = rng.normal(values.mean(axis=0), values.std(axis=0), size=values.shape)
            for i, row in enumerate(gaussian):
                row["views"][name] = surrogate[i].tolist()
        result["nulls"]["matched_gaussian_white_noise"].append(summarize(gaussian, cal))
    observed = result["observed"]["persistent_regimes"]
    result["regime_count_noise_gameable"] = any(
        null["persistent_regimes"] >= observed
        for distribution in result["nulls"].values() for null in distribution)
    block_dwells = [item["median_dwell_samples"] for item in result["nulls"]["four_sample_block_shuffle"]
                    if item["median_dwell_samples"] is not None]
    result["observed_dwell_exceeds_block_null_max"] = bool(block_dwells) and (
        result["observed"]["median_dwell_samples"] > max(block_dwells))
    result["interpretation"] = (
        "A null matching or exceeding observed persistent regime count blocks a discovery claim. "
        "This is one trajectory; surrogate repeats test descriptor gaming, not seed generalization."
    )
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"observed": observed, "null_count_ranges": {
        name: [min(item["persistent_regimes"] for item in data),
               max(item["persistent_regimes"] for item in data)]
        for name, data in result["nulls"].items()},
        "gameable": result["regime_count_noise_gameable"],
        "dwell_exceeds_block_null_max": result["observed_dwell_exceeds_block_null_max"]}))


if __name__ == "__main__":
    main()
