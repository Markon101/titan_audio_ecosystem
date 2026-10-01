#!/usr/bin/env python3
"""Compare per-view trajectory dimensions across matched observer captures."""

import argparse
import json
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np

from checkpoint_geometry import covariance_pr, detrend
from regime_archive import read_capture, sha256


def summarize(paths, seed):
    rows = list(read_capture(paths))
    rng = np.random.default_rng(seed)
    views = {}
    for name in sorted(rows[0]["views"]):
        x = np.asarray([row["views"][name] for row in rows], dtype=np.float64)
        null = [covariance_pr(np.column_stack([rng.permutation(x[:, i])
                                                for i in range(x.shape[1])])) for _ in range(4)]
        views[name] = {
            "samples": len(x), "dimensions": x.shape[1],
            "covariance_participation": covariance_pr(x),
            "first_difference_participation": covariance_pr(np.diff(x, axis=0)),
            "detrended_participation": covariance_pr(detrend(x)),
            "independent_channel_shuffle_null_pr": null,
            "median_channel_std": float(np.median(x.std(axis=0))),
            "first_to_last_quarter_centroid_distance": float(np.linalg.norm(
                x[:max(1, len(x) // 4)].mean(axis=0) -
                x[-max(1, len(x) // 4):].mean(axis=0)) / np.sqrt(x.shape[1])),
        }
    return {"capture_paths": [str(p) for p in paths],
            "capture_sha256": [sha256(p) for p in paths],
            "first_step": rows[0]["global_step"], "last_step": rows[-1]["global_step"],
            "views": views}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", nargs="+", required=True)
    parser.add_argument("--b", nargs="+", required=True)
    parser.add_argument("--c", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    arms = {label: summarize(getattr(args, label), 6001 + i)
            for i, label in enumerate(("a", "b", "c"))}
    report = {"schema": 1, "arms": arms,
              "scope": "raw orthogonal sketches/pooled maps; dimension proxies during training, not proof of full-space capacity"}
    with open(args.out, "w") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({label: {name: round(item["first_difference_participation"], 2)
                              for name, item in arm["views"].items()
                              if name in ("field_fine", "field_meso", "field_coarse", "gru", "morphic_l16", "audio_behavior")}
                      for label, arm in arms.items()}))


if __name__ == "__main__":
    main()
