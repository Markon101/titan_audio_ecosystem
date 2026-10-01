#!/usr/bin/env python3
"""Compare matched corpus-arm parameter changes from the same source model."""

import argparse
import json
from pathlib import Path

import numpy as np

from checkpoint_geometry import SafeTensorStore, group, matrix_view, spectral_metrics, sha256


def compare(source, target):
    if set(source.header) != set(target.header):
        raise ValueError("model tensor sets differ across corpus arms")
    sums = {}
    matrix_metrics = {}
    for name in source.header:
        if name == "__metadata__":
            continue
        before = np.asarray(source.array(name), dtype=np.float64)
        after = np.asarray(target.array(name), dtype=np.float64)
        delta = after - before
        g = group(name)
        item = sums.setdefault(g, {"weight_sq": 0.0, "delta_sq": 0.0, "parameters": 0})
        item["weight_sq"] += float(np.sum(before * before))
        item["delta_sq"] += float(np.sum(delta * delta))
        item["parameters"] += int(before.size)
        matrix = matrix_view(delta)
        if matrix is not None and min(matrix.shape) >= 8:
            matrix_metrics[name] = {"group": g,
                                    "delta_participation_rank": spectral_metrics(matrix)["participation_rank"]}
    groups = {g: {"parameters": item["parameters"],
                  "relative_l2_delta": float(np.sqrt(item["delta_sq"] / max(item["weight_sq"], 1e-24)))}
              for g, item in sums.items()}
    return {"groups": groups, "matrices": matrix_metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--a", required=True, type=Path)
    parser.add_argument("--b", required=True, type=Path)
    parser.add_argument("--c", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    source = SafeTensorStore(args.source)
    arms = {}
    for label, path in (("a", args.a), ("b", args.b), ("c", args.c)):
        arms[label] = {"model_sha256": sha256(path), **compare(source, SafeTensorStore(path))}
    report = {"schema": 1, "source_model_sha256": sha256(args.source),
              "arms": arms,
              "scope": "net parameter deltas over matched budgets; matrix delta rank is not gradient covariance rank"}
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({label: {g: round(item["relative_l2_delta"], 4)
                              for g, item in arm["groups"].items() if g in ("gru", "msfield", "temporal_decoder")}
                      for label, arm in arms.items()}))


if __name__ == "__main__":
    main()
