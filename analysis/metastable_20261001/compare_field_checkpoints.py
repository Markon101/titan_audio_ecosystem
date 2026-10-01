#!/usr/bin/env python3
"""Compare exact v10 field states across matched corpus forks."""

import argparse
import json
from pathlib import Path

import numpy as np


def load(root):
    root = Path(root)
    meta = json.loads((root / "manifest.json").read_text())
    fields = {}
    for name, shape in meta["shape"].items():
        values = np.fromfile(root / f"{name}.f32le", dtype="<f4")
        if values.size != int(np.prod(shape)):
            raise ValueError(f"field export is malformed: {root}/{name}")
        fields[name] = values.reshape(shape).astype(np.float64)
    return meta["global_step"], fields


def compare(a, b):
    result = {}
    for name in a:
        left, right = a[name], b[name]
        delta = right - left
        norm_left = np.linalg.norm(left)
        norm_right = np.linalg.norm(right)
        map_left = np.sqrt(np.mean(left * left, axis=0)).reshape(-1)
        map_right = np.sqrt(np.mean(right * right, axis=0)).reshape(-1)
        result[name] = {
            "initial_rms": float(np.sqrt(np.mean(left * left))),
            "final_rms": float(np.sqrt(np.mean(right * right))),
            "delta_rms": float(np.sqrt(np.mean(delta * delta))),
            "relative_delta_l2": float(np.linalg.norm(delta) / max(norm_left, 1e-12)),
            "full_state_cosine": float(np.sum(left * right) / max(norm_left * norm_right, 1e-12)),
            "channel_rms_map_correlation": float(np.corrcoef(map_left, map_right)[0, 1]),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--a", required=True, type=Path)
    parser.add_argument("--b", required=True, type=Path)
    parser.add_argument("--c", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    step, source = load(args.source)
    arms = {}
    for label, path in (("a", args.a), ("b", args.b), ("c", args.c)):
        final_step, fields = load(path)
        arms[label] = {"final_step": final_step, "vs_source": compare(source, fields)}
    report = {"schema": 1, "source_step": step, "arms": arms,
              "scope": "descriptive full-state and channel-RMS-map change; different corpus target schedules are not paired"}
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({label: {name: round(data["relative_delta_l2"], 3)
                              for name, data in arm["vs_source"].items()}
                      for label, arm in arms.items()}))


if __name__ == "__main__":
    main()
