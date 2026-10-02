#!/usr/bin/env python3
"""One-chunk oracle for the fixed-schedule feature's disabled path."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PRIOR = ROOT / "analysis/metastable_20261001"
sys.path.insert(0, str(PRIOR))
from checkpoint_geometry import SafeTensorStore  # noqa: E402

TAG = "v10-msfield-fresh-20260930-02"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_max_difference(old, new):
    left, right = SafeTensorStore(old), SafeTensorStore(new)
    if set(left.header) != set(right.header):
        raise ValueError("checkpoint tensor names differ")
    maximum = 0.0
    for name in left.header:
        if name == "__metadata__":
            continue
        a, b = left.array(name), right.array(name)
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"checkpoint tensor shape/dtype differs: {name}")
        maximum = max(maximum, float(np.max(np.abs(a.astype(float) - b.astype(float)))))
    return maximum


def main():
    target = HERE / "runs/disabled_new"
    subprocess.run(["python3", str(PRIOR / "fork_checkpoint.py"),
                    str(PRIOR / "frozen_step_58107"), str(target)], cwd=ROOT, check=True,
                   capture_output=True)
    binary = ROOT / "target/release/titan"
    args = [str(binary), "--base-dir", str(target),
            "--corpus-dir", "/sdcard/Download/OLD_WAVS",
            "--corpus-manifest", "/sdcard/Download/titan_corpus_manifest_v7_sml.json",
            "--substrate", "msfield", "--run-tag", TAG,
            "--duration", "0.09", "--threads", "4", "--seed", "44", "--lr", "0.00045",
            "--bptt", "64", "--core-update-every", "1", "--morph-layers", "16",
            "--morph-width", "512", "--max-morph-depth", "16", "--motif-capacity", "512"]
    with (target / "run.log").open("xb") as log:
        subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    metadata = json.loads((target / f"titan_run_metadata_v10_msfield_{TAG}.json").read_text())
    if (metadata["run"]["start_global_step"], metadata["run"]["end_global_step"]) != (58107, 58108):
        raise RuntimeError("new binary did not advance exactly one chunk")
    reference = json.loads((PRIOR / "regression_gate_result.json").read_text())["runs"]["old"]["artifact_sha256"]
    old_dir = PRIOR / "runs/regression_old"
    output = {"schema": 1, "new_binary_sha256": sha(binary),
              "old_binary_sha256": json.loads((PRIOR / "regression_gate_result.json").read_text())["runs"]["old"]["binary_sha256"],
              "world_byte_equal": sha(target / f"titan_world_v10_msfield_{TAG}.bin") == reference["titan_world_v10_msfield"],
              "audio_byte_equal": sha(metadata["outputs"]["audio"]) == reference["audio"],
              "model_max_abs_difference": tensor_max_difference(
                  old_dir / f"titan_model_v10_msfield_{TAG}.safetensors",
                  target / f"titan_model_v10_msfield_{TAG}.safetensors"),
              "optimizer_max_abs_difference": tensor_max_difference(
                  old_dir / f"titan_optimizer_v10_msfield_{TAG}.safetensors",
                  target / f"titan_optimizer_v10_msfield_{TAG}.safetensors")}
    output["passed"] = (output["world_byte_equal"] and output["audio_byte_equal"] and
                        output["model_max_abs_difference"] <= 1e-7 and
                        output["optimizer_max_abs_difference"] <= 1e-7)
    (HERE / "disabled_path_gate.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output))
    if not output["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
