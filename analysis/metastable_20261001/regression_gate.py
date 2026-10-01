#!/usr/bin/env python3
"""Run matched one-chunk old/off/on continuations from frozen forks."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from checkpoint_geometry import SafeTensorStore

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TAG = "v10-msfield-fresh-20260930-02"
STEMS = ("titan_model_v10_msfield", "titan_world_v10_msfield",
         "titan_optimizer_v10_msfield")
EXTS = ("safetensors", "bin", "safetensors")


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def max_tensor_difference(first, second):
    a, b = SafeTensorStore(first), SafeTensorStore(second)
    if set(a.header) != set(b.header):
        raise RuntimeError("checkpoint tensor names differ")
    largest = 0.0
    changed = 0
    for name in a.header:
        if name == "__metadata__":
            continue
        left, right = a.array(name), b.array(name)
        if left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError(f"checkpoint tensor schema differs: {name}")
        if not np.array_equal(left, right):
            changed += 1
            largest = max(largest, float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))))
    return {"changed_tensors": changed, "max_abs_difference": largest}


def main():
    # Android's shared /sdcard mount is noexec. Preserve exact bytes in the
    # workspace before executing the old binary.
    source_binary = Path("/sdcard/Download/TITAN_v10_msfield_fresh_20260930_02_cont/titan_run_binary")
    legacy_binary = HERE / "runs" / "titan_legacy_72a4d69"
    if not legacy_binary.exists():
        shutil.copyfile(source_binary, legacy_binary)
        legacy_binary.chmod(0o700)
    if sha(legacy_binary) != sha(source_binary):
        raise RuntimeError("legacy executable copy hash mismatch")
    results = {}
    for label, binary, flag in [
        ("old", legacy_binary, []),
        ("new_off", ROOT / "target/release/titan", []),
        ("new_on", ROOT / "target/release/titan", ["--regime-capture", "--regime-stride", "1"]),
    ]:
        base = HERE / "runs" / f"regression_{label}"
        metadata = base / f"titan_run_metadata_v10_msfield_{TAG}.json"
        prior_step = json.loads(metadata.read_text())["run"]["end_global_step"]
        if prior_step not in (58107, 58108):
            raise RuntimeError(f"{label} is not a step-58107/58108 fork")
        args = [str(binary), "--base-dir", str(base),
                "--corpus-dir", "/sdcard/Download/OLD_WAVS",
                "--corpus-manifest", "/sdcard/Download/titan_corpus_manifest_v7_sml.json",
                "--substrate", "msfield", "--run-tag", TAG,
                "--duration", "0.09", "--threads", "4", "--seed", "44", "--lr", "0.00045",
                "--bptt", "64", "--core-update-every", "1", "--morph-layers", "16",
                "--morph-width", "512", "--max-morph-depth", "16", "--motif-capacity", "512"] + flag
        if prior_step == 58107:
            log_path = base / "regression_run.log"
            if log_path.exists() and log_path.stat().st_size == 0:
                log_path.unlink()
            with log_path.open("x") as log:
                completed = subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError(f"{label} failed with exit {completed.returncode}; see its log")
        finished = json.loads(metadata.read_text())
        if finished["run"]["start_global_step"] != 58107 or finished["run"]["end_global_step"] != 58108:
            raise RuntimeError(f"{label} did not advance exactly one chunk")
        artifacts = {stem: sha(base / f"{stem}_{TAG}.{ext}") for stem, ext in zip(STEMS, EXTS)}
        artifacts["audio"] = sha(finished["outputs"]["audio"])
        results[label] = {"binary_sha256": sha(binary), "argv": args,
                          "artifact_sha256": artifacts, "run_id": finished["run_id"],
                          "optimizer_updates": finished["run"]["optimizer_updates_cumulative"]}
    expected = results["old"]["artifact_sha256"]
    equality = {label: {name: value == expected[name] for name, value in result["artifact_sha256"].items()}
                for label, result in results.items() if label != "old"}
    report = {"schema": 1, "source_step": 58107, "end_step": 58108,
              "runs": results, "byte_equal_to_old": equality,
              "all_equal": all(all(items.values()) for items in equality.values())}
    numeric = {}
    for label in ("new_off", "new_on"):
        numeric[label] = {}
        for stem, ext in zip(STEMS, EXTS):
            if ext == "safetensors":
                numeric[label][stem] = max_tensor_difference(
                    HERE / "runs/regression_old" / f"{stem}_{TAG}.{ext}",
                    HERE / "runs" / f"regression_{label}" / f"{stem}_{TAG}.{ext}")
    report["tensor_differences"] = numeric
    report["disabled_equivalent_within_tolerance"] = all(
        equality[label]["audio"] and equality[label]["titan_world_v10_msfield"] and
        all(item["max_abs_difference"] <= 1e-7 for item in numeric[label].values())
        for label in ("new_off", "new_on"))
    (HERE / "regression_gate_result.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"all_equal": report["all_equal"],
                      "equivalent_within_tolerance": report["disabled_equivalent_within_tolerance"],
                      "byte_equal_to_old": equality, "tensor_differences": numeric}))
    if not report["disabled_equivalent_within_tolerance"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
