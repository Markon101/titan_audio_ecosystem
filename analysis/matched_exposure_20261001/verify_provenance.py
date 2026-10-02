#!/usr/bin/env python3
"""Check frozen parent, aliases, schedules, packaged traces, and six runs."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIOR = ROOT / "analysis/metastable_20261001"
TAG = "v10-msfield-fresh-20260930-02"


def sha(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parent = json.loads((PRIOR / "frozen_step_58107_receipt.json").read_text())
    exposure = json.loads((HERE / "matched_exposure_receipt.json").read_text())
    campaign = json.loads((HERE / "matched_run_receipts.json").read_text())
    package = json.loads((HERE / "package_receipt.json").read_text())
    for item in parent["files"]:
        if sha(item["source"]) != item["sha256"] or sha(PRIOR / "frozen_step_58107" / item["name"]) != item["sha256"]:
            raise RuntimeError(f"frozen parent changed: {item['name']}")
    for arm in "abc":
        directory = HERE / f"runs/corpus/{arm}"
        for item in exposure["arms"][arm]["slots"]:
            if sha(directory / item["alias"]) != item["source_sha256"]:
                raise RuntimeError(f"slot source changed: {arm}:{item['slot']}")
        if sha(HERE / f"runs/corpus/{arm}_manifest.json") != exposure["arms"][arm]["manifest_sha256"]:
            raise RuntimeError(f"manifest changed: {arm}")
        for name, expected in exposure["heldout_sha256"].items():
            if sha(directory / name) != expected:
                raise RuntimeError(f"held-out source changed: {arm}:{name}")
    for seed, item in exposure["schedules"].items():
        if sha(HERE / f"schedule_{seed}.json") != item["sha256"]:
            raise RuntimeError(f"schedule changed: {seed}")
    if set(campaign["runs"]) != set(package["runs"]) or len(campaign["runs"]) != 6:
        raise RuntimeError("run/package inventory mismatch")
    for key, item in campaign["runs"].items():
        run = HERE / f"runs/seed_{key}"
        metadata = run / f"titan_run_metadata_v10_msfield_{TAG}.json"
        for name, stem, ext in (("model", "titan_model_v10_msfield", "safetensors"),
                                ("world", "titan_world_v10_msfield", "bin"),
                                ("optimizer", "titan_optimizer_v10_msfield", "safetensors")):
            if sha(run / f"{stem}_{TAG}.{ext}") != item[f"{name}_sha256"]:
                raise RuntimeError(f"run {key} {name} checkpoint changed")
        if sha(metadata) != item["metadata_sha256"] or sha(run / "titan_run_binary") != item["binary_sha256"]:
            raise RuntimeError(f"run {key} metadata or binary changed")
        if sha(HERE / f"{key}_metadata.json") != package["runs"][key]["metadata_sha256"]:
            raise RuntimeError(f"packaged metadata changed: {key}")
        for label, artifact in package["runs"][key]["artifacts"].items():
            ext = "jsonl.gz" if label == "capture" else "csv.gz"
            if sha(HERE / f"{key}_{label}.{ext}") != artifact["gzip_sha256"]:
                raise RuntimeError(f"packaged {label} changed: {key}")
        if not item["target_schedule_verified"]:
            raise RuntimeError(f"scheduled target trace was not verified: {key}")
    report = {"schema": 1, "canonical_source_unchanged": True,
              "frozen_copy_unchanged": True, "alias_and_heldout_bytes_unchanged": True,
              "schedule_bytes_unchanged": True, "six_runs_and_packages_unchanged": True,
              "run_count": len(campaign["runs"]),
              "binary_sha256": next(iter(campaign["runs"].values()))["binary_sha256"]}
    (HERE / "provenance_gate.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
