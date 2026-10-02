#!/usr/bin/env python3
"""Package the six run traces with deterministic gzip and SHA-256 receipts."""

import gzip
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
TAG = "v10-msfield-fresh-20260930-02"
KEYS = ("uncertainty_trace", "msfield_trace", "topology_index",
        "topology_values", "morph_events")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def compress(source, target):
    if target.exists():
        raise RuntimeError(f"package output exists: {target}")
    before = sha(source)
    with open(source, "rb") as src, target.open("xb") as dst:
        with gzip.GzipFile(filename="", mode="wb", fileobj=dst,
                           mtime=0, compresslevel=9) as gz:
            shutil.copyfileobj(src, gz, 4 * 1024 * 1024)
    if sha(source) != before:
        raise RuntimeError(f"source changed while packaging: {source}")
    return {"source": str(source), "source_sha256": before,
            "gzip_sha256": sha(target), "gzip_bytes": target.stat().st_size}


def main():
    campaign = json.loads((HERE / "matched_run_receipts.json").read_text())
    if len(campaign["runs"]) != 6:
        raise RuntimeError("six completed runs are required before packaging")
    receipt = {"schema": 1, "runs": {}}
    for key in sorted(campaign["runs"]):
        run = HERE / f"runs/seed_{key}"
        metadata_path = run / f"titan_run_metadata_v10_msfield_{TAG}.json"
        metadata = json.loads(metadata_path.read_text())
        if sha(metadata_path) != campaign["runs"][key]["metadata_sha256"]:
            raise RuntimeError(f"metadata changed for {key}")
        artifacts = {}
        artifacts["capture"] = compress(metadata["telemetry"]["regime_capture"],
                                         HERE / f"{key}_capture.jsonl.gz")
        for name in KEYS:
            artifacts[name] = compress(metadata["telemetry"][name],
                                       HERE / f"{key}_{name}.csv.gz")
        snapshot = HERE / f"{key}_metadata.json"
        if snapshot.exists():
            raise RuntimeError(f"metadata snapshot exists: {snapshot}")
        shutil.copyfile(metadata_path, snapshot)
        resource = run / "resource_guard.json"
        guard_copy = HERE / f"{key}_resource.json"
        shutil.copyfile(resource, guard_copy)
        receipt["runs"][key] = {"run_id": metadata["run_id"],
            "metadata_sha256": sha(snapshot), "resource_guard_sha256": sha(guard_copy),
            "artifacts": artifacts}
        print(f"packaged {key}", flush=True)
    (HERE / "package_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"runs": len(receipt["runs"])}))


if __name__ == "__main__":
    main()
