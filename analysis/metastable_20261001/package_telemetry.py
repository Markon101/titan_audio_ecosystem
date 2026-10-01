#!/usr/bin/env python3
"""Preserve run telemetry across a tagged continuation that rewrites CSVs."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--start-step", type=int, required=True)
    parser.add_argument("--end-step", type=int, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9_]*", args.name):
        parser.error("--name must be a safe lowercase artifact stem")
    run = args.run_dir.resolve(strict=True)
    metadata_path = run / f"titan_run_metadata_v10_msfield_{TAG}.json"
    metadata = json.loads(metadata_path.read_text())
    if (metadata["run"]["start_global_step"] != args.start_step or
            metadata["run"]["end_global_step"] != args.end_step):
        parser.error("metadata does not match requested segment")
    records = {}
    for key in KEYS:
        source = Path(metadata["telemetry"][key])
        if not source.is_absolute():
            source = HERE.parents[1] / source
        source = source.resolve(strict=True)
        target = HERE / f"{args.name}_{key}.csv.gz"
        if target.exists():
            parser.error(f"target exists: {target}")
        source_hash = sha(source)
        with source.open("rb") as src, target.open("xb") as dst:
            with gzip.GzipFile(filename="", mode="wb", fileobj=dst, mtime=0,
                               compresslevel=9) as gz:
                shutil.copyfileobj(src, gz, 4 * 1024 * 1024)
        if sha(source) != source_hash:
            raise RuntimeError(f"telemetry changed during packaging: {source}")
        records[key] = {"source": str(source), "source_sha256": source_hash,
                        "gzip_sha256": sha(target), "gzip_bytes": target.stat().st_size}
    receipt = {"schema": 1, "run_id": metadata["run_id"],
               "start_step": args.start_step, "end_step": args.end_step,
               "metadata_sha256": sha(metadata_path), "artifacts": records}
    (HERE / f"{args.name}_telemetry_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"name": args.name, "run_id": metadata["run_id"],
                      "artifacts": len(records)}))


if __name__ == "__main__":
    main()
