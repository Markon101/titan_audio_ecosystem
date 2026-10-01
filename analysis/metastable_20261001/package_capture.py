#!/usr/bin/env python3
"""Publish a deterministic compressed observer capture with source receipts."""

import gzip
import hashlib
import json
from pathlib import Path
import shutil
import argparse
import re

HERE = Path(__file__).resolve().parent
DEFAULT_RUN = HERE / "runs/observer_step_58107"
TAG = "v10-msfield-fresh-20260930-02"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-step", type=int, default=58107)
    parser.add_argument("--end-step", type=int, default=59513)
    parser.add_argument("--name", default="observer_capture")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9_]*", args.name):
        parser.error("--name must be a safe lowercase artifact stem")
    run = args.run_dir.resolve(strict=True)
    metadata_path = run / f"titan_run_metadata_v10_msfield_{TAG}.json"
    metadata = json.loads(metadata_path.read_text())
    capture = (HERE.parents[1] / metadata["telemetry"]["regime_capture"]).resolve()
    if (not capture.is_file() or metadata["run"]["start_global_step"] != args.start_step or
            metadata["run"]["end_global_step"] != args.end_step):
        raise RuntimeError("observer capture is incomplete or has the wrong step interval")
    output = HERE / f"{args.name}.jsonl.gz"
    if output.exists():
        raise RuntimeError("packaged capture already exists")
    source_hash = sha(capture)
    with capture.open("rb") as source, output.open("xb") as destination:
        with gzip.GzipFile(filename="", mode="wb", fileobj=destination, mtime=0, compresslevel=9) as gz:
            shutil.copyfileobj(source, gz, 4 * 1024 * 1024)
    if sha(capture) != source_hash:
        raise RuntimeError("capture changed during packaging")
    with gzip.open(output, "rt") as stream:
        lines = sum(1 for line in stream if line.strip())
    receipt = {"schema": 1, "start_step": args.start_step, "end_step": args.end_step,
               "samples": lines, "sample_stride": 4,
               "run_id": metadata["run_id"],
               "source_capture": str(capture), "source_sha256": source_hash,
               "gzip_sha256": sha(output), "gzip_bytes": output.stat().st_size,
               "binary_sha256": sha(run / "titan_run_binary"),
               "run_metadata_sha256": sha(metadata_path)}
    (HERE / f"{args.name}_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"samples": lines, "gzip_bytes": receipt["gzip_bytes"],
                      "gzip_sha256": receipt["gzip_sha256"]}))


if __name__ == "__main__":
    main()
