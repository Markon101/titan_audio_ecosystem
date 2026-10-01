#!/usr/bin/env python3
"""Copy a v10 checkpoint set once, byte-for-byte, with a SHA-256 receipt."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

STEMS = (
    "titan_model_v10_msfield",
    "titan_world_v10_msfield",
    "titan_optimizer_v10_msfield",
    "titan_morph_state_v10_msfield",
    "titan_run_metadata_v10_msfield",
)


def digest(path):
    sha = hashlib.sha256()
    with open(path, "rb") as src:
        for block in iter(lambda: src.read(4 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    if args.destination.exists():
        parser.error("destination already exists; frozen sets are immutable")
    source = args.source.resolve(strict=True)
    tag = "v10-msfield-fresh-20260930-02"
    extensions = ("safetensors", "bin", "safetensors", "json", "json")
    files = [source / f"{stem}_{tag}.{ext}" for stem, ext in zip(STEMS, extensions)]
    for path in files:
        if not path.is_file():
            parser.error(f"missing source: {path}")
    metadata = json.loads(files[-1].read_text())
    actual = metadata.get("run", {}).get("end_global_step")
    if actual != args.step or metadata.get("invocation", {}).get("substrate") != "msfield":
        parser.error(f"source metadata is not msfield step {args.step}: {actual}")
    args.destination.mkdir(parents=True)
    records = []
    for path in files:
        target = args.destination / path.name
        source_hash = digest(path)
        with path.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst, 4 * 1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        if digest(target) != source_hash or digest(path) != source_hash:
            raise RuntimeError(f"copy verification failed: {path}")
        target.chmod(0o444)
        records.append({"name": path.name, "source": str(path), "bytes": path.stat().st_size,
                        "sha256": source_hash})
    receipt = {"schema": 1, "tag": tag, "global_step": args.step,
               "metadata_run_id": metadata.get("run_id"),
               "source_git_commit": metadata.get("build", {}).get("git_commit"),
               "files": records}
    (args.destination.parent / f"{args.destination.name}_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"global_step": args.step, "files": len(records),
                      "total_bytes": sum(r["bytes"] for r in records)}, sort_keys=True))


if __name__ == "__main__":
    main()
