#!/usr/bin/env python3
"""Create an isolated writable run fork from a verified frozen checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    frozen = args.frozen.resolve(strict=True)
    receipt = json.loads((frozen.parent / "frozen_step_58107_receipt.json").read_text())
    if args.destination.exists():
        parser.error("destination exists; a run fork is created once")
    for item in receipt["files"]:
        source = frozen / item["name"]
        if sha256(source) != item["sha256"]:
            parser.error(f"frozen checkpoint hash mismatch: {source}")
    args.destination.mkdir(parents=True)
    for item in receipt["files"]:
        source = frozen / item["name"]
        target = args.destination / item["name"]
        with source.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst, 4 * 1024 * 1024)
        if sha256(target) != item["sha256"]:
            raise RuntimeError(f"fork copy hash mismatch: {target}")
        target.chmod(0o600)
    (args.destination / "fork_receipt.json").write_text(json.dumps({
        "schema": 1, "parent": str(frozen), "parent_step": receipt["global_step"],
        "parent_hashes": {item["name"]: item["sha256"] for item in receipt["files"]},
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"fork": str(args.destination), "step": receipt["global_step"],
                      "files": len(receipt["files"])}, sort_keys=True))


if __name__ == "__main__":
    main()
