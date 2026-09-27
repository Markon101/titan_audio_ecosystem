#!/usr/bin/env python3
"""Compare two isolated one-run TITAN Audio outputs without changing them."""

import argparse
import hashlib
import json
import mmap
from pathlib import Path
import struct

import numpy as np


DTYPES = {"F32": "<f4", "F64": "<f8", "I32": "<i4", "I64": "<i8", "U64": "<u8"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safetensors(path: Path):
    handle = path.open("rb")
    mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    header_len = struct.unpack_from("<Q", mapped)[0]
    header = json.loads(mapped[8:8 + header_len])
    return handle, mapped, header, 8 + header_len


def compare_tensors(old_path: Path, new_path: Path) -> dict:
    old = safetensors(old_path)
    new = safetensors(new_path)
    if old[2] != new[2]:
        raise ValueError(f"tensor layout changed: {old_path.name}")
    changed = over_tolerance = 0
    max_abs_delta = 0.0
    tensor_count = 0
    worst = []
    for name, description in old[2].items():
        if name == "__metadata__":
            continue
        dtype = DTYPES.get(description["dtype"])
        if dtype is None:
            raise ValueError(f"unsupported tensor dtype: {description['dtype']}")
        start, end = description["data_offsets"]
        count = (end - start) // np.dtype(dtype).itemsize
        a = np.frombuffer(old[1], dtype=dtype, count=count, offset=old[3] + start)
        b = np.frombuffer(new[1], dtype=dtype, count=count, offset=new[3] + start)
        n_changed = int(np.count_nonzero(a != b))
        tensor_count += 1
        if n_changed:
            delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
            if np.issubdtype(a.dtype, np.floating):
                tolerance = 1e-7 + 1e-6 * np.maximum(np.abs(a), np.abs(b))
                n_over = int(np.count_nonzero(delta > tolerance))
            else:
                n_over = n_changed
            changed += n_changed
            over_tolerance += n_over
            max_abs_delta = max(max_abs_delta, float(delta.max()))
            worst.append({"tensor": name, "changed": n_changed,
                          "over_tolerance": n_over, "max_abs_delta": float(delta.max())})
    return {"layout_equal": True, "tensor_count": tensor_count,
            "changed_elements": changed, "over_tolerance": over_tolerance,
            "max_abs_delta": max_abs_delta,
            "largest_deltas": sorted(worst, key=lambda item: item["max_abs_delta"], reverse=True)[:8],
            "tolerance": "1e-7 + 1e-6 * max(abs(old),abs(new)) for floating tensors; exact integers"}


def one(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {pattern} in {root}, found {len(matches)}")
    return matches[0]


def compare(old_root: Path, new_root: Path, old_binary: Path, new_binary: Path) -> dict:
    result = {"old_root": str(old_root.resolve()), "new_root": str(new_root.resolve()),
              "old_binary_sha256": sha256(old_binary), "new_binary_sha256": sha256(new_binary),
              "files": {}, "tensors": {}}
    for name, pattern in (("full_audio", "rust_ecosystem_out_*.wav"),
                          ("prime_audio", "titan_prime_*.wav"),
                          ("model", "titan_model_v9.safetensors"),
                          ("optimizer", "titan_optimizer_v9.safetensors"),
                          ("world", "titan_world_v9.bin"),
                          ("morph", "titan_morph_state_v9.json")):
        old_path, new_path = one(old_root, pattern), one(new_root, pattern)
        old_hash, new_hash = sha256(old_path), sha256(new_path)
        result["files"][name] = {"old_sha256": old_hash, "new_sha256": new_hash,
                                 "byte_identical": old_hash == new_hash,
                                 "old_bytes": old_path.stat().st_size,
                                 "new_bytes": new_path.stat().st_size}
        if name in {"model", "optimizer"}:
            result["tensors"][name] = compare_tensors(old_path, new_path)
    old_world = one(old_root, "titan_world_v9.bin").read_bytes()
    new_world = one(new_root, "titan_world_v9.bin").read_bytes()
    if len(old_world) < 24 or len(new_world) < 24 or old_world[:8] != b"TITANW9\0":
        raise ValueError("unexpected world-checkpoint wrapper")
    result["world"] = {"payload_equal": old_world[24:] == new_world[24:],
                       "payload_differing_bytes": sum(a != b for a, b in zip(old_world[24:], new_world[24:]))
                       + abs(len(old_world) - len(new_world)),
                       "wrapper_header_equal": old_world[:16] == new_world[:16]}
    for arm, root in (("old", old_root), ("new", new_root)):
        metadata = json.loads(one(root, "titan_run_metadata_v9.json").read_text())
        result.setdefault("metadata", {})[arm] = {
            "end_global_step": metadata["run"]["end_global_step"],
            "optimizer_updates_cumulative": metadata["run"]["optimizer_updates_cumulative"],
            "experiments": metadata["invocation"].get("experiments"),
        }
    result["legacy_gate"] = {
        "audio_equal": result["files"]["full_audio"]["byte_identical"]
                       and result["files"]["prime_audio"]["byte_identical"],
        "world_payload_equal": result["world"]["payload_equal"],
        "model_within_tolerance": result["tensors"]["model"]["over_tolerance"] == 0,
        "optimizer_within_tolerance": result["tensors"]["optimizer"]["over_tolerance"] == 0,
        "same_steps": result["metadata"]["old"]["end_global_step"] == result["metadata"]["new"]["end_global_step"]
                      and result["metadata"]["old"]["optimizer_updates_cumulative"] == result["metadata"]["new"]["optimizer_updates_cumulative"],
    }
    result["legacy_gate"]["pass"] = all(result["legacy_gate"].values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--old-binary", type=Path, required=True)
    parser.add_argument("--new-binary", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    if args.report_out.exists():
        parser.exit(1, "report already exists\n")
    result = compare(args.old_root, args.new_root, args.old_binary, args.new_binary)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result["legacy_gate"], sort_keys=True))


if __name__ == "__main__":
    main()
