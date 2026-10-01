#!/usr/bin/env python3
"""Collect exact commands and SHA-256 identities for completed fork runs."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TAG = "v10-msfield-fresh-20260930-02"
STEMS = (
    ("model", "titan_model_v10_msfield", "safetensors"),
    ("world", "titan_world_v10_msfield", "bin"),
    ("optimizer", "titan_optimizer_v10_msfield", "safetensors"),
    ("morph", "titan_morph_state_v10_msfield", "json"),
    ("metadata", "titan_run_metadata_v10_msfield", "json"),
)
RUNS = {
    "a_segment1": HERE / "runs/segment1_step_59513",
    "a_segment2": HERE / "runs/observer_step_58107",
    "b_segment1": HERE / "runs/corpus_b_segment1_step_59513",
    "b_segment2": HERE / "runs/corpus_new_family_run",
    "c_segment1": HERE / "runs/corpus_c_segment1_step_59513",
    "c_segment2": HERE / "runs/corpus_existing_variant_run",
    "acute_a": HERE / "runs/acute_a_run",
    "acute_b": HERE / "runs/acute_b_run",
    "acute_c": HERE / "runs/acute_c_run",
}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main():
    source = json.loads((HERE / "frozen_step_58107_receipt.json").read_text())
    source_ok = all(sha(item["source"]) == item["sha256"] for item in source["files"])
    if not source_ok:
        raise RuntimeError("canonical source checkpoint changed")
    result = {"schema": 1, "source_step": 58107,
              "source_receipt_sha256": sha(HERE / "frozen_step_58107_receipt.json"),
              "canonical_source_unchanged": source_ok, "runs": {}}
    for label, directory in RUNS.items():
        metadata_path = directory / f"titan_run_metadata_v10_msfield_{TAG}.json"
        metadata = json.loads(metadata_path.read_text())
        run = metadata["run"]
        expected_start = 59513 if label.endswith("segment2") else 58107
        expected_end = 60919 if label.endswith("segment2") else 58810 if label.startswith("acute_") else 59513
        if run["start_global_step"] != expected_start or run["end_global_step"] != expected_end:
            raise RuntimeError(f"run interval mismatch: {label}")
        files = {short: {"path": str(directory / f"{stem}_{TAG}.{ext}"),
                         "sha256": sha(directory / f"{stem}_{TAG}.{ext}")}
                 for short, stem, ext in STEMS}
        binary_dir = directory
        if label == "a_segment1": binary_dir = RUNS["a_segment2"]
        if label == "b_segment1": binary_dir = RUNS["b_segment2"]
        if label == "c_segment1": binary_dir = RUNS["c_segment2"]
        binary = binary_dir / "titan_run_binary"
        corpus_manifest = resolve(metadata["corpus"]["manifest"])
        result["runs"][label] = {
            "run_id": metadata["run_id"], "start_step": expected_start,
            "end_step": expected_end, "completed_chunks": run["completed_chunks"],
            "optimizer_updates_cumulative": run["optimizer_updates_cumulative"],
            "model_exact_tensors_loaded": metadata["invocation"]["model_migration"]["exact"],
            "optimizer_exact_moments_loaded": run["optimizer_migration"]["exact"],
            "binary_sha256": sha(binary),
            "args": metadata["invocation"]["args"],
            "corpus_manifest_sha256": sha(corpus_manifest),
            "training_families": metadata["corpus"]["training_families"],
            "validation_is_strict": metadata["corpus"]["validation_is_strict"],
            "files": files,
            "audio_sha256": sha(resolve(metadata["outputs"]["audio"])),
            "prime_sha256": sha(resolve(metadata["outputs"]["prime"])),
        }
    (HERE / "run_receipts.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"runs": len(result["runs"]),
                      "canonical_source_unchanged": source_ok,
                      "binary_hashes": len({entry["binary_sha256"] for entry in result["runs"].values()})}))


if __name__ == "__main__":
    main()
