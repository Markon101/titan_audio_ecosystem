#!/usr/bin/env python3
"""Build three six-family guaranteed-exposure corpora from audited sources."""

import hashlib
import json
from pathlib import Path

from prepare_corpus_forks import (EXISTING_FILES, NEW_FILES, NEW_DIR, ORIGINAL_DIR,
                                  ORIGINAL_MANIFEST, HERE, wav_info)

OUTPUT = HERE / "runs/acute_exposure"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if OUTPUT.exists():
        raise RuntimeError("acute exposure directory exists")
    original = json.loads(ORIGINAL_MANIFEST.read_text())
    by_name = {entry["file"]: entry for entry in original["entries"]}
    heldout = [dict(entry) for entry in original["entries"]
               if entry["role"] in ("development", "validation")]
    if len(heldout) != 11:
        raise RuntimeError("held-out manifest roles changed")
    variants = json.loads((HERE / "corpus_forks_receipt.json").read_text())["existing_variants"]
    variant_by_source = {item["file"]: item for item in variants}
    arms = {"a": [], "b": [], "c": []}
    for name in EXISTING_FILES:
        entry = by_name[name]
        if entry["role"] != "train":
            raise RuntimeError(f"not a source training file: {name}")
        arms["a"].append((name, ORIGINAL_DIR / name, entry["family"], "existing_original"))
        variant = variant_by_source[name]
        path = HERE / "runs/corpus_forks/existing_variants" / variant["variant_file"]
        if sha(path) != variant["variant_sha256"]:
            raise RuntimeError(f"variant hash changed: {path}")
        arms["c"].append((variant["variant_file"], path.resolve(), entry["family"], "existing_eq_variant"))
    for name, family in NEW_FILES:
        arms["b"].append((name, NEW_DIR / name, family, "new_user_audio_family"))
    receipt = {"schema": 1, "source_manifest_sha256": sha(ORIGINAL_MANIFEST),
               "arms": {}, "heldout_files": [entry["file"] for entry in heldout]}
    OUTPUT.mkdir(parents=True)
    for label, files in arms.items():
        directory = OUTPUT / label
        directory.mkdir()
        entries = [dict(entry) for entry in heldout]
        records = []
        for entry in heldout:
            (directory / entry["file"]).symlink_to(ORIGINAL_DIR / entry["file"])
        for name, source, family, provenance in files:
            if not source.is_file():
                raise RuntimeError(f"acute source missing: {source}")
            (directory / name).symlink_to(source.resolve())
            entries.append({"file": name, "role": "train", "family": family,
                            "provenance": f"acute_exposure_{provenance}_20261001"})
            records.append({"file": name, "family": family, "source": str(source),
                            "sha256": sha(source), **wav_info(source)})
        if len({record["family"] for record in records}) != 6:
            raise RuntimeError(f"acute arm {label} does not contain six distinct families")
        manifest = {**original, "generated_by": "acute_exposure_fork_20261001",
                    "entries": entries}
        manifest_path = OUTPUT / f"{label}_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        receipt["arms"][label] = {"manifest_sha256": sha(manifest_path),
                                  "train_files": records,
                                  "training_seconds": sum(item["duration_seconds"] for item in records)}
    a_hash = {item["sha256"] for item in receipt["arms"]["a"]["train_files"]}
    b_hash = {item["sha256"] for item in receipt["arms"]["b"]["train_files"]}
    c_hash = {item["sha256"] for item in receipt["arms"]["c"]["train_files"]}
    heldout_hash = {sha(ORIGINAL_DIR / entry["file"]) for entry in heldout}
    if (a_hash & b_hash or a_hash & c_hash or b_hash & c_hash or
            a_hash & heldout_hash or b_hash & heldout_hash or c_hash & heldout_hash):
        raise RuntimeError("acute exposure has train/heldout or new/existing byte overlap")
    (HERE / "acute_exposure_forks_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({label: {"files": len(item["train_files"]),
                              "seconds": item["training_seconds"]}
                      for label, item in receipt["arms"].items()}))


if __name__ == "__main__":
    main()
