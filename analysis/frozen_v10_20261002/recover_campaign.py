#!/usr/bin/env python3
"""Recover completed matched runs, distinguishing immutable parent from live alias."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / "analysis/metastable_20261001"
MATCHED = ROOT / "analysis/matched_exposure_20261001"
TAG = "v10-msfield-fresh-20260930-02"

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(4194304), b""):
            h.update(block)
    return h.hexdigest()

def main():
    parent = json.loads((OLD / "frozen_step_58107_receipt.json").read_text())
    plan = json.loads((MATCHED / "matched_exposure_receipt.json").read_text())
    runs = json.loads((MATCHED / "matched_run_receipts.json").read_text())["runs"]
    result = {"schema": 1, "parent_step": 58107, "parent": [], "runs": {}}
    source_hash_cache = {}
    for item in parent["files"]:
        archived = OLD / "frozen_step_58107" / item["name"]
        if sha(archived) != item["sha256"]:
            raise RuntimeError(f"archived parent changed: {archived}")
        live_hash = sha(item["source"])
        result["parent"].append({"file": item["name"], "archived_sha256": item["sha256"],
                                 "live_alias_sha256": live_hash,
                                 "live_alias_still_matches": live_hash == item["sha256"]})
        if "metadata" in item["name"]:
            live = json.loads(Path(item["source"]).read_text())
            result["live_alias_end_step"] = live["run"]["end_global_step"]
    for key, receipt in runs.items():
        base = MATCHED / f"runs/seed_{key}"
        arm = key.rsplit("_", 1)[-1]
        manifest = MATCHED / f"runs/corpus/{arm}_manifest.json"
        if sha(manifest) != plan["arms"][arm]["manifest_sha256"]:
            raise RuntimeError(f"corpus manifest changed: {arm}")
        slot_sources = {}
        for slot in plan["arms"][arm]["slots"]:
            alias = MATCHED / "runs/corpus" / arm / slot["alias"]
            source = Path(slot["source"])
            if alias.resolve(strict=True) != source.resolve(strict=True):
                raise RuntimeError(f"source mapping changed: {alias}")
            resolved = source.resolve(strict=True)
            if str(resolved) not in source_hash_cache:
                source_hash_cache[str(resolved)] = sha(resolved)
            if source_hash_cache[str(resolved)] != slot["source_sha256"]:
                raise RuntimeError(f"source audio changed: {source}")
            slot_sources[slot["slot"]] = slot["source_sha256"]
        metadata_path = base / f"titan_run_metadata_v10_msfield_{TAG}.json"
        metadata = json.loads(metadata_path.read_text())
        if sha(metadata_path) != receipt["metadata_sha256"]:
            raise RuntimeError(f"metadata changed: {key}")
        for name in ("model", "world", "optimizer", "audio", "prime"):
            if sha(metadata["outputs"][name]) != receipt[f"{name}_sha256"]:
                raise RuntimeError(f"{key} {name} changed")
        if (metadata["run"]["start_global_step"], metadata["run"]["end_global_step"],
            metadata["run"]["optimizer_updates_cumulative"]) != (58107, 59131, 924):
            raise RuntimeError(f"run interval mismatch: {key}")
        schedule = Path(metadata["invocation"]["target_schedule"])
        if sha(schedule) != receipt["schedule_sha256"]:
            raise RuntimeError(f"schedule changed: {key}")
        episodes = json.loads(schedule.read_text())["episodes"]
        rows = list(csv.DictReader(open(metadata["telemetry"]["uncertainty_trace"])))
        for row in rows:
            step = int(row["step"])
            episode = next(e for e in episodes if e["start_step"] <= step < e["start_step"] + e["chunks"])
            offset = step - episode["start_step"]
            assert (row["target_file"], int(row["target_frame"]), int(row["target_chunks_left"])) == (
                f"slot_{episode['slot']:02}.wav", episode["source_frame"] + offset * 4096,
                episode["chunks"] - offset)
        result["runs"][key] = {"complete": True, "hashes_match": True,
                              "scheduled_rows_verified": len(rows),
                              "source_slots_verified": slot_sources, "rerun_needed": False}
    if len(result["runs"]) != 6:
        raise RuntimeError("missing matched arms")
    result["decision"] = "all six complete; no rerun; use archived step 58107; live alias has advanced independently"
    (HERE / "recovery.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"complete_arms": len(runs), "archived_parent_verified": True,
                      "live_alias_end_step": result["live_alias_end_step"], "rerun_needed": False}))

if __name__ == "__main__":
    main()
