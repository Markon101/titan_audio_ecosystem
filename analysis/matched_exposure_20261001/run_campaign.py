#!/usr/bin/env python3
"""Execute six matched, resource-guarded v10 exposure runs in fixed order."""

import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIOR = ROOT / "analysis/metastable_20261001"
FROZEN = PRIOR / "frozen_step_58107"
TAG = "v10-msfield-fresh-20260930-02"
RUN_ORDER = ((20261002, "a"), (20261002, "c"), (20261002, "b"),
             (20261003, "c"), (20261003, "b"), (20261003, "a"))
START, END = 58107, 59131


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_schedule_rows(path, schedule):
    rows = list(csv.DictReader(open(path)))
    if len(rows) < 100:
        raise RuntimeError("scheduled run has too few target trace samples")
    for row in rows:
        step = int(row["step"])
        episode = next((item for item in schedule["episodes"]
                        if item["start_step"] <= step < item["start_step"] + item["chunks"]), None)
        if episode is None:
            raise RuntimeError(f"no schedule episode covers trace step {step}")
        offset = step - episode["start_step"]
        expected = (f"slot_{episode['slot']:02}.wav",
                    episode["source_frame"] + offset * 4096,
                    episode["chunks"] - offset)
        observed = (row["target_file"], int(row["target_frame"]),
                    int(row["target_chunks_left"]))
        if observed != expected:
            raise RuntimeError(f"target schedule mismatch at step {step}: {observed} != {expected}")
    return len(rows)


def main():
    receipt_path = HERE / "matched_run_receipts.json"
    if receipt_path.exists():
        progress = json.loads(receipt_path.read_text())
    else:
        progress = {"schema": 1, "source_step": START, "end_step": END, "runs": {}}
    plan = json.loads((HERE / "matched_exposure_receipt.json").read_text())
    binary = ROOT / "target/release/titan"
    binary_hash = sha(binary)
    for seed, arm in RUN_ORDER:
        key = f"{seed}_{arm}"
        if key in progress["runs"]:
            print(f"verified completed run {key}", flush=True)
            continue
        base = HERE / f"runs/seed_{seed}_{arm}"
        schedule_path = HERE / f"schedule_{seed}.json"
        schedule_hash = sha(schedule_path)
        if schedule_hash != plan["schedules"][str(seed)]["sha256"]:
            raise RuntimeError(f"schedule bytes changed: {seed}")
        corpus_dir = HERE / f"runs/corpus/{arm}"
        manifest = HERE / f"runs/corpus/{arm}_manifest.json"
        if sha(manifest) != plan["arms"][arm]["manifest_sha256"]:
            raise RuntimeError(f"corpus manifest changed: {arm}")
        subprocess.run(["python3", str(PRIOR / "fork_checkpoint.py"),
                        str(FROZEN), str(base)], cwd=ROOT, check=True, capture_output=True)
        run_binary = base / "titan_run_binary"
        shutil.copyfile(binary, run_binary)
        run_binary.chmod(0o700)
        if sha(run_binary) != binary_hash:
            raise RuntimeError("run executable copy hash mismatch")
        command = [str(run_binary), "--base-dir", str(base),
                   "--corpus-dir", str(corpus_dir),
                   "--corpus-manifest", str(manifest),
                   "--target-schedule", str(schedule_path),
                   "--substrate", "msfield", "--run-tag", TAG,
                   "--duration", "87.4", "--threads", "4", "--seed", "44",
                   "--lr", "0.00045", "--bptt", "64",
                   "--max-autograd-tape", "4", "--core-update-every", "1",
                   "--morph-layers", "16", "--morph-width", "512",
                   "--max-morph-depth", "16", "--motif-capacity", "512",
                   "--regime-capture", "--regime-stride", "4"]
        print(f"starting {key} with schedule {schedule_hash}", flush=True)
        subprocess.run(["python3", str(HERE / "resource_guard.py"),
                        "--log", str(base / "run.log"),
                        "--receipt", str(base / "resource_guard.json"),
                        "--min-available-mib", "500", "--min-swap-mib", "100",
                        "--", *command], cwd=ROOT, check=True)
        metadata_path = base / f"titan_run_metadata_v10_msfield_{TAG}.json"
        metadata = json.loads(metadata_path.read_text())
        invocation = metadata["invocation"]
        if ((metadata["run"]["start_global_step"], metadata["run"]["end_global_step"],
             metadata["run"]["completed_chunks"], metadata["run"]["optimizer_updates_cumulative"])
                != (START, END, 1024, 924)
                or invocation["autograd_tape_chunks"] != 4
                or invocation["threads"] != 4
                or invocation["target_schedule_sha256"] != schedule_hash
                or invocation["model_migration"]["exact"] != 229
                or metadata["run"]["optimizer_migration"]["exact"] != 229
                or not metadata["corpus"]["validation_is_strict"]):
            raise RuntimeError(f"completed run identity mismatch: {key}")
        schedule = json.loads(schedule_path.read_text())
        trace_rows = verify_schedule_rows(metadata["telemetry"]["uncertainty_trace"], schedule)
        paths = metadata["outputs"]
        progress["runs"][key] = {"run_id": metadata["run_id"],
            "binary_sha256": binary_hash, "schedule_sha256": schedule_hash,
            "corpus_manifest_sha256": sha(manifest),
            "model_sha256": sha(paths["model"]), "world_sha256": sha(paths["world"]),
            "optimizer_sha256": sha(paths["optimizer"]),
            "audio_sha256": sha(paths["audio"]), "prime_sha256": sha(paths["prime"]),
            "metadata_sha256": sha(metadata_path), "trace_rows": trace_rows,
            "target_schedule_verified": True,
            "resource_guard": json.loads((base / "resource_guard.json").read_text())}
        temporary = receipt_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
        temporary.replace(receipt_path)
        print(f"completed {key}: {trace_rows} scheduled target samples verified", flush=True)


if __name__ == "__main__":
    main()
