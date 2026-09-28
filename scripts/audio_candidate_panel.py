#!/usr/bin/env python3
"""Blindly package two or three existing TITAN WAVs from any substrate.

The package compares audio candidates, not learned weights under a controlled
state. Inputs are opened read-only. No model is loaded, trained, or queried.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import re

import audio_listening_panel as panel
import prime_package as prime


SCHEMA = "titan_audio_candidate_panel_v1"
LABELS = "ABC"
CHUNK_FRAMES = 4096


def file_id(path: Path) -> dict:
    return prime.file_identity(path)


def resolve_file(value: str | None, base: Path) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("file paths must be nonempty strings or null")
    path = Path(value)
    resolved = (path if path.is_absolute() else base / path).resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"expected a file: {resolved}")
    return resolved


def load_config(path: Path) -> dict:
    path = path.resolve(strict=True)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"seed", "clip_frames", "candidates"}:
        raise ValueError("config requires exactly seed, clip_frames, candidates")
    if type(raw["seed"]) is not int or raw["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(raw["clip_frames"]) is not int or raw["clip_frames"] < 1:
        raise ValueError("clip_frames must be a positive integer")
    candidates = raw["candidates"]
    if not isinstance(candidates, list) or not 2 <= len(candidates) <= 3:
        raise ValueError("candidates must contain two or three arms")
    arms = []
    ids = set()
    for item in candidates:
        required = {"id", "wav", "start_frame"}
        allowed = required | {"checkpoint", "world", "run_metadata", "executable",
                              "processing_note"}
        if not isinstance(item, dict) or not required <= set(item) or not set(item) <= allowed:
            raise ValueError("candidate requires id, wav, start_frame; optional checkpoint, world, run_metadata, executable, processing_note")
        arm_id = item["id"]
        if not isinstance(arm_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", arm_id):
            raise ValueError("invalid candidate id")
        if arm_id.lower() in ids:
            raise ValueError("duplicate candidate id")
        ids.add(arm_id.lower())
        start = item["start_frame"]
        if type(start) is not int or start < 0:
            raise ValueError("start_frame must be a nonnegative integer")
        note = item.get("processing_note", "existing TITAN WAV; no additional source processing declared")
        if not isinstance(note, str) or not note.strip():
            raise ValueError("processing_note must be nonempty text")
        files = {field: resolve_file(item.get(field), path.parent)
                 for field in ("wav", "checkpoint", "world", "run_metadata", "executable")}
        if files["wav"] is None:
            raise ValueError("candidate wav must be a file")
        info = prime.wav_info(files["wav"])
        if start + raw["clip_frames"] > info["frames"]:
            raise ValueError(f"candidate {arm_id} interval extends beyond source WAV")
        arms.append({"id": arm_id, "start_frame": start, "processing_note": note,
                     "files": files, "wav_info": info})
    config_identity = file_id(path)
    return {"seed": raw["seed"], "clip_frames": raw["clip_frames"],
            "candidates": arms, "config": config_identity}


def validate_metadata(arm: dict, identities: dict) -> dict | None:
    path = arm["files"]["run_metadata"]
    if path is None:
        return None
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("run metadata must be a JSON object")
    reported = metadata.get("outputs", {})
    if not isinstance(reported, dict):
        raise ValueError("run metadata outputs must be an object")
    matches = {}
    for field, reported_name in (("wav", "audio"), ("checkpoint", "model"),
                                 ("world", "world")):
        declared = identities.get(field)
        output = reported.get(reported_name)
        if declared is None or output is None:
            matches[field] = None
            continue
        reported_path = Path(output)
        if not reported_path.is_absolute():
            reported_path = path.parent / reported_path
        if reported_path.is_file():
            matches[field] = file_id(reported_path)["sha256"] == declared["sha256"]
            if not matches[field]:
                raise ValueError(f"{arm['id']} {field} differs from run metadata output")
        else:
            matches[field] = None
    run = metadata.get("run", {})
    field = metadata.get("field", {})
    if matches["wav"] is True and "completed_chunks" in run:
        chunk_size = field.get("chunk_size")
        if (type(chunk_size) is not int or chunk_size <= 0
                or type(run["completed_chunks"]) is not int
                or run["completed_chunks"] * chunk_size != arm["wav_info"]["frames"]):
            raise ValueError(f"{arm['id']} WAV frame count disagrees with run metadata")
    return {"run_id": metadata.get("run_id"), "build": metadata.get("build"),
            "run_start_step": run.get("start_global_step"),
            "run_end_step": run.get("end_global_step"),
            "run_completed_chunks": run.get("completed_chunks"),
            "chunk_size": field.get("chunk_size"),
            "corpus": metadata.get("corpus"),
            "reported_outputs": {key: reported.get(key) for key in ("audio", "model", "world")},
            "declared_files_match_reported_outputs": matches,
            "scope": "run metadata describes the trajectory; a final checkpoint does not reproduce every emitted training chunk"}


def build_panel(config: dict, output: Path) -> dict:
    output = output.resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    arms = []
    canonical = [config["config"]]
    for arm in config["candidates"]:
        identities = {field: file_id(path) if path else None
                      for field, path in arm["files"].items()}
        canonical.extend(item for item in identities.values() if item is not None)
        metadata = validate_metadata(arm, identities)
        arms.append({**arm, "identities": identities, "metadata": metadata})
    if any(Path(item["path"]).is_relative_to(output) for item in canonical):
        raise ValueError("output directory overlaps a canonical input")
    output.mkdir(parents=True, exist_ok=False)
    incomplete = output / "INCOMPLETE"
    incomplete.write_text("Package incomplete; do not use BLIND files.\n", encoding="utf-8")
    (output / "BLIND").mkdir()
    (output / "source_intervals").mkdir()
    for arm in arms:
        interval = output / "source_intervals" / f"{arm['id']}.wav"
        arm["fade_frames_per_edge"] = prime.export_clip(
            arm["files"]["wav"], interval, arm["start_frame"], config["clip_frames"])
        arm["interval"] = interval
        arm["interval_identity"] = file_id(interval)
        arm["interval_metrics"] = panel.inspect_wav(interval)
    target_rms = min(
        min(arm["interval_metrics"]["rms_pcm"] for arm in arms),
        min(arm["interval_metrics"]["rms_pcm"] * panel.HEADROOM_SAMPLE
            / arm["interval_metrics"]["peak_pcm"] for arm in arms),
    )
    shuffled = list(arms)
    random.Random(config["seed"]).shuffle(shuffled)
    key = []
    for label, arm in zip(LABELS, shuffled):
        gain = target_rms / arm["interval_metrics"]["rms_pcm"]
        blind = output / "BLIND" / f"{label}.wav"
        exported = panel.write_scaled(arm["interval"], blind, gain, config["clip_frames"])
        key.append({"anonymous_id": label, "candidate_id": arm["id"],
                    "source_audio": arm["identities"]["wav"],
                    "source_checkpoint": arm["identities"]["checkpoint"],
                    "source_world": arm["identities"]["world"],
                    "declared_executable": arm["identities"]["executable"],
                    "run_metadata": arm["identities"]["run_metadata"],
                    "run_declaration": arm["metadata"],
                    "processing_note": arm["processing_note"],
                    "source_interval_frames": [arm["start_frame"],
                                               arm["start_frame"] + config["clip_frames"]],
                    "edge_fade_frames_per_side": arm["fade_frames_per_edge"],
                    "interval_pcm_sha256": arm["interval_identity"]["sha256"],
                    "gain": gain, "blind_audio": file_id(blind),
                    "output_rms_pcm": exported["rms_pcm"],
                    "descriptive_pcm": prime.signal_checks(blind)})
    levels = [item["output_rms_pcm"] for item in key]
    difference_db = 20 * math.log10(max(levels) / min(levels))
    if difference_db > 0.01:
        raise ValueError("quantized panel levels differ by more than 0.01 dB RMS")
    for original in canonical:
        if file_id(Path(original["path"])) != original:
            raise ValueError(f"canonical input changed while packaging: {original['path']}")
    (output / "BLIND" / "LISTEN.txt").write_text(
        "Blind TITAN audio candidate panel\n\n"
        "Listen to A.wav through the final letter at the same playback setting.\n"
        "Record timing, transitions, timbre, harshness, stereo, and usefulness as a prime.\n"
        "Keep key.json private until listening notes are complete.\n",
        encoding="utf-8")
    receipt = {"schema": SCHEMA, "seed": config["seed"],
               "clip_frames": config["clip_frames"],
               "clip_seconds": config["clip_frames"] / panel.RATE,
               "config": config["config"],
               "comparison_design": "pre-existing output candidates with independent source trajectories",
               "strict_state_or_weight_equivalence": False,
               "interpretation_limits": [
                   "audio preference does not isolate substrate from training state, trajectory, interval, or random initialization",
                   "a final checkpoint is an associated lineage artifact, not necessarily the exact weight state used for every emitted chunk",
                   "a supplied executable is a hashable declaration; the package cannot prove it produced a pre-existing WAV",
                   "signal metrics are descriptive, not perceptual scores",
                   "no downstream generation or Suno transfer is measured"],
               "processing": "same sample-exact PCM16 interval extraction with integer edge fades; global RMS matching and common -1 dBFS peak guard; no other mastering",
               "level_target_rms_pcm": target_rms,
               "maximum_level_difference_db": difference_db,
               "key": key}
    (output / "key.json").write_text(json.dumps(receipt, indent=2,
                                                 sort_keys=True, allow_nan=False) + "\n",
                                          encoding="utf-8")
    downstream = {"schema": "titan_downstream_attempts_v1",
                  "note": "Private manual template; add one record for every attempt, including rejected rerolls. No Suno API is used.",
                  "attempts": [
                      {"anonymous_id": item["anonymous_id"],
                       "titan_source_checkpoint": item["source_checkpoint"],
                       "titan_source_audio_hash": item["blind_audio"]["sha256"],
                       "titan_source_audio_path": item["blind_audio"]["path"],
                       "suno_model_version": None, "suno_mode": None,
                       "suno_custom_model": None, "prompt_style_settings": None,
                       "weirdness_slider": None, "style_slider": None,
                       "upload_trim_frames": None, "attempt_order": None,
                       "generation_seed_if_available": None,
                       "output_identifiers": [], "output_audio_sha256": None,
                       "accepted_or_rejected": None, "human_preference_notes": None}
                      for item in key]}
    (output / "downstream_receipt_template.json").write_text(
        json.dumps(downstream, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    incomplete.unlink()
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        receipt = build_panel(load_config(args.config), args.output_dir)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        parser.exit(1, f"candidate panel failed: {error}\n")
    print(f"wrote {len(receipt['key'])} blind WAVs to {args.output_dir / 'BLIND'}")


if __name__ == "__main__":
    main()
