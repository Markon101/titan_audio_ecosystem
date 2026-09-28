#!/usr/bin/env python3
"""Run frozen v9 same-world evaluations and make a blinded audio package.

This is a sidecar. It invokes TITAN's existing --analysis-only path, never a
training invocation, and does not contact a downstream service.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import subprocess

import audio_listening_panel as panel
import prime_package as prime


SCHEMA = "titan_matched_frozen_eval_v1"
RATE = 48_000
CHUNK_FRAMES = 4096
LABELS = "ABC"


def identity(path: Path) -> dict:
    return prime.file_identity(path)


def require_path(raw: str, base: Path) -> Path:
    path = Path(raw)
    return (path if path.is_absolute() else base / path).resolve(strict=True)


def load_config(path: Path) -> dict:
    path = path.resolve(strict=True)
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {"binary", "base_dir", "corpus_manifest", "corpus_dir",
                "common_world", "seed", "chunks", "candidates"}
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError(f"config requires exactly {sorted(expected)}")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(config["chunks"]) is not int or not 1 <= config["chunks"] <= 4096:
        raise ValueError("chunks must be an integer from 1 to 4096")
    candidates = config["candidates"]
    if not isinstance(candidates, list) or not 2 <= len(candidates) <= 3:
        raise ValueError("candidates must contain two or three arms")
    seen = set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != {"id", "model"}:
            raise ValueError("each candidate requires exactly id and model")
        arm_id = candidate["id"]
        if not isinstance(arm_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", arm_id):
            raise ValueError("invalid candidate id")
        if arm_id.lower() in seen:
            raise ValueError("duplicate candidate id")
        seen.add(arm_id.lower())
    base = path.parent
    resolved = {name: require_path(config[name], base) for name in
                ("binary", "base_dir", "corpus_manifest", "corpus_dir", "common_world")}
    resolved["candidates"] = [
        {"id": candidate["id"], "model": require_path(candidate["model"], base)}
        for candidate in candidates
    ]
    resolved["seed"] = config["seed"]
    resolved["chunks"] = config["chunks"]
    resolved["config"] = path
    if not resolved["binary"].is_file() or not resolved["common_world"].is_file():
        raise ValueError("binary and common_world must be files")
    if not resolved["base_dir"].is_dir() or not resolved["corpus_dir"].is_dir():
        raise ValueError("base_dir and corpus_dir must be directories")
    if not resolved["corpus_manifest"].is_file():
        raise ValueError("corpus_manifest must be a file")
    for candidate in resolved["candidates"]:
        if not candidate["model"].is_file():
            raise ValueError(f"model is missing: {candidate['model']}")
    return resolved


def schedule(analysis_dir: Path) -> list[dict]:
    trace = analysis_dir / "rollouts" / "trace.csv"
    with trace.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty rollout trace: {trace}")
    return [{key: row[key] for key in
             ("rollout_offset", "absolute_step", "target_file", "target_frame")}
            for row in rows]


def check_analysis(analysis_dir: Path, expected: dict, model: Path) -> dict:
    report = json.loads((analysis_dir / "analysis_report.json").read_text(encoding="utf-8"))
    provenance = json.loads((analysis_dir / "provenance.json").read_text(encoding="utf-8"))
    analysis = report["analysis"]
    if (analysis.get("weights_frozen") is not True
            or analysis.get("optimizer_constructed") is not False
            or analysis.get("optimizer_steps") != 0
            or analysis.get("backward_passes") != 0):
        raise ValueError("evaluation did not certify frozen, optimizer-free execution")
    if analysis.get("analysis_seed") != expected["seed"]:
        raise ValueError("analysis seed differs from requested seed")
    config = provenance["analysis_configuration"]
    if (config.get("frozen_rollouts") != [expected["chunks"]]
            or config.get("analysis_stride") != 1
            or config.get("model_path") != str(model)
            or config.get("state_path") != str(expected["common_world"])):
        raise ValueError("analysis invocation or checkpoint paths differ from request")
    if (provenance["paths"]["corpus_manifest"] != str(expected["corpus_manifest"])
            or provenance["paths"]["wav_dir"] != str(expected["eval_base"] / "OLD_WAVS")):
        raise ValueError("analysis corpus paths differ from request")
    corpus = provenance["corpus"]
    if corpus.get("fast_hash_mode") or not corpus.get("name_role_candidate_order"):
        raise ValueError("corpus provenance is incomplete or has no candidates")
    wav = analysis_dir / "rollouts" / "post_dsp.wav"
    info = prime.wav_info(wav)
    if info["frames"] != expected["chunks"] * CHUNK_FRAMES:
        raise ValueError("frozen render has unexpected frame count")
    trace = schedule(analysis_dir)
    if len(trace) != expected["chunks"] or any(not row["target_file"] for row in trace):
        raise ValueError("full target schedule is absent")
    if (trace[0]["rollout_offset"] != "1"
            or trace[-1]["rollout_offset"] != str(expected["chunks"])):
        raise ValueError("rollout trace has unexpected offsets")
    return {"report": report, "provenance": provenance, "wav": wav,
            "wav_info": info, "schedule": trace}


def run_analysis(config: dict, output: Path, candidate: dict) -> None:
    analysis_dir = output / "analyses" / candidate["id"]
    command = [str(config["binary"]), "--analysis-only", "--base-dir",
               str(config["eval_base"]), "--model", str(candidate["model"]),
               "--state", str(config["common_world"]), "--corpus-manifest",
               str(config["corpus_manifest"]), "--analysis-dir", str(analysis_dir),
               "--analysis-seed", str(config["seed"]), "--analysis-stride", "1",
               "--frozen-rollout", str(config["chunks"]), "--analysis-terminal", "quiet"]
    log = output / "analyses" / f"{candidate['id']}.stdout.log"
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"analysis failed with exit {result.returncode}; inspect {log}")


def package(config: dict, output: Path, run: bool = True) -> dict:
    """Create an isolated evaluation base and a private key beside blind WAVs."""
    output = output.resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    inputs = {key: identity(config[key]) for key in
              ("config", "binary", "corpus_manifest", "common_world")}
    inputs["models"] = {arm["id"]: identity(arm["model"])
                        for arm in config["candidates"]}
    all_inputs = [inputs[key] for key in ("config", "binary", "corpus_manifest", "common_world")]
    all_inputs.extend(inputs["models"].values())
    if any(Path(item["path"]).is_relative_to(output) for item in all_inputs):
        raise ValueError("output directory overlaps a canonical input")
    if output.is_relative_to(config["base_dir"]) or output.is_relative_to(config["corpus_dir"]):
        raise ValueError("output directory must be outside base_dir and corpus_dir")
    output.mkdir(parents=True, exist_ok=False)
    incomplete = output / "INCOMPLETE"
    incomplete.write_text("Evaluation or packaging did not finish. Do not use BLIND files yet.\n",
                          encoding="utf-8")
    (output / "analyses").mkdir()
    (output / "BLIND").mkdir()
    eval_base = output / "eval_base"
    eval_base.mkdir()
    (eval_base / "OLD_WAVS").symlink_to(config["corpus_dir"], target_is_directory=True)
    config = {**config, "eval_base": eval_base}
    if run:
        for arm in config["candidates"]:
            run_analysis(config, output, arm)
    arms = []
    for arm in config["candidates"]:
        analysis_dir = output / "analyses" / arm["id"]
        arms.append({**arm, **check_analysis(analysis_dir, config, arm["model"])})
    schedule0 = arms[0]["schedule"]
    if any(arm["schedule"] != schedule0 for arm in arms[1:]):
        raise ValueError("target file/frame schedules differ; package is not matched")
    corpus_hashes = [arm["provenance"]["corpus"]["aggregate_identity_sha256"]
                     for arm in arms]
    if len(set(corpus_hashes)) != 1:
        raise ValueError("corpus content inventories differ across arms")
    build_commits = [arm["report"]["identity"]["build_commit"] for arm in arms]
    if len(set(build_commits)) != 1:
        raise ValueError("analysis executables report different build commits")
    world_hashes = []
    for arm in arms:
        records = arm["provenance"]["checkpoint_files"]
        world_records = [entry for entry in records
                         if entry["path"] == str(config["common_world"])]
        if len(world_records) != 1:
            raise ValueError("common world hash absent from analysis provenance")
        world_hashes.append(world_records[0]["sha256"])
    if len(set(world_hashes)) != 1 or world_hashes[0] != inputs["common_world"]["sha256"]:
        raise ValueError("world content differs across arms")
    for original in all_inputs:
        if identity(Path(original["path"])) != original:
            raise ValueError(f"canonical input changed during evaluation: {original['path']}")
    verified_wavs = 0
    for entry in arms[0]["provenance"]["corpus"]["entries_in_manifest_order"]:
        filename = entry["file"]
        if Path(filename).name != filename:
            raise ValueError(f"unsafe corpus filename in provenance: {filename}")
        if entry["present"] and entry["sha256"]:
            wav = config["corpus_dir"] / filename
            if identity(wav)["sha256"] != entry["sha256"]:
                raise ValueError(f"corpus WAV changed during evaluation: {wav}")
            verified_wavs += 1
    metrics = {arm["id"]: panel.inspect_wav(arm["wav"]) for arm in arms}
    target_rms = min(
        min(item["rms_pcm"] for item in metrics.values()),
        min(item["rms_pcm"] * panel.HEADROOM_SAMPLE / item["peak_pcm"]
            for item in metrics.values()),
    )
    shuffled = list(arms)
    random.Random(config["seed"]).shuffle(shuffled)
    key = []
    for label, arm in zip(LABELS, shuffled):
        gain = target_rms / metrics[arm["id"]]["rms_pcm"]
        blind = output / "BLIND" / f"{label}.wav"
        output_metrics = panel.write_scaled(arm["wav"], blind, gain,
                                            config["chunks"] * CHUNK_FRAMES)
        key.append({"anonymous_id": label, "candidate_id": arm["id"],
                    "model": inputs["models"][arm["id"]],
                    "source_audio": identity(arm["wav"]),
                    "source_interval_frames": [0, config["chunks"] * CHUNK_FRAMES],
                    "source_trajectory_start_step": arm["provenance"]["world"]["global_step"],
                    "level_gain": gain, "blind_audio": identity(blind),
                    "descriptive_pcm": prime.signal_checks(blind),
                    "rms_pcm": output_metrics["rms_pcm"]})
    levels = [item["rms_pcm"] for item in key]
    if 20 * math.log10(max(levels) / min(levels)) > 0.01:
        raise ValueError("level-matched WAVs differ by more than 0.01 dB RMS")
    (output / "BLIND" / "LISTEN.txt").write_text(
        "Play A.wav through the final letter at the same playback level.\n"
        "Record useful structure, timing, harshness, stereo, and defects before opening ../key.json.\n"
        "The files are level matched. Letter assignment reveals no checkpoint order.\n",
        encoding="utf-8",
    )
    schedule_hash = hashlib.sha256(json.dumps(schedule0, sort_keys=True,
                                             separators=(",", ":")).encode()).hexdigest()
    receipt = {"schema": SCHEMA, "status": "complete", "interpretation":
               "frozen same-world v9 comparison; metrics describe signals, not musical quality",
               "strict_equivalence": False,
               "limits": ["same saved world can favor one learned model",
                          "different learned weights induce different post-forward target-error feedback",
                          "identical target schedule does not equal identical state trajectory",
                          "no Suno output or downstream transfer is evaluated"],
               "inputs": inputs, "corpus_dir": str(config["corpus_dir"]),
               "executable_reported_build_commit": build_commits[0],
               "corpus_content_inventory_sha256": corpus_hashes[0],
               "corpus_wavs_verified_after": verified_wavs,
               "common_world_sha256": world_hashes[0], "analysis_seed": config["seed"],
               "render_chunks": config["chunks"],
               "render_frames": config["chunks"] * CHUNK_FRAMES,
               "mastering": "identical analysis post-DSP path; no final mastering normalization",
               "level_matching": {"method": "global RMS, common -1 dBFS peak guard, PCM16 rounding",
                                  "target_rms_pcm": target_rms,
                                  "maximum_pair_difference_db": 20 * math.log10(max(levels) / min(levels))},
               "target_schedule_sha256": schedule_hash,
               "target_schedule": schedule0,
               "key": key}
    (output / "key.json").write_text(json.dumps(receipt, indent=2,
                                                 sort_keys=True, allow_nan=False) + "\n",
                                          encoding="utf-8")
    downstream = {"schema": "titan_downstream_attempts_v1",
                  "note": "Private template; fill one entry per generation attempt, including rejected rerolls. No Suno API is used.",
                  "attempts": [
                      {"anonymous_id": item["anonymous_id"],
                       "titan_source_checkpoint": item["model"],
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
        result = package(load_config(args.config), args.output_dir)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        parser.exit(1, f"matched evaluation failed: {error}\n")
    print(f"wrote {len(result['key'])} blind clips to {args.output_dir / 'BLIND'}")


if __name__ == "__main__":
    main()
