#!/usr/bin/env python3
"""Package existing TITAN WAVs for local, offline priming comparisons.

This tool reads canonical inputs and creates a new sidecar directory. It never
loads a model, trains, calls a downstream service, or chooses a musical winner.
"""

from __future__ import annotations

import argparse
from array import array
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import wave


SCHEMA_VERSION = 1
GENERATED_NAME_MARKERS = (
    "rust_ecosystem_out",
    "titan_prime",
    "titan_output",
    "ecosystem_out",
)
NEUTRAL_PROMPT = (
    "Create an original instrumental passage with a coherent progression "
    "and clear development.\n"
)
DESCRIPTOR_PROMPT_DEFAULT = (
    "Listener worksheet: describe this clip after listening before using these words as a prompt.\n"
    "Audible timbre: \n"
    "Pulse or rhythm: \n"
    "Development: \n"
    "Spatial image: \n"
)



def file_identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"input changed while hashing: {path}")
    return {
        "path": str(path),
        "bytes": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def output_identity(path: Path, output_dir: Path) -> dict:
    identity = file_identity(path)
    return {
        "relative_path": str(path.relative_to(output_dir)),
        "bytes": identity["bytes"],
        "sha256": identity["sha256"],
    }


def wav_info(path: Path) -> dict:
    with wave.open(str(path), "rb") as reader:
        info = {
            "channels": reader.getnchannels(),
            "sample_rate": reader.getframerate(),
            "sample_width_bytes": reader.getsampwidth(),
            "frames": reader.getnframes(),
            "compression": reader.getcomptype(),
        }
    if (
        info["channels"] != 2
        or info["sample_rate"] != 48_000
        or info["sample_width_bytes"] != 2
        or info["compression"] != "NONE"
    ):
        raise ValueError(f"expected 48 kHz stereo PCM16 WAV: {path}: {info}")
    info["duration_seconds"] = info["frames"] / info["sample_rate"]
    return info


def pcm16_samples(raw: bytes) -> array:
    samples = array("h")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples


def pcm16_bytes(samples: array) -> bytes:
    if sys.byteorder == "little":
        return samples.tobytes()
    copied = array("h", samples)
    copied.byteswap()
    return copied.tobytes()


def signal_checks(path: Path) -> dict:
    info = wav_info(path)
    frames = 0
    sum_l = sum_r = sum_ll = sum_rr = sum_lr = 0
    peak = clipped = both_zero = 0
    with wave.open(str(path), "rb") as reader:
        while raw := reader.readframes(65_536):
            samples = pcm16_samples(raw)
            for offset in range(0, len(samples), 2):
                left, right = samples[offset], samples[offset + 1]
                frames += 1
                sum_l += left
                sum_r += right
                sum_ll += left * left
                sum_rr += right * right
                sum_lr += left * right
                peak = max(peak, abs(left), abs(right))
                clipped += int(abs(left) >= 32_767 or abs(right) >= 32_767)
                both_zero += int(left == 0 and right == 0)
    if frames != info["frames"] or frames == 0:
        raise ValueError(f"empty or incomplete WAV: {path}")
    mean_l, mean_r = sum_l / frames, sum_r / frames
    var_l = sum_ll / frames - mean_l * mean_l
    var_r = sum_rr / frames - mean_r * mean_r
    covariance = sum_lr / frames - mean_l * mean_r
    correlation = (
        max(-1.0, min(1.0, covariance / math.sqrt(var_l * var_r)))
        if var_l > 0 and var_r > 0
        else None
    )

    def dbfs(power: float) -> float | None:
        return 10 * math.log10(power / (32_768 * 32_768)) if power > 0 else None

    return {
        "duration_seconds": info["duration_seconds"],
        "peak_dbfs": 20 * math.log10(peak / 32_768) if peak else None,
        "rms_left_dbfs": dbfs(sum_ll / frames),
        "rms_right_dbfs": dbfs(sum_rr / frames),
        "dc_left_fraction_full_scale": mean_l / 32_768,
        "dc_right_fraction_full_scale": mean_r / 32_768,
        "stereo_correlation": correlation,
        "clipped_frames": clipped,
        "both_channels_zero_fraction": both_zero / frames,
        "scope": "basic PCM checks only; no musical quality or downstream claim",
    }


def export_clip(source: Path, destination: Path, start: int, count: int) -> int:
    """Copy a sample-exact interval and add the legacy-sized edge fades."""
    fade = min(2048, count // 4)
    with wave.open(str(source), "rb") as reader, wave.open(str(destination), "wb") as writer:
        writer.setparams(reader.getparams())
        reader.setpos(start)
        remaining = count
        offset = 0
        while remaining:
            raw = reader.readframes(min(65_536, remaining))
            if not raw:
                raise ValueError("source WAV ended before requested clip")
            samples = pcm16_samples(raw)
            block_frames = len(samples) // 2
            if fade:
                for index in range(block_frames):
                    position = offset + index
                    if position < fade:
                        gain = position / fade
                    elif position >= count - fade:
                        gain = (count - position - 1) / fade
                    else:
                        continue
                    samples[2 * index] = int(samples[2 * index] * gain)
                    samples[2 * index + 1] = int(samples[2 * index + 1] * gain)
            writer.writeframesraw(pcm16_bytes(samples))
            remaining -= block_frames
            offset += block_frames
    return fade


def generated_name(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in GENERATED_NAME_MARKERS)


def candidate_specs(raw_specs: list[str]) -> list[tuple[str, Path, dict]]:
    """Validate candidate names and inputs before creating package outputs."""
    candidates = []
    seen = set()
    for spec in raw_specs:
        if "=" in spec:
            label, path_str = spec.split("=", 1)
        else:
            path_str = spec
            label = Path(spec).stem
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", label):
            raise ValueError(f"invalid candidate label: {label!r}")
        folded = label.lower()
        if folded == "legacy_prime" or re.fullmatch(r"clip_[0-9]+", folded):
            raise ValueError(f"reserved candidate label: {label!r}")
        if folded in seen:
            raise ValueError(f"duplicate candidate label: {label!r}")
        seen.add(folded)
        path = Path(path_str).resolve(strict=True)
        wav_info(path)
        candidates.append((label, path, file_identity(path)))
    return candidates


def corpus_audit(path: Path, wav_dir: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("corpus manifest has no entries array")
    declared: dict[str, int] = {}
    effective_candidates: dict[str, int] = {}
    conflicts = []
    family_roles: dict[str, set[str]] = {}
    for entry in entries:
        name = entry["file"]
        role = entry["role"]
        declared[role] = declared.get(role, 0) + 1
        family_roles.setdefault(entry["family"], set()).add(role)
        present = (wav_dir / name).is_file()
        eligible = (
            present
            and name.lower().endswith(".wav")
            and role != "exclude"
            and not generated_name(name)
        )
        if eligible:
            effective_candidates[role] = effective_candidates.get(role, 0) + 1
        if entry.get("provenance") == "titan_generated_quarantine" and role != "exclude":
            conflicts.append({"file": name, "declared_role": role, "production_filename_filter_excludes": generated_name(name)})
    return {
        "manifest": file_identity(path),
        "declared_role_counts": declared,
        "present_name_role_candidates": effective_candidates,
        "candidate_count_scope": "present WAVs after production name/role filtering; format and length indexing not checked",
        "quarantine_role_conflicts": conflicts,
        "mixed_role_families": sorted(name for name, roles in family_roles.items() if len(roles) > 1),
        "manifest_edited": False,
    }


def git_source() -> dict:
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def package(args: argparse.Namespace) -> dict:
    source = args.source_wav.resolve(strict=True)
    info = wav_info(source)
    if info["frames"] == 0:
        raise ValueError("source WAV has no audio frames")
    if not math.isfinite(args.clip_seconds) or args.clip_seconds <= 0:
        raise ValueError("--clip-seconds must be finite and positive")
    requested_frames = round(args.clip_seconds * info["sample_rate"])
    if requested_frames < 1:
        raise ValueError("--clip-seconds is shorter than one sample")
    count = min(requested_frames, info["frames"])
    source_identity = file_identity(source)
    candidates = candidate_specs(getattr(args, "candidate_clip", None) or [])
    metadata_data = None
    metadata_identity = None
    checkpoint_inputs = {}
    reported_audio_matches = None
    if args.metadata_json:
        metadata_identity = file_identity(args.metadata_json)
        metadata_data = json.loads(args.metadata_json.read_text(encoding="utf-8"))
        outputs = metadata_data.get("outputs", {})
        reported_audio = outputs.get("audio")
        if reported_audio and Path(reported_audio).is_file():
            reported_audio_matches = file_identity(Path(reported_audio))["sha256"] == source_identity["sha256"]
            if not reported_audio_matches:
                raise ValueError("metadata output audio bytes do not match --source-wav")
        run = metadata_data.get("run", {})
        if "completed_chunks" in run:
            chunks = run["completed_chunks"]
            if isinstance(chunks, bool) or not isinstance(chunks, int) or chunks < 0:
                raise ValueError("metadata run completed_chunks must be a nonnegative integer")
            chunk_size = metadata_data.get("field", {}).get("chunk_size", 4096)
            if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
                raise ValueError("metadata chunk_size must be a positive integer")
            if chunks * chunk_size != info["frames"]:
                raise ValueError("metadata run frame count does not match --source-wav")
        if "rendered_seconds" in run:
            rendered = run["rendered_seconds"]
            if not isinstance(rendered, (int, float)) or isinstance(rendered, bool) or not math.isfinite(rendered):
                raise ValueError("metadata rendered_seconds must be finite")
            tolerance = max(1 / info["sample_rate"], info["duration_seconds"] * 1e-6)
            if abs(rendered - info["duration_seconds"]) > tolerance:
                raise ValueError("metadata run duration does not match --source-wav")
        for key in ("model", "world", "optimizer", "morph_state"):
            reported = outputs.get(key)
            if reported:
                checkpoint_inputs[key] = file_identity(Path(reported)) if Path(reported).is_file() else {"path": reported, "present": False}
    corpus = corpus_audit(args.corpus_manifest, args.corpus_wav_dir) if args.corpus_manifest else None
    legacy_identity = file_identity(args.legacy_prime) if args.legacy_prime else None
    legacy_prompt_identity = file_identity(args.legacy_prompt) if args.legacy_prompt else None
    if args.legacy_prime:
        wav_info(args.legacy_prime)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    (output / "clips").mkdir()
    (output / "prompts").mkdir()
    maximum_start = info["frames"] - count
    seeded_start = random.Random(args.seed ^ int(source_identity["sha256"][:16], 16)).randint(0, maximum_start)
    selections = (
        ("opening", 0),
        ("middle", maximum_start // 2),
        ("ending", maximum_start),
        ("seeded", seeded_start),
    )
    unique: dict[int, list[str]] = {}
    for label, start in selections:
        unique.setdefault(start, []).append(label)
    clips = []
    for index, (start, labels) in enumerate(unique.items(), 1):
        clip_id = f"clip_{index:02d}"
        target = output / "clips" / f"{clip_id}.wav"
        fade = export_clip(source, target, start, count)
        clips.append({
            "id": clip_id,
            "selection_labels": labels,
            "start_frame": start,
            "end_frame_exclusive": start + count,
            "fade_frames_per_edge": fade,
            "processing": "sample-exact PCM16 interval with integer edge fades; no gain or resampling",
            "wav": output_identity(target, output),
            "signal_checks": signal_checks(target),
        })
    if args.legacy_prime:
        target = output / "clips" / "legacy_prime.wav"
        shutil.copyfile(args.legacy_prime, target)
        clips.append({
            "id": "legacy_prime",
            "selection_labels": ["existing_export"],
            "source": legacy_identity,
            "start_frame": None,
            "end_frame_exclusive": None,
            "processing": "byte-for-byte copy of existing prime; source offsets unknown",
            "wav": output_identity(target, output),
            "signal_checks": signal_checks(target),
        })

    candidate_identities = []
    for label, cand_path, cand_identity in candidates:
        candidate_identities.append(cand_identity)
        target = output / "clips" / f"{label}.wav"
        shutil.copyfile(cand_path, target)
        clips.append({
            "id": label,
            "selection_labels": ["candidate_remediation"],
            "source": cand_identity,
            "start_frame": None,
            "end_frame_exclusive": None,
            "processing": "candidate prime; checked 48 kHz stereo PCM16 WAV",
            "wav": output_identity(target, output),
            "signal_checks": signal_checks(target),
        })

    prompts = []
    neutral = output / "prompts" / "neutral.txt"
    neutral.write_text(NEUTRAL_PROMPT, encoding="utf-8")
    prompts.append({"id": "neutral", "applies_to": [clip["id"] for clip in clips], "file": output_identity(neutral, output)})

    desc_identity = None
    if getattr(args, "descriptor_prompt", None) or getattr(args, "include_descriptor_prompt", False):
        desc_target = output / "prompts" / "audible_descriptors.txt"
        if getattr(args, "descriptor_prompt", None):
            desc_source = Path(args.descriptor_prompt).resolve(strict=True)
            desc_identity = file_identity(desc_source)
            shutil.copyfile(desc_source, desc_target)
            prompts.append({
                "id": "audible_descriptors",
                "applies_to": [clip["id"] for clip in clips],
                "source": desc_identity,
                "file": output_identity(desc_target, output),
                "status": "provided_unverified",
                "warning": "verify descriptors against each clip before use",
            })
        else:
            desc_target.write_text(DESCRIPTOR_PROMPT_DEFAULT, encoding="utf-8")
            prompts.append({
                "id": "audible_descriptors",
                "applies_to": [clip["id"] for clip in clips],
                "file": output_identity(desc_target, output),
                "status": "blank_listener_template",
                "warning": "fill and verify clip-specific descriptors before use",
            })

    if args.legacy_prompt:
        target = output / "prompts" / "legacy_run_prompt.txt"
        shutil.copyfile(args.legacy_prompt, target)
        prompts.append({"id": "legacy_run_prompt", "applies_to": ["legacy_prime"] if args.legacy_prime else [], "source": legacy_prompt_identity, "file": output_identity(target, output), "warning": "whole-run proxy prompt; confirm audible descriptors before use"})
    notes = output / "listening_notes.csv"
    with notes.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("clip_id", "audible_timbre", "pulse_or_rhythm", "development", "defects", "usable_as_reference", "listener"))
        for clip in clips:
            writer.writerow((clip["id"], "", "", "", "", "", ""))
    canonical_inputs = [source_identity, metadata_identity, legacy_identity, legacy_prompt_identity]
    canonical_inputs.extend(candidate_identities)
    if desc_identity:
        canonical_inputs.append(desc_identity)
    canonical_inputs.extend(checkpoint_inputs.values())
    if corpus:
        canonical_inputs.append(corpus["manifest"])

    checked_paths = set()
    for identity in canonical_inputs:
        if identity is None or "sha256" not in identity or identity["path"] in checked_paths:
            continue
        checked_paths.add(identity["path"])
        if file_identity(Path(identity["path"])) != identity:
            raise RuntimeError(f"canonical input changed while packaging: {identity['path']}")
    receipt = {
        "schema": {"name": "titan_prime_package", "version": SCHEMA_VERSION},
        "interpretation": "candidate clips and basic PCM checks; no musical ranking or downstream result",
        "tool": file_identity(Path(__file__)),
        "git_source": git_source(),
        "source_wav": source_identity,
        "source_wav_format": info,
        "clip_seconds_requested": args.clip_seconds,
        "clip_frames": count,
        "selection_seed": args.seed,
        "run_metadata": metadata_identity,
        "run_metadata_reported_build": metadata_data.get("build") if metadata_data else None,
        "run_metadata_reported_run": metadata_data.get("run") if metadata_data else None,
        "metadata_audio_bytes_match": reported_audio_matches,
        "checkpoint_inputs": checkpoint_inputs,
        "corpus_audit": corpus,
        "clips": clips,
        "prompts": prompts,
        "listening_notes": output_identity(notes, output),
        "canonical_inputs_mutated": False,
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-wav", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--clip-seconds", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--corpus-manifest", type=Path)
    parser.add_argument("--corpus-wav-dir", type=Path, default=Path("/sdcard/Download/OLD_WAVS"))
    parser.add_argument("--legacy-prime", type=Path)
    parser.add_argument("--legacy-prompt", type=Path)
    parser.add_argument("--candidate-clip", action="append", default=[], help="Candidate clip in format LABEL=PATH")
    parser.add_argument("--descriptor-prompt", type=Path, help="Path to custom audible descriptor prompt text file")
    parser.add_argument("--include-descriptor-prompt", action="store_true", help="Include default audible descriptor prompt")
    arguments = parser.parse_args()
    try:
        result = package(arguments)
    except (OSError, ValueError, RuntimeError, wave.Error, json.JSONDecodeError) as error:
        parser.exit(1, f"prime package failed: {error}\n")
    print(f"wrote {len(result['clips'])} clips and receipt to {arguments.output_dir}")


if __name__ == "__main__":
    main()
