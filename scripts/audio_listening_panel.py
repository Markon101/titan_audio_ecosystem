#!/usr/bin/env python3
"""Make a local blinded listening panel from three fixed raw/EQ WAV pairs.

Config JSON: {"pairs": [{"id": "window_1", "raw": "raw.wav", "eq": "eq.wav"}, ...]}
Relative WAV paths are resolved beside the config file. Give listeners only
LISTEN.txt and listen_*.wav; keep key.json separate from their materials.
All six listening files share one RMS target, subject to a common peak guard.
"""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import wave


RATE = 48_000
HEADROOM_DBFS = -1.0
HEADROOM_SAMPLE = math.floor(32768.0 * 10 ** (HEADROOM_DBFS / 20.0))
LABELS = "ABCDEF"


def identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"input changed while hashing: {path}")
    return {"path": str(path), "bytes": before.st_size,
            "mtime_ns": before.st_mtime_ns, "sha256": digest.hexdigest()}


def pcm_samples(raw: bytes) -> array:
    samples = array("h")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples


def pcm_bytes(samples: array) -> bytes:
    if sys.byteorder == "little":
        return samples.tobytes()
    copy = array("h", samples)
    copy.byteswap()
    return copy.tobytes()


def inspect_wav(path: Path) -> dict:
    with wave.open(str(path), "rb") as reader:
        if (reader.getnchannels(), reader.getframerate(), reader.getsampwidth(),
                reader.getcomptype()) != (2, RATE, 2, "NONE"):
            raise ValueError(f"expected 48 kHz stereo PCM16 WAV: {path}")
        frames_expected = reader.getnframes()
        sum_squares = peak = sample_count = 0
        while raw := reader.readframes(65_536):
            samples = pcm_samples(raw)
            sample_count += len(samples)
            sum_squares += sum(sample * sample for sample in samples)
            peak = max(peak, max((abs(sample) for sample in samples), default=0))
    if frames_expected == 0 or sample_count != frames_expected * 2:
        raise ValueError(f"empty or truncated WAV: {path}")
    rms = math.sqrt(sum_squares / sample_count)
    if rms == 0:
        raise ValueError(f"silent WAV cannot form a level-matched panel: {path}")
    return {"frames": frames_expected, "rms_pcm": rms, "peak_pcm": peak}


def load_pairs(config_path: Path) -> tuple[dict, list[dict]]:
    config_path = config_path.resolve(strict=True)
    config_identity = identity(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != {"pairs"} or not isinstance(config["pairs"], list) or len(config["pairs"]) != 3:
        raise ValueError("config must contain exactly three pairs")
    pairs = []
    ids = set()
    used_paths = {config_path}
    for item in config["pairs"]:
        if not isinstance(item, dict) or set(item) != {"id", "raw", "eq"}:
            raise ValueError("each pair requires exactly id, raw, and eq")
        pair_id = item["id"]
        if not isinstance(pair_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", pair_id):
            raise ValueError(f"invalid pair id: {pair_id!r}")
        if pair_id.lower() in ids:
            raise ValueError(f"duplicate pair id: {pair_id}")
        ids.add(pair_id.lower())
        arms = {}
        for name in ("raw", "eq"):
            value = item[name]
            if not isinstance(value, str) or not value:
                raise ValueError(f"{pair_id}.{name} must be a nonempty path")
            candidate = Path(value)
            path = (candidate if candidate.is_absolute() else config_path.parent / candidate).resolve(strict=True)
            if path in used_paths:
                raise ValueError(f"source path overlap: {path}")
            used_paths.add(path)
            arms[name] = {"identity": identity(path), "metrics": inspect_wav(path)}
        if arms["raw"]["metrics"]["frames"] != arms["eq"]["metrics"]["frames"]:
            raise ValueError(f"pair {pair_id} has unequal frame counts")
        raw = arms["raw"]["metrics"]
        eq = arms["eq"]["metrics"]
        pairs.append({"id": pair_id, "arms": arms,
                      "requested_rms_pcm": min(raw["rms_pcm"], eq["rms_pcm"])})
    all_arms = [arm["metrics"] for pair in pairs for arm in pair["arms"].values()]
    requested_global_rms = min(arm["rms_pcm"] for arm in all_arms)
    peak_safe_rms = min(
        arm["rms_pcm"] * HEADROOM_SAMPLE / arm["peak_pcm"] for arm in all_arms
    )
    target_rms = min(requested_global_rms, peak_safe_rms)
    for pair in pairs:
        pair["target_rms_pcm"] = target_rms
        for name in ("raw", "eq"):
            pair["arms"][name]["gain"] = target_rms / pair["arms"][name]["metrics"]["rms_pcm"]
    if identity(config_path) != config_identity:
        raise ValueError("config changed while reading")
    return config_identity, pairs


def write_scaled(source: Path, destination: Path, gain: float, frames: int) -> dict:
    sum_squares = peak = count = 0
    with wave.open(str(source), "rb") as reader, wave.open(str(destination), "wb") as writer:
        writer.setnchannels(2)
        writer.setframerate(RATE)
        writer.setsampwidth(2)
        while raw := reader.readframes(65_536):
            samples = pcm_samples(raw)
            for index, sample in enumerate(samples):
                value = round(sample * gain)
                samples[index] = value
                sum_squares += value * value
                peak = max(peak, abs(value))
                count += 1
            writer.writeframesraw(pcm_bytes(samples))
    if count != frames * 2 or peak > HEADROOM_SAMPLE:
        raise ValueError(f"output frame count or peak constraint failed: {destination}")
    return {"frames": frames, "rms_pcm": math.sqrt(sum_squares / count),
            "peak_pcm": peak, "sha256": identity(destination)["sha256"]}


def build_panel(config_path: Path, output_dir: Path, seed: int) -> dict:
    config_identity, pairs = load_pairs(config_path)
    output_dir = output_dir.resolve(strict=False)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    for pair in pairs:
        for arm in pair["arms"].values():
            if Path(arm["identity"]["path"]).is_relative_to(output_dir):
                raise ValueError("output directory overlaps an input WAV")
    assignments = [(pair, name) for pair in pairs for name in ("raw", "eq")]
    random.Random(seed).shuffle(assignments)
    output_dir.mkdir(parents=True, exist_ok=False)
    key = {"schema": "titan_audio_listening_panel_v1", "seed": seed,
           "config_before": config_identity, "sample_peak_headroom_dbfs": HEADROOM_DBFS,
           "requested_global_rms_pcm": min(pair["requested_rms_pcm"] for pair in pairs),
           "applied_global_rms_pcm": pairs[0]["target_rms_pcm"],
           "pairs": [], "files": []}
    for pair in pairs:
        key["pairs"].append({
            "id": pair["id"],
            "requested_target_rms_pcm": pair["requested_rms_pcm"],
            "applied_target_rms_pcm": pair["target_rms_pcm"],
        })
    for label, (pair, name) in zip(LABELS, assignments):
        arm = pair["arms"][name]
        source = Path(arm["identity"]["path"])
        filename = f"listen_{label}.wav"
        output = write_scaled(source, output_dir / filename,
                              arm["gain"], arm["metrics"]["frames"])
        key["files"].append({"file": filename, "pair_id": pair["id"],
                             "arm": name, "source": arm["identity"],
                             "source_metrics": arm["metrics"],
                             "gain": arm["gain"], "output": output})
    for pair in pairs:
        results = [entry["output"]["rms_pcm"] for entry in key["files"] if entry["pair_id"] == pair["id"]]
        mismatch_db = abs(20 * math.log10(results[0] / results[1]))
        if mismatch_db > 0.01:
            raise ValueError(f"quantized pair RMS mismatch exceeds 0.01 dB: {pair['id']}")
    all_levels = [entry["output"]["rms_pcm"] for entry in key["files"]]
    if 20 * math.log10(max(all_levels) / min(all_levels)) > 0.01:
        raise ValueError("global panel RMS mismatch exceeds 0.01 dB")
    for pair in pairs:
        for arm in pair["arms"].values():
            path = Path(arm["identity"]["path"])
            after = identity(path)
            if after != arm["identity"]:
                raise ValueError(f"source changed while making panel: {path}")
            entry = next(item for item in key["files"] if item["source"]["path"] == str(path))
            entry["source_after"] = after
    config_after = identity(Path(config_identity["path"]))
    if config_after != config_identity:
        raise ValueError("config changed while making panel")
    key["config_after"] = config_after
    (output_dir / "LISTEN.txt").write_text(
        "Blind listening panel\n\n"
        "Play listen_A.wav through listen_F.wav with the same playback settings.\n"
        "For each file, record perceived harshness, stereo image, transient detail,\n"
        "and usefulness as an audio reference. Add free-form notes.\n"
        "Use key.json only after all ratings are complete.\n",
        encoding="utf-8",
    )
    (output_dir / "key.json").write_text(json.dumps(key, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        result = build_panel(args.config_json, args.output_dir, args.seed)
    except (OSError, ValueError, wave.Error, json.JSONDecodeError) as exc:
        parser.exit(1, f"listening panel failed: {exc}\n")
    print(f"wrote {len(result['files'])} blinded WAVs to {args.output_dir}")


if __name__ == "__main__":
    main()
