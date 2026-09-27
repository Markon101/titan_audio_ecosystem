#!/usr/bin/env python3
"""Create seam-matched ordered and block-shuffled audio upload controls.

This is an offline input study. It changes phrase order while retaining each
local block; it does not measure downstream transfer by itself.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import wave

import numpy as np


RATE = 48_000
PEAK_LIMIT = 10 ** (-1.0 / 20.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_pcm(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as reader:
        if (reader.getnchannels(), reader.getframerate(), reader.getsampwidth(),
                reader.getcomptype()) != (2, RATE, 2, "NONE"):
            raise ValueError("expected 48 kHz stereo PCM16 WAV")
        frames = reader.getnframes()
        if not 4 * RATE <= frames <= 60 * RATE:
            raise ValueError("input duration must be 4 to 60 seconds")
        data = reader.readframes(frames)
    if len(data) != frames * 4:
        raise ValueError("truncated input WAV")
    return np.frombuffer(data, dtype="<i2").reshape(-1, 2).astype(np.float64) / 32768.0


def add_seams(samples: np.ndarray, boundaries: list[int], fade_frames: int) -> np.ndarray:
    out = samples.copy()
    if fade_frames == 0:
        return out
    phase = (np.arange(fade_frames, dtype=np.float64) + 1) / (fade_frames + 1)
    before = np.cos(np.pi * phase / 2) ** 2
    after = np.sin(np.pi * phase / 2) ** 2
    for boundary in boundaries:
        out[boundary - fade_frames:boundary] *= before[:, None]
        out[boundary:boundary + fade_frames] *= after[:, None]
    return out


def write_pcm(path: Path, samples: np.ndarray) -> None:
    pcm = np.rint(samples * 32768).clip(-32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(RATE)
        writer.writeframes(pcm.tobytes())


def rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples * samples)))


def make_controls(source: Path, output_dir: Path, block_seconds: float = 4.0,
                  fade_ms: float = 10.0, seed: int = 4242) -> dict:
    if not math.isfinite(block_seconds) or not 0.25 <= block_seconds <= 15:
        raise ValueError("block_seconds must be finite and in [0.25, 15]")
    if not math.isfinite(fade_ms) or not 0 <= fade_ms <= 50:
        raise ValueError("fade_ms must be finite and in [0, 50]")
    source = source.resolve(strict=True)
    output_dir = output_dir.resolve(strict=False)
    if output_dir.exists() or source.is_relative_to(output_dir):
        raise ValueError("output directory must be new and separate from source")
    initial_hash = sha256(source)
    original = read_pcm(source)
    block_frames = round(block_seconds * RATE)
    fade_frames = round(fade_ms * RATE / 1000)
    full_blocks, tail_frames = divmod(len(original), block_frames)
    if full_blocks < 4 or fade_frames >= block_frames // 4:
        raise ValueError("need four full blocks and a fade shorter than one-quarter block")
    order = list(range(full_blocks))
    random.Random(seed).shuffle(order)
    if order == list(range(full_blocks)):
        order = order[1:] + order[:1]
    head = [original[index * block_frames:(index + 1) * block_frames] for index in order]
    if tail_frames:
        head.append(original[full_blocks * block_frames:])
    shuffled = np.concatenate(head, axis=0)
    boundaries = [index * block_frames for index in range(1, full_blocks)]
    if tail_frames:
        boundaries.append(full_blocks * block_frames)
    ordered_seamed = add_seams(original, boundaries, fade_frames)
    shuffled_seamed = add_seams(shuffled, boundaries, fade_frames)
    target_rms = min(rms(ordered_seamed), rms(shuffled_seamed))
    if target_rms <= 0:
        raise ValueError("silent input cannot form a matched control")
    gains = [target_rms / rms(item) for item in (ordered_seamed, shuffled_seamed)]
    highest = max(float(np.max(np.abs(item))) * gain
                  for item, gain in zip((ordered_seamed, shuffled_seamed), gains))
    headroom_gain = min(1.0, PEAK_LIMIT / highest) if highest > 0 else 1.0
    ordered_seamed *= gains[0] * headroom_gain
    shuffled_seamed *= gains[1] * headroom_gain
    output_dir.mkdir(parents=True, exist_ok=False)
    exact_copy = output_dir / "B_original_exact.wav"
    shutil.copyfile(source, exact_copy)
    if sha256(exact_copy) != initial_hash:
        raise RuntimeError("source copy did not match")
    files = []
    for name, array in (("O_ordered_seams.wav", ordered_seamed),
                        ("S_shuffled_4s.wav", shuffled_seamed)):
        path = output_dir / name
        write_pcm(path, array)
        decoded = read_pcm(path)
        files.append({"file": name, "sha256": sha256(path), "frames": len(decoded),
                      "stereo_rms": rms(decoded),
                      "sample_peak": float(np.max(np.abs(decoded)))})
    if sha256(source) != initial_hash:
        raise RuntimeError("source changed while making controls")
    spread_db = 20 * math.log10(max(item["stereo_rms"] for item in files)
                                / min(item["stereo_rms"] for item in files))
    if spread_db > 0.01 or any(item["sample_peak"] > PEAK_LIMIT + 1 / 32768 for item in files):
        raise RuntimeError("controls failed RMS or peak gate")
    receipt = {"schema": "titan_audio_temporal_control_v1",
               "source": str(source), "source_sha256": initial_hash,
               "exact_copy": exact_copy.name, "exact_copy_sha256": sha256(exact_copy),
               "sample_rate": RATE, "frames": len(original),
               "block_seconds_requested": block_seconds, "block_frames": block_frames,
               "full_blocks_permuted": full_blocks, "unchanged_tail_frames": tail_frames,
               "permutation": order, "seed": seed,
               "fade_ms_requested": fade_ms, "fade_frames_each_side": fade_frames,
               "seam_boundaries_frames": boundaries,
               "rms_match_gains": {"ordered": gains[0], "shuffled": gains[1]},
               "shared_peak_guard_gain": headroom_gain,
               "output_rms_spread_db": spread_db,
               "outputs": files,
               "interpretation": "Only order of full blocks was permuted. Both conditions received the same seam-window positions and matched total stereo RMS. Local within-block timing remains, and edit seams remain a possible cue. No downstream output was analyzed."}
    (output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (output_dir / "UPLOAD_NOTES.txt").write_text(
        "B_original_exact.wav: selected practical TITAN B clip, copied byte for byte.\n"
        "O_ordered_seams.wav: same block order with seam fades.\n"
        "S_shuffled_4s.wav: full blocks permuted, same seam fades.\n"
        "Use O versus S for a timing-order comparison; B is the practical input.\n"
        "Keep downstream model, mode, text prompt, upload length and settings fixed.\n"
        "Save every generated output with its input label and settings.\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-wav", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-seconds", type=float, default=4.0)
    parser.add_argument("--fade-ms", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()
    result = make_controls(args.source_wav, args.output_dir,
                           args.block_seconds, args.fade_ms, args.seed)
    print(f"wrote {len(result['outputs'])} matched control WAVs to {args.output_dir}")


if __name__ == "__main__":
    main()
