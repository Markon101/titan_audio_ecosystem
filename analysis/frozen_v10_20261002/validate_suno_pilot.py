#!/usr/bin/env python3
"""Check blinded manual-pilot controls and expose their residual confounds."""
import argparse
import hashlib
import json
import wave
from pathlib import Path

import numpy as np


def read(path):
    with wave.open(str(path), "rb") as wav:
        assert (wav.getnchannels(), wav.getframerate(), wav.getsampwidth()) == (2, 48000, 2)
        frames = wav.getnframes()
        data = np.frombuffer(wav.readframes(frames), "<i2").copy()
    return data.reshape(frames, 2).astype(np.float64) / 32768


def rms(values):
    return float(np.sqrt(np.mean(values * values)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", required=True, type=Path)
    args = parser.parse_args()
    root = args.pilot
    key = json.loads((root / "private/answer_key.json").read_text())
    audio = {}
    for code, item in key["conditions"].items():
        path = root / "uploads" / f"{code}.wav"
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"pilot WAV changed: {code}")
        audio[item["identity"]] = read(path)
    if any(len(wav) != key["frames"] for wav in audio.values()):
        raise RuntimeError("pilot durations differ")
    block = key["time_block_frames"]
    order = np.asarray(key["time_permutation"])
    if not np.array_equal(np.sort(order), np.arange(len(order))):
        raise RuntimeError("time control lost or duplicated blocks")
    intact = audio["intact_frozen_titan"]
    reordered = audio["short_block_time_order"]
    boundaries = np.arange(block, len(intact), block)
    def jump(values):
        return float(np.abs(values[boundaries] - values[boundaries - 1]).mean())
    # Edge fades and gain differ slightly; interior block waveform agreement
    # after mapping and RMS normalization is still a useful manipulation check.
    intact_blocks = intact.reshape(-1, block, 2)
    reordered_blocks = reordered.reshape(-1, block, 2)
    interior = slice(2400, block - 2400)
    local_correlations = []
    for index, source in enumerate(order):
        a = intact_blocks[source, interior].ravel()
        b = reordered_blocks[index, interior].ravel()
        local_correlations.append(float(np.dot(a, b) / max(1e-12, np.linalg.norm(a) * np.linalg.norm(b))))
    validation = {
        "schema": 1, "frames": key["frames"], "duration_seconds": key["duration_seconds"],
        "all_hashes_and_durations_match": True,
        "time_blocks_are_a_permutation": True,
        "time_fraction_moved": float(np.mean(order != np.arange(len(order)))),
        "time_median_interior_block_correlation": float(np.median(local_correlations)),
        "intact_boundary_mean_jump": jump(intact),
        "time_control_boundary_mean_jump": jump(reordered),
        "time_boundary_jump_ratio": jump(reordered) / max(1e-12, jump(intact)),
        "phase_surrogate_preprocessing": key["surrogate_validation"],
        "rms": {name: rms(wav) for name, wav in audio.items()},
        "limitation": "time-block reordering introduces boundary discontinuities; ratings cannot isolate global order from seams",
    }
    (root / "control_validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"fraction_moved": validation["time_fraction_moved"],
                      "block_correlation": validation["time_median_interior_block_correlation"],
                      "boundary_jump_ratio": validation["time_boundary_jump_ratio"]}))


if __name__ == "__main__":
    main()
