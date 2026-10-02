#!/usr/bin/env python3
"""Build byte-audited six-slot corpora and two fixed exogenous schedules."""

import hashlib
import json
from pathlib import Path
import random
import wave

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PRIOR = ROOT / "analysis/metastable_20261001"
OUTPUT = HERE / "runs/corpus"
SOURCE_MANIFEST = Path("/sdcard/Download/titan_corpus_manifest_v7_sml.json")
ORIGINAL_DIR = Path("/sdcard/Download/OLD_WAVS")
SOURCE_STEP = 58107
EPISODE_CHUNKS = 256
CHUNK_SIZE = 4096
SCHEDULE_SLOTS = {20261002: [0, 1, 2, 3], 20261003: [2, 3, 4, 5]}


def sha(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def frames(path):
    with wave.open(str(path), "rb") as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (48000, 2, 2):
            raise ValueError(f"expected 48 kHz stereo PCM16 WAV: {path}")
        return source.getnframes()


def main():
    if OUTPUT.exists():
        raise RuntimeError("matched exposure corpus already exists")
    prior = json.loads((PRIOR / "acute_exposure_forks_receipt.json").read_text())
    source_manifest = json.loads(SOURCE_MANIFEST.read_text())
    heldout = [dict(item) for item in source_manifest["entries"]
               if item["role"] in ("development", "validation")]
    if len(heldout) != 11:
        raise RuntimeError("held-out roles changed")
    heldout_hashes = {item["file"]: sha(ORIGINAL_DIR / item["file"]) for item in heldout}
    source = {arm: prior["arms"][arm]["train_files"] for arm in "abc"}
    for arm in "abc":
        if len(source[arm]) != 6 or len({item["sha256"] for item in source[arm]}) != 6:
            raise RuntimeError(f"arm {arm} lacks six unique sources")
        for item in source[arm]:
            if sha(item["source"]) != item["sha256"]:
                raise RuntimeError(f"source bytes changed: {item['source']}")
            if item["sha256"] in heldout_hashes.values():
                raise RuntimeError(f"training bytes overlap held-out source: {item['source']}")
    OUTPUT.mkdir(parents=True)
    receipt = {"schema": 1, "source_step": SOURCE_STEP,
               "source_checkpoint_receipt_sha256": sha(PRIOR / "frozen_step_58107_receipt.json"),
               "source_manifest_sha256": sha(SOURCE_MANIFEST),
               "heldout_sha256": heldout_hashes,
               "arms": {}, "schedules": {}}
    for arm in "abc":
        directory = OUTPUT / arm
        directory.mkdir()
        entries = list(heldout)
        for item in heldout:
            (directory / item["file"]).symlink_to(ORIGINAL_DIR / item["file"])
        slots = []
        for slot, item in enumerate(source[arm]):
            alias = f"slot_{slot:02}.wav"
            content = Path(item["source"]).resolve(strict=True)
            (directory / alias).symlink_to(content)
            entries.append({"file": alias, "role": "train", "family": f"slot_{slot:02}",
                            "provenance": f"matched_exposure_{arm}_20261001"})
            slots.append({"slot": slot, "alias": alias, "source": str(content),
                          "source_sha256": item["sha256"], "source_family": item["family"],
                          "frames": frames(content)})
        document = {"schema_version": source_manifest["schema_version"],
                    "generated_by": "matched_exposure_20261001", "entries": entries}
        manifest = OUTPUT / f"{arm}_manifest.json"
        manifest.write_text(json.dumps(document, indent=2) + "\n")
        receipt["arms"][arm] = {"manifest_sha256": sha(manifest), "slots": slots}
    for seed, choices in SCHEDULE_SLOTS.items():
        rng = random.Random(seed)
        slots = choices.copy()
        rng.shuffle(slots)
        episodes = []
        for index, slot in enumerate(slots):
            shortest = min(receipt["arms"][arm]["slots"][slot]["frames"] for arm in "abc")
            max_start_chunk = (shortest - EPISODE_CHUNKS * CHUNK_SIZE) // CHUNK_SIZE
            if max_start_chunk < 0:
                raise RuntimeError(f"slot {slot} is shorter than a full target episode")
            start_frame = rng.randrange(max_start_chunk + 1) * CHUNK_SIZE
            episodes.append({"start_step": SOURCE_STEP + index * EPISODE_CHUNKS,
                             "chunks": EPISODE_CHUNKS, "slot": slot,
                             "source_frame": start_frame})
        schedule = {"schema": 1, "seed": seed,
                    "slots": [f"slot_{i:02}.wav" for i in range(6)],
                    "episodes": episodes}
        path = HERE / f"schedule_{seed}.json"
        path.write_text(json.dumps(schedule, indent=2, sort_keys=True) + "\n")
        receipt["schedules"][str(seed)] = {"sha256": sha(path), "episodes": episodes,
                                            "end_step": SOURCE_STEP + len(episodes) * EPISODE_CHUNKS}
    (HERE / "matched_exposure_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"arms": list(receipt["arms"]),
                      "seeds": list(receipt["schedules"]),
                      "end_step": SOURCE_STEP + 4 * EPISODE_CHUNKS}))


if __name__ == "__main__":
    main()
