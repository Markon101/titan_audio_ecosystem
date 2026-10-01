#!/usr/bin/env python3
"""Audit and prepare separate new-family and existing-family corpus controls.

The original 41 WAVs and manifest are read only. New-family files are local
user audio outside the small v10 manifest; existing-family controls are
deterministic -0.8 dB bell-EQ variants of six *training* files. Neither arm
changes development or validation membership.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import wave

HERE = Path(__file__).resolve().parent
ORIGINAL_DIR = Path("/sdcard/Download/OLD_WAVS")
ORIGINAL_MANIFEST = Path("/sdcard/Download/titan_corpus_manifest_v7_sml.json")
NEW_DIR = Path("/sdcard/Music/Audio_Lab/SOUND_MASTERING_AUDIO")
OUTPUT = HERE / "runs/corpus_forks"

NEW_FILES = [
    ("Echo Pulse_223417525.wav", "echo_pulse"),
    ("Run Free ext v2_231733492.wav", "run_free"),
    ("Bubblegum Dreams remix v1_230429505.wav", "bubblegum_dreams"),
    ("Path of the Wise master_225134336.wav", "path_of_the_wise"),
    ("Measured in Quanta 2_213908974.wav", "measured_in_quanta"),
    ("Infinite Love ext v1-master1_030403250.wav", "infinite_love"),
]
EXISTING_FILES = [
    "Architect of Change_014405555.wav",
    "Beats of the Night_014407065.wav",
    "Concrete Desires_014428538.wav",
    "Beyond the Veil_014411812.wav",
    "Chaos in Capitol_014414870.wav",
    "AlgoRhythm s in Digital Motion - Hearts Align_0144.wav",
]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def wav_info(path):
    with wave.open(str(path), "rb") as stream:
        return {"sample_rate": stream.getframerate(), "channels": stream.getnchannels(),
                "sample_width_bytes": stream.getsampwidth(),
                "duration_seconds": stream.getnframes() / stream.getframerate()}


def audit():
    manifest = json.loads(ORIGINAL_MANIFEST.read_text())
    entries = manifest["entries"]
    by_name = {entry["file"]: entry for entry in entries}
    if len(by_name) != 41 or len(by_name) != len(entries):
        raise RuntimeError("source manifest is not the expected 41-file small corpus")
    all_families = {entry["family"] for entry in entries}
    training_families = {entry["family"] for entry in entries if entry["role"] == "train"}
    source = {entry["file"]: ORIGINAL_DIR / entry["file"] for entry in entries}
    if any(not path.is_file() for path in source.values()):
        raise RuntimeError("original corpus has a missing manifest file")
    original_hashes = {name: sha(path) for name, path in source.items()}
    duplicate_groups = {}
    for name, digest in original_hashes.items():
        duplicate_groups.setdefault(digest, []).append(name)
    duplicate_groups = {digest: names for digest, names in duplicate_groups.items() if len(names) > 1}
    for names in duplicate_groups.values():
        identities = {(by_name[name]["role"], by_name[name]["family"]) for name in names}
        if len(identities) > 1:
            raise RuntimeError(f"cross-role or cross-family duplicate bytes: {names}")
    additions = {"new_families": [], "existing_variants": []}
    for name, family in NEW_FILES:
        path = NEW_DIR / name
        if name in by_name or family in all_families or not path.is_file():
            raise RuntimeError(f"new-family identity conflicts or is absent: {name}")
        digest = sha(path)
        if digest in original_hashes.values():
            raise RuntimeError(f"new-family audio duplicates source corpus: {name}")
        info = wav_info(path)
        if (info["sample_rate"], info["channels"], info["sample_width_bytes"]) != (48000, 2, 2):
            raise RuntimeError(f"new-family WAV format differs: {name}")
        additions["new_families"].append({"file": name, "family": family,
                                          "source": str(path), "source_sha256": digest, **info})
    if (len({item["source_sha256"] for item in additions["new_families"]}) != len(NEW_FILES) or
            len({item["family"] for item in additions["new_families"]}) != len(NEW_FILES)):
        raise RuntimeError("new-family fork contains duplicate bytes or family IDs")
    for name in EXISTING_FILES:
        entry = by_name[name]
        if entry["role"] != "train" or entry["family"] not in training_families:
            raise RuntimeError(f"existing-family control uses a held-out family: {name}")
        info = wav_info(source[name])
        if (info["sample_rate"], info["channels"], info["sample_width_bytes"]) != (48000, 2, 2):
            raise RuntimeError(f"existing-family WAV format differs: {name}")
        additions["existing_variants"].append({"file": name, "family": entry["family"],
                                                "source": str(source[name]),
                                                "source_sha256": original_hashes[name], **info})
    duration_new = sum(item["duration_seconds"] for item in additions["new_families"])
    duration_existing = sum(item["duration_seconds"] for item in additions["existing_variants"])
    if abs(duration_new - duration_existing) / duration_new > 0.10:
        raise RuntimeError("added audio durations differ by more than 10 percent")
    return manifest, source, original_hashes, duplicate_groups, additions, duration_new, duration_existing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="create isolated corpus directories")
    args = parser.parse_args()
    manifest, source, original_hashes, duplicate_groups, additions, duration_new, duration_existing = audit()
    if not args.build:
        print(json.dumps({"audited": True, "new_families": len(NEW_FILES),
                          "existing_variants": len(EXISTING_FILES),
                          "within_family_duplicate_pairs": len(duplicate_groups),
                          "new_seconds": duration_new, "existing_seconds": duration_existing}))
        return
    if OUTPUT.exists():
        parser.error("corpus fork output already exists")
    OUTPUT.mkdir(parents=True)
    env = os.environ.copy()
    env["LD_PRELOAD"] = "/data/data/com.termux/files/usr/lib/libc++_shared.so"
    for arm in ("new_families", "existing_variants"):
        directory = OUTPUT / arm
        directory.mkdir()
        for name, path in source.items():
            (directory / name).symlink_to(path)
        entries = [dict(entry) for entry in manifest["entries"]]
        if arm == "new_families":
            for item in additions[arm]:
                (directory / item["file"]).symlink_to(item["source"])
                entries.append({"file": item["file"], "role": "train",
                                "family": item["family"],
                                "provenance": "local_user_audio_new_family_fork_20261001"})
        else:
            for item in additions[arm]:
                name = "EQ08_" + item["file"]
                output = directory / name
                subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i",
                                item["source"], "-af", "equalizer=f=3500:t=q:w=0.7:g=-0.8",
                                "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(output)],
                               env=env, check=True)
                if wav_info(output)["duration_seconds"] != item["duration_seconds"]:
                    raise RuntimeError(f"variant duration changed: {name}")
                item["variant_file"] = name
                item["variant_sha256"] = sha(output)
                entries.append({"file": name, "role": "train", "family": item["family"],
                                "provenance": "existing_family_mild_eq_control_20261001"})
        fork_manifest = {**manifest, "generated_by": "metastable_corpus_fork_20261001",
                         "entries": entries}
        (OUTPUT / f"{arm}_manifest.json").write_text(json.dumps(fork_manifest, indent=2) + "\n")
    receipt = {"schema": 1, "source_manifest": str(ORIGINAL_MANIFEST),
               "source_manifest_sha256": sha(ORIGINAL_MANIFEST),
               "fork_manifest_sha256": {
                   arm: sha(OUTPUT / f"{arm}_manifest.json")
                   for arm in ("new_families", "existing_variants")},
               "source_file_sha256": original_hashes,
               "within_family_duplicate_groups": duplicate_groups,
               "new_families": additions["new_families"],
               "existing_variants": additions["existing_variants"],
               "added_duration_seconds": {"new_families": duration_new,
                                          "existing_variants": duration_existing},
               "variant_filter": "equalizer=f=3500:t=q:w=0.7:g=-0.8; 48k stereo pcm_s16le"}
    (HERE / "corpus_forks_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"built": True, "output": str(OUTPUT),
                      "new_seconds": duration_new, "existing_seconds": duration_existing}))


if __name__ == "__main__":
    main()
