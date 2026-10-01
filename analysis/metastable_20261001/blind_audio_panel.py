#!/usr/bin/env python3
"""Copy final 60-second primes into a local blinded listening panel."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import wave


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def wav_info(path):
    with wave.open(str(path), "rb") as stream:
        return {"sample_rate": stream.getframerate(), "channels": stream.getnchannels(),
                "bits_per_sample": stream.getsampwidth() * 8,
                "duration_seconds": stream.getnframes() / stream.getframerate()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--arm", nargs=2, action="append", required=True,
                        metavar=("LABEL", "PRIME_WAV"))
    args = parser.parse_args()
    if len(args.arm) != 3 or len({item[0] for item in args.arm}) != 3:
        parser.error("provide exactly three distinct --arm LABEL PRIME_WAV pairs")
    if args.out.exists():
        parser.error("panel directory exists")
    sources = [(label, Path(path).resolve(strict=True)) for label, path in args.arm]
    for _, path in sources:
        info = wav_info(path)
        if (info["sample_rate"], info["channels"], info["bits_per_sample"]) != (48000, 2, 16):
            parser.error(f"prime WAV format mismatch: {path}")
        if not 59.0 <= info["duration_seconds"] <= 60.1:
            parser.error(f"prime WAV duration mismatch: {path}")
    random.Random(args.seed).shuffle(sources)
    args.out.mkdir(parents=True)
    private_key = {}
    public = []
    for index, (label, path) in enumerate(sources, 1):
        name = f"clip_{index:02d}.wav"
        target = args.out / name
        source_hash = sha(path)
        shutil.copyfile(path, target)
        if sha(target) != source_hash:
            raise RuntimeError(f"panel copy verification failed: {name}")
        private_key[name] = {"label": label, "source": str(path),
                             "source_sha256": source_hash}
        public.append({"file": name, "sha256": source_hash,
                       **wav_info(target)})
    (args.out / "key.private.json").write_text(json.dumps({"seed": args.seed,
        "mapping": private_key}, indent=2, sort_keys=True) + "\n")
    (args.out / "receipt.json").write_text(json.dumps({"schema": 1,
        "blinded_clips": public,
        "scope": "subjective output panel; different source intervals and corpus targets prevent causal quality attribution"},
        indent=2, sort_keys=True) + "\n")
    (args.out / "LISTENING.md").write_text(
        "# Blind TITAN audio panel\n\n"
        "Listen to clip_01.wav, clip_02.wav, and clip_03.wav in a fresh order. "
        "Rate coherence, beauty, strangeness, morphing structure, stereo comfort, "
        "and usefulness as a Suno conditioning upload. Note any clipping, harshness, "
        "or abrupt discontinuities. Reveal key.private.json only after ratings are saved.\n"
    )
    print(json.dumps({"panel": str(args.out), "clips": len(public),
                      "key": "key.private.json (keep private until rating)"}))


if __name__ == "__main__":
    main()
