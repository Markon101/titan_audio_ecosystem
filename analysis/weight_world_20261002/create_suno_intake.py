#!/usr/bin/env python3
"""Create the user-requested Downloads intake, keeping the answer key local."""
import argparse
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = ROOT / "analysis/frozen_host_clamp_20261002/runs/suno_quietcut_pilot"
    old_key = json.loads((source / "private/answer_key.json").read_text())
    wanted = {"intact_frozen_titan", "short_block_time_order", "shared_phase_spectral_surrogate"}
    candidates = [(code, item) for code, item in old_key["conditions"].items()
                  if item["identity"] in wanted]
    assert len(candidates) == 3
    rng = random.Random(20261003)
    rng.shuffle(candidates)
    private = HERE / "runs/suno_intake_private"
    if args.output.exists() or private.exists():
        raise FileExistsError("intake or private key already exists; preserve it")
    args.output.mkdir(parents=True)
    private.mkdir(parents=True)
    for name in ("anonymous_inputs", "generated_wavs", "notes"):
        (args.output / name).mkdir()
    key = {"schema": 1, "conditions": {}, "output_directory": str(args.output)}
    public = {"schema": 1, "duration_seconds": old_key["duration_seconds"],
              "sample_rate": old_key["sample_rate"], "inputs": {}}
    for index, (old_code, item) in enumerate(candidates, 1):
        code = f"S{index:02}"
        original = source / "uploads" / f"{old_code}.wav"
        if sha(original) != item["sha256"]:
            raise RuntimeError("source pilot WAV changed")
        destination = args.output / "anonymous_inputs" / f"{code}.wav"
        shutil.copyfile(original, destination)
        public["inputs"][code] = {"file": f"anonymous_inputs/{code}.wav", "sha256": sha(destination)}
        key["conditions"][code] = {"identity": item["identity"], "sha256": sha(destination)}
    (private / "answer_key.json").write_text(json.dumps(key, indent=2) + "\n")
    (args.output / "input_receipt.json").write_text(json.dumps(public, indent=2) + "\n")
    schedule = []
    for repeat in range(1, 3):
        ids = list(public["inputs"])
        rng.shuffle(ids)
        ids.append("PROMPT_ONLY")
        rng.shuffle(ids)
        for code in ids:
            schedule.append((len(schedule) + 1, code, repeat))
    with (args.output / "suggested_upload_order.csv").open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["order", "anonymous_input", "repeat"])
        writer.writerows(schedule)
    (args.output / "README.md").write_text(
        "# Titan manual Suno pilot\n\nPut generated WAVs in `generated_wavs/` and your MD notes in `notes/`. "
        "The three 21.845-second inputs in `anonymous_inputs/` are anonymous. "
        "Use the same instrumental prompt, model/custom model, mode, and sliders for all inputs. "
        "Record any setting the interface does not expose as 'not exposed'. "
        "`suggested_upload_order.csv` gives two randomized repeats per input plus "
        "a separate prompt-only baseline; a smaller first batch is fine. "
        "Retain every output, including rejected rerolls, and record upload trims. "
        "Rate outputs before condition identities are revealed. The answer key stays "
        "in the research workspace. No generation has been submitted automatically.\n"
    )
    (args.output / "notes/pilot_notes.md").write_text(
        "# Suno instrumental pilot notes\n\n"
        "## Shared generation settings\n\n"
        "- Date/time:\n- Suno model/version:\n- Custom model selection (or none):\n"
        "- Exact prompt:\n- Lyrics/instrumental selection:\n- Inspiration/cover/other mode:\n"
        "- Reference/audio strength:\n- Style/prompt strength:\n- Weirdness:\n"
        "- Other settings:\n- Seed/variation IDs exposed?\n- Upload trim/duration:\n\n"
        "## Every attempt, including rejected rerolls\n\n"
        "| Output WAV filename | Anonymous input or PROMPT_ONLY | Order | Seed/variation ID | Settings changed | Kept/rejected | Notes |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n| | | | | | | |\n\n"
        "## Blind ratings\n\nUse 1–5, higher is better; artifacts uses higher = more artifacts. "
        "Use N/A for vocal interest in instrumental outputs.\n\n"
        "| Output WAV | Preference | Fullness | Cleanliness | Structured weirdness | Stereo motion | Vocal interest | Transitions | Aliveness | Artifacts | Notes |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        "| | | | | | | | | | | |\n"
    )
    (HERE / "suno_intake_receipt.json").write_text(json.dumps({
        "schema": 1, "public_directory": str(args.output), "public_inputs": public,
        "private_answer_key_sha256": sha(private / "answer_key.json"),
        "generation_outputs_recorded": 0,
    }, indent=2) + "\n")
    print(json.dumps({"directory": str(args.output), "anonymous_inputs": list(public["inputs"])}))


if __name__ == "__main__":
    main()
