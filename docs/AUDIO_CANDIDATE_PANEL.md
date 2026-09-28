# Blind panel for existing TITAN WAVs

Use `scripts/audio_candidate_panel.py` to compare two or three existing 48 kHz
stereo PCM16 outputs, including outputs from different substrates. It only
reads files and writes a new sidecar directory. It does not load models,
train, render new trajectories, or call Suno.

Create a JSON config with one interval per candidate. Use exact sample frames
so all clips have equal duration. `checkpoint`, `world`, `run_metadata`, and
`executable` can be `null` or omitted when unknown; record what is known
instead of inferring missing lineage.

```json
{
  "seed": 20260928,
  "clip_frames": 1437696,
  "candidates": [
    {
      "id": "legacy",
      "wav": "/path/to/legacy/full_output.wav",
      "start_frame": 0,
      "checkpoint": "/path/to/legacy/final_model.safetensors",
      "world": "/path/to/legacy/final_world.bin",
      "run_metadata": "/path/to/legacy/run_metadata.json",
      "executable": null,
      "processing_note": "full short-run output"
    },
    {
      "id": "msfield",
      "wav": "/path/to/msfield/full_output.wav",
      "start_frame": 0,
      "checkpoint": "/path/to/msfield/final_model.safetensors",
      "world": "/path/to/msfield/final_world.bin",
      "run_metadata": "/path/to/msfield/run_metadata.json",
      "executable": null,
      "processing_note": "full short-run output"
    }
  ]
}
```

Run:

```bash
python scripts/audio_candidate_panel.py \
  --config /path/to/candidates.json \
  --output-dir /path/to/new_private_package
```

`BLIND/` contains only anonymous WAVs and `LISTEN.txt`. Keep `key.json`,
`source_intervals/`, and `downstream_receipt_template.json` private until
listening ratings are recorded. The tool refuses existing output directories.
If packaging fails after output creation, `INCOMPLETE` remains and the blind
files should not be used.

Each candidate receives the same sample-exact interval extraction, integer
edge fades of up to 2,048 frames per side, and global RMS matching with a
common −1 dBFS PCM peak guard. The receipt records source interval frames,
source/interval/blind SHA-256 hashes, checkpoint and world declarations,
actual gain, descriptive PCM checks, and maximum output RMS mismatch. When
run metadata is supplied, the tool checks any still-accessible reported audio,
model, and world outputs against those declarations and checks the full WAV
frame count. The metadata's reported build identifier is retained; an
optional executable hash is only a declaration unless a separate run receipt
establishes that those exact binary bytes produced the WAV.

This is an **unmatched trajectory comparison**. A final checkpoint is an
associated lineage artifact; audio emitted during training need not have been
rendered by that final weight state. Preference for one anonymous clip cannot
isolate substrate effects from its initialization, target schedule, world
state, or training path. Use the frozen same-world v9 harness in
[`MATCHED_AUDIO_EVAL.md`](MATCHED_AUDIO_EVAL.md) for the closer same-architecture
weight comparison. Neither panel measures downstream transfer by itself.

The private downstream template has fields for Suno model/version and mode,
custom model, prompt/style settings, slider values, output IDs and hashes,
attempt order, rejection status, and human notes. Fill one entry for every
manual generation attempt, including rerolls. Keep those settings fixed
between anonymous sources when testing the effect of the uploaded audio.
