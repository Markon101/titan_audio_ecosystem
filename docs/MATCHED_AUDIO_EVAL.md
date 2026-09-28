# Frozen same-world audio comparison

`scripts/matched_eval_package.py` is an experimental, read-only v9 evaluation
wrapper. It runs the existing `--analysis-only --frozen-rollout` path once per
model, using one saved v9 world for all arms. It does not build an optimizer,
perform backward passes, update checkpoints, or contact Suno. Keep its output
outside the TITAN base directory and corpus directory.

Use two or three **same-architecture v9** model checkpoints. For example,
compare the small-corpus parent and child weights under one explicitly chosen
saved world. This is a conditional comparison, not a replay of each model's
original trajectory. The common world may favor one model, and differing
weights can produce different post-forward target-error feedback even with the
same sampled target files and frames.

Create a JSON config like this. The September 28 checkout has the small-corpus
manifest's WAVs under `OLD_WAVS`; `Titan_Audio_Corpus_SML` currently contains
no regular files. Set `corpus_dir` to the location that actually holds those
WAVs on the machine running the evaluation.

```json
{
  "binary": "./target/release/titan",
  "base_dir": "/sdcard/Download",
  "corpus_manifest": "/sdcard/Download/titan_corpus_manifest_v7_sml.json",
  "corpus_dir": "/sdcard/Download/OLD_WAVS",
  "common_world": "/sdcard/Download/titan_world_v9_v9-newcorpus-retain-01.bin",
  "seed": 424242,
  "chunks": 352,
  "candidates": [
    {"id": "retain_01", "model": "/sdcard/Download/titan_model_v9_v9-newcorpus-retain-01.safetensors"},
    {"id": "retain_02", "model": "/sdcard/Download/titan_model_v9_v9-newcorpus-retain-02.safetensors"}
  ]
}
```

`352` chunks at 4,096 frames and 48 kHz produce 30.037 seconds. The wrapper
creates an isolated `eval_base/OLD_WAVS` symlink to the configured corpus
directory because the current v9 analysis runner expects that conventional
name. It does not move or modify corpus files. The original `base_dir` is
used only as a path constraint: the package must be elsewhere. To run:

```bash
python scripts/matched_eval_package.py \
  --config /path/to/matched_eval.json \
  --output-dir analysis/matched_eval_01
```

The tool refuses an existing output directory. Each arm's `analyses/` folder
contains TITAN's full read-only report, provenance, trace, and post-DSP WAV.
The wrapper verifies the frozen/optimizer-free report, exact requested model
and common-world paths, full per-WAV corpus provenance, fixed render length,
and identical sampled target file/frame schedule. It also hashes the actual
executable, models, world, manifest, source WAVs, and anonymous output WAVs.
If any check fails, it leaves an `INCOMPLETE` marker and does not produce a
finished key.

Share only `BLIND/` for listening. `key.json` remains private until ratings
are recorded. The anonymous WAVs are processed identically: TITAN's analysis
post-DSP render, then global RMS matching with a common −1 dBFS PCM peak
guard and no other mastering. `key.json` records the exact gain, sample
interval, checkpoint/world identities, corpus inventory and target schedule.
Its signal metrics are descriptive, not perceptual quality scores.

`downstream_receipt_template.json` is a private per-attempt template for later
manual Suno experiments. Fill one row for every generation, including rerolls
or rejections: exact anonymous input hash, Suno model/version and mode, custom
model if any, prompt/style text, slider settings, output IDs/hashes, and human
preference notes. Keep model/version, prompt, mode, and sliders fixed across
arms when testing input effects. No downstream transfer conclusion is valid
without an exact input-to-output mapping and repeated controls.

This harness cannot compare the different v9 and msfield state representations
under a common world. For that comparison, use controlled fresh starts and
report any unmatched initialization or feedback semantics explicitly.
