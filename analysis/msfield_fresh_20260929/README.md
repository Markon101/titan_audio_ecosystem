# Fresh msfield seed-43 run — 2026-09-29

This isolated `v10-msfield-exp` run started from fresh weights and a fresh
world. Its 180 shared non-substrate tensors were matched to a deterministic
**fresh** v9 initialization at seed 43; no mature v9 weights or world state
were imported. The physical 16×512 MorphicStack and decoder parameter counts
match the earlier short v10 comparison. Active Morphic depth was fixed at 10.

The clean release binary at commit `fb54bb1` rendered 703 chunks (59.989 s)
with BPTT 64, core update every 4, learning rate 0.0005, four threads, and
11 optimizer updates. The model, `TITANM10` world, and optimizer agree at
global step 703. The stereo WAV has no clipped frames. Exact paths, hashes,
parameter groups, invocation, and descriptive signal checks are in
`run_receipt.json`. The exact executable is also stored as `titan_run_binary`
beside the WAV and checkpoint set on the sdcard.

The user reported that the **full 60-second WAV** already sounds better and
subjectively "more diffusiony." The receipt ties that note to its audio hash.
This is listening feedback about the sound, not a claim
that TITAN is sampling from a diffusion audio model or that the downstream
effect has been measured.

The completed baseline lives in
`/sdcard/Download/TITAN_v10_msfield_fresh_20260929_01/`. The copied model,
world, optimizer, metadata, WAVs, and executable matched their source hashes
at step 703. This copy is ready for continuation in the sibling
`/sdcard/Download/TITAN_v10_msfield_fresh_20260929_01_cont/` directory.

## Continue for one bounded 120-second block

From the repository root, run:

```sh
./target/release/titan \
  --base-dir /sdcard/Download/TITAN_v10_msfield_fresh_20260929_01_cont \
  --corpus-dir /sdcard/Download/OLD_WAVS \
  --corpus-manifest /sdcard/Download/titan_corpus_manifest_v7_sml.json \
  --substrate msfield --run-tag v10-msfield-fresh-20260929-01 \
  --duration 120 --threads 4 --seed 43 --lr 0.0005 \
  --bptt 64 --core-update-every 4 \
  --morph-layers 16 --morph-width 512 --max-morph-depth 10 \
  --freeze-morph --motif-capacity 512
```

Reuse that command for later blocks, changing only `--duration` after
listening and reviewing the traces. Keep the same tag and settings; omit
`--fresh-model` and `--morph-depth` because the copied world supplies the
active state. `--core-update-every 4` matches the existing v9 cadence and
the earlier v10 pilot. A separate tagged fork can later compare cadence 2;
changing it here would mix a training-schedule change into this lineage.
