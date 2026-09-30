# Fresh msfield seed-44 run — 2026-09-30

This is a separate fresh `v10-msfield-exp` run. It uses 16 physical Morphic
blocks at width 512, the same 16,826,171 total parameters and the same decoder
parameter count as the seed-43 pilot. It starts at L01 with adaptive growth
allowed through L16. No mature v9 weights or world state were imported.

The clean release binary at commit `72a4d69` completed 703 chunks (59.989 s)
with BPTT 64, core update every 2, learning rate 0.0005, and **six actual
worker threads at startup**. The active depth remained L01; no morph event
occurred during this short block. The saved model, `TITANM10` world, and AdamW
optimizer agree at step 703 after 11 updates. The stereo WAV has no clipped
frames. Exact paths, hashes, invocation, parameter groups, and descriptive
signal checks are in `run_receipt.json`.

The completed baseline is in
`/sdcard/Download/TITAN_v10_msfield_fresh_20260930_02/`, including the exact
executable as `titan_run_binary`. A verified copy at step 703 is ready for
continuation in `/sdcard/Download/TITAN_v10_msfield_fresh_20260930_02_cont/`.

## Continue for one bounded 120-second block

From the repository root:

```sh
./target/release/titan \
  --base-dir /sdcard/Download/TITAN_v10_msfield_fresh_20260930_02_cont \
  --corpus-dir /sdcard/Download/OLD_WAVS \
  --corpus-manifest /sdcard/Download/titan_corpus_manifest_v7_sml.json \
  --substrate msfield --run-tag v10-msfield-fresh-20260930-02 \
  --duration 120 --threads 6 --seed 44 --lr 0.0005 \
  --bptt 64 --core-update-every 2 \
  --morph-layers 16 --morph-width 512 --max-morph-depth 16 \
  --motif-capacity 512
```

Keep this tag and omit `--fresh-model`, `--fresh-world`, `--morph-depth`, and
`--freeze-morph` when continuing. Android may restrict Termux to four CPUs in
the background. TITAN caps requested threads to the CPUs available **at
startup**; check the startup line for `Threads: 6` if that setting matters to
the comparison. Later affinity changes can alter throughput without changing
the initialized worker count.
