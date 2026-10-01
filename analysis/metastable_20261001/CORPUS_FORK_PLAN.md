# Preregistered small-corpus ecology contrast

## Start and arms

All arms start from the same frozen v10 `TITANM10` step-58,107 model, world,
AdamW optimizer, Morphic state, and metadata. The experiment binary is SHA-256
`3b3c172aaf662b70e55dd4282059eb288a128e97d1a01f4c483cb6a6bba0c75d`.
Every arm requests 4 threads, seed 44, LR 0.00045, BPTT 64/tape 8, core update
every tape, L16/width 512/max depth 16, motif capacity 512, and the
measurement-only regime capture at stride 4. No open-endedness intervention
or layer regularizer is active.

Each arm receives two consecutive 120-second blocks (1,406 chunks each),
with an exact checkpoint save/resume at the block boundary. This matches the
existing A run's optimizer flush and process-boundary target sampling. The
total budget is 2,812 chunks per arm.

| Arm | Training corpus change | Expected loader identity |
| --- | --- | --- |
| A | Original small manifest and `/sdcard/Download/OLD_WAVS` | 30 listed training files, 23 training families |
| B | Add six distinct user-audio families from outside the small manifest | 36 training files, 29 families |
| C | Add six deterministic `-0.8 dB` 3.5 kHz bell-EQ variants of six existing train files | 36 training files, 23 families |

The six B files total 818.15 seconds; the six C source files total 786.18
seconds, a 3.9% difference. All are 48 kHz, stereo, 16-bit PCM. Original
files are linked read-only into separate corpus directories. The source
manifest's development and validation entries, filenames, families, and
source bytes are copied without alteration. A source-byte audit found four
duplicate pairs in the original 41-file corpus, each within the same
role/family; no train/development/validation byte overlap was found. Exact
source and derived hashes are in `corpus_forks_receipt.json`. The C filter
changes timbre slightly without adding new composition; it is a control for
additional variants, not an equal-audio-distribution guarantee.

## Fixed analysis plan

The regime archive uses `regime_calibration.json`, fitted only on the first
half of A's first measurement interval. No B/C observation can tune a
threshold or select an intervention. Analyze both blocks of each arm from
scratch with this file, and verify exact archive save/resume across the
block boundary. The principal descriptive comparisons are:

1. Number of **confirmed persistent** regimes in the second block and whole
   run; dwell distribution, revisits, and transitions. Four-sample candidate
   count is excluded as a headline because a block-shuffle null gamed it.
2. Median and time course of trajectory participation ratio; per-view field,
   GRU, Morphic activation/delta dimensions; and field topology diversity.
3. New motif count and similarity rejection fraction, plus learned parameter
   update magnitudes by group/layer from exact checkpoints.
4. Health, stagnation, rail/NaN events, gradient norm/clipping, fixed
   development and strict validation probes, and audible output/prime assets.

B is **exposure-qualified** only if its trace shows at least two distinct new
families as selected training targets during the matched budget. If fewer
occur, the contrast is inconclusive, regardless of regime count. The B/C
comparison additionally requires C to sample at least two distinct
added existing-family variants; otherwise the variants control is not an
exposure-matched comparator. A strong
directional result would be B gaining persistent field-including regimes or
trajectory dimensions beyond both A and C while health and fixed probes stay
comparable. A response limited to decoder/audio views would support a
narrower output-adaptation claim. A single checkpoint/seed and only two
blocks per arm cannot establish a general capacity limit or downstream Suno
transfer; no significance claim is planned.

Source targets are selected by TITAN's ordinary uniform-family scheduler.
Different manifests intentionally yield different target schedules. This is
the ecological exposure being tested, not a paired-target causal substrate
ablation. All roles and exact target filenames will be recorded in normal
telemetry for post-run interpretation.
