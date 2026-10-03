# Parameter versus world maturation, 2026-10-02

## Source and controls

The clean 2x2 comparison uses archived step-58,107 P and its completed
observer descendant step-60,919 C, separated by 2,812 training chunks and 44
optimizer updates. Both use identical L16 msfield architecture. The preserved
step-703 L1 checkpoint E has the same physical 16-block tensor inventory and
is a secondary long-history comparison. No intermediate growth-stage world
was found; the preserved step-59,513 checkpoint is already L16 and was not
needed for this endpoint contrast. All source sets were copied/frozen and
verified with SHA-256 receipts. No tensor resizing or initialization was used.

Every condition ran 384 chunks (32.768 seconds) against six identical
train-only family/frame intervals, with fixed RNG and relative event clocks.
World chronology/ages, full DSP/recurrent/controller/episodic/motif states
were preserved; weights alone were swapped within each common world. Each
condition had an independent no-op clone, exact parameter hashes before/after,
strict four-development/four-validation probe identities, and source-file
non-mutation checks. All passed. The normal disabled path and measurement
enabled/disabled paths also matched raw/post audio and final world fingerprint
exactly against the preserved working binary.

This is conditional frozen inference, not a native resume of a crossed
weight/world pair. The generic report's optimizer/world consistency refers
only to those two files; it does not certify joint model/world provenance for
a transplant. Historical training choices changed across the long lineage,
so E-to-P is accumulated parameter maturation, not a randomized constant-
objective training experiment. The original fresh manifest hash matches the
matched-corpus source manifest; all eleven retained held-out WAVs reverified.

## Preregistered results

Episode means receive equal weight; the six temporal episodes are correlated
observations, not independent model seeds. Lower error is better. Predictor
means below use each frozen model's own subsequent trajectory; first-offset
predictor scores use the exact same saved history within a world.

| Weights / world, active depth | Validation spectral | Validation chroma | Predictor MSE |
| --- | ---: | ---: | ---: |
| P / P, L16 | 0.79442 | 0.88834 | 0.001782 |
| C / P, L16 | 0.79128 | 0.97468 | 0.005951 |
| P / C, L16 | 0.79006 | 0.89818 | 0.001791 |
| C / C, L16 | 0.78305 | 0.97300 | 0.004472 |
| E / P, L16 | 0.79909 | 0.42531 | 0.126598 |
| E / C, L16 | 0.81530 | 0.44054 | 0.123924 |
| E / P, L1 | 0.79463 | 0.42247 | 0.125163 |
| P / P, L1 | 0.78918 | 0.87983 | 0.001125 |

The 44 later updates yield an average weight effect of -0.00508 in validation
spectral error (-0.64%), +0.08058 in chroma error (+9.02%), and +0.003425 in
self-prediction MSE (+191.7%). The world effect on spectral error is -0.00630
and the interaction is -0.00386. Therefore both parameter and persistent-state
changes influence the score, with no unambiguous overall later-learning win.
The initial common-history MSE also worsens with C weights: 0.002078 to
0.003246 in P world, and 0.000126 to 0.000301 in C world. NLL need not follow
MSE because predicted variance changes. Full effects for development,
validation, stereo, health, advantage, field dynamics, and audio are in
`RESULTS.json`.

E-to-P shows a much larger predictor improvement under common history and
common state. This supports learned predictor information in the parameters,
limited to these recurrent/audio observations; it does not establish useful
musical timing. E has much better aggregate chroma scores, even in the L1
sensitivity pair. The apparent advantage could be a flat/noisy spectrum
exploiting the proxy. Matched external waveform nulls are being scored before
interpreting it as superior audio learning. Enabling E's untrained upper
layers at L16 and disabling P's trained upper layers at L1 each has its own
coordinate-context limitation; neither is a native performance comparison.

## Trajectories and archive limits

Compact multi-view trajectory participation, first-difference participation,
recurrence, per-scale structure, and motif admissions/rejections were measured
without probe feedback. The archive's original persistence criterion spans
far longer than this evaluation. A zero persistent-region counter is not a
saturation result. Candidate counts and Gaussian/time-shuffle descriptor
nulls are exploratory; descriptor Gaussian noise is not a physical world.
All eight cells were bounded, finite, and free of audio clipping. Audio
distance ratios are not additive causal fractions, and none of these
measurements rates coherence or downstream conditioning.

Fine-field trajectory PR is about 2.45–2.84 for mature weights and 1.15–1.17
for E weights under these mature worlds. Minimum measured health is about
0.75 for mature L16 versus 0.60–0.61 for E. E admits four additional motifs
per window while mature weights admit zero or one. These observations do not
make motif count a quality metric. The short-run archive often creates one
candidate region for matched Gaussian descriptors as well as the observed
trajectory, so its candidate count fails to distinguish those controls here.
No organized-regime claim is based on that count.

## Telemetry and manual pilot

`TELEMETRY_AUDIT.md` documents named-column rail, confidence, clipping, HOLD,
reward, low-band, and delta evidence and sampling limits. It also preserves
the unexpected mismatched live trace/metadata/65,536 autosave provenance.
The canonical files were inspected read-only and left intact.

The user-requested `/sdcard/Download/TITAN_Suno_Pilot_20261002` contains three
anonymous inputs, a randomized two-repeat schedule plus prompt-only baseline,
generated-WAV and notes directories, and the complete settings/rating MD
template. The answer key stays in ignored workspace storage. The user is
generating instrumental examples manually. No Suno API or automatic upload
was used, and no output or rating has yet been analyzed. `ingest_suno_outputs.py`
checks hashes and acoustic descriptors without opening the key.

## Resource record and commands

Available RAM was about 2.1–2.3 GiB and swap was nearly full before builds.
Two guarded optimized builds stopped at the 1-GiB RAM floor; a mixed cached
non-LTO build exposed Android TLS linker incompatibility. All logs/receipts
are preserved. A Cargo-consistent ThinLTO build succeeded, peak tree RSS
1.06 GiB with minimum available RAM 1.21 GiB. The first failed build's
process-tree RSS reading was unavailable (reported zero); only its available-
RAM and stop evidence are usable. Numerical behavior remained byte exact in
the independent regression. Frozen inference ran sequentially under a RAM
guard; no training run was launched.

From the repository root:

```sh
python3 analysis/weight_world_20261002/run_matrix.py --prepare-only
python3 analysis/weight_world_20261002/build_guard.py --name build_thin --thin
python3 analysis/weight_world_20261002/run_matrix.py --smoke
python3 analysis/weight_world_20261002/run_regression.py
python3 analysis/weight_world_20261002/run_matrix.py
python3 analysis/weight_world_20261002/analyze_matrix.py
python3 analysis/weight_world_20261002/audit_existing.py
python3 analysis/weight_world_20261002/ingest_suno_outputs.py --out analysis/weight_world_20261002/runs/suno_outputs_batch1
```

The runner verifies and preserves completed cells. It refuses partial output
directories instead of overwriting them. Frozen copies, full WAVs, captures,
logs, and private keys are ignored; compact reports and hashes are tracked.
