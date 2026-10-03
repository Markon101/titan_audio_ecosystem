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
Controller RNG streams start identically, but branch-dependent draw consumption
can diverge after decisions differ. Forcing draws use a common per-offset
stream; state-dependent hazard thresholds and magnitudes can still differ.
These are total closed-loop frozen effects, not direct readout effects under
the earlier host-input clamp.

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
The C world itself developed while weights were updating. The factorial
separates their current frozen causal channels; it does not show that C's
state could have developed identically in a no-SGD training control.
The initial common-history MSE also worsens with C weights: 0.002078 to
0.003246 in P world, and 0.000126 to 0.000301 in C world. NLL need not follow
MSE because predicted variance changes. Full effects for development,
validation, stereo, health, advantage, field dynamics, and audio are in
`RESULTS.json`.

E-to-P shows a much larger predictor improvement under common history and
common state. This supports learned predictor information in the parameters,
limited to these recurrent/audio observations; it does not establish useful
musical timing. The saved histories are endogenous organism observations,
not an independently collected corpus-prediction test. E has much better aggregate chroma scores, even in the L1
sensitivity pair. The apparent advantage could be a flat/noisy spectrum
exploiting the proxy; the completed null test below confirms that concern.
Enabling E's untrained upper
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

## Fixed-probe noise and phase controls

The external scorer uses the exact same Rust bank/projectors, with no model
forward, backward, optimizer, or feedback. Repeated scoring of the same clip
is exact. Re-scored PCM16 raw WAVs differ from online values by at most
3.0e-5 in chroma and 8.9e-6 in spectral error. Null RMS ratios are
0.999969–0.999970 after encoding; channel covariance/stereo correlation is
matched, and the shared-phase control preserves preprocessing global channel
and cross-spectra. No peak-safety attenuation was needed.

| External clip | Validation spectral | Validation chroma |
| --- | ---: | ---: |
| Parent raw | 0.79442 | 0.88833 |
| Parent covariance-matched Gaussian | 0.86349 | 0.34261 |
| Parent shared-phase surrogate | 0.77047 | 0.92125 |
| Child raw | 0.79127 | 0.97466 |
| Early raw | 0.79910 | 0.42530 |
| Early covariance-matched Gaussian | 0.96560 | 0.33717 |
| Early shared-phase surrogate | 0.80716 | 0.41869 |

Gaussian noise obtains lower chroma distance than both early and mature
signals, while worsening spectral distance. Therefore the early chroma
advantage does not establish useful music learning. The parent phase
surrogate improves spectral error by 0.02395, more than the small later
weight effect, despite altering temporal structure. These proxies describe
different local/statistical relations and do not certify long-range
organization. Neither null dominates all metrics. The result weakens a simple
loss-based account of useful maturation; it does not establish that Titan's
music or downstream conditioning is poor. `NULL_RESULTS.json` preserves
scores, hashes, matching properties, and serialization-error checks.

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
python3 analysis/weight_world_20261002/prepare_audio_nulls.py
python3 analysis/weight_world_20261002/score_audio_nulls.py
python3 analysis/weight_world_20261002/audit_existing.py
python3 analysis/weight_world_20261002/ingest_suno_outputs.py --out analysis/weight_world_20261002/runs/suno_outputs_batch1
```

The runner verifies and preserves completed cells. It refuses partial output
directories instead of overwriting them. Frozen copies, full WAVs, captures,
logs, and private keys are ignored; compact reports and hashes are tracked.
All 92 Rust tests passed sequentially under a resource guard after the final
scorer change (peak process-tree RSS 564 MiB, minimum available RAM 1.91 GiB).
The final disabled-binary and measurement-only regressions again matched
raw/post audio and final world state exactly. The external WAV ingestion
fixture passed without opening the answer key; it is explicitly marked as
test data, not a Suno generation. Python compilation, Rust formatting, and
diff checks passed.
The matrix binary is preserved as `runs/matrix_titan`, corresponding to
source commit `f5165cc`. A later binary adds only external audio scoring;
missing matrix cells must use the preserved binary identity. Rechecking a
completed cell leaves its resource and command receipts intact. To inspect
or recover the original matrix, use `--binary
analysis/weight_world_20261002/runs/matrix_titan`. Standalone score output
directories are also preserved rather than overwritten.
