# Titan predictive-density experiment

This is an opt-in **offline frozen-future selector**, not a training loss,
checkpoint migration, neural critic, or runtime attractor-escape controller.
Normal Titan code is untouched. It uses the existing TITANM10 frozen harness,
deterministic descriptors, tiny ridge probes, and chronological held-out
probability coding. Source/model/world/optimizer artifacts remain read-only.

## Commands from the repository root

```sh
OPENBLAS_NUM_THREADS=1 python3 analysis/predictive_density_20261003/test_predictive.py
# These create new immutable reports; rerunning them refuses overwrite:
OPENBLAS_NUM_THREADS=1 python3 analysis/predictive_density_20261003/validate_metric.py
OPENBLAS_NUM_THREADS=1 python3 analysis/predictive_density_20261003/challenge_metric.py
OPENBLAS_NUM_THREADS=1 python3 analysis/predictive_density_20261003/validate_metric.py --hardened
OPENBLAS_NUM_THREADS=1 python3 analysis/predictive_density_20261003/challenge_metric.py --hardened
# Frozen arms are restart-safe only after full receipt/configuration verification:
python3 analysis/predictive_density_20261003/run_frozen.py --parity
python3 analysis/predictive_density_20261003/run_frozen.py --snapshot w0
python3 analysis/predictive_density_20261003/run_frozen.py --snapshot w128
OPENBLAS_NUM_THREADS=1 python3 analysis/predictive_density_20261003/analyze.py \
  --include-lesions --output analysis/predictive_density_20261003/RESULTS.json
OPENBLAS_NUM_THREADS=1 python3 analysis/predictive_density_20261003/freeze_summary.py
```

Choose an already-rendered future, copying its exact WAV bytes:

```sh
python3 analysis/predictive_density_20261003/selection.py \
  --results analysis/predictive_density_20261003/RESULTS.json --snapshot w0 \
  --pd-weight 0 --output analysis/predictive_density_20261003/runs/selected/baseline.wav
python3 analysis/predictive_density_20261003/selection.py \
  --results analysis/predictive_density_20261003/RESULTS.json --snapshot w0 \
  --pd-weight 0.02 --output analysis/predictive_density_20261003/runs/selected/weak_pd.wav
python3 analysis/predictive_density_20261003/selection.py \
  --results analysis/predictive_density_20261003/RESULTS.json --snapshot w0 \
  --naive --output analysis/predictive_density_20261003/runs/selected/naive.wav
```

Completed arms and exports refuse overwrite. Do not delete partially finished
arms to make the runner proceed; audit them and choose a new explicitly named
study if necessary. The source interval schedule, parameter hashes, binary
hash, resource readings, target-frame identities and no-op equivalence are
part of the receipts. Local WAVs/large captures live in ignored `runs/`.

## Method boundaries

Training/calibration/selection/audit occupy 40/20/20/20 percent of each
trajectory, with horizon purging. No validation probes select predictors,
normalizers, reward or candidates. Quantization and standardization use the
training prefix. Gaussian residual scales and baseline choice use calibration.
Proper discrete probabilities are integrated over learned quantile bins;
their factorized log loss is measured in bits **of these descriptors**.
Dependencies among target dimensions make this a product code, not a joint
entropy or mutual-information estimate. Coefficient cost is an assumed
16 bits per parameter, without quantizing or serializing the double-precision
fit: net bits and density are operational costs, not an actual MDL compressor.
Wall time and multiply-add proxies remain separately labeled.

The primary baseline minimizes calibration code length plus coefficient cost.
Also report unpenalized calibration-winner and output-history contrasts, so a
cost penalty cannot disguise a stronger affordable simple forecast. This
additional readout was added after the first snapshot revealed that its
cost-aware comparator chose iid; it does not change the primary reward or
the preregistered decision. The first readout is retained in W0_RESULTS.json.

The final framework uses v2 affine-periodic controls. A v1 sign/gain-changing
delayed loop earned nonzero reward (0.25), falsifying a general anti-triviality
claim. v2 fits cheap per-coordinate affine periodic baselines and rejects that
witness (0.0). Nonzero selection requires v2; zero-weight bypass stays exact.
The retained v1 and v2 validation reports separate this correction from the
original preregistered result. No additional Titan run was needed.

A constructed long-memory Gaussian process still earns v2 reward. That is a
positive test of predictive relations, and a limitation for artistic reward:
PD identifies compressible stochastic structure, not musical usefulness.
The coefficient examples were explored as synthetic fixtures, not Titan
result selection. A finite battery cannot prove immunity to all noise-based
reward gaming. Do not deploy a PD-only controller from these results.

Normalizers are frozen; changing the audit suffix cannot change selection
scores. Shuffled internal views are explicitly artificial negative controls.
Joint-minus-sum is an operational prediction statistic, not a PID decomposition.
Progress compares smaller/larger probe fits on the same audit targets and is
not Titan SGD learning. It is measured separately, not fed into selection.

The reward requires positive net gain after the stated cost and positive
gain in both halves of the selection interval. Physical guards reject quiet,
railed, nonfinite and spectrally concentrated single-oscillator candidates.
These are finite-tested protections, not a proof against every adversarial
signal. A zero coefficient bypasses metrics entirely and returns baseline.
The weak coefficient multiplies a bounded coding reward, penalized by audible
waveform displacement. That penalty is conservative and is not musical quality.
Naive selection omits these protections and optimizes local persistence MSE.

The candidate bank consists of two small initial host-energy offsets and
exact no-op, under one parent and two declared warmup snapshots. It is not a
study of a learned PD controller or independently trained seeds. No future
world is committed. Early lesion/clamp clips use a separately labeled short
history protocol; insufficient long horizons remain unsupported.

The intact baseline windows overlap by 896/1,024 chunks (87.5%), exactly in
raw/post audio. They are shifted windows on one trajectory. Refitted code
gains differ despite this large overlap, exposing fitting-window sensitivity.
`density_curves.csv` includes normalized gains and forecast counts;
`NOVELTY_RESULTS.json` separately freezes descriptive novelty/covariance data.
`runs/selected/w0_baseline.wav` and `w0_naive.wav` (and w128 equivalents) are
the available listening pair; A/B/C duplicates should not be rated as replicates.
