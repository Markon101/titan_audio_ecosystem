# Predictive information and density: first frozen battery

October 3, 2026. Canonical parent: archived TITANM10 step 58,107,
v10-msfield-fresh-20260930-02, L16, 16,826,171 parameters.

## Decision and what was implemented

Implemented a small, opt-in, external predictive-coding framework and frozen
future selector. No Rust training/forward/checkpoint code changed. There is
no optimizer construction/backward pass in these frozen rollouts, no new
training lineage, and no runtime escape controller. `--pd-weight 0` bypasses
metrics and selects intact bytes. `--pd-weight 0.02` scores tiny candidate
futures with a bounded coding reward and waveform-displacement cost. All
source checkpoint files, including Adam/world/morph state, remain unchanged.

The initial battery demonstrates **no generation benefit** from PD selection.
Protected C equals intact A/B in both windows; naive D chooses +0.03 host
energy. All candidates remain healthy/bounded, so D did not demonstrably
collapse Titan. Silence/constants win the separate naive synthetic control.

The meter detects constructed complementary predictive relations and exposes
its own false positives. It does **not** establish information-dense musical
abstractions in this mature Titan trajectory. Do not add this reward to the
training loss or promote it into runtime control from this evidence.

## Rationale and method

Predictive information concerns dependencies between past and future;
our code-length difference is a finite probe approximation, not an estimator
of that quantity. Background: [Bialek, Nemenman and Tishby](https://arxiv.org/abs/physics/0007070).
Description-cost-aware probing is motivated by
[Voita and Titov](https://arxiv.org/abs/2003.12298), but this study is not their
online MDL estimator or a complete executable compressor.

Targets are thirteen deterministic chunk descriptors: mid/side log RMS,
eight log-band fractions, flux, stereo correlation and level balance.
Forecast horizons are 1/4/16/64 chunks (0.085/0.341/1.365/5.461 seconds);
representation sensitivities also include 128 chunks. Internal observations
are available every eight chunks, supporting 8/16/64/128-chunk forecasts.
Each added view is reduced to four deterministic orthogonal sketch
coordinates: GRU, three field scales, L1/L8/L16 Morphic deltas, Morphic output,
decoder/control, cyclic synthesis phase, and host/motif observational proxies.
Joint GRU/coarse, host/coarse, early/late deltas, and motif/GRU probes are
reported. No claim is made to observing full motif or host memory contents.

The chronological split is training/calibration/selection/audit = 40/20/20/20,
with horizon purging. Quantizers/scalers fit training only; residual variances
and baseline selection fit calibration only. Tiny ridge probes forecast
future output, conditioned on current output history plus optional state.
Baseline competitors include iid marginal, persistence, trend, local-history
ridge, fixed/data-estimated periodic copy, and in v2 affine periodic forecasts.
Gaussians are integrated over learned quantile bins and renormalized to proper
discrete distributions. The product code does not capture dependencies among
target dimensions. Current training-target error is an allowed host summary;
target identity and future forcing are absent from the probe inputs.
Development/validation scores are readout-only and never select anything.

Density = saved bits / assumed 16-bit coefficient charge. This charge is
explicitly a proxy: coefficients remain float64, and feature/normalizer side
information is not a complete transmitted code. Both total and per-feature,
per-forecast gains, cost, net gain and compute proxies are available. Forecast
counts differ by horizon: use normalized columns in `density_curves.csv`
before interpreting decay. No exponential fit or confident predictive-depth
estimate is justified here. No net density horizon passes the cost gate.

Progress compares smaller/larger prefix-trained probes on identical audit
targets. [Compression-progress motivation](https://arxiv.org/abs/0812.4360)
motivates that diagnostic, but it measures the probe's acquisition, not Titan
parameter learning. It is not used as reward; different-window PD subtraction
would confound task drift with learning, and repeated probe resets could be
gamed. Operational joint-minus-sum and joint-beyond-best gains are not PID.
Shared history in the probes and unequal joint capacity also limit that
interpretation; shuffled-view controls have the same added capacity.

## Important falsifications and implementation repairs

1. A suffix-mutation regression exposed circular lag padding contaminating
   prefix normalization. Causal padding fixed this; selection is invariant
   to audit-target changes.
2. v1 rejects seven simple audio controls and three additional noise/tone
   challenges, but a sign-changing 65-lag process earns reward 0.25.
   This falsifies a general anti-triviality claim for copy-only baselines.
   v2 adds cheap affine-periodic predictors and a broader training-only period
   search; the same witness receives 0.0. Both versions and the first v1
   Titan readout (`W0_RESULTS.json`) are retained. Final selection requires v2.
3. A constructed two-lag Gaussian process receives v2 reward about 0.207.
   This is a positive test for compact stochastic predictive structure and
   a warning for artistic reward: long-memory colored processes can score
   well. Three toy coefficient pairs were explored and documented. This is
   neither a preregistered Titan discovery nor a music-quality result.
4. A short-clip sensitivity initially attempted an affine lag longer than
   its training prefix. The analysis stopped without a result file; supported
   fitting lags are now bounded and a short-window regression test covers it.
   Unsupported horizons are reported as missing rather than fabricated.

No metric was tuned to make a Titan candidate win. The zero-reward synthetic
controls are evidence for those finite cases, not a proof against all noise
or adversarial signals. Peak concentration/activity guards and coding baselines
remain a limited defense. Predictive structure alone cannot certify beauty.

## Frozen battery and exact controls

The 64-chunk A/B instrumentation comparison has byte-identical raw and
post-DSP WAVs and full final world fingerprints. Each has an independent
no-op clone. Two 1,024-chunk banks use intact plus cloned initial energy
offsets 0/+0.01/+0.03 with fixed weights/morphology and existing energy bounds.
Offsets are host interventions, not substrate-specific interventions or
energy-preserving field pulses. Common forcing RNG/event clocks are used;
state-dependent hazards/magnitudes can still differ. These are total
closed-loop futures, not host-clamped direct effects.

The zero-offset bank arm exactly matches intact raw/post audio and full
world state. All arms pass nonfinite/bounds checks and have zero clipping
and zero optimizer updates. Six corpus aliases match their original source
WAV hashes exactly; 1,068 sampled target file/frame rows were independently
verified. These are sampled-row checks, not a claim of full per-chunk logging.
Eight A/B/C/D exports match their measured source WAV hashes exactly.

**The baseline windows overlap by 896 chunks (87.5%) with exact raw and
post-DSP samples.** Warmup 128 advances the same trajectory rather than
creating an independent replicate. Differences in coding gains largely
compare shifted fitting/calibration/audit windows on this overlapping path.
Candidate offsets occur at different snapshots, but this remains one model
and one world lineage. `overlap_receipt.json` preserves the direct check.

## Results

| Window | A/B/C choice | D naive choice | Intact audit gains at horizons 1/4/16/64, bits | Predictor cost |
| --- | --- | --- | --- | --- |
| warmup 0 | Intact | Energy +0.03 | +276 / +138 / +9 / +18 | 3,120 bits |
| warmup 128 | Intact | Energy +0.03 | -106 / -111 / -324 / -289 | 3,120 bits |

The cost-aware comparator chooses iid in these intact full-output curves.
The unpenalized calibration winner and output-history contrasts are also
reported. Most gains disappear against those stronger forecasts: additional
slow history yields about -2/+7/-24/-13 bits over local history in window 0,
and -20/+12/-14/-18 in window 128. The near-horizon positive iid result is
not evidence of a substantial new abstraction. None pays the declared cost.
Small representation-specific envelope/spectral/stereo gains are preserved
without using their maximum to select a candidate.

Adding internal views mostly worsens out-of-time coding. At 64 chunks, coarse
state adds about +10/+11 bits beyond history; its raw iid gains are +19/+12.
The two shuffled-view gains are -3/+18 in window 0 and -12/-27 in window 128.
This is a weak, small hint under tiny forecast support, not a robust compact
mechanism claim. GRU and Morphic views generally hurt this linear probe;
negative gain is estimator/forecast failure, not negative information in the
organism. Joint-minus-sum does not provide a stable beneficial interaction.
Do not infer that causal pathways are useless or that Titan lacks nonlinear
predictive information; limited data, sketches, stationarity and model class
can all hide it.

Both shuffled-chunk and shared-phase controls also receive zero protected
reward. This meter therefore does not establish an intact ordering advantage
over them in this budget. Descriptor entropy, zlib token ratio, covariance
participation and recent nearest distances are supporting measurements in
`NOVELTY_RESULTS.json`; they are not raw-waveform compression or artistic
novelty. Noise can have high descriptor entropy/dimension without coded gain.

The naive choice lowers local descriptor-change MSE by about 7%/16% relative
to intact. Its full RMS changes by only about 0.07%/0.14%, but normalized
selection-window waveform displacement is 1.38/1.19: a small host pulse can
produce a very different later waveform without much level change. Those
distances are not additive causal fractions or preference scores. The weak
selector's conservative displacement penalty is an additional reason to
avoid these branches; reward already equals zero before that penalty.
The sample-distance penalty is too phase-sensitive to serve as a general
musical-quality anchor: even a useful phase-changing transition may be far
from intact. Removing that penalty would still select intact here because
all protected rewards are zero and ties choose no-op. A later positive
selection test should use a declared spectral/envelope health constraint
instead of treating waveform closeness as quality.

Existing 256-chunk host-clamp/closed-loop lesions were rescored under a
separately labeled short-history sensitivity, with horizon 64 unsupported.
Coarse hold can improve next-chunk code gain (48 to 60 bits in one clamped
window), illustrating that higher predictability need not mean a more useful
organism. No intervention shows a supported net density benefit here.

## Audio, resources and next decision

Exact role exports are under local ignored `runs/selected/`. A/B/C are
duplicates; listening should compare intact with naive, without pretending
duplicates are independent tracks. This interface did not listen or produce
coherence/preference ratings, and no new Suno generation was requested.

The bank renders 87.381 seconds per arm. Wall time was 596/644 seconds per
four-arm bank; peak RSS 895/917 MiB, minimum available RAM 1,579/1,317 MiB.
No safety floor fired. The short measurement-on/off timing ratio is about
1.18, including startup and existing fixed-probe/capture work; it is not a
steady-state PD overhead estimate. The external bank analysis took about
9/7 seconds, roughly 1–2% of bank-render wall time, plus reused-lesion analysis.
Compute proxies are labeled separately from actual time. No build was needed.

The cheapest next discriminating test is **probe transfer across trajectories**
with shared quantization/scaling and a training-only episode bank: train a
tiny predictor once, evaluate it on genuinely separate frozen trajectories,
and compare state/history/affine-periodic predictors plus matched nulls. Reuse
available artifacts before rendering more. This tests whether apparent gains
are reusable abstractions or fit-window artifacts. Only if positive structure
survives that gate should PD influence a training objective or a broader
candidate bank. Any learning-progress reward needs a persistent, non-resetting
probe and a held-out reward-calibration stream, separate from validation.

Commands and safe overwrite behavior are in `README.md`. `PLAN.md` records
the original protocol; v1/v2 result files, target/resource receipts, curve CSV,
null batteries, eight exact exports and nine deterministic tests preserve
positive controls, failures and negative Titan results. The production loss,
canonical checkpoint, v9 sibling and existing world-rejection protections are
unchanged.
