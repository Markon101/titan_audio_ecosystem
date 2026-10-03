# Frozen parameter and world maturation, 2026-10-02

## Questions and comparison limits

The clean primary contrast uses mature L16 weights/world at step 58,107
and its observer-only descendant at 60,919: 2,812 training chunks and 44
Adam updates, with unchanged physical architecture and active depth. A
2x2 swap estimates parameter change, world development, and their interaction.
World transplantation is conditional inference: all tensor shapes, runtime
state, controller/DSP/motif interfaces, and schema must match exactly. It is
not an exact training resume of the crossed pair.

The preserved original step-703 weights are a secondary long-history check.
They physically contain the same 16 blocks, but only L1 was active. Testing
them at L16 includes enabling upper blocks that were never trained. Therefore
also compare step-703 and step-58,107 weights at a fixed L1 under the same
mature world. No architecture resizing or tensor initialization is allowed.
No preserved intermediate morphogenic or early-L16 checkpoint has yet been
found; absence is reported rather than synthesizing one. Comparisons use one
lineage and historical training objectives, not independent model seeds.

## Evaluation protocol, written before the substantive runs

Freeze all source checkpoint sets with byte/hash receipts before use. Keep
the existing archived parent unchanged. Evaluate sequentially, with no
optimizer, backward pass, or new training. Every cell has an independent
no-op clone and exact parameter hashes before/after rollout. Require common
initial world fingerprints across weight swaps within each world/depth.

An opt-in standardized evaluation RNG uses the declared analysis seed for
controller and forcing streams across worlds. A relative evaluation event
clock synchronizes planner, shear, motif, and episodic event phases while
preserving each world's native absolute chronology and historical ages.
Targets use a separate relative clock over one checksummed schedule: six
train-only family aliases, 64 chunks each, total 384 chunks (32.768 seconds).
The normal frozen/training paths keep their existing semantics when these
flags are absent. Repeat the mature factorial on another schedule/RNG only
if the first result leaves a concrete ambiguity worth resolving.

Primary cells at fixed L16:

- P weights / P world (58,107 / 58,107);
- C weights / P world (60,919 / 58,107);
- P weights / C world (58,107 / 60,919);
- C weights / C world (60,919 / 60,919).

Secondary cells: step-703 weights in each mature world at L16, then step-703
versus parent weights at L1 in the parent world. All cells use identical
relative target frames and RNG settings. Host feedback can evolve normally
after the common start; comparisons concern frozen closed-loop dynamics.

## Measurements and decision rule

At fixed stride eight, score the same four development and four strict
validation probes using existing spectral/chroma semantics. Also report
target spectral/chroma/stereo relations, predictor MSE/NLL/calibration,
health/stagnation, unclamped/log confidence, exact action advantage, per-scale
RMS/rails/deltas, motif admissions/rejections, compact internal regime views,
trajectory participation ratio, recurrence, and full audio/stereo distances.
Probe scores never select targets, regime definitions, or interventions.

For each metric report the paired weight effect within each world, the world
effect within each weight, their averaged main effects, and difference of
differences interaction. Time windows are correlated repeated observations,
not independent experimental replicates; do not manufacture significance
from them. Lower probe/predictor error is supporting evidence of parameter
learning, not proof of preferred music or uniquely reconstructing arbitrary
phrases in a model with no direct reference input. Preserve negative or mixed
results and short-window/archive limitations.

## Other work and resources

Audit the Space Bunny objections from named CSV columns and existing source
and run metadata. Distinguish exact facts from sampled evidence and unavailable
per-chunk history. No claim of an uninterrupted HOLD lock or spike causality
may rely on sparse rows alone. The Suno generation interface is unavailable
to these tools; the user will supply manual outputs/settings/ratings in the
requested Downloads intake. The answer key stays private until ratings end.

Check live processes/RAM/swap/storage before each expensive build or rollout.
Frozen inference has previously used about 0.3 GiB RSS; use two threads and
sequential cells. Preserve partial runs and never overwrite completed cells.

## Prespecified follow-up after the chroma surprise

The completed matrix shows lower aggregate chroma error for E weights despite
poor predictor performance. Before interpreting that result, score the same
raw WAVs and matched external controls against the **identical Rust probe
bank/projectors**. The controls match channel covariance/RMS or preserve
global channel/cross-spectra through shared phase randomization. They are
never fed into Titan. Score parent, child, early, parent-noise, early-noise,
parent-phase and early-phase clips at the same offsets. Validate raw-WAV
scores against their online metrics to quantify PCM16 serialization error.
If noise or phase surrogates beat structured outputs, the affected probe
metric cannot alone support useful audio learning. No checkpoint is selected
or retrained from these scores.
