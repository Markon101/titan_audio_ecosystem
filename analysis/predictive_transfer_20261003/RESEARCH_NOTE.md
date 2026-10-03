# Can compact predictive probes transfer?

October 3, 2026. Follow-up to the frozen predictive-density battery, source
step 58,107 TITANM10 L16. No new Titan inference or training was performed.

## Main decision

The earlier coarse-field hint does **not** transfer to the primary trajectory
with different forcing/controller RNG. Positive results in shifted worlds
or weights either remain weak, are matched/exceeded by shuffled controls, or
include a large static uncertainty-calibration benefit. These data do not
justify predictive-density training reward or an attractor-escape controller.
They also do not establish that Titan has no predictive information: a small
linear model, sparse state sketches and these coarse audio targets can miss it.

The most useful new result is methodological: **a change in fixed residual
sigma can look like information carried by a state even when the forecast
mean gets worse**. The decomposition retains that apparent positive result
and explains it without crediting a new long-horizon abstraction.

## Protocol and controls

`PLAN.md` preceded transfer scoring. Fit on PD w0's first 40%, calibrate on
the next 20%, then freeze all bins/scalers/projections/weights/variances and
source-chosen comparators. Serialize and reload before any test scoring.
The six new deterministic tests cover exact round-trip, no evaluation refit,
causal episode-local history, horizon purging, finite short adaptation, and
transfer of a known complementary synthetic relationship to a new random
sequence. That positive control beats shuffled inputs; the estimator is not
simply incapable of detecting any transferable signal.

Source-window codes match the previous v2 report within 1e-8 bits; predictions
and scores before/after serialization match exactly. Fixed-bank hashes and
canonical parent hashes remain unchanged after evaluation. Nine prior tests
also pass. Runtime source-test indices use horizon purging; no target IDs,
future forcing or development/validation scores are predictor inputs. The
world's current training-target error remains an allowed host proxy.

Eleven existing episodes were read sequentially. The primary is the historical
L16_WP_SP run: same parent weights/world, seed 20261002 instead of 20261003.
Only one full PCM chunk equals a source chunk, and test origins start at 128
chunks. This is a separate forced trajectory of one organism, not independent
training replication. Other cells vary world and/or weights; the distant
step-703 weights at L16 are an OOD sensitivity with previously untrained upper
layers, not native early-model performance. The warmup-128 reference remains
overlapping: 486 of its chunks occur in source fit/calibration and 896 in the
full source, so it is excluded from a claim of independent transfer.

The two initial implementation stops were a padded-versus-cropped scoring
shape mismatch caught by tests and a source comparison that expected a
standalone history row in the prior report. Both were repaired before scores
were accepted. The saved partial source probes were retained and adopted only
after exact full-state equality. No scientific threshold or feature was tuned
to make a Titan view win.

## Primary transfer results

Gain is over the SAME frozen output-history predictor. State observations
have stride eight; horizon 64 is 5.461 seconds and horizon 128 is 10.923 seconds.
The primary state-to-output test has 24 and 16 forecast origins respectively.
These are correlated samples, not condition replicates.

| View / horizon | Source audit gain | Primary transfer gain | Shared-sigma mean-only gain | Three shuffled transfer gains |
| --- | ---: | ---: | ---: | --- |
| Coarse / 64 | +10.4 bits | -9.5 bits | -12.4 bits | -2.3 / +18.3 / -29.3 |
| Fine / 64 | — | +2.1 bits | +4.1 bits | -14.4 / -8.6 / +1.9 |
| Phase / 64 | — | +8.0 bits | +5.8 bits | +4.5 / -14.7 / -4.7 |
| Phase / 128 | — | +18.3 bits | -1.3 bits | -3.5 / -113.2 / -25.0 |

The coarse hint fails the main transfer test. Fine and phase have small
positive contrasts, including a little mean improvement at horizon 64. The
fine mean MSE improves about 2.4% and phase about 1.4% versus history, but a
null nearly matches fine's code gain. None pays the 832-bit assumed charge
for its 52 extra parameters. These hints cannot establish a substantial
information-dense abstraction or better music.

Output-only slow history also contributes little beyond local history: on
the primary test it saves about -10.4 bits at one chunk and -27.6 at 64 chunks.
It nevertheless saves +441/+40 bits against iid, illustrating why gain over
an easy marginal comparator can overstate new structure. Raw/factorized
codes, same-capacity slow-history controls and normalized forecast gains are
all preserved in `curves.csv` and the compressed full report.

## The apparent long-horizon phase result

For the +18.2679-bit phase result at horizon 128, use identical history means
but replace ONLY their sigma with the phase probe's fixed source sigma:

- History mean + phase sigma saves **19.5408 bits** over ordinary history.
- Adding the phase-dependent mean then **loses 1.2729 bits**.
- Their sum exactly reconstructs the original +18.2679 bits.

Most variance-only benefit comes from one spectral band (+20.6 bits). Phase
mean predictions have slightly worse overall MSE than history at this horizon.
The proper probabilistic code gain is real for the specified predictors, but
it does not establish extra future information requiring phase input. Static
residual calibration is available without that input. At horizon 64 the
same decomposition is +3.55 bits from sigma and +4.46 from the changed mean:
a small possible signal, not the larger long-horizon claim.

`VARIANCE_CONTROLS.json` records exact decompositions for all eleven episodes
and mean-only raw descriptor errors. Those raw errors mix descriptor units
and are not perceptual audio losses. Joint-minus-sum now uses HISTORY-relative
gains, preventing shared-history double counting. It still is not PID, and
there is no stable beneficial joint state contribution supported here.

## World/weight shift and false reassurance from adaptation

The small source-world normalizer is not portable across all coordinates.
Coarse standardized RMS distance is about 0.97 in the primary path, 9.82 in
the child world under parent weights, and 22.54/28.41 under later weights in
parent/child worlds. Corresponding coarse clipping fractions are 0%, 3.7%,
19.1%, 25.0%. The step-703 OOD cell is far outside this coordinate distribution.
Large negative state-code results in those cells therefore combine linear
model failure and representation drift. They cannot establish absent capacity.
The primary coarse failure occurs without that input clipping, however.

Meso state appears to help the later-weight cells at horizon 64 (+85/+90 bits
over history; about 10%/12% better mean MSE), but shuffled gains are
+182/+129/+87 and +133/+83/+63. These are also still worse than iid coding.
Preserve those directional positives, but do not call them new organization:
the nulls weaken the state-specific interpretation. Static regime/coordinate
offsets and a failing transferred history model can create such contrasts.

A declared diagnostic recalibrates ONLY sigma on each test's first 256 chunks,
then evaluates the remaining origins; compare it to strict transfer on those
exact origins. Weights, scalers, bins, mean predictions and MSE remain fixed.
For primary GRU at horizon 64, code gain flips from **-11.2 to +7.0 bits**,
while MSE is exactly unchanged at 1.204 and shared-sigma mean gain remains
negative. Under child-world transfer, GRU's horizon-eight code deficit shrinks
from -1,266 to -201 bits while MSE stays 11.965. Better calibration can rescue
probabilistic coding without rescuing a predictive abstraction. Unsupported
sigma-adaptation horizons are explicitly missing; no future audit labels
enter the calibration prefix.

## Resources, artifacts and next experiment

The main pass took about 110 seconds, peak Python RSS 286 MiB. No build,
training, inference or audio generation ran. New learned source probes occupy
about 520 KiB of JSON, including stored projections and normalization; this
actual artifact size is separate from the older 16-bit coefficient proxy.
The full 19-MiB diagnostic report is preserved in a deterministic 3.4-MiB
gzip archive with byte hashes. Compact summary, curves, drift and variance
controls are tracked; raw WAVs and learned probe files stay local.

Up to four CPU threads are authorized for later work when beneficial and RAM
permits. These small deterministic fits used one BLAS thread; no completed
work was rerun solely to use more threads. Existing other sessions and nearly
full swap were respected. No canonical checkpoint or normal Titan path changed.

The next informative measurement should use **a shared uncertainty model and
multiple declared source trajectories**, then a genuinely disjoint trajectory
for evaluation. That distinguishes poor single-path normalization from an
absence of reusable mean-predictive structure. Prefer existing artifacts and
causal audio/control/phase features before adding critic capacity. Require
gain over history, matched-capacity nulls and calibration-only controls;
do not use a source-selected favorable maximum as reward. Fixed probe weights
alone are not enough if apparent information comes from fixed sigma changes.

No runtime PD reward or attractor escape is promoted. Useful learned weights,
nonlinear abstractions, audio beauty and downstream conditioning remain
separate questions. Exact commands and archive-reading examples are in
`README.md`.
