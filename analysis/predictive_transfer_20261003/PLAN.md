# Fixed predictive-probe transfer, October 3, 2026

Written before transfer scores. Reuse existing optimizer-free artifacts;
no Titan render, build, training, controller promotion or checkpoint writes.
This is held out from the new probe fitting, not previously uninspected data.

## Source and tests

Train/normalize/quantize on the first 40% of PD w0 intact (source: step-58107
parent, forcing seed 20261003); calibrate residual scales and choose simple
comparators on its next 20%. Freeze all state, serialize and reload it before
scoring. Ignore the source selection interval and score its final 20% as the
within-trajectory reference. Source tests must reproduce existing v2 codes.

Primary transfer is historical L16_WP_SP: same archived parent weights/world,
different declared forcing/controller seed 20261002 and 384-chunk schedule.
Then L16_WP_SC tests those weights in the mature child world. L16_WC_SP and
L16_WC_SC are weight/world-shift sensitivities, not evidence of learning from
probe error. L16_WE_SP is a distant untrained-depth/OOD sensitivity, not native
early-model performance. PD +0.01/+0.03 energy paths in w0 and w128 test
within-family intervention transfer. w128 intact is a known overlapping
reference only. Verify byte identity and record overlap instead of treating
these as independent model seeds or trajectories chosen by scores.

All test histories are causal and episode-local. Warmup adjusts the nuisance
elapsed-time coordinate; world absolute age is not a predictor feature.
At least 128 audio chunks / 16 stride-eight observations precede test origins.
Evaluate all available later origins at horizons 1/4/16/64 for audio and
8/16/64/128 for state-to-audio. Report support and normalized gains. The
384-chunk clips cannot establish tens-of-seconds musical predictive depth.
No source fitting pairs cross train/calibration boundaries; target probes,
validation scores, future forcing and target identities are never inputs.

## Frozen measuring stick

Keep the v2 feature definitions, fixed projections, ridge=10, sigma floor=.15,
13 audio targets, and 16-bit coefficient proxy. Freeze source quantizers,
normalizers, projection conventions, affine-period coefficients, selected
baseline families, linear weights and residual scales. Test length/content
must not refit anything. Save a complete JSON probe artifact and compare
predictions before/after restoration exactly. Normal production behavior and
previous study artifacts remain unchanged.

Probe all previously declared state views and four declared pairs. Report
gain over iid, source-selected simple comparator, unpenalized source comparator,
history, and same-capacity slow history. Operational synergy uses incremental
gain over the common HISTORY comparator, avoiding double-counting shared
history in joint-minus-sum. This is still not PID. Compare source-fitted
shuffled-view predictors, with their own train/calibration shuffles, against
shuffled test views; three seeds, no selection of a favorable null.

Also report mean-only gains when history and state share history's source
residual sigma. Gaussian confidence differences can otherwise masquerade as
new predictive information. Report raw/scaled MSE, quantizer edge saturation,
scaler clipping and uncertainty-calibration drift. Product code lengths and
assumed coefficient costs remain operational, not literal organism MDL/MI.

## Cheap diagnostic if strict transfer fails

Predetermined sensitivity: retain ALL weights/normalizers/quantizers and
recalibrate ONLY sigma on a test's first 256 chunks; evaluate origins after
that prefix. Purge future labels from the calibration boundary and require
eight eligible pairs. Unsupported horizons remain missing. The source
chronological reference receives no test recalibration. Keep this adapted
result separate from strict transfer and do not use it for generation/reward.
If this rescues code gain without improving MSE, interpret calibration, not
new predictive abstractions. No new coefficients, feature or horizon tuning.

## Gates and reporting

Tests: train/calibration horizon purging; suffix/test mutation leaves frozen
parameters and comparator choice unchanged; exact save/load and deterministic
prediction; one trajectory cannot use another's history; no nonfinite codes;
known transferable complementary process vs shuffled controls; calibrated
failure under distribution shift. Recheck nine prior tests and source hashes.
Stream one WAV at a time; no giant array/cache. Check RAM/processes before
work, respect Hermes, and record elapsed/RSS. Preserve original checkpoints,
completed results and partial outputs. Freeze full results, readable curves,
positive/negative/null evidence and exact commands; commit/push dedicated work.
Do not manufacture condition p-values from correlated time points.
