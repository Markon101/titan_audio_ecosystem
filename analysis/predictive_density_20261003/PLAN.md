# Predictive density: frozen measurement and bounded selection

Written before the substantive battery, October 3, 2026.

## Decisions

Use the existing optimizer-free TITANM10 harness and a small external NumPy
probe, not a neural critic or a new training loss. The opt-in generation
objective selects among frozen futures from the same parent. It renders an
output choice; it does not replace a runtime world or enable attractor escape.
The zero-coefficient path chooses the untouched baseline exactly. No Rust
forward equation, optimizer, checkpoint format, or normal training path changes.

Frozen source: archived step 58,107, tag v10-msfield-fresh-20260930-02. Hash
all source files before/after. Reuse existing fixed-source intervals and
repeat them in a declared train-only schedule for longer frozen inference.
Record model/world/RNG/binary identities, resource limits, every candidate,
and optimizer_steps=0. Do not use development/validation scores for fitting,
normalization, candidate choice, reward, or coefficient calibration.

## Measuring stick

At chunk horizons 1,4,16,64, estimate proper discrete predictive code lengths
for deterministic low-cost audio descriptors (envelope, log bands, flux,
stereo geometry). Learn quantization and normalizers on the chronological
training prefix only. Use disjoint training/calibration/selection/audit
segments with horizon purging. Fit tiny ridge predictors; calibrate Gaussian
residual scales and choose simple baselines only in the calibration segment.
Integrate probabilities over learned bins and renormalize; do not call raw
MSE, a log variance ratio, or an uncalibrated Gaussian density literal bits.

Compare to iid marginal, persistence, local trend, output-history ridge, and
periodic-copy baselines. Report iid gain separately from gain beyond the
calibration-chosen simple baseline. Report total bits, bits/feature/forecast,
assumed 16-bit coefficient cost, net bits after that cost, parameter count,
fit/predict multiply-add estimates, and wall time separately. PD is an
operational coding-efficiency ratio, not an estimator of true mutual
information. No fitted code length alone establishes music quality.

Probe current GRU, fine/meso/coarse pooled structure, early/middle/late Morphic
residual sketches, and host/controller/motif observational summaries for
future output, not merely themselves. Limit each added view to four fixed
projection coordinates. Use equal dimensions and report individual/joint
gains, incremental gain above either member, and G(A,B)-G(A)-G(B). The latter
is operational and can be negative under redundancy; it is not PID.

Existing regime captures describe forward-proposed state and reconstructed
Morphic residuals. Under holds/bypasses they need not equal committed state
or the actually used Morphic readout. Internal contributions will therefore
be measured on intact candidates only; earlier lesion/clamp audio can receive
output-only analysis with an explicit short-window support warning. Host and
motif summaries are proxies, not the full memory contents. No target identity
or future forcing is a probe input. Current output history is available at t.

## Falsification and anti-collapse

Use silence, constant state/audio, a sinusoid, exact repeated textures,
colored noise, white noise, shuffled time, shared-phase audio surrogates,
and a synthetic delayed cross-view process. Include same-capacity shifted
and time-shuffled internal-view nulls. A structure reward requires coding
gain beyond the simple baselines AND enough gain to cover its declared
coefficient cost on the selection segment, with positive gain in both halves.
Reject silence/nonfinite/rails; report entropy, activity, compression and
repetition independently rather than adding many heuristic bonuses.
If noise or trivial controls receive reward, disable nonzero selection until
the error is understood. Persist negative gains, not just positive summaries.

Learning progress must compare predictors trained on smaller/larger prefixes
on the SAME future segment. PD(t)-PD(t-k) on different data confounds task
drift with acquisition. Report the fixed-target progress diagnostic; do not
reward repeated probe resets or claim it is Titan parameter learning.

## Initial battery

1. Deterministic synthetic/null and leakage tests before reward use.
2. Short A baseline versus B measurement-only no-op parity, including raw and
   post-DSP WAVs, weights, complete world fingerprint, exact target frames.
3. Two declared snapshots (warmup 0 and 128, one parent, not model seeds),
   1,024 chunks each, stride 8: intact plus two bounded initial energy-state
   offsets +0.01,+0.03 using existing cloned-state perturbation code. These
   are host-state interventions, not proof of a substrate-specific reward.
   Energy is clipped to existing bounds; log the RMS/energy confound.
4. A/B use the intact output. C uses --pd-weight 0.02, bounded reward, and a
   normalized waveform-displacement cost relative to the intact candidate.
   D uses deliberately naive predictability with that protection omitted.
   Selection sees only the selection prefix; report the untouched audit
   suffix separately. Always include no-op and deterministic tie-to-no-op.
   The same candidate bank serves A/B/C/D; do not invent independent runs.
5. Output-only reuse of prior matched host-clamp/closed-loop lesions. Do not
   pool across interventions as independent observations or mistake existing
   short clips for long-horizon evidence. Package declared selected audio for
   later listening; this interface cannot provide subjective audio ratings.

Before analyzing those short reused clips, declare a separate sensitivity
protocol with maximum history lag 32 (rather than 128). Their 256 chunks
cannot support the primary long-context split. Keep this sensitivity labeled
and outside generation selection; unsupported long horizons remain missing.

These modest frozen runs fit the measured device envelope. No compilation is
needed if the existing binary/harness meets the protocol. Sequential runs,
startup available RAM >=1.5 GiB, stop below 0.75 GiB, preserve partial output,
never overwrite a completed arm. Respect other sessions and check processes
before each run. Publish exact commands, overhead, failures, decision gates,
and both positive and negative evidence. Commit/push on this dedicated branch.

## Post-battery protocol record

After the first v1 window was frozen, a sign/gain-changing delayed-loop
falsifier exposed a nonzero reward. Preserve W0_RESULTS.json and the v1
control reports. The final v2 readout adds affine-periodic baselines, broader
training-prefix period search and an unpenalized-baseline contrast, reusing
the exact same frozen banks. Nonzero export selection requires v2. This is
an explicitly revised estimator, not a preregistered success claim. Decoder
control/cyclic phase views and normalized curve columns are measurement-only
extensions. No coefficient, candidate budget or intervention was changed to
produce a winning Titan future. Both C decisions remain intact.

Direct comparison subsequently established 87.5% exact audio overlap between
the intact snapshot windows. Treat them as shifted fitting/audit windows of
one trajectory, with this confound preserved in overlap_receipt.json.
