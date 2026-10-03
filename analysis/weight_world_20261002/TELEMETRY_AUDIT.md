# Named-column audit of the supplied objections

`audit_existing.py` checks the six completed matched-exposure arms, the final
observer block, the finalized live continuation scalar trace, and the preserved
fresh-run trace. It validates CSV widths and hashes each inspected file.
The original anonymous criticism document was not supplied; the audit addresses
the user's listed objections. Detailed statistics are in `telemetry_audit.json`.

- **Rings and depth:** msfield deliberately bypasses the legacy neural-CA
  dilated ring machinery. Its reported ring count/gain are 0/0, and its
  full manifold depth is 4: all 64 channels at each scale remain available.
  Local transport/diffusion/reaction and adjacent-scale exchange remain active.
  These flags are not estimates of emergent dimensionality.
- **Rails:** the six mature matched arms have maximum named
  `field_rail_excess` between 0 and 4.45e-5; the observer maximum is 3.61e-5
  and the finalized continuation maximum is 7.54e-5. These do not support
  substantial mature railing. The fresh-run maximum is 0.0262, which is a
  different early regime and is preserved in the audit. `field_signed_mean`
  and RMS are separate columns and cannot be read as rail excess.
- **Confidence:** 86–100% of sampled mature rows reach the explicit 0.98
  clamp (depending on run). Reconstruction from the recorded error and
  calibration EMAs matches the formula within 3.3e-8. Fresh initialization
  starts from a manually defined confidence before the first predictor update;
  this explains its first-row formula discrepancy. "Raw" confidence in prior
  summaries means before authority gating, not before the confidence clamp.
  This is saturation of a heuristic, not demonstrated calibrated certainty.
- **Clipping:** accepted finite updates use `min(1,5/max(grad_norm,1e-6))`
  before Adam; this is not a parameter-update fraction. Recorded scales
  match within 3e-8. None of these traces has a scale at zero or <=1e-6.
  The smallest mature matched value is 0.3309; the finalized continuation
  minimum is 0.3860. Tiny positive scales could be legitimate under much
  larger finite gradients. Startup norm 0/scale 1 is the initialized telemetry
  state. Values repeat between optimizer horizons and must be deduplicated.
- **Triple HOLD:** no planner/bandit/selected triple HOLD appears in the six
  matched arms or final observer block. The finalized continuation has a
  longest nine-row streak spanning 80 chunks (6.83 seconds between endpoints).
  Sparse rows do not establish an uninterrupted episode between samples or
  a long-term lock. The claim must retain that sampling limit.
- **Rewards/advantages:** action-conditioned reward variation is present;
  absolute negative reward does not imply action invariance. Historical traces
  lack the pre-update bandit reward EMA, so they cannot reconstruct exact
  advantage. The reported reward-minus-adaptive-mean is explicitly a different
  centered score. New opt-in frozen evaluation records exact bandit advantage.
- **Low band:** the mismatch depends on family. In the finalized continuation,
  mean output-minus-target ratio is about +0.0406 for Chasing Tomorrow,
  -0.0894 for Concrete Desires, and -0.0525 for Echoes Through Dimensions.
  The matched A schedule also shows large negative mismatch for Architect of
  Change (-0.2730). This is a systematic family-conditioned mismatch in the
  first supervised log-band proxy, not proof of one universal bass excess.
- **Spikes and provenance:** existing fine-delta CSVs report the proposed
  learned step, before the current forcing. Their ten-chunk sample cadence
  generally misses tape/update boundaries, so even a spike on a different
  sampled phase cannot exclude boundary effects. Exact hidden deltas and
  realized forcing events are absent from these old scalar traces. Do not
  infer spike causation from them.

The live directory has an additional real provenance failure: its finalized
metadata/scalar CSV describe run `1790916563564-p8687-s44` ending 63,966,
while `msfield_trace` belongs to later run `1790919377781-p25937-s44`, with
152 complete rows and one truncated row near step 65,486. That process is no
longer present. The audit labels the field trace unmatched and incomplete,
leaves its tape/horizon attribution null, and does not join it to the older
metadata. The canonical files are not repaired or overwritten. All learning
comparisons use verified archived checkpoint copies instead.

A subsequent read-only checkpoint inspection found that the live **world and
Adam files agree at step 65,536**, while the finalized run metadata remains
at 63,966. Thus the live directory contains a consistent later autosave plus
older finalized metadata and an unfinished streamed field trace. This is not
evidence that the autosave tensors are numerically corrupt. Its exact run
provenance needs recovery before treating it as a promoted research parent.
The inspection's pre/post hashes were unchanged.
