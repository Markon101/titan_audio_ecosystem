# TITAN telemetry semantics

TITAN's telemetry mixes direct signal measurements with control heuristics and
artistic interpretation. The distinction matters when comparing experiments.

## Files and schema

Telemetry schema v2 writes five complementary artifacts per run:

- `uncertainty_trace_rust.csv` is the sampled scalar trace. `raw_movement` is
  the mean absolute micro-field delta printed as `Move:` in the console.
  `uncertainty_movement` is a different bounded feature derived from movement
  trend and model surprise. The old ambiguous `movement` heading is removed.
- `ca_topology_rust.csv` remains a headerless matrix of 4,096 macro-field
  values per sampled row so numerical consumers retain a fixed shape.
- `ca_topology_index_rust.csv` maps every topology row to `run_id`, sample
  index, global and local step, depth, radiation amplitude, field entropy, and
  any morph event observed since the prior sample.
- `morph_events_rust.csv` records neurogenesis and pruning at their exact
  global and local steps, including the resulting depth and radiation value.
- `titan_run_metadata_v6.json` records the build commit, dirty/release flags,
  invocation, reset mode, seed, thread count, requested BPTT and bounded tape,
  field dimensions, start/end state, output paths, and trace semantics.

The CSV trace is intentionally overwritten per process. Use the metadata
`run_id` when archiving or joining artifacts from multiple runs.

## Measurements and estimators

- `sigma` is a mean-centered, multi-lag propagation-slope estimate over recent
  movement. It is not a literal physical branching ratio. Its accompanying
  `criticality_confidence` discounts low-variance, non-stationary, short, or
  cross-lag-inconsistent windows.
- `phi` is a bounded audio-structure proxy derived from observable signal
  statistics. It is not Integrated Information Theory's Phi.
- `pi_proxy` summarizes multi-lag predictive structure. It is not causal proof
  of information integration.
- `empowerment` is a transition-variance heuristic coupled to movement. It is
  not channel-capacity empowerment.
- `novelty_dmin` is distance to recent, level-normalized log-spectral shapes.
  It detects timbral change, not semantic or compositional novelty.

## Controllers

- `V` is a designed control potential over operating-state summaries, not
  thermodynamic free energy.
- `temp` controls perturbation, plasticity, and exploration. It is an adaptive
  control variable, not physical temperature.
- `energy` is a bounded synthesis/control budget.
- `temp_stuck`, `temp_subcritical`, `temp_curiosity`, `temp_stagnation`, and
  `temp_motion` are the additive drives of the temperature target before its
  final clamp and EMA. They are the first place to inspect persistent heat.
- `controlled_shear_rms` is the requested and phase-normalized RMS of the
  structured macro-field forcing.
- `field_signed_mean` detects sign bias in the micro field;
  `field_rail_excess` is the mean amount by which cell magnitude exceeds 0.9.
  A weak global-mean damper removes only 3.5% of the field's DC mode per chunk,
  so local signed structure remains free while population drift is bounded.
- `radiation_probability` is the per-chunk sparse-radiation hazard and is
  independent of the selected BPTT window.
- `grad_norm` is the requested-horizon mean gradient norm before global
  clipping and `clip_scale` is the factor applied before AdamW updates its
  moments. Horizons above 8 accumulate bounded, detached tape segments.
- `stereo_corr` is the post-DSP Pearson correlation between channels. Strongly
  negative values warn of mono cancellation even when width sounds impressive;
  Haas side gain is capped at 1.25 to bound that risk.
- `field_entropy` is the channel-archetype entropy in bits for the current
  micro field. It is not the entropy of the rendered waveform.

Claims about improved sound should be supported by repeated seeded runs,
ablation comparisons, objective audio measurements, and blinded listening—not
by these internal metrics alone.
