# TITAN telemetry semantics

TITAN's telemetry mixes direct signal measurements with control heuristics and
artistic interpretation. The distinction matters when comparing experiments.

## Files and schema

Telemetry schema v10 writes five complementary artifacts per run. With
`--run-tag NAME`, telemetry and checkpoint filenames receive that tag instead
of overwriting the untagged run. Finalized audio filenames always add their
shared random hash after any run tag:

- `uncertainty_trace_rust.csv` is the sampled scalar trace. `raw_movement` is
  the mean absolute micro-field delta printed as `Move:` in the console.
  `uncertainty_movement` is a different bounded feature derived from movement
  trend and model surprise. The old ambiguous `movement` heading is removed.
- `ca_topology_rust.csv` is a headerless matrix of 1,024 macro-field values
  per sampled row. v9's macro CA is 32x32; this is intentionally incompatible
  with the 4,096-column v7 topology matrix.
- `ca_topology_index_rust.csv` maps every topology row to `run_id`, sample
  index, global and local step, morph and manifold depths, active spatial-ring
  count, far-ring gain, radiation amplitude, field entropy, and any morph event
  observed since the prior sample.
- `morph_events_rust.csv` records neurogenesis and pruning at their exact
  global and local steps, including before/after manifold and ring state.
- `titan_run_metadata_v9.json` records the build commit, dirty/release flags,
  invocation, reset mode, seed, thread count, requested BPTT and bounded tape,
  field dimensions, start/end state, output paths, and trace semantics.
  It also records the corpus manifest summary and optimizer resume/update
  counts. Resize runs include exact/resized/initialized/dropped model-tensor and
  optimizer-moment counts. v9 additionally records micro/macro dimensions,
  folded topology, active/max manifold depth and neighborhood-ring state,
  parameter count, core-update cadence, decoder control rate, phase timings,
  and the random hash shared by that run's finalized audio filenames.

The large topology matrix is written incrementally. Scalar and topology-index
records are streamed to bounded temporary JSONL spools and converted to the
stable CSV schemas during finalization; they are no longer retained as an
ever-growing in-memory JSON collection.

An untagged CSV trace is intentionally overwritten by the next untagged
process. Use `--run-tag` when retaining telemetry from multiple runs and use
the metadata `run_id` when joining their rows.

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
- `carrier_freq_l`, `carrier_freq_r`, and `carrier_beat_hz` expose low-frequency
  beating directly. `mimic_coarse`, `mimic_fine`, `band_loss`, `chroma_loss`,
  `onset_loss`, `modulation_loss`, `recurrence_loss`, `boundary_loss`, and
  `level_loss` separate source-grounding terms instead of collapsing them into
  one score.
- `output_low_band_ratio` and `target_low_band_ratio` compare the first
  supervised 20 Hz log band. They expose the sub-bass/RMS shortcut directly.
- `output_side_mid_log_ratio` and `target_side_mid_log_ratio` expose stereo
  side dominance. `decoder_stereo_corr`/`target_stereo_corr` and the two
  `stereo_level_log_ratio` fields close the panned-mono loophole: channel gain
  imbalance can no longer masquerade as spatial width. `stereo_balance_loss`
  combines all three relations without mixing target audio into the renderer.
  `stereo_side_geometry_loss`, `stereo_correlation_loss`, and
  `stereo_level_loss` expose its unweighted components. `decoder_pan` is the
  bounded global residual after the 4x8 field-to-stereo map. v8.1 limits it to
  +/-0.10. `decoder_pan_raw` exposes saturation before that map, while
  `pan_center_loss` measures squared energy in the bounded audible residual.
  It deliberately does not grow with an arbitrarily large raw coordinate.
  `decoder_side_control`, `decoder_width_control`, and `decoder_width_raw`
  distinguish collapsed side excitation, an active width rail, and saturation
  in the underlying width head. A version marker makes the first v8.1 resume
  reset only the changed pan/width controls' incompatible Adam moments.
- `development_best_spectral`, `development_mean_spectral`, and
  `development_mean_chroma` use fixed, gradient-excluded development families.
  `development_score`, `development_plateau_ready`, and
  `development_relative_improvement` make the morphic-growth gate auditable.
- `validation_best_spectral`, `validation_mean_spectral`, and
  `validation_mean_chroma` score every emitted chunk against fixed probes.
  `validation_is_strict` in run metadata must be true before these are called
  held-out results. A training-family fallback remains available for tiny or
  manually incomplete corpora, but is only a within-run diagnostic. Validation
  metrics never select a training target or an architecture transition.
- `target_file`, `target_frame`, and `target_chunks_left` identify the coherent
  source episode in force at each sample. This makes temporal-supervision bugs
  and corpus bias auditable.
- `ultrasonic_ratio` is the fraction of pre-master magnitude above 20 kHz. It
  is an aliasing/foldback guardrail, not a musical brightness score.

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
- `optimizer_updates` is the persisted cumulative AdamW step count;
  `optimizer_updates_run` is the current process count and `optimizer_resumed`
  states whether compatible moments were restored. `optimizer_migration` in
  run metadata separates exact, resized, and newly initialized moment pairs.
- `core_update_every_tapes` in run metadata is the two-timescale optimization
  cadence. Decoder weights receive every optimizer horizon; CA, GRU, episodic,
  and MorphicStack gradients are retained only on the scheduled full tapes.
- `phase_profile` reports average milliseconds per completed chunk for model
  forward, target loading, loss/metrics, backward, optimizer, output I/O, and
  checkpoint work. Target-loading time is a measured subset of loss/metrics,
  not an additional mutually exclusive bucket.
- `side_energy_width` is the old RMS channel-difference measure. It can be high
  for panned mono and is retained as a diagnostic, not called true width.
  `width` multiplies it by interchannel incoherence, so perfectly correlated
  unequal-gain channels measure near zero. `stereo_corr` is the final post-DC,
  pre-master normalized correlation. Strongly negative values warn of mono
  cancellation; strongly positive values warn of mono collapse.
- `morph_frozen` and `morph_max_depth` state the run's structural policy.
  Growth additionally requires a strict development split and a completed
  plateau window; validation metrics never participate in that decision.
  Run metadata additionally records the physically constructed `morph_layers`
  and internal `morph_width`; these determine parameter and optimizer size but
  do not change the MorphicStack's 512-dimensional external interface.
- `manifold_depth` is the number of active 16-feature CA sheets derived from
  morph depth. `active_spatial_rings` is one at L01 and two once the dilated
  far ring has nonzero gain. `far_ring_gain` exposes its gradual fade-in. These
  are architecture controls, not estimators of intrinsic physical dimension.
- `motif_capacity` in run metadata is the selected host-memory limit (1--4096,
  default 64). `motifs_active` is the retained entry count at finalization.
  Resume-time growth retains every motif; shrinking retains the newest entries
  and discards the oldest first.
- `field_entropy` is the channel-archetype entropy in bits for the current
  micro field. It is not the entropy of the rendered waveform.
- `crit_gain` is fixed at 1.0 in v9. `sigma` remains useful evidence, but no
  learning-rate singularity is applied until calibration establishes a real
  critical surface rather than merely naming one.

Claims about improved sound should be supported by repeated seeded runs,
ablation comparisons, objective audio measurements, and blinded listening—not
by these internal metrics alone.
