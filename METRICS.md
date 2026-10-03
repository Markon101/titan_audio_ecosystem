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

## Opt-in v10 metastable-regime measurements

For msfield, `active_spatial_rings=0` and `far_ring_gain=0` mean that the
legacy v9 dilated neural-CA ring mechanism is not used. They do not mean
absence of spatial interaction: msfield has learned local advection,
diffusion, reaction, and fine/meso/coarse exchange. All 64 channels at each
scale remain active, so msfield reports full manifold depth 4 independently
of Morphic depth. These are substrate flags, not learned dimensionality.

`model_confidence` is the explicit 0.02..0.98-clamped heuristic from predictor
error/calibration EMAs. The term "raw" in existing summaries means before
the adaptive authority gate, not before that clamp. Saturation at 0.98 is
not calibrated evidence of certainty. Opt-in frozen evaluation records
`log_confidence_unclamped` and `confidence_unclamped` from the same EMAs.

`clip_scale` is held at the most recent successful optimizer horizon and
repeated on intermediate sampled rows. At startup it is 1 even though
`grad_norm` is 0. For finite norms it is `min(1, 5 / max(norm, 1e-6))`, and
it scales gradients before Adam, not the learning rate or parameter delta.
A tiny positive scale can be valid for a huge finite gradient; exact zero
cannot arise from the normal finite accepted update. Sparse trace rows must
be deduplicated by cumulative optimizer update to estimate clipping rates.

Frozen weight/world studies additionally support `--analysis-common-rng`,
`--analysis-target-origin N`, `--analysis-active-depth N`, and
`--analysis-evaluation-metrics`. These are msfield-only analysis flags. The
first fixes controller/forcing seed and relative event phases across worlds;
the second maps targets to a relative clock without changing native world
timestamps; the third uses a fixed active prefix without changing tensors.
The last scores strict held-out probes observationally, captures compact
trajectory views, records predictor/action/health details, and checks exact
parameter hashes before and after rollout. This standardized protocol is an
artificial matched-state evaluation, not an exact replay of native training.

`--regime-capture --regime-stride N` writes a run-ID-specific JSONL sidecar
for the isolated msfield substrate only. It samples pooled signed and RMS
spatial maps at all three scales, seeded orthogonal sketches of GRU and
representative Morphic activations and layer deltas, decoder seed/control
state, and post-DSP audio descriptors. Internal values are captured before
that chunk's optimizer update. Sampling has no feedback into training,
ecology, motif storage, or target selection; the normal path remains off.

`analysis/metastable_20261001/regime_archive.py` turns that raw capture into
an offline open-endedness trace and separate saveable regime archive. Five
domains (field, GRU, Morphic, decoder, audio) have equal total distance
weight after calibration/online normalization. A candidate region needs four
consecutive samples, while `persistent_new_regime` fires only after the
calibrated long dwell and mostly healthy visits. `regime_id` can therefore
name an unconfirmed candidate; check `regime_persistent` before counting it.
The trace records nearest archive distance, short/long novelty, dwell,
revisits/transitions, covariance participation, topology/entropy change,
motif rejection, trap components, and no-intervention fields. An absent
candidate score is `null`, not zero. The DOT/JSON/SVG graph records region
transitions after each offline replay.

The trap score is a descriptive offline diagnostic. Its hysteretic
`over_resident` label requires sustained corroboration from independent
dynamic indicators; a healthy long dwell, saturated confidence, or motif
similarity rejection alone cannot trigger it. There is currently no runtime
escape controller. Thresholds come from the first half of the first
observer-only mature continuation and are held fixed for later forks; that
calibration half is in-sample. Do not treat a high trajectory dimension or
high regime count as intrinsically good: white noise raises dimension, and
short block shuffling can create candidate regions. The companion null report
and fixed-probe/audio evidence are required for an organization claim.

`--target-schedule FILE` is a separate v10-only, opt-in corpus experiment.
It selects a single **training** alias and source frame by absolute global
step from a checksummed six-slot JSON file. Target selection consumes no
runtime RNG; sampled target filename, frame, and remaining chunks must agree
with the file in every arm. The loader rejects gaps, invalid slots, episodes
beyond WAV length, and non-strict held-out probes. It does not schedule or
inspect development/validation targets. When disabled, the original
uniform-family episode sampler is unchanged. Run metadata records schedule
path, SHA-256, and seed only when the flag is active.

`--max-autograd-tape N` optionally lowers the actual v10 graph length from
the default eight chunks, while `--bptt 64` still accumulates gradients to a
64-chunk optimizer horizon. This changes temporal credit assignment. Compare
arms only when both requested horizon and actual tape length match; do not
attribute an absolute difference from an eight-chunk run to the corpus.

## Analysis report schema v1

Frozen instrumentation writes `titan_audio_analysis` schema v1 sidecars. The
top level records analysis/build/checkpoint identity, corpus provenance,
configuration, conditions, developmental/rollout age, model statistics,
rollouts, ablations, perturbations, benchmarks, trained/init controls,
separability, attribution, warnings, artifacts, interpretation constraints,
and the canonical pre/post non-mutation result.

Every analysis report states:

- `weights_frozen: true`, `backward_passes: 0`,
  `optimizer_constructed: false`, and `optimizer_steps: 0`;
- whether direct reference input exists (it does not in Audio v9);
- the target-error-feedback, morphology, controller, and forcing semantics;
- checkpoint/world step, physical and active morphology, analysis seed, and
  independent RNG policy; and
- validity and a reason for measurements that are mathematically undefined.

Analysis JSON step records include `recurrent_proposed_delta_rms`, the GRU
update before a clone-only hold, and `recurrent_committed_delta_rms`, the
update after that intervention. These fields distinguish a zero learned
update from a hold that merely fails to affect audible output. They are
observational and do not alter the normal runtime or the v1 trace CSV columns.

State distance and phenotype/audio distance are separate domains.
`latent_recovery_ratio(t)` is distance to the paired unperturbed baseline at
time `t`, divided by the non-negligible initial distance. Time-to-half is valid
only if the ratio remains at or below 0.5 for two sampled observations. Small
audio distance with persistent state distance is `phenotype robustness`, not
state recovery or self-healing.

Approximate recurrence uses a compact normalized state signature and a minimum
temporal separation. It is reported as an `approximate_cycle_candidate` with a
validity flag, not proof of a limit cycle or attractor. Separability clusters
are operational phenotype regimes, not concepts. Benchmark distance is signal
descriptor proximity, not semantic understanding.

Paired analysis conditions use deterministic exogenous force-macro, radiation,
and kick streams keyed by checkpoint RNG identity, absolute step, and stream
tag. Controller and target sampling have separate matched analysis streams.
These streams never advance the saved normal-runtime RNG. Perturbation
directions use `--analysis-seed`, independently of all of them.

Corpus provenance retains `role_counts` and `scheduler_order` for compatibility:
these describe declared manifest roles and listed WAVs in sorted disk order.
`name_role_candidate_role_counts` and `name_role_candidate_order` apply the
production loader's role and generated-filename exclusion before WAV format
and length checks. `quarantine_role_conflicts` names entries marked as TITAN
generated while assigned a non-exclude role. These additive fields report the
disagreement without editing the manifest or claiming that every candidate
passed production WAV indexing.

The harness exports fixed-column CSV traces, structured JSON, 48 kHz stereo
WAVs, fixed-scale spectrograms, and fixed-range micro/macro state atlases.
Artifact-index entries include media type, condition, offsets, processing
status, byte size, and SHA-256. JSON never uses NaN or infinity as a numeric
value.
