use super::metrics::{audio_metrics, AudioMetrics};
use super::state::AnalysisWorld;
use anyhow::Result;
use candle_core::{DType, Tensor};
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct Intervention {
    pub name: String,
    pub micro_nca_hold: bool,
    pub macro_nca_hold: bool,
    pub gru_hold: bool,
    pub episodic_read_zero: bool,
    pub episodic_empty: bool,
    pub planner_model_zero: bool,
    pub bandit_zero: bool,
    pub motif_recall_disabled: bool,
    pub radiation_disabled: bool,
    pub shear_disabled: bool,
    pub micro_kick_disabled: bool,
    pub potential_gains_identity: bool,
    pub target_feedback_disabled: bool,
}

impl Intervention {
    pub(crate) fn full() -> Self {
        Self {
            name: "full".to_string(),
            ..Self::default()
        }
    }

    pub(crate) fn parse(name: &str) -> Result<Self> {
        let mut intervention = Self {
            name: name.to_string(),
            ..Self::default()
        };
        match name {
            "full" | "none" => {}
            "micro_nca_hold" => intervention.micro_nca_hold = true,
            "macro_nca_hold" => intervention.macro_nca_hold = true,
            "gru_hold" => intervention.gru_hold = true,
            "episodic_read_zero" => intervention.episodic_read_zero = true,
            "episodic_empty" => {
                intervention.episodic_empty = true;
                intervention.episodic_read_zero = true;
            }
            "planner_model_zero" => intervention.planner_model_zero = true,
            "bandit_zero" => intervention.bandit_zero = true,
            "motif_recall_disabled" => intervention.motif_recall_disabled = true,
            "radiation_disabled" => intervention.radiation_disabled = true,
            "structured_shear_disabled" => intervention.shear_disabled = true,
            "micro_kick_disabled" => intervention.micro_kick_disabled = true,
            "potential_gains_identity" => intervention.potential_gains_identity = true,
            "target_independent_feedback_ablation" => intervention.target_feedback_disabled = true,
            "arbiter_bypass" => {
                // The arbiter participates only in loss mixing. With no loss,
                // backward, or optimizer in this runner, its output path is absent.
            }
            unsupported => anyhow::bail!(
                "unsupported safe Phase 1 ablation {unsupported:?}; no production-forward hook was added"
            ),
        }
        Ok(intervention)
    }

    pub(crate) fn subsystem_class(&self) -> &'static str {
        if self.micro_nca_hold || self.macro_nca_hold || self.gru_hold || self.episodic_read_zero {
            "learned_neural_state_transition"
        } else if self.planner_model_zero {
            "learned_world_model_host_effect"
        } else if self.bandit_zero || self.motif_recall_disabled || self.potential_gains_identity {
            "host_controller_or_memory"
        } else if self.radiation_disabled || self.shear_disabled || self.micro_kick_disabled {
            "host_stochastic_forcing"
        } else if self.target_feedback_disabled {
            "target_error_feedback"
        } else if self.name == "arbiter_bypass" {
            "training_only_null_control"
        } else {
            "full_baseline"
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct StepRecord {
    pub rollout_offset: usize,
    pub absolute_step: u64,
    pub action: String,
    pub model_proposal: String,
    pub bandit_proposal: String,
    pub force_macro: bool,
    pub target_file: Option<String>,
    pub target_frame: Option<usize>,
    pub target_error_feedback: Option<f32>,
    pub movement: f32,
    pub micro_rms: f32,
    pub macro_rms: f32,
    pub recurrent_rms: f32,
    pub micro_near_bound_fraction: f32,
    pub macro_near_bound_fraction: f32,
    pub synergy: f32,
    pub sigma: f32,
    pub energy: f32,
    pub temperature: f32,
    pub radiation_probability: f32,
    pub radiation_realized: bool,
    pub shear_amplitude: f32,
    pub kick_amplitude: f32,
    pub morph_active_depth: usize,
    pub manifold_depth: usize,
    pub far_ring_gain: f64,
    pub episodic_slots: usize,
    pub motif_occupancy: usize,
    pub model_confidence_raw: f32,
    pub model_confidence_effective: f32,
    pub audio_raw: AudioMetrics,
    pub audio_post_dsp: AudioMetrics,
    pub warnings: Vec<String>,
}

pub(crate) struct StepOutput {
    pub record: StepRecord,
    pub left: Vec<f32>,
    pub right: Vec<f32>,
    pub raw_left: Vec<f32>,
    pub raw_right: Vec<f32>,
}

pub(crate) fn step(
    world: &mut AnalysisWorld,
    rollout_offset: usize,
    intervention: &Intervention,
) -> Result<StepOutput> {
    let device = world.micro.device().clone();
    let absolute_step = world.absolute_step;
    let escape_strength = world.adaptive.escape_strength();
    let curiosity =
        super::super::ecological_curiosity(&world.adaptive, world.host.stagnation_ticks);
    let aperture = world.uncertainty.branch_aperture();
    let control_slew =
        (0.10 + 0.18 * world.controller.meta.surprise() + 0.14 * escape_strength).clamp(0.08, 0.38);

    let prediction = if let (Some(input), Some(_actual)) =
        (&world.pending_predictor_input, &world.last_observation)
    {
        let (mean, log_variance) = world.bundle.monitor.forward(input)?;
        Some((
            mean.reshape((super::super::OBS_DIM,))?.to_vec1::<f32>()?,
            log_variance
                .reshape((super::super::OBS_DIM,))?
                .to_vec1::<f32>()?,
        ))
    } else {
        None
    };
    if absolute_step.is_multiple_of(super::super::PLAN_EVERY as u64) {
        if intervention.planner_model_zero {
            world.controller.cached_model_scores = [0.0; super::super::ACTION_COUNT];
        } else {
            world.controller.cached_model_scores = super::super::plan_action_scores(
                &world.bundle.monitor,
                &world.hidden.detach(),
                &device,
                &world.adaptive,
                world.host.smoothed_control,
                control_slew,
            )?;
        }
    }
    if intervention.bandit_zero {
        world.controller.bandit.q = [0.0; super::super::ACTION_COUNT];
    }
    let motif_available =
        !intervention.motif_recall_disabled && world.motifs.has_recallable(absolute_step);
    let action = world.controller.choose(
        world.host.last_temp,
        &mut world.adaptive,
        motif_available,
        &mut world.controller_rng,
    );
    let mut target_control = super::super::SynthesisControl::for_action(action);
    if action == super::super::ControlAction::Recall && !intervention.motif_recall_disabled {
        if let Some((remembered, strength)) = world.motifs.recall(
            world.last_observation.as_ref(),
            absolute_step,
            &mut world.motif_diagnostics,
        ) {
            target_control =
                target_control.blend(remembered, (0.35 + 0.55 * strength).clamp(0.0, 0.9));
        }
    }
    if escape_strength > 0.0 {
        target_control = target_control.blend(
            super::super::SynthesisControl::for_action(super::super::ControlAction::Turbulence),
            (0.18 + 0.62 * escape_strength).clamp(0.0, 0.82),
        );
    }
    let control = world
        .host
        .smoothed_control
        .blend(target_control, control_slew);
    world.host.smoothed_control = control;
    let predictor_input =
        super::super::predictor_input(&world.hidden.detach(), action, control, &device)?.detach();
    let force_probability = (0.18
        + aperture * 0.48
        + 0.12 * (control.shear_mult - 1.0).max(0.0)
        + 0.10 * world.controller.meta.surprise()
        + 0.32 * escape_strength)
        .clamp(0.05, 0.98);
    let mut macro_rng = forcing_rng(world.forcing_seed, absolute_step, 0x4D41_4352);
    let force_macro = absolute_step.is_multiple_of(super::super::MACRO_UPDATE_EVERY)
        && macro_rng.gen_range(0.0f32..1.0) < force_probability;
    let episodic_out = if intervention.episodic_read_zero {
        Tensor::zeros((1, super::super::EPI_DIM), DType::F32, &device)?
    } else {
        world.bundle.episodic.read(&world.hidden, &device)?
    };
    let old_micro = world.micro.clone();
    let old_macro = world.macro_t.clone();
    let old_hidden = world.hidden.clone();
    let output = world.bundle.model.forward(
        &world.micro,
        &world.macro_t,
        &world.hidden,
        &episodic_out,
        world.phases,
        world.theta_prev,
        world.theta_prev2,
        force_macro,
        absolute_step,
        false,
        world.energy,
        &control,
    )?;
    let raw_stereo = output.stereo.tanh()?;
    let raw = raw_stereo.to_vec2::<f32>()?;
    let raw_left = raw[0].clone();
    let raw_right = raw[1].clone();
    let movement = output.movement_t.to_scalar::<f32>()?;
    let synergy =
        super::super::calculate_cross_layer_synergy_tensor(&output.next_micro, &output.next_macro)?
            .to_scalar::<f32>()?;
    let field_means = output
        .next_micro
        .mean(candle_core::D::Minus1)?
        .mean(candle_core::D::Minus1)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let (_, field_entropy, _) = super::super::SemanticField::archetype_field(
        &field_means
            .iter()
            .map(|value| (value + 1.0) * 0.5)
            .collect::<Vec<_>>(),
    );
    if let (Some(actual), Some((mean, log_variance))) =
        (&world.last_observation, prediction.as_ref())
    {
        world.controller.meta.update(actual, mean, log_variance);
    }

    let frequencies = [
        first_f32(&output.cur_freq_l)?,
        first_f32(&output.cur_freq_r)?,
        first_f32(&output.mod_freq_l)?,
        first_f32(&output.mod_freq_r)?,
    ];
    world.bundle.model.current_freq_l = frequencies[0];
    world.bundle.model.current_freq_r = frequencies[1];
    world.bundle.model.last_pan = first_f32(&output.pan)?;
    let chunk_dt = super::super::CHUNK_SIZE as f32 / super::super::SAMPLE_RATE as f32;
    for (phase, frequency) in world.phases.iter_mut().zip(frequencies) {
        *phase =
            (*phase + super::super::TWO_PI * frequency * chunk_dt).rem_euclid(super::super::TWO_PI);
    }
    let pair = output.pair_sums.to_vec1::<f32>()?;
    world.theta_prev2 = world.theta_prev;
    world.theta_prev = pair[1].atan2(pair[0] + 1e-6);
    let aux_left = output.aux_freqs_l.to_vec1::<f32>()?;
    let aux_right = output.aux_freqs_r.to_vec1::<f32>()?;
    for index in 0..3 {
        world.bundle.model.aux_phase_l[index] = (world.bundle.model.aux_phase_l[index]
            + super::super::TWO_PI * aux_left[index] * chunk_dt)
            .rem_euclid(super::super::TWO_PI);
        world.bundle.model.aux_phase_r[index] = (world.bundle.model.aux_phase_r[index]
            + super::super::TWO_PI * aux_right[index] * chunk_dt)
            .rem_euclid(super::super::TWO_PI);
    }
    let scan_left = output.scan_freqs_l.to_vec1::<f32>()?;
    let scan_right = output.scan_freqs_r.to_vec1::<f32>()?;
    for index in 0..super::super::SCAN_PARTIALS {
        world.bundle.model.scan_phase_l[index] = (world.bundle.model.scan_phase_l[index]
            + super::super::TWO_PI * scan_left[index] * chunk_dt)
            .rem_euclid(super::super::TWO_PI);
        world.bundle.model.scan_phase_r[index] = (world.bundle.model.scan_phase_r[index]
            + super::super::TWO_PI * scan_right[index] * chunk_dt)
            .rem_euclid(super::super::TWO_PI);
    }

    let raw_metrics = audio_metrics(&raw_left, &raw_right);
    let boost_target = if raw_metrics.peak < 0.25 {
        (0.25 / (raw_metrics.peak + 1e-6)).clamp(1.0, 4.0)
    } else {
        1.0
    };
    world.host.boost_state = world.host.boost_state * 0.9 + boost_target * 0.1;
    let mut left = raw_left
        .iter()
        .map(|sample| (sample * world.host.boost_state * 0.92).tanh())
        .collect::<Vec<_>>();
    let mut right = raw_right
        .iter()
        .map(|sample| (sample * world.host.boost_state * 0.92).tanh())
        .collect::<Vec<_>>();
    for (left, right) in left.iter_mut().zip(&mut right) {
        let next_left = *left - world.host.dc_x1_l + 0.998 * world.host.dc_y1_l;
        world.host.dc_x1_l = *left;
        world.host.dc_y1_l = next_left;
        *left = next_left;
        let next_right = *right - world.host.dc_x1_r + 0.998 * world.host.dc_y1_r;
        world.host.dc_x1_r = *right;
        world.host.dc_y1_r = next_right;
        *right = next_right;
    }

    let target_feedback = if intervention.target_feedback_disabled {
        None
    } else if let Some(loader) = world.target_loader.as_mut() {
        let targets =
            loader.sample_chunks(super::super::TARGET_K, &mut world.target_rng, &device)?;
        loader.commit_selection(0);
        let target = targets
            .narrow(0, 0, 1)?
            .reshape((2, super::super::CHUNK_SIZE))?;
        let output_spectrum = world.target_projector.log_mag(&raw_stereo)?;
        let target_spectrum = world.target_projector.log_mag(&target)?.detach();
        Some(
            super::super::robust_distance(&output_spectrum.sub(&target_spectrum)?, 0.03)?
                .to_scalar::<f32>()?,
        )
    } else {
        None
    };

    let sigma = world.criticality.update(movement);
    let movement_signal = world.movement_monitor.analyze(movement)?;
    let post =
        world
            .spectral_monitor
            .analyze(&left, &right, movement, synergy, field_entropy, sigma);
    let mimic_normalized = target_feedback
        .map(|value| value / (1.0 + value))
        .unwrap_or(0.0);
    if !intervention.target_feedback_disabled {
        world.uncertainty.update(
            &post.json,
            &movement_signal,
            mimic_normalized,
            synergy,
            0.0,
            &world.controller.meta,
        );
    }
    let region_change = output.region_change.mean_all()?.to_scalar::<f32>()?;
    let observation_delta = world
        .last_observation
        .as_ref()
        .map(|previous| post.observation.distance(previous))
        .unwrap_or(0.08);
    let structured_complexity = post.observation.structured_complexity();
    world.adaptive.observe(
        movement,
        region_change,
        observation_delta,
        structured_complexity,
        sigma,
        world.controller.meta.confidence,
    );
    let recurrence = world.motifs.recurrence(&post.observation, absolute_step);
    let reward = post.observation.reward_against(
        world.last_observation.as_ref(),
        recurrence,
        &world.adaptive,
        world.controller.meta.confidence,
        world.controller.action_age,
    );
    if !intervention.bandit_zero {
        world.controller.bandit.update(action, reward);
    }
    world.adaptive.observe_reward(reward);
    if absolute_step.is_multiple_of(super::super::MOTIF_EVERY as u64) {
        world.motifs.maybe_store(
            &post.observation,
            control,
            absolute_step,
            super::super::MOTIF_DEFAULT_CAPACITY.max(world.motifs.entries.len()),
            &world.adaptive,
            &mut world.motif_diagnostics,
        );
    }
    world.last_observation = Some(post.observation);
    world.pending_predictor_input = Some(predictor_input);

    let micro_abs = output.next_micro.abs()?.mean_all()?.to_scalar::<f32>()?;
    let macro_abs = output.next_macro.abs()?.mean_all()?.to_scalar::<f32>()?;
    let rms_mid = audio_metrics(&left, &right).rms_mid;
    let metabolic_cost = 0.0020
        + rms_mid.clamp(0.0, 1.0) * 0.0060
        + movement.clamp(0.0, 0.20) * 0.045
        + (control.kick_mult - 1.0).max(0.0) * 0.0015;
    world.energy = (world.energy - metabolic_cost).max(0.18);
    let recharge_capacity = (1.0 - world.energy).max(0.0);
    world.energy += recharge_capacity * (1.0 - rms_mid).clamp(0.0, 1.0) * 0.012;
    world.energy += super::super::ENERGY_HOMEO_RATE * (super::super::POT_ENERGY_SET - world.energy);
    world.energy = world.energy.clamp(0.18, 0.96);
    let pot = world.potential.update(super::super::PotentialState {
        micro_amp: micro_abs,
        macro_amp: macro_abs,
        coupling: synergy,
        movement,
        energy: world.energy,
        sigma,
        curiosity,
        stagnation: world.adaptive.stagnation,
    });
    world.host.last_temp = pot.temp;
    let rad_target =
        (0.18 + 0.18 * escape_strength + 0.08 * world.controller.meta.surprise() + 0.05 * pot.temp)
            .clamp(super::super::RAD_AMP_MIN, super::super::RAD_AMP_MAX);
    world.rad_amp += 0.012 * (rad_target - world.rad_amp);

    world.micro = if intervention.micro_nca_hold {
        old_micro.detach()
    } else {
        output.next_micro.detach()
    };
    world.macro_t = if intervention.macro_nca_hold {
        old_macro.detach()
    } else {
        output.next_macro.detach()
    };
    world.hidden = if intervention.gru_hold {
        old_hidden.detach()
    } else {
        output.next_hidden.detach()
    };
    let radiation_window_probability = (super::super::RADIATE_PROB
        + curiosity * 0.04
        + (control.kick_mult - 1.0).max(0.0) * 0.03
        + world.controller.meta.surprise() * 0.02
        + escape_strength * 0.12)
        .clamp(0.0, 0.30);
    let radiation_probability =
        super::super::reference_window_probability_to_chunk(radiation_window_probability);
    let mut radiation_rng = forcing_rng(world.forcing_seed, absolute_step, 0x5241_4449);
    let radiation_draw = radiation_rng.gen::<f32>();
    let radiation_realized =
        !intervention.radiation_disabled && radiation_draw < radiation_probability;
    if radiation_realized {
        world.micro = super::super::levy_radiate(
            &world.micro,
            world.rad_amp
                * (1.0
                    + curiosity * 0.15
                    + world.controller.meta.surprise() * 0.10
                    + escape_strength * 0.20),
            &mut radiation_rng,
        )?
        .detach();
    }
    let controlled_shear = if intervention.shear_disabled {
        0.0
    } else {
        (pot.shear_amp * control.shear_mult).clamp(0.0, 0.75)
    };
    if absolute_step.is_multiple_of(super::super::MACRO_UPDATE_EVERY) {
        world.shear_phase += super::super::SHEAR_PHASE_VEL
            * super::super::MACRO_UPDATE_EVERY as f32
            * (0.82 + 0.28 * control.shear_mult);
        let shear = world
            .shear
            .generate(controlled_shear, world.shear_phase, &device)?;
        let gain = if intervention.potential_gains_identity {
            1.0
        } else {
            pot.macro_gain
        };
        world.macro_t = world
            .macro_t
            .add(&shear)?
            .tanh()?
            .affine(gain as f64, 0.0)?;
    }
    let controlled_kick = if intervention.micro_kick_disabled {
        0.0
    } else {
        (pot.micro_kick * control.kick_mult).clamp(0.0, 0.10)
    };
    if controlled_kick > 1e-3 {
        let mut kick_rng = forcing_rng(world.forcing_seed, absolute_step, 0x4B49_434B);
        let kick = super::super::randn_t(
            &mut kick_rng,
            &[
                1,
                super::super::CA_CHANNELS,
                super::super::GRID_H,
                super::super::GRID_W,
            ],
            controlled_kick,
            &device,
        )?;
        world.micro = world.micro.add(&kick)?;
    }
    let micro_gain = if intervention.potential_gains_identity {
        1.0
    } else {
        pot.micro_gain
    };
    world.micro = world
        .micro
        .affine(micro_gain as f64, 0.0)?
        .clamp(-1.0f32, 1.0f32)?;
    if !intervention.episodic_empty
        && absolute_step.is_multiple_of(super::super::EPI_SNAP_EVERY as u64)
        && absolute_step > 0
    {
        world.bundle.episodic.snapshot(&output.refined_hidden);
    }

    let micro_values = super::super::flatten_tensor(&world.micro)?;
    let macro_values = super::super::flatten_tensor(&world.macro_t)?;
    let hidden_values = super::super::flatten_tensor(&world.hidden)?;
    let signature = |values: &[f32]| {
        let count = values.len().max(1) as f32;
        let rms = (values.iter().map(|value| value * value).sum::<f32>() / count).sqrt();
        let near = values.iter().filter(|value| value.abs() >= 0.99).count() as f32 / count;
        (rms, near)
    };
    let (micro_rms, micro_near) = signature(&micro_values);
    let (macro_rms, macro_near) = signature(&macro_values);
    let (recurrent_rms, _) = signature(&hidden_values);
    let target_info = world
        .target_loader
        .as_ref()
        .map(super::super::TargetAudioLoader::episode_info);
    let post_metrics = audio_metrics(&left, &right);
    let mut warnings = Vec::new();
    if target_feedback.is_some() {
        warnings.push(
            "target feedback uses frozen coarse spectral error only; no objective, backward, or optimizer step is evaluated"
                .to_string(),
        );
    } else if !intervention.target_feedback_disabled {
        warnings.push(
            "no read-only corpus target was available; rollout is target-independent".to_string(),
        );
    }
    if post_metrics.nonfinite > 0 || micro_values.iter().any(|value| !value.is_finite()) {
        warnings.push(
            "nonfinite state or audio detected; no bio-reset is performed in analysis".to_string(),
        );
    }
    let record = StepRecord {
        rollout_offset,
        absolute_step,
        action: action.label().to_string(),
        model_proposal: world.controller.model_proposal().label().to_string(),
        bandit_proposal: world.controller.bandit_proposal().label().to_string(),
        force_macro,
        target_file: target_info.as_ref().map(|(name, _, _)| name.clone()),
        target_frame: target_info.as_ref().map(|(_, frame, _)| *frame),
        target_error_feedback: target_feedback,
        movement,
        micro_rms,
        macro_rms,
        recurrent_rms,
        micro_near_bound_fraction: micro_near,
        macro_near_bound_fraction: macro_near,
        synergy,
        sigma,
        energy: world.energy,
        temperature: pot.temp,
        radiation_probability,
        radiation_realized,
        shear_amplitude: controlled_shear,
        kick_amplitude: controlled_kick,
        morph_active_depth: world.bundle.model.depth(),
        manifold_depth: world.bundle.model.manifold_depth(),
        far_ring_gain: world.bundle.model.far_ring_gain(),
        episodic_slots: world.bundle.episodic.slots.len(),
        motif_occupancy: world.motifs.entries.len(),
        model_confidence_raw: world.controller.meta.confidence,
        model_confidence_effective: world.adaptive.effective_model_weight,
        audio_raw: raw_metrics,
        audio_post_dsp: post_metrics,
        warnings,
    };
    world.absolute_step = world.absolute_step.saturating_add(1);
    Ok(StepOutput {
        record,
        left,
        right,
        raw_left,
        raw_right,
    })
}

fn forcing_rng(seed: u64, absolute_step: u64, stream_tag: u64) -> super::super::RuntimeRng {
    let mut mixed = seed
        ^ absolute_step.wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ stream_tag.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    mixed ^= mixed >> 31;
    super::super::RuntimeRng::seed_from_u64(mixed)
}

fn first_f32(tensor: &Tensor) -> Result<f32> {
    tensor
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("analysis expected a non-empty scalar-like tensor"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ablations_are_typed_and_unknown_hooks_are_rejected() -> Result<()> {
        assert_eq!(
            Intervention::parse("gru_hold")?.subsystem_class(),
            "learned_neural_state_transition"
        );
        assert_eq!(
            Intervention::parse("radiation_disabled")?.subsystem_class(),
            "host_stochastic_forcing"
        );
        assert!(Intervention::parse("invented_neural_module").is_err());
        Ok(())
    }

    #[test]
    fn arbiter_is_a_training_only_null_control() -> Result<()> {
        assert_eq!(
            Intervention::parse("arbiter_bypass")?.subsystem_class(),
            "training_only_null_control"
        );
        Ok(())
    }
}
