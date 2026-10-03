use super::load::LoadedOrigin;
use super::metrics::{self, AudioMetrics, RecurrenceCandidate, StateDistance};
use super::state::AnalysisWorld;
use super::step::{self, Intervention, StepRecord};
use anyhow::Result;
use serde::Serialize;
use std::collections::BTreeSet;
use std::time::Instant;

#[derive(Clone, Debug)]
pub(crate) struct StateFrame {
    pub offset: usize,
    pub absolute_step: u64,
    pub micro: Vec<f32>,
    pub macro_t: Vec<f32>,
    pub coarse: Option<Vec<f32>>,
    pub hidden: Vec<f32>,
    pub left: Vec<f32>,
    pub right: Vec<f32>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct HorizonSummary {
    pub horizon_chunks: usize,
    pub absolute_step: u64,
    pub state_distance_from_origin: StateDistance,
    pub coarse_distance_from_origin: Option<f32>,
    pub audio: AudioMetrics,
    pub recurrence: RecurrenceCandidate,
    pub bounded: bool,
    pub nonfinite_detected: bool,
    pub clipping_detected: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RolloutSummary {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evaluation_protocol: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub probe_manifest: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weights_hash_before: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weights_hash_after: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub regime_capture_path: Option<String>,
    pub condition: String,
    pub subsystem_class: String,
    pub horizon_chunks: usize,
    pub stride_chunks: usize,
    pub checkpoint_start_step: u64,
    pub warmup_chunks: usize,
    pub initial_world_fingerprint: String,
    pub final_world_fingerprint: String,
    pub fixed_weights: bool,
    pub optimizer_steps: usize,
    pub fixed_morphology: bool,
    pub target_feedback_semantics: String,
    pub forcing_protocol: String,
    pub elapsed_seconds: f64,
    pub sampled_steps: Vec<StepRecord>,
    pub horizons: Vec<HorizonSummary>,
    pub warnings: Vec<String>,
}

pub(crate) struct RolloutRun {
    pub summary: RolloutSummary,
    pub frames: Vec<StateFrame>,
    pub all_left: Vec<f32>,
    pub all_right: Vec<f32>,
    pub all_raw_left: Vec<f32>,
    pub all_raw_right: Vec<f32>,
    pub all_feedback: Vec<Option<f32>>,
    pub all_forcing: Vec<step::ForcingFrame>,
    pub all_control_energy: Vec<step::ControlEnergyFrame>,
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn run(
    origin: &LoadedOrigin,
    condition: &str,
    intervention: &Intervention,
    horizons: &[usize],
    stride: usize,
    initialized_weights: bool,
    initialization_seed: u64,
    fresh_world: bool,
    initial_transform: Option<&dyn Fn(&mut AnalysisWorld) -> Result<()>>,
    progress: impl Fn(usize, usize, &StepRecord),
) -> Result<RolloutRun> {
    let maximum_horizon = horizons.iter().copied().max().unwrap_or(0);
    if maximum_horizon == 0 {
        anyhow::bail!("frozen rollout requires at least one positive horizon");
    }
    let requested: BTreeSet<usize> = horizons.iter().copied().collect();
    let mut world = AnalysisWorld::from_saved(
        origin,
        initialized_weights,
        initialization_seed,
        fresh_world,
    )?;
    if let Some(schedule) = &world.target_schedule {
        schedule.validate_interval(
            origin.target_origin.unwrap_or(world.absolute_step),
            origin.warmup_chunks + maximum_horizon,
        )?;
    }
    for offset in 1..=origin.warmup_chunks {
        step::step(&mut world, offset, &Intervention::full())?;
    }
    if let Some(transform) = initial_transform {
        transform(&mut world)?;
    }
    let weights_hash_before = if origin.evaluation_metrics {
        Some(super::load::model_inventory_hash(&world.bundle.varmap)?)
    } else {
        None
    };
    if origin.evaluation_metrics {
        let directory = origin
            .capture_root
            .as_ref()
            .unwrap()
            .join("trajectory")
            .join(condition);
        std::fs::create_dir_all(&directory)?;
        world.capture = Some(super::super::regime_capture::RegimeCapture::new(
            &directory.display().to_string(),
            condition,
            stride,
        )?);
    }
    let checkpoint_start_step = world.absolute_step;
    let initial_world_fingerprint = world.fingerprint()?;
    let initial = world.state_vectors()?;
    let initial_coarse = world.coarse_values()?;
    let mut frames = vec![StateFrame {
        offset: 0,
        absolute_step: world.absolute_step,
        micro: initial.0.clone(),
        macro_t: initial.1.clone(),
        coarse: initial_coarse.clone(),
        hidden: initial.2.clone(),
        left: Vec::new(),
        right: Vec::new(),
    }];
    let mut sampled_steps = Vec::new();
    let mut all_left = Vec::with_capacity(maximum_horizon * super::super::CHUNK_SIZE);
    let mut all_right = Vec::with_capacity(maximum_horizon * super::super::CHUNK_SIZE);
    let mut all_raw_left = Vec::with_capacity(maximum_horizon * super::super::CHUNK_SIZE);
    let mut all_raw_right = Vec::with_capacity(maximum_horizon * super::super::CHUNK_SIZE);
    let mut all_feedback = Vec::with_capacity(maximum_horizon);
    let mut all_forcing = Vec::with_capacity(maximum_horizon);
    let mut all_control_energy = Vec::with_capacity(maximum_horizon);
    let mut recurrence_history = vec![(
        checkpoint_start_step,
        metrics::compact_signature(&initial.0, &initial.1, &initial.2),
    )];
    let mut horizon_summaries = Vec::new();
    let mut warnings = Vec::new();
    let started = Instant::now();
    for offset in 1..=maximum_horizon {
        let output = step::step(&mut world, offset, intervention)?;
        all_left.extend_from_slice(&output.left);
        all_right.extend_from_slice(&output.right);
        all_raw_left.extend_from_slice(&output.raw_left);
        all_raw_right.extend_from_slice(&output.raw_right);
        all_feedback.push(output.target_feedback);
        all_forcing.push(output.forcing);
        all_control_energy.push(output.control_energy);
        let observe = offset.is_multiple_of(stride) || requested.contains(&offset) || offset == 1;
        if observe {
            progress(offset, maximum_horizon, &output.record);
            warnings.extend(output.record.warnings.iter().cloned());
            sampled_steps.push(output.record.clone());
            let state = world.state_vectors()?;
            frames.push(StateFrame {
                offset,
                absolute_step: world.absolute_step,
                micro: state.0.clone(),
                macro_t: state.1.clone(),
                coarse: world.coarse_values()?,
                hidden: state.2.clone(),
                left: output.left.clone(),
                right: output.right.clone(),
            });
            let signature = metrics::compact_signature(&state.0, &state.1, &state.2);
            let recurrence = metrics::recurrence_candidate(
                world.absolute_step,
                &signature,
                &recurrence_history,
                stride.saturating_mul(4).max(16),
            );
            recurrence_history.push((world.absolute_step, signature));
            if requested.contains(&offset) {
                let audio_start = offset.saturating_sub(stride) * super::super::CHUNK_SIZE;
                let audio = metrics::audio_metrics(
                    &all_left[audio_start.min(all_left.len())..],
                    &all_right[audio_start.min(all_right.len())..],
                );
                let distance = metrics::state_distance(
                    &state.0, &state.1, &state.2, &initial.0, &initial.1, &initial.2,
                );
                let bounded = output.record.micro_near_bound_fraction < 0.50
                    && output.record.macro_near_bound_fraction < 0.50
                    && audio.nonfinite == 0;
                horizon_summaries.push(HorizonSummary {
                    horizon_chunks: offset,
                    absolute_step: world.absolute_step,
                    state_distance_from_origin: distance,
                    coarse_distance_from_origin: world
                        .coarse_values()?
                        .as_ref()
                        .zip(initial_coarse.as_ref())
                        .and_then(|(current, prior)| metrics::relative_l2(current, prior)),
                    audio: audio.clone(),
                    recurrence,
                    bounded,
                    nonfinite_detected: audio.nonfinite > 0,
                    clipping_detected: audio.clipping_fraction > 0.0,
                });
            }
        }
    }
    warnings.sort();
    warnings.dedup();
    let target_feedback_semantics = if intervention.target_feedback_disabled {
        "target_independent_feedback_ablation"
    } else if intervention.feedback_zero {
        "target_error_scalar_zero_with_observation_updates_preserved"
    } else if intervention.feedback_replay.is_some() {
        "artificial_baseline_feedback_scalar_replay"
    } else if world.target_feedback_available {
        "coarse_target_error_feedback_without_loss_or_optimizer"
    } else {
        "target_independent_no_read_only_target_available"
    };
    if let Some(capture) = world.capture.as_mut() {
        capture.flush()?;
    }
    let weights_hash_after = if origin.evaluation_metrics {
        Some(super::load::model_inventory_hash(&world.bundle.varmap)?)
    } else {
        None
    };
    anyhow::ensure!(
        weights_hash_before == weights_hash_after,
        "frozen model parameters changed during rollout"
    );
    Ok(RolloutRun {
        summary: RolloutSummary {
            evaluation_protocol: if origin.common_rng || origin.target_origin.is_some() || origin.active_depth.is_some() {
                Some(serde_json::json!({"common_rng":origin.common_rng,
                    "event_clock":if origin.common_rng { "relative_evaluation_clock" } else { "native_absolute_step" },
                    "target_origin":origin.target_origin, "active_depth_override":origin.active_depth,
                    "native_world_chronology_preserved":true}))
            } else { None },
            probe_manifest: world.evaluation.as_ref().map(|evaluation| evaluation.manifest.clone()),
            weights_hash_before, weights_hash_after,
            regime_capture_path: world.capture.as_ref().map(|capture| capture.path().to_string()),
            condition: condition.to_string(),
            subsystem_class: intervention.subsystem_class().to_string(),
            horizon_chunks: maximum_horizon,
            stride_chunks: stride,
            checkpoint_start_step,
            warmup_chunks: origin.warmup_chunks,
            initial_world_fingerprint,
            final_world_fingerprint: world.fingerprint()?,
            fixed_weights: true,
            optimizer_steps: 0,
            fixed_morphology: true,
            target_feedback_semantics: target_feedback_semantics.to_string(),
            forcing_protocol: if intervention.forcing_replay.is_some() {
                "baseline_realized_host_forcing_replayed_by_offset_with_fixed_target_schedule"
            } else {
                "common_exogenous_forcing_by_absolute_step_with_separate_controller_and_target_rng_streams"
            }.to_string(),
            elapsed_seconds: started.elapsed().as_secs_f64(),
            sampled_steps,
            horizons: horizon_summaries,
            warnings,
        },
        frames,
        all_left,
        all_right,
        all_raw_left,
        all_raw_right,
        all_feedback,
        all_forcing,
        all_control_energy,
    })
}

pub(crate) fn comparison(baseline: &RolloutRun, intervention: &RolloutRun) -> serde_json::Value {
    let mut points = Vec::new();
    for intervention_frame in &intervention.frames {
        let Some(baseline_frame) = baseline
            .frames
            .iter()
            .find(|frame| frame.offset == intervention_frame.offset)
        else {
            continue;
        };
        points.push(serde_json::json!({
            "offset": intervention_frame.offset,
            "absolute_step": intervention_frame.absolute_step,
            "state_distance": metrics::state_distance(
                &intervention_frame.micro,
                &intervention_frame.macro_t,
                &intervention_frame.hidden,
                &baseline_frame.micro,
                &baseline_frame.macro_t,
                &baseline_frame.hidden,
            ),
            "coarse_state_distance": intervention_frame.coarse.as_ref()
                .zip(baseline_frame.coarse.as_ref())
                .and_then(|(left, right)| metrics::relative_l2(left, right)),
            "audio_distance": metrics::audio_distance(
                &intervention_frame.left,
                &intervention_frame.right,
                &baseline_frame.left,
                &baseline_frame.right,
            ),
        }));
    }
    serde_json::json!({
        "baseline": baseline.summary.condition,
        "condition": intervention.summary.condition,
        "subsystem_class": intervention.summary.subsystem_class,
        "initial_state_equal": baseline.frames.first().zip(intervention.frames.first()).is_some_and(|(left, right)| {
            left.micro == right.micro && left.macro_t == right.macro_t && left.coarse == right.coarse && left.hidden == right.hidden
        }),
        "points": points,
        "interpretation_constraint": "closed-loop compensation may mask or amplify a direct subsystem effect"
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparison_distinguishes_latent_and_audio_domains() {
        let state = StateFrame {
            offset: 0,
            absolute_step: 0,
            micro: vec![0.0],
            macro_t: vec![0.0],
            coarse: None,
            hidden: vec![0.0],
            left: vec![0.0],
            right: vec![0.0],
        };
        assert_eq!(state.micro, vec![0.0]);
    }
}
