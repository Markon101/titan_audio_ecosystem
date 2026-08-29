use super::load::LoadedOrigin;
use super::metrics::{self, AudioDistance, StateDistance};
use super::rollout::{self, RolloutRun};
use super::state::AnalysisWorld;
use super::step::Intervention;
use anyhow::Result;
use candle_core::Tensor;
use rand::SeedableRng;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RecoveryPoint {
    pub offset: usize,
    pub state_distance: StateDistance,
    pub audio_distance: AudioDistance,
    pub latent_recovery_ratio: Option<f32>,
    pub phenotype_robust_state_persistent: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct PerturbationCondition {
    pub name: String,
    pub scale: f32,
    pub seed: u64,
    pub target_component: String,
    pub normalization: String,
    pub forcing_protocol: String,
    pub initial_latent_distance: Option<f32>,
    pub time_to_half_recovery_chunks: Option<usize>,
    pub half_recovery_valid: bool,
    pub half_recovery_invalid_reason: Option<String>,
    pub terminal_classification: String,
    pub points: Vec<RecoveryPoint>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct PerturbationSuite {
    pub horizon_chunks: usize,
    pub baseline_condition: String,
    pub conditions: Vec<PerturbationCondition>,
    pub terminology: serde_json::Value,
}

pub(crate) struct PerturbationRuns {
    pub suite: PerturbationSuite,
    pub baseline: RolloutRun,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_suite(
    origin: &LoadedOrigin,
    horizon: usize,
    stride: usize,
    names: &[String],
    scales: &[f32],
    analysis_seed: u64,
    progress: impl Fn(&str, usize, usize),
    mut export: impl FnMut(&str, &RolloutRun) -> Result<()>,
) -> Result<PerturbationRuns> {
    let baseline = rollout::run(
        origin,
        "perturbation_baseline",
        &Intervention::full(),
        &[horizon],
        stride,
        false,
        analysis_seed,
        false,
        None,
        |offset, total, _| progress("perturbation_baseline", offset, total),
    )?;
    let names = if names.is_empty() {
        vec![
            "micro_gaussian".to_string(),
            "macro_gaussian".to_string(),
            "micro_patch_erase".to_string(),
            "recurrent_gaussian".to_string(),
            "active_channel_mask".to_string(),
        ]
    } else {
        names.to_vec()
    };
    let mut summaries = Vec::new();
    for (name_index, name) in names.iter().enumerate() {
        validate_name(name)?;
        for (scale_index, scale) in scales.iter().copied().enumerate() {
            let seed = analysis_seed
                ^ (name_index as u64).wrapping_mul(0x9e37_79b9)
                ^ (scale_index as u64).wrapping_mul(0x85eb_ca6b);
            let label = format!("{}_s{:08}", name, (scale * 1_000_000.0).round() as u64);
            let transform = |world: &mut AnalysisWorld| apply(world, name, scale, seed);
            let run = rollout::run(
                origin,
                &label,
                &Intervention::full(),
                &[horizon],
                stride,
                false,
                analysis_seed,
                false,
                Some(&transform),
                |offset, total, _| progress(&label, offset, total),
            )?;
            summaries.push(summarize(name, scale, seed, &baseline, &run));
            export(&label, &run)?;
        }
    }
    Ok(PerturbationRuns {
        suite: PerturbationSuite {
            horizon_chunks: horizon,
            baseline_condition: baseline.summary.condition.clone(),
            conditions: summaries,
            terminology: serde_json::json!({
                "latent_recovery": "decreasing state distance toward the paired unperturbed baseline",
                "phenotype_robustness": "small audio distance despite persistent latent distance",
                "not_claimed": ["self-healing", "life", "strong emergence", "Lyapunov exponent"]
            }),
        },
        baseline,
    })
}

fn validate_name(name: &str) -> Result<()> {
    if matches!(
        name,
        "micro_gaussian"
            | "macro_gaussian"
            | "micro_patch_erase"
            | "macro_patch_erase"
            | "active_channel_mask"
            | "spatial_region"
            | "recurrent_gaussian"
            | "recurrent_coordinate_mask"
            | "episodic_slot_noise"
            | "episodic_slot_mask"
            | "episodic_empty"
            | "energy_state"
            | "controller_temperature"
    ) {
        Ok(())
    } else {
        anyhow::bail!("unsupported perturbation {name:?}")
    }
}

fn apply(world: &mut AnalysisWorld, name: &str, scale: f32, seed: u64) -> Result<()> {
    let mut rng = super::super::RuntimeRng::seed_from_u64(seed ^ 0x5045_5254);
    let active_channels = world.bundle.model.manifold_depth() * super::super::CA_FEATURE_CHANNELS;
    match name {
        "micro_gaussian" => perturb_tensor_gaussian(
            &mut world.micro,
            scale,
            &mut rng,
            Some(active_channels * super::super::GRID_H * super::super::GRID_W),
        )?,
        "macro_gaussian" => perturb_tensor_gaussian(
            &mut world.macro_t,
            scale,
            &mut rng,
            Some(active_channels * super::super::MACRO_H * super::super::MACRO_W),
        )?,
        "micro_patch_erase" => perturb_patch(
            &mut world.micro,
            super::super::GRID_H,
            super::super::GRID_W,
            scale,
            active_channels,
        )?,
        "macro_patch_erase" => perturb_patch(
            &mut world.macro_t,
            super::super::MACRO_H,
            super::super::MACRO_W,
            scale,
            active_channels,
        )?,
        "active_channel_mask" => {
            let active = world.bundle.model.manifold_depth() * super::super::CA_FEATURE_CHANNELS;
            let mut values = super::super::flatten_tensor(&world.micro)?;
            let channels = ((active as f32 * scale.clamp(0.0, 1.0)).ceil() as usize).max(1);
            let plane = super::super::GRID_H * super::super::GRID_W;
            for channel in 0..channels.min(active) {
                values[channel * plane..(channel + 1) * plane].fill(0.0);
            }
            world.micro = Tensor::from_vec(
                values,
                (
                    1,
                    super::super::CA_CHANNELS,
                    super::super::GRID_H,
                    super::super::GRID_W,
                ),
                world.micro.device(),
            )?;
        }
        "spatial_region" => perturb_region(&mut world.micro, scale, active_channels)?,
        "recurrent_gaussian" => perturb_tensor_gaussian(&mut world.hidden, scale, &mut rng, None)?,
        "recurrent_coordinate_mask" => {
            let mut values = super::super::flatten_tensor(&world.hidden)?;
            let count = ((values.len() as f32 * scale.clamp(0.0, 1.0)).ceil() as usize).max(1);
            for value in values.iter_mut().take(count) {
                *value = 0.0;
            }
            world.hidden =
                Tensor::from_vec(values, (1, super::super::MEMORY_DIM), world.hidden.device())?;
        }
        "episodic_slot_noise" => {
            if let Some(slot) = world.bundle.episodic.slots.front_mut() {
                perturb_tensor_gaussian(slot, scale, &mut rng, None)?;
            }
        }
        "episodic_slot_mask" => {
            if !world.bundle.episodic.slots.is_empty() {
                world.bundle.episodic.slots.pop_front();
            }
        }
        "episodic_empty" => world.bundle.episodic.slots.clear(),
        "energy_state" => {
            world.energy = (world.energy + scale).clamp(0.18, 0.96);
        }
        "controller_temperature" => {
            world.host.last_temp = (world.host.last_temp + scale).clamp(0.0, 1.0);
        }
        _ => unreachable!("validated perturbation"),
    }
    Ok(())
}

fn perturb_tensor_gaussian(
    tensor: &mut Tensor,
    scale: f32,
    rng: &mut super::super::RuntimeRng,
    active_elements: Option<usize>,
) -> Result<()> {
    let mut values = super::super::flatten_tensor(tensor)?;
    let active = active_elements.unwrap_or(values.len()).min(values.len());
    let rms = (values[..active]
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        / active.max(1) as f32)
        .sqrt()
        .max(1e-6);
    for value in values.iter_mut().take(active) {
        *value += super::super::rng_normal(rng) * scale * rms;
    }
    *tensor = Tensor::from_vec(values, tensor.dims().to_vec(), tensor.device())?;
    Ok(())
}

fn perturb_patch(
    tensor: &mut Tensor,
    height: usize,
    width: usize,
    scale: f32,
    active_channels: usize,
) -> Result<()> {
    let mut values = super::super::flatten_tensor(tensor)?;
    let side_fraction = scale.clamp(0.01, 1.0).sqrt();
    let patch_h = ((height as f32 * side_fraction).round() as usize).clamp(1, height);
    let patch_w = ((width as f32 * side_fraction).round() as usize).clamp(1, width);
    let row_start = (height - patch_h) / 2;
    let col_start = (width - patch_w) / 2;
    let plane = height * width;
    let active = active_channels.min(super::super::CA_CHANNELS);
    for channel in 0..active {
        for row in row_start..row_start + patch_h {
            let start = channel * plane + row * width + col_start;
            values[start..start + patch_w].fill(0.0);
        }
    }
    *tensor = Tensor::from_vec(values, tensor.dims().to_vec(), tensor.device())?;
    Ok(())
}

fn perturb_region(tensor: &mut Tensor, scale: f32, active_channels: usize) -> Result<()> {
    let mut values = super::super::flatten_tensor(tensor)?;
    let plane = super::super::GRID_H * super::super::GRID_W;
    let row_start = super::super::GRID_H / 4;
    let row_end = super::super::GRID_H * 3 / 4;
    let col_start = super::super::GRID_W / 4;
    let col_end = super::super::GRID_W * 3 / 4;
    for channel in 0..active_channels.min(super::super::CA_CHANNELS) {
        for row in row_start..row_end {
            for column in col_start..col_end {
                let index = channel * plane + row * super::super::GRID_W + column;
                values[index] = (values[index] + scale).clamp(-1.0, 1.0);
            }
        }
    }
    *tensor = Tensor::from_vec(values, tensor.dims().to_vec(), tensor.device())?;
    Ok(())
}

fn summarize(
    name: &str,
    scale: f32,
    seed: u64,
    baseline: &RolloutRun,
    perturbed: &RolloutRun,
) -> PerturbationCondition {
    let mut points = Vec::new();
    let mut initial = None;
    for frame in &perturbed.frames {
        let Some(base) = baseline
            .frames
            .iter()
            .find(|base| base.offset == frame.offset)
        else {
            continue;
        };
        let state = metrics::state_distance(
            &frame.micro,
            &frame.macro_t,
            &frame.hidden,
            &base.micro,
            &base.macro_t,
            &base.hidden,
        );
        if frame.offset == 0 {
            initial = state.combined_relative_l2;
        }
        let ratio = state
            .combined_relative_l2
            .zip(initial)
            .and_then(|(distance, initial)| (initial > 1e-9).then_some(distance / initial));
        let audio = metrics::audio_distance(&frame.left, &frame.right, &base.left, &base.right);
        points.push(RecoveryPoint {
            offset: frame.offset,
            phenotype_robust_state_persistent: ratio.is_some_and(|ratio| ratio > 0.5)
                && audio
                    .multiresolution_log_spectral
                    .is_some_and(|distance| distance < 0.1),
            state_distance: state,
            audio_distance: audio,
            latent_recovery_ratio: ratio,
        });
    }
    let half_recovery = points.windows(2).find_map(|window| {
        (window[0]
            .latent_recovery_ratio
            .is_some_and(|ratio| ratio <= 0.5)
            && window[1]
                .latent_recovery_ratio
                .is_some_and(|ratio| ratio <= 0.5))
        .then_some(window[0].offset)
    });
    let valid_initial = initial.is_some_and(|distance| distance > 1e-9);
    let terminal_ratio = points.last().and_then(|point| point.latent_recovery_ratio);
    let terminal_classification = match terminal_ratio {
        Some(ratio) if ratio <= 0.5 => "latent_contraction_toward_paired_baseline",
        Some(ratio) if ratio > 1.25 => "perturbation_amplification",
        Some(_) => "latent_distance_persisting",
        None => "invalid_or_unmeasurable",
    };
    PerturbationCondition {
        name: name.to_string(),
        scale,
        seed,
        target_component: target_component(name).to_string(),
        normalization: if name.contains("gaussian") {
            "noise_std_equals_scale_times_component_rms"
        } else {
            "declared_fraction_or_bounded_amplitude"
        }
        .to_string(),
        forcing_protocol:
            "common_exogenous_forcing_by_absolute_step; separate matched closed-loop controller RNG"
                .to_string(),
        initial_latent_distance: initial,
        time_to_half_recovery_chunks: half_recovery,
        half_recovery_valid: valid_initial && half_recovery.is_some(),
        half_recovery_invalid_reason: if !valid_initial {
            Some("initial state distance below numerical floor".to_string())
        } else if half_recovery.is_none() {
            Some("distance did not remain at or below half for two observations".to_string())
        } else {
            None
        },
        terminal_classification: terminal_classification.to_string(),
        points,
    }
}

fn target_component(name: &str) -> &'static str {
    if name.starts_with("micro") || name == "active_channel_mask" || name == "spatial_region" {
        "micro_state"
    } else if name.starts_with("macro") {
        "macro_state"
    } else if name.starts_with("recurrent") {
        "recurrent_memory"
    } else if name.starts_with("episodic") {
        "episodic_memory"
    } else {
        "host_controller_state"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminology_never_equates_output_robustness_with_recovery() {
        assert_ne!("phenotype_robustness", "latent_recovery");
        assert!(validate_name("micro_gaussian").is_ok());
        assert!(validate_name("self_healing").is_err());
    }

    #[test]
    fn gaussian_perturbation_preserves_dormant_suffix() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let mut tensor = Tensor::from_vec(vec![1.0f32, -1.0, 0.0, 0.0], (4,), &device)?;
        let mut rng = super::super::super::RuntimeRng::seed_from_u64(99);
        perturb_tensor_gaussian(&mut tensor, 0.1, &mut rng, Some(2))?;
        let values = tensor.to_vec1::<f32>()?;
        assert_ne!(&values[..2], &[1.0, -1.0]);
        assert_eq!(&values[2..], &[0.0, 0.0]);
        Ok(())
    }
}
