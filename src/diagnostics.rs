use anyhow::Result;
use candle_core::DType;
use candle_nn::VarMap;
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize)]
pub(crate) struct TensorStatistics {
    pub name: String,
    pub subsystem: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub elements: usize,
    pub bytes: usize,
    pub active: bool,
    pub minimum: Option<f32>,
    pub maximum: Option<f32>,
    pub rms: Option<f32>,
    pub zero_fraction: Option<f32>,
    pub nonfinite: usize,
}

pub(crate) fn model_statistics(
    varmap: &VarMap,
    world: Option<&super::WorldCheckpoint>,
    physical_depth: usize,
    morph_width: usize,
) -> Result<serde_json::Value> {
    let active_depth = world
        .map(|checkpoint| checkpoint.active_depth.min(physical_depth))
        .unwrap_or(1);
    let data = varmap.data().lock().unwrap();
    let mut names: Vec<String> = data.keys().cloned().collect();
    names.sort();
    let mut tensors = Vec::with_capacity(names.len());
    let mut subsystem_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut total_parameters = 0usize;
    let mut active_parameters = 0usize;
    let mut allocated_bytes = 0usize;
    let mut nonfinite_parameters = 0usize;
    for name in names {
        let tensor = data[&name].as_tensor();
        let active = morph_block_index(&name).is_none_or(|index| index < active_depth);
        let elements = tensor.elem_count();
        let bytes = elements * dtype_bytes(tensor.dtype());
        let subsystem = subsystem_for(&name).to_string();
        let values = tensor
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut minimum = f32::INFINITY;
        let mut maximum = f32::NEG_INFINITY;
        let mut square_sum = 0.0f64;
        let mut zero_count = 0usize;
        let mut finite_count = 0usize;
        let mut nonfinite = 0usize;
        for value in values {
            if value.is_finite() {
                minimum = minimum.min(value);
                maximum = maximum.max(value);
                square_sum += value as f64 * value as f64;
                zero_count += usize::from(value == 0.0);
                finite_count += 1;
            } else {
                nonfinite += 1;
            }
        }
        total_parameters += elements;
        active_parameters += if active { elements } else { 0 };
        allocated_bytes += bytes;
        nonfinite_parameters += nonfinite;
        *subsystem_counts.entry(subsystem.clone()).or_default() += elements;
        tensors.push(TensorStatistics {
            name,
            subsystem,
            shape: tensor.dims().to_vec(),
            dtype: format!("{:?}", tensor.dtype()).to_ascii_lowercase(),
            elements,
            bytes,
            active,
            minimum: (finite_count > 0).then_some(minimum),
            maximum: (finite_count > 0).then_some(maximum),
            rms: (finite_count > 0).then_some((square_sum / finite_count as f64).sqrt() as f32),
            zero_fraction: (finite_count > 0).then_some(zero_count as f32 / finite_count as f32),
            nonfinite,
        });
    }
    let persistent = world.map(persistent_state_statistics);
    let inactive_parameters = total_parameters.saturating_sub(active_parameters);
    let world_bytes = world
        .and_then(|checkpoint| bincode::serialized_size(checkpoint).ok())
        .unwrap_or(0);
    Ok(serde_json::json!({
        "learned_parameters": {
            "total_allocated": total_parameters,
            "active_exact_whole_blocks": active_parameters,
            "inactive_reserved_whole_blocks": inactive_parameters,
            "active_count_scope": "only wholly inactive MorphicStack blocks are excluded; shared folded-manifold tensors remain allocated",
            "nonfinite": nonfinite_parameters,
            "fp32_weight_bytes": allocated_bytes,
            "adam_two_moment_bytes_estimate": allocated_bytes.saturating_mul(2),
            "subsystems": subsystem_counts,
            "tensors": tensors,
        },
        "architecture": {
            "morphic_physical_depth": physical_depth,
            "morphic_active_depth": active_depth,
            "morphic_width": morph_width,
            "manifold_storage_channels": super::CA_CHANNELS,
            "manifold_active_sheets": super::manifold_depth_for_morph_depth(active_depth),
            "manifold_feature_channels_per_sheet": super::CA_FEATURE_CHANNELS,
            "micro_state": [1, super::CA_CHANNELS, super::GRID_H, super::GRID_W],
            "macro_state": [1, super::CA_CHANNELS, super::MACRO_H, super::MACRO_W],
            "recurrent_memory": super::MEMORY_DIM,
            "episodic_readout": super::EPI_DIM,
            "episodic_capacity": super::EPI_SLOTS,
            "novelty_capacity": super::NOVELTY_SLOTS,
            "world_model_observation": super::OBS_DIM,
            "controller_actions": super::ACTION_COUNT,
            "decoder_control_frames": super::DECODER_CONTROL_FRAMES,
            "decoder_controls": super::DECODER_CONTROL_COUNT,
            "regional_partials": super::SCAN_PARTIALS,
        },
        "persistent_non_learned_state": persistent,
        "memory_estimate": {
            "weights_bytes": allocated_bytes,
            "weights_plus_adam_bytes": allocated_bytes.saturating_mul(3),
            "serialized_world_payload_bytes": world_bytes,
            "one_fp32_analysis_clone_minimum_bytes": allocated_bytes.saturating_add(world_bytes as usize),
            "note": "Candle graph temporaries, FFT buffers, audio streams, and allocator overhead are additional and workload-dependent"
        },
        "interpretation_constraint": "parameter count is anatomy, not causal importance"
    }))
}

fn dtype_bytes(dtype: DType) -> usize {
    dtype.size_in_bytes()
}

pub(crate) fn morph_block_index(name: &str) -> Option<usize> {
    [".morphic.l", ".morphic.norm"]
        .into_iter()
        .find_map(|marker| {
            let start = name.find(marker)? + marker.len();
            let digits: String = name[start..]
                .chars()
                .take_while(char::is_ascii_digit)
                .collect();
            (!digits.is_empty()).then(|| digits.parse().ok()).flatten()
        })
}

pub(crate) fn subsystem_for(name: &str) -> &'static str {
    if name.contains("micro_ca") {
        "micro_nca"
    } else if name.contains("macro_ca") {
        "macro_nca"
    } else if name.contains("gru_memory") {
        "gru_recurrent_memory"
    } else if name.contains("morphic") {
        "morphic_stack"
    } else if name.contains("episodic") {
        "episodic_attention"
    } else if name.contains("asymp_contract") {
        "recurrent_to_micro"
    } else if name.contains("temporal_decoder") {
        "spatial_temporal_decoder"
    } else if name.contains("monitor_head") {
        "world_model_planner"
    } else if name.contains("arbiter") {
        "objective_arbiter"
    } else if name.contains("wavefolder") {
        "wavefolders"
    } else if name.contains("partial_") {
        "regional_scan_synthesis"
    } else if name.contains("spatial_panner")
        || name.contains("stereo_width")
        || name.contains("pan")
    {
        "stereo_control"
    } else if name.contains("fm_mod")
        || name.contains("pitch_head")
        || name.contains("oscillator_gain")
        || name.contains("wave_morph")
        || name.contains("base_freq")
    {
        "oscillator_synthesis"
    } else {
        "other_learned"
    }
}

fn persistent_state_statistics(world: &super::WorldCheckpoint) -> serde_json::Value {
    let f32_bytes = std::mem::size_of::<f32>();
    serde_json::json!({
        "micro": {"elements": world.micro_tape.len(), "bytes": world.micro_tape.len() * f32_bytes},
        "macro": {"elements": world.macro_tape.len(), "bytes": world.macro_tape.len() * f32_bytes},
        "recurrent": {"elements": world.hidden_mem.len(), "bytes": world.hidden_mem.len() * f32_bytes},
        "episodic": {"slots": world.episodic_slots.len(), "capacity": super::EPI_SLOTS, "elements": world.episodic_slots.iter().map(Vec::len).sum::<usize>()},
        "novelty": {"slots": world.novelty_spectra.len(), "capacity": super::NOVELTY_SLOTS, "elements": world.novelty_spectra.iter().map(Vec::len).sum::<usize>()},
        "motifs": {"occupancy": world.motifs.entries.len(), "configured_capacity_not_serialized": true},
        "model_runtime": {"haas_history_elements": world.model_runtime.prev_haas_side.len(), "scan_phase_elements": world.model_runtime.scan_phase_l.len() + world.model_runtime.scan_phase_r.len()},
        "monitor_history": {"spectral": world.spectral_history.len(), "spectral_bins": world.spectral_prev_mags.len(), "movement": world.movement_history.len()},
        "host_controller_state_serialized_inside_world": true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn morphic_indices_are_parsed_without_confusing_other_layers() {
        assert_eq!(morph_block_index("model.morphic.l0_1.weight"), Some(0));
        assert_eq!(morph_block_index("model.morphic.l12_2.bias"), Some(12));
        assert_eq!(morph_block_index("model.morphic.norm7.weight"), Some(7));
        assert_eq!(morph_block_index("model.micro_ca.l0.weight"), None);
    }

    #[test]
    fn subsystem_labels_are_operational() {
        assert_eq!(subsystem_for("model.micro_ca.update.weight"), "micro_nca");
        assert_eq!(subsystem_for("arbiter.net.0.weight"), "objective_arbiter");
    }
}
