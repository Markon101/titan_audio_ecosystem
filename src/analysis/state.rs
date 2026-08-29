use super::load::{LoadedOrigin, ModelBundle};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use rand::SeedableRng;
use std::collections::{BTreeMap, HashMap, VecDeque};

pub(crate) struct AnalysisWorld {
    pub bundle: ModelBundle,
    pub micro: Tensor,
    pub macro_t: Tensor,
    pub hidden: Tensor,
    pub phases: [f32; 4],
    pub theta_prev: f32,
    pub theta_prev2: f32,
    pub energy: f32,
    pub rad_amp: f32,
    pub shear_phase: f32,
    pub controller_rng: super::super::RuntimeRng,
    pub target_rng: super::super::RuntimeRng,
    pub forcing_seed: u64,
    pub absolute_step: u64,
    pub uncertainty: super::super::AudioUncertaintyState,
    pub potential: super::super::PotentialController,
    pub criticality: super::super::CriticalityEstimator,
    pub controller: super::super::HybridController,
    pub adaptive: super::super::AdaptiveDynamics,
    pub motifs: super::super::MotifMemory,
    pub motif_diagnostics: super::super::MotifDiagnostics,
    pub last_observation: Option<super::super::AudioObservation>,
    pub pending_predictor_input: Option<Tensor>,
    pub host: super::super::HostRuntimeState,
    pub spectral_monitor: super::super::SpectralEntropyMonitor,
    pub movement_monitor: super::super::MovementCoherenceMonitor,
    pub shear: super::super::ShearField2D,
    pub target_loader: Option<super::super::TargetAudioLoader>,
    pub target_projector: super::super::SpectralProjector,
    pub target_feedback_available: bool,
}

impl AnalysisWorld {
    pub(crate) fn from_saved(
        origin: &LoadedOrigin,
        initialized_weights: bool,
        initialization_seed: u64,
        fresh_world: bool,
    ) -> Result<Self> {
        let checkpoint = origin
            .world
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("analysis world is required"))?;
        let device = Device::Cpu;
        let mut bundle = super::load::load_model_bundle(
            origin,
            initialized_weights,
            initialization_seed,
            !fresh_world,
        )?;
        bundle.model.set_depth(checkpoint.active_depth);
        let target_loader =
            read_only_target_loader(&origin.paths.wav_dir, &origin.paths.corpus_manifest)?;
        let target_feedback_available = target_loader.is_some();
        let forcing_seed = if fresh_world {
            initialization_seed ^ 0xF0C1_6A11
        } else {
            super::super::checkpoint_checksum(&bincode::serialize(&checkpoint.rng)?)
        };
        if fresh_world {
            let mut state_rng = super::super::RuntimeRng::seed_from_u64(initialization_seed);
            let micro = bundle.model.project_manifold(&super::super::randn_t(
                &mut state_rng,
                &[
                    1,
                    super::super::CA_CHANNELS,
                    super::super::GRID_H,
                    super::super::GRID_W,
                ],
                1.0,
                &device,
            )?)?;
            let macro_t = bundle.model.project_manifold(&super::super::randn_t(
                &mut state_rng,
                &[
                    1,
                    super::super::CA_CHANNELS,
                    super::super::MACRO_H,
                    super::super::MACRO_W,
                ],
                1.0,
                &device,
            )?)?;
            return Ok(Self {
                bundle,
                micro,
                macro_t,
                hidden: Tensor::zeros((1, super::super::MEMORY_DIM), DType::F32, &device)?,
                phases: [0.0; 4],
                theta_prev: 0.0,
                theta_prev2: 0.0,
                energy: super::super::POT_ENERGY_SET,
                rad_amp: super::super::RAD_AMP_INIT,
                shear_phase: 0.0,
                controller_rng: super::super::RuntimeRng::seed_from_u64(forcing_seed ^ 0xC017_2011),
                target_rng: super::super::RuntimeRng::seed_from_u64(forcing_seed ^ 0x7A26_E700),
                forcing_seed,
                absolute_step: 0,
                uncertainty: super::super::AudioUncertaintyState::new(),
                potential: super::super::PotentialController::new(),
                criticality: super::super::CriticalityEstimator::default(),
                controller: super::super::HybridController::default(),
                adaptive: super::super::AdaptiveDynamics::default(),
                motifs: super::super::MotifMemory::with_capacity(
                    super::super::MOTIF_DEFAULT_CAPACITY,
                ),
                motif_diagnostics: super::super::MotifDiagnostics::default(),
                last_observation: None,
                pending_predictor_input: None,
                host: super::super::HostRuntimeState::default(),
                spectral_monitor: super::super::SpectralEntropyMonitor::new(20),
                movement_monitor: super::super::MovementCoherenceMonitor::new(20),
                shear: super::super::ShearField2D::new(
                    super::super::CA_CHANNELS,
                    super::super::MACRO_H,
                    super::super::MACRO_W,
                ),
                target_loader,
                target_projector: super::super::SpectralProjector::new(&device)
                    .map_err(anyhow::Error::msg)?,
                target_feedback_available,
            });
        }
        let mut spectral_monitor = super::super::SpectralEntropyMonitor::new(20);
        spectral_monitor.restore_history(&checkpoint.spectral_history);
        spectral_monitor.restore_prev_mags(&checkpoint.spectral_prev_mags);
        let mut movement_monitor = super::super::MovementCoherenceMonitor::new(20);
        movement_monitor.restore_history(&checkpoint.movement_history);
        Ok(Self {
            bundle,
            micro: Tensor::from_vec(
                checkpoint.micro_tape.clone(),
                (
                    1,
                    super::super::CA_CHANNELS,
                    super::super::GRID_H,
                    super::super::GRID_W,
                ),
                &device,
            )?,
            macro_t: Tensor::from_vec(
                checkpoint.macro_tape.clone(),
                (
                    1,
                    super::super::CA_CHANNELS,
                    super::super::MACRO_H,
                    super::super::MACRO_W,
                ),
                &device,
            )?,
            hidden: Tensor::from_vec(
                checkpoint.hidden_mem.clone(),
                (1, super::super::MEMORY_DIM),
                &device,
            )?,
            phases: checkpoint.phases,
            theta_prev: checkpoint.theta_prev,
            theta_prev2: checkpoint.theta_prev2,
            energy: checkpoint.energy_state,
            rad_amp: checkpoint.rad_amp,
            shear_phase: checkpoint.shear_phase,
            controller_rng: super::super::RuntimeRng::seed_from_u64(forcing_seed ^ 0xC017_2011),
            target_rng: super::super::RuntimeRng::seed_from_u64(forcing_seed ^ 0x7A26_E700),
            forcing_seed,
            absolute_step: checkpoint.global_step,
            uncertainty: checkpoint.uncertainty.clone(),
            potential: checkpoint.potential.clone(),
            criticality: checkpoint.criticality.clone(),
            controller: checkpoint.controller.clone(),
            adaptive: checkpoint.adaptive_dynamics.clone(),
            motifs: checkpoint.motifs.clone(),
            motif_diagnostics: checkpoint.motif_diagnostics.clone(),
            last_observation: checkpoint.last_observation.clone(),
            pending_predictor_input: checkpoint
                .pending_predictor_input
                .as_ref()
                .filter(|values| {
                    values.len() == super::super::MEMORY_DIM + super::super::ACTION_COUNT
                })
                .map(|values| {
                    Tensor::from_vec(
                        values.clone(),
                        (1, super::super::MEMORY_DIM + super::super::ACTION_COUNT),
                        &device,
                    )
                })
                .transpose()?,
            host: checkpoint.host_runtime.clone(),
            spectral_monitor,
            movement_monitor,
            shear: super::super::ShearField2D::new(
                super::super::CA_CHANNELS,
                super::super::MACRO_H,
                super::super::MACRO_W,
            ),
            target_loader,
            target_projector: super::super::SpectralProjector::new(&device)
                .map_err(anyhow::Error::msg)?,
            target_feedback_available,
        })
    }

    pub(crate) fn state_vectors(&self) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        Ok((
            super::super::flatten_tensor(&self.micro)?,
            super::super::flatten_tensor(&self.macro_t)?,
            super::super::flatten_tensor(&self.hidden)?,
        ))
    }
}

fn read_only_target_loader(
    wav_dir: &std::path::Path,
    manifest_path: &std::path::Path,
) -> Result<Option<super::super::TargetAudioLoader>> {
    if !wav_dir.is_dir() || !manifest_path.is_file() {
        return Ok(None);
    }
    let manifest: super::super::CorpusManifest =
        serde_json::from_slice(&std::fs::read(manifest_path)?)?;
    if !(1..=2).contains(&manifest.schema_version) {
        anyhow::bail!(
            "unsupported corpus manifest schema {}",
            manifest.schema_version
        );
    }
    let entries: HashMap<&str, &super::super::CorpusEntry> = manifest
        .entries
        .iter()
        .map(|entry| (entry.file.as_str(), entry))
        .collect();
    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(wav_dir)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .collect();
    paths.sort();
    let mut files = Vec::new();
    let mut roles = Vec::new();
    let mut families = Vec::new();
    for path in paths {
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some(entry) = entries.get(name).copied() else {
            continue;
        };
        if path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("wav"))
            && entry.role != super::super::CorpusRole::Exclude
            && !super::super::is_generated_audio_name(name)
        {
            if let Ok(file) = super::super::TargetAudioLoader::index_wav(&path) {
                if file.output_frames >= super::super::CHUNK_SIZE {
                    files.push(file);
                    roles.push(entry.role);
                    families.push(entry.family.clone());
                }
            }
        }
    }
    if files.is_empty() {
        return Ok(None);
    }
    let mut train: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut development: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut validation: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (index, role) in roles.iter().enumerate() {
        match role {
            super::super::CorpusRole::Train => train
                .entry(families[index].clone())
                .or_default()
                .push(index),
            super::super::CorpusRole::Development => development
                .entry(families[index].clone())
                .or_default()
                .push(index),
            super::super::CorpusRole::Validation => validation
                .entry(families[index].clone())
                .or_default()
                .push(index),
            super::super::CorpusRole::Exclude => {}
        }
    }
    if train.is_empty() {
        for (index, family) in families.iter().enumerate() {
            train.entry(family.clone()).or_default().push(index);
        }
    }
    let mut development_files: Vec<usize> = development
        .values()
        .filter_map(|group| group.first().copied())
        .collect();
    let development_is_strict = !development_files.is_empty();
    if development_files.is_empty() {
        development_files.extend(
            train
                .values()
                .filter_map(|group| group.first().copied())
                .take(1),
        );
    }
    let mut validation_files: Vec<usize> = validation
        .values()
        .filter_map(|group| group.first().copied())
        .collect();
    let validation_is_strict = !validation_files.is_empty();
    if validation_files.is_empty() {
        validation_files.extend(
            train
                .values()
                .filter_map(|group| group.first().copied())
                .take(1),
        );
    }
    Ok(Some(super::super::TargetAudioLoader {
        files,
        train_groups: train.into_values().collect(),
        development_files,
        validation_files,
        development_held_out_files: roles
            .iter()
            .filter(|role| **role == super::super::CorpusRole::Development)
            .count(),
        held_out_files: roles
            .iter()
            .filter(|role| **role == super::super::CorpusRole::Validation)
            .count(),
        development_is_strict,
        validation_is_strict,
        manifest_path: manifest_path.display().to_string(),
        active: None,
        pending: Vec::with_capacity(super::super::TARGET_K),
        prefetched: VecDeque::with_capacity(super::super::TARGET_PREFETCH_CHUNKS),
        last_served: None,
    }))
}
