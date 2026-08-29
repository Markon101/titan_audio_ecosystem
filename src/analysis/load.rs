use super::config::AnalysisRequest;
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder as VBV, VarMap};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ResolvedPaths {
    pub base_dir: PathBuf,
    pub wav_dir: PathBuf,
    pub model: PathBuf,
    pub world: PathBuf,
    pub optimizer: PathBuf,
    pub morph: PathBuf,
    pub run_metadata: PathBuf,
    pub corpus_manifest: PathBuf,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct InferredArchitecture {
    pub morph_blocks: usize,
    pub morph_width: usize,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct OptimizerSummary {
    pub present: bool,
    pub global_step: Option<u64>,
    pub cumulative_updates: Option<u64>,
    pub renderer_control_version: Option<i64>,
    pub moment_tensors: usize,
    pub checkpoint_consistent_with_world: Option<bool>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct CorpusFileProvenance {
    pub manifest_index: usize,
    pub file: String,
    pub role: String,
    pub family: String,
    pub declared_provenance: String,
    pub present: bool,
    pub scheduler_index: Option<usize>,
    pub byte_size: Option<u64>,
    pub sha256: Option<String>,
    pub sample_rate: Option<u32>,
    pub channels: Option<u16>,
    pub bits_per_sample: Option<u16>,
    pub sample_format: Option<String>,
    pub source_frames: Option<usize>,
    pub output_frames_48k: Option<usize>,
    pub supported: bool,
    pub status: String,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct CorpusProvenance {
    pub manifest: super::super::provenance::FileIdentity,
    pub schema_version: Option<u32>,
    pub generated_by: Option<String>,
    pub entries_in_manifest_order: Vec<CorpusFileProvenance>,
    pub scheduler_order: Vec<String>,
    pub role_counts: BTreeMap<String, usize>,
    pub ordered_families: BTreeMap<String, Vec<String>>,
    pub unlisted_wavs: Vec<String>,
    pub missing_entries: Vec<String>,
    pub scheduler_semantics: String,
    pub coherent_episode_chunks: usize,
    pub aggregate_identity_sha256: String,
    pub fast_hash_mode: bool,
}

pub(crate) struct LoadedOrigin {
    pub paths: ResolvedPaths,
    pub architecture: InferredArchitecture,
    pub world: Option<super::super::WorldCheckpoint>,
    pub optimizer: OptimizerSummary,
    pub corpus: CorpusProvenance,
    pub canonical_before: Vec<super::super::provenance::FileIdentity>,
}

pub(crate) struct ModelBundle {
    pub varmap: VarMap,
    pub model: super::super::ComplexAudioEcosystem,
    pub monitor: super::super::MonitorHead,
    pub episodic: super::super::EpisodicMemory,
}

pub(crate) fn resolve_paths(request: &AnalysisRequest) -> ResolvedPaths {
    let base = &request.base_dir;
    let tag = request.run_tag.as_deref();
    ResolvedPaths {
        base_dir: base.clone(),
        wav_dir: base.join("OLD_WAVS"),
        model: request.model_path.clone().unwrap_or_else(|| {
            PathBuf::from(super::super::artifacts::artifact_path(
                &base.display().to_string(),
                "titan_model_v9",
                "safetensors",
                tag,
            ))
        }),
        world: request.state_path.clone().unwrap_or_else(|| {
            PathBuf::from(super::super::artifacts::artifact_path(
                &base.display().to_string(),
                "titan_world_v9",
                "bin",
                tag,
            ))
        }),
        optimizer: PathBuf::from(super::super::artifacts::artifact_path(
            &base.display().to_string(),
            "titan_optimizer_v9",
            "safetensors",
            tag,
        )),
        morph: PathBuf::from(super::super::artifacts::artifact_path(
            &base.display().to_string(),
            "titan_morph_state_v9",
            "json",
            tag,
        )),
        run_metadata: PathBuf::from(super::super::artifacts::artifact_path(
            &base.display().to_string(),
            "titan_run_metadata_v9",
            "json",
            tag,
        )),
        corpus_manifest: request
            .corpus_manifest
            .clone()
            .unwrap_or_else(|| base.join("titan_corpus_manifest_v7.json")),
    }
}

pub(crate) fn load_origin(request: &AnalysisRequest) -> Result<LoadedOrigin> {
    let paths = resolve_paths(request);
    if !paths.model.is_file() {
        anyhow::bail!("analysis model does not exist: {}", paths.model.display());
    }
    let architecture = infer_architecture(&paths.model)?;
    let needs_world = !request.frozen_rollouts.is_empty()
        || request.dynamics_ablation.is_some()
        || request.perturbation_analysis.is_some()
        || request.trained_init_control
        || request.target_feedback_switch.is_some()
        || request.render_attribution;
    let world = if paths.world.is_file() {
        let checkpoint = super::super::load_world(&paths.world.display().to_string())?;
        if checkpoint.active_depth > architecture.morph_blocks {
            anyhow::bail!(
                "world active depth L{:02} exceeds checkpoint physical depth L{:02}",
                checkpoint.active_depth,
                architecture.morph_blocks
            );
        }
        Some(checkpoint)
    } else if needs_world {
        anyhow::bail!(
            "stateful analysis requires v9 world: {}",
            paths.world.display()
        );
    } else {
        None
    };
    let optimizer = inspect_optimizer(&paths.optimizer, world.as_ref())?;
    let corpus = inspect_corpus(&paths, request.fast_provenance)?;
    let canonical_paths = canonical_paths(&paths)?;
    let canonical_before = super::super::provenance::identify_files(&canonical_paths)?;
    Ok(LoadedOrigin {
        paths,
        architecture,
        world,
        optimizer,
        corpus,
        canonical_before,
    })
}

pub(crate) fn load_model_bundle(
    origin: &LoadedOrigin,
    initialized_only: bool,
    initialization_seed: u64,
    restore_world_runtime: bool,
) -> Result<ModelBundle> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VBV::from_varmap(&varmap, DType::F32, &device);
    let architecture = super::super::MorphArchitecture {
        blocks: origin.architecture.morph_blocks,
        width: origin.architecture.morph_width,
    };
    let mut model =
        super::super::ComplexAudioEcosystem::new(vb.pp("model"), &device, architecture)?;
    let _arbiter = super::super::AudioArbiter::new(vb.pp("arbiter"))?;
    let monitor = super::super::MonitorHead::new(vb.pp("monitor_head"))?;
    let mut episodic = super::super::EpisodicMemory::new(vb.pp("episodic"))?;
    super::super::deterministic_reinit(&varmap, initialization_seed, &device)?;
    if !initialized_only {
        let report = super::super::load_into_varmap(
            &varmap,
            &origin.paths.model.display().to_string(),
            &device,
        )?;
        if !report.fully_exact() {
            anyhow::bail!(
                "analysis requires an exact model load: exact={} resized={} new={} non-morphic-incompatible={}",
                report.exact,
                report.morph_resized,
                report.morph_initialized,
                report.non_morph_missing + report.non_morph_mismatch + report.source_non_morph_dropped
            );
        }
    }
    if let Some(world) = &origin.world {
        model.set_depth(world.active_depth);
        if !initialized_only && restore_world_runtime {
            model.restore_runtime_state(&world.model_runtime, &device)?;
            episodic.restore_host(&world.episodic_slots, &device)?;
        }
    }
    Ok(ModelBundle {
        varmap,
        model,
        monitor,
        episodic,
    })
}

fn infer_architecture(model_path: &Path) -> Result<InferredArchitecture> {
    let tensors = candle_core::safetensors::load(model_path, &Device::Cpu)
        .map_err(anyhow::Error::msg)
        .with_context(|| {
            format!(
                "reading model tensor inventory from {}",
                model_path.display()
            )
        })?;
    let mut maximum_index = None;
    let mut morph_width = None;
    for (name, tensor) in &tensors {
        if let Some(index) = super::super::diagnostics::morph_block_index(name) {
            maximum_index = Some(maximum_index.map_or(index, |current: usize| current.max(index)));
        }
        if name.ends_with(".morphic.l0_1.weight") {
            morph_width = tensor.dims().first().copied();
        }
    }
    let morph_blocks = maximum_index
        .map(|index| index + 1)
        .ok_or_else(|| anyhow::anyhow!("model has no v9 MorphicStack block tensors"))?;
    let morph_width = morph_width
        .ok_or_else(|| anyhow::anyhow!("model has no v9 MorphicStack l0 expansion tensor"))?;
    if !(1..=super::super::MORPH_RUNTIME_MAX_BLOCKS).contains(&morph_blocks)
        || !(super::super::MORPH_RUNTIME_MIN_WIDTH..=super::super::MORPH_RUNTIME_MAX_WIDTH)
            .contains(&morph_width)
    {
        anyhow::bail!(
            "inferred morphology {} blocks x {} width is outside v9 runtime bounds",
            morph_blocks,
            morph_width
        );
    }
    Ok(InferredArchitecture {
        morph_blocks,
        morph_width,
    })
}

fn inspect_optimizer(
    path: &Path,
    world: Option<&super::super::WorldCheckpoint>,
) -> Result<OptimizerSummary> {
    if !path.is_file() {
        return Ok(OptimizerSummary {
            present: false,
            global_step: None,
            cumulative_updates: None,
            renderer_control_version: None,
            moment_tensors: 0,
            checkpoint_consistent_with_world: None,
        });
    }
    let tensors = candle_core::safetensors::load(path, &Device::Cpu)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("reading optimizer metadata from {}", path.display()))?;
    let scalar = |name: &str| -> Result<Option<i64>> {
        tensors
            .get(name)
            .map(|tensor| tensor.to_scalar::<i64>().map_err(anyhow::Error::msg))
            .transpose()
    };
    let global_step = scalar("optimizer.global_step")?.map(|value| value as u64);
    Ok(OptimizerSummary {
        present: true,
        global_step,
        cumulative_updates: scalar("optimizer.step_t")?.map(|value| value as u64),
        renderer_control_version: scalar("optimizer.renderer_control_version")?,
        moment_tensors: tensors
            .keys()
            .filter(|name| name.starts_with("optimizer.m.") || name.starts_with("optimizer.v."))
            .count(),
        checkpoint_consistent_with_world: world
            .map(|checkpoint| global_step.is_some_and(|step| step == checkpoint.global_step)),
    })
}

fn inspect_corpus(paths: &ResolvedPaths, fast_hash_mode: bool) -> Result<CorpusProvenance> {
    let manifest_identity = super::super::provenance::identify_file(&paths.corpus_manifest)?;
    if !paths.corpus_manifest.is_file() {
        return Ok(CorpusProvenance {
            manifest: manifest_identity,
            schema_version: None,
            generated_by: None,
            entries_in_manifest_order: Vec::new(),
            scheduler_order: Vec::new(),
            role_counts: BTreeMap::new(),
            ordered_families: BTreeMap::new(),
            unlisted_wavs: Vec::new(),
            missing_entries: Vec::new(),
            scheduler_semantics: "unavailable_no_manifest".to_string(),
            coherent_episode_chunks: super::super::TARGET_EPISODE_CHUNKS,
            aggregate_identity_sha256: super::super::provenance::sha256_bytes(b"no-manifest"),
            fast_hash_mode,
        });
    }
    let bytes = std::fs::read(&paths.corpus_manifest)?;
    let manifest: super::super::CorpusManifest = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing {} read-only", paths.corpus_manifest.display()))?;
    if !(1..=2).contains(&manifest.schema_version) {
        anyhow::bail!(
            "unsupported corpus manifest schema {}",
            manifest.schema_version
        );
    }
    let listed: HashMap<&str, usize> = manifest
        .entries
        .iter()
        .enumerate()
        .map(|(index, entry)| (entry.file.as_str(), index))
        .collect();
    let mut disk_paths: Vec<PathBuf> = if paths.wav_dir.is_dir() {
        std::fs::read_dir(&paths.wav_dir)?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| {
                path.extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("wav"))
            })
            .collect()
    } else {
        Vec::new()
    };
    disk_paths.sort();
    let scheduler_order: Vec<String> = disk_paths
        .iter()
        .filter_map(|path| path.file_name().and_then(|name| name.to_str()))
        .filter(|name| listed.contains_key(name))
        .map(str::to_string)
        .collect();
    let scheduler_indices: HashMap<&str, usize> = scheduler_order
        .iter()
        .enumerate()
        .map(|(index, name)| (name.as_str(), index))
        .collect();
    let unlisted_wavs = disk_paths
        .iter()
        .filter_map(|path| path.file_name().and_then(|name| name.to_str()))
        .filter(|name| !listed.contains_key(*name))
        .map(str::to_string)
        .collect::<Vec<_>>();
    let mut records = Vec::with_capacity(manifest.entries.len());
    let mut role_counts = BTreeMap::new();
    let mut ordered_families: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut missing_entries = Vec::new();
    let mut aggregate = Vec::new();
    aggregate.extend_from_slice(&bytes);
    for (manifest_index, entry) in manifest.entries.iter().enumerate() {
        let path = paths.wav_dir.join(&entry.file);
        let present = path.is_file();
        let role = format!("{:?}", entry.role).to_ascii_lowercase();
        *role_counts.entry(role.clone()).or_insert(0) += 1;
        ordered_families
            .entry(format!("{}:{}", role, entry.family))
            .or_default()
            .push(entry.file.clone());
        if !present {
            missing_entries.push(entry.file.clone());
        }
        let metadata = present.then(|| std::fs::metadata(&path)).transpose()?;
        let wav = present.then(|| hound::WavReader::open(&path)).transpose();
        let (spec, source_frames, status) = match wav {
            Ok(Some(reader)) => {
                let spec = reader.spec();
                let supported = matches!(
                    (spec.sample_format, spec.bits_per_sample),
                    (hound::SampleFormat::Float, 32)
                        | (hound::SampleFormat::Int, 16)
                        | (hound::SampleFormat::Int, 24 | 32)
                );
                let status = if supported {
                    "supported"
                } else {
                    "unsupported_format"
                };
                (Some(spec), Some(reader.duration() as usize), status)
            }
            Ok(None) => (None, None, "missing"),
            Err(_) => (None, None, "invalid_wav"),
        };
        let supported = status == "supported";
        let hash = if present && !fast_hash_mode {
            Some(super::super::provenance::sha256_file(&path)?)
        } else {
            None
        };
        aggregate.extend_from_slice(entry.file.as_bytes());
        aggregate.extend_from_slice(hash.as_deref().unwrap_or("hash-skipped").as_bytes());
        records.push(CorpusFileProvenance {
            manifest_index,
            file: entry.file.clone(),
            role,
            family: entry.family.clone(),
            declared_provenance: entry.provenance.clone(),
            present,
            scheduler_index: scheduler_indices.get(entry.file.as_str()).copied(),
            byte_size: metadata.as_ref().map(std::fs::Metadata::len),
            sha256: hash,
            sample_rate: spec.map(|value| value.sample_rate),
            channels: spec.map(|value| value.channels),
            bits_per_sample: spec.map(|value| value.bits_per_sample),
            sample_format: spec
                .map(|value| format!("{:?}", value.sample_format).to_ascii_lowercase()),
            source_frames,
            output_frames_48k: spec.zip(source_frames).map(|(value, frames)| {
                (frames as u64 * super::super::SAMPLE_RATE as u64 / value.sample_rate as u64)
                    as usize
            }),
            supported,
            status: status.to_string(),
        });
    }
    Ok(CorpusProvenance {
        manifest: manifest_identity,
        schema_version: Some(manifest.schema_version),
        generated_by: Some(manifest.generated_by),
        entries_in_manifest_order: records,
        scheduler_order,
        role_counts,
        ordered_families,
        unlisted_wavs,
        missing_entries,
        scheduler_semantics: "filesystem_paths_sorted_then_uniform_family_then_uniform_variant_with_coherent_episodes".to_string(),
        coherent_episode_chunks: super::super::TARGET_EPISODE_CHUNKS,
        aggregate_identity_sha256: super::super::provenance::sha256_bytes(&aggregate),
        fast_hash_mode,
    })
}

pub(crate) fn canonical_paths(paths: &ResolvedPaths) -> Result<Vec<PathBuf>> {
    let mut selected = vec![
        paths.model.clone(),
        paths.world.clone(),
        paths.optimizer.clone(),
        paths.morph.clone(),
        paths.run_metadata.clone(),
        paths.corpus_manifest.clone(),
    ];
    if paths.base_dir.is_dir() {
        for entry in std::fs::read_dir(&paths.base_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                selected.push(path);
            }
        }
    }
    selected.sort();
    selected.dedup();
    Ok(selected)
}

pub(crate) fn model_inventory_hash(varmap: &VarMap) -> Result<String> {
    let data = varmap.data().lock().unwrap();
    let mut names: Vec<&String> = data.keys().collect();
    names.sort();
    let mut normalized = Vec::new();
    for name in names {
        let tensor: &Tensor = data[name].as_tensor();
        normalized.extend_from_slice(name.as_bytes());
        normalized
            .extend_from_slice(format!("|{:?}|{:?}|", tensor.dtype(), tensor.dims()).as_bytes());
        for value in tensor
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?
        {
            normalized.extend_from_slice(&value.to_le_bytes());
        }
    }
    Ok(super::super::provenance::sha256_bytes(&normalized))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_paths_do_not_create_directories() {
        let request = AnalysisRequest {
            base_dir: PathBuf::from("/definitely/not/created/by/titan-analysis"),
            ..AnalysisRequest::default()
        };
        let paths = resolve_paths(&request);
        assert_eq!(paths.world.file_name().unwrap(), "titan_world_v9.bin");
        assert!(!request.base_dir.exists());
    }
}
