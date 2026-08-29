mod ablation;
mod attribution;
mod benchmark;
pub(crate) mod config;
mod controls;
mod load;
mod metrics;
mod perturbation;
mod report;
mod rollout;
mod separability;
mod state;
mod step;
mod terminal;

use anyhow::{Context, Result};
use config::{AnalysisRequest, TerminalMode};
use report::ArtifactEntry;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub(crate) fn is_analysis_flag(argument: &str) -> bool {
    matches!(
        argument,
        "--analysis-only"
            | "--analysis-help"
            | "--analysis-dir"
            | "--analysis-tag"
            | "--analysis-seed"
            | "--analysis-stride"
            | "--model-stats"
            | "--frozen-rollout"
            | "--analysis-morph-policy"
            | "--dynamics-ablation"
            | "--ablation"
            | "--perturbation-analysis"
            | "--perturbation"
            | "--perturbation-scales"
            | "--benchmark"
            | "--trained-init-control"
            | "--separability-analysis"
            | "--target-feedback-switch"
            | "--render-attribution"
            | "--analysis-expensive"
            | "--analysis-no-audio"
            | "--analysis-terminal"
            | "--analysis-fast-provenance"
            | "--analysis-overwrite"
    )
}

#[derive(Serialize)]
struct AnalysisSchema {
    name: &'static str,
    version: u32,
    modality: &'static str,
}

#[derive(Serialize)]
struct NonMutationReport {
    canonical_before: Vec<super::provenance::FileIdentity>,
    canonical_after: Vec<super::provenance::FileIdentity>,
    unchanged: bool,
    canonical_world_written: bool,
    canonical_checkpoint_bytes_unchanged: bool,
}

pub(crate) fn run_from_args(args: &[String]) -> Result<()> {
    let request = config::parse(args)?;
    if request.help {
        config::print_help();
        return Ok(());
    }
    run(request)
}

fn run(request: AnalysisRequest) -> Result<()> {
    let started = Instant::now();
    let effective_threads = request
        .threads
        .min(
            std::thread::available_parallelism()
                .map(|threads| threads.get())
                .unwrap_or(1),
        )
        .max(1);
    rayon::ThreadPoolBuilder::new()
        .num_threads(effective_threads)
        .build_global()
        .context("initializing the analysis-only CPU pool")?;
    let _wake_lock = super::WakeLockGuard::acquire();

    // This entire load phase is read-only. The first directory mutation occurs
    // only after exact checkpoint/corpus resolution and canonical fingerprints.
    let origin = load::load_origin(&request)?;
    let source_identity = super::provenance::source_identity();
    let normalized_configuration = serde_json::to_vec(&request)?;
    let model_hash = origin
        .canonical_before
        .iter()
        .find(|identity| identity.path == origin.paths.model.display().to_string())
        .and_then(|identity| identity.sha256.as_deref())
        .unwrap_or("model-hash-unavailable");
    let mut identity_bytes = model_hash.as_bytes().to_vec();
    identity_bytes.extend_from_slice(&normalized_configuration);
    identity_bytes.extend_from_slice(&request.analysis_seed.to_le_bytes());
    let digest = super::provenance::sha256_bytes(&identity_bytes);
    let short_digest = &digest[..16];
    let analysis_id = request
        .analysis_tag
        .as_ref()
        .map(|tag| format!("{tag}-{short_digest}"))
        .unwrap_or_else(|| short_digest.to_string());
    let output = request.analysis_dir.clone().unwrap_or_else(|| {
        origin
            .paths
            .base_dir
            .join("titan_audio_analysis_v1")
            .join(&analysis_id)
    });
    validate_output_root(&output, &origin)?;
    report::prepare_output(&output, request.overwrite)?;
    let incomplete_marker = output.join("INCOMPLETE");
    report::write_text_atomic(
        &incomplete_marker,
        "Analysis did not reach its atomic final report. Canonical inputs were opened read-only; inspect run.log and rerun into a new directory.\n",
    )?;
    report::write_text_atomic(
        &output.join("run.log"),
        &format!(
            "Titan Audio scientific instrumentation v1\nanalysis_id={analysis_id}\nfrozen_weights=true\noptimizer_steps=0\nbackward_passes=0\ndirect_reference_input=false\n"
        ),
    )?;
    let requested_names = request.requested_analyses();
    terminal::banner(
        request.terminal,
        &analysis_id,
        &output,
        &origin,
        &requested_names,
        request.analysis_seed,
    );

    let mut warnings = source_identity.warnings.clone();
    if request.fast_provenance {
        warnings.push(
            "fast provenance skipped per-WAV byte hashes; corpus identity is incomplete"
                .to_string(),
        );
    }
    if origin.optimizer.present && origin.optimizer.checkpoint_consistent_with_world == Some(false)
    {
        warnings.push(
            "optimizer global step does not match world step; checkpoint set is not exact-resume consistent"
                .to_string(),
        );
    }
    warnings.push(
        "Audio has no direct reference input; target effects in frozen rollouts use post-forward coarse error feedback only"
            .to_string(),
    );
    let mut artifacts = Vec::<ArtifactEntry>::new();
    let provenance_value = serde_json::json!({
        "schema": {"name":"titan_audio_provenance","version":1},
        "source": source_identity,
        "package": {"name": env!("CARGO_PKG_NAME"), "version": env!("CARGO_PKG_VERSION")},
        "target": std::env::consts::ARCH,
        "paths": origin.paths,
        "checkpoint_files": origin.canonical_before,
        "world": origin.world.as_ref().map(|world| serde_json::json!({
            "version": world.version,
            "global_step": world.global_step,
            "seed": world.seed,
            "active_depth": world.active_depth,
        })),
        "optimizer": origin.optimizer,
        "corpus": origin.corpus,
        "analysis_invocation": request.invocation,
        "analysis_configuration": request,
        "analysis_id": analysis_id,
    });
    let provenance_path = output.join("provenance.json");
    report::write_json_atomic(&provenance_path, &provenance_value)?;
    artifacts.push(report::artifact_entry(
        &output,
        &provenance_path,
        "application/json",
        None,
        None,
        None,
        "read-only identity report",
    )?);

    let mut model_statistics = None;
    if request.model_stats {
        println_unless_quiet(
            request.terminal,
            "[model_stats] loading exact tensor inventory",
        );
        let bundle = load::load_model_bundle(&origin, false, request.analysis_seed, true)?;
        let inventory_hash = load::model_inventory_hash(&bundle.varmap)?;
        let mut statistics = super::diagnostics::model_statistics(
            &bundle.varmap,
            origin.world.as_ref(),
            origin.architecture.morph_blocks,
            origin.architecture.morph_width,
        )?;
        statistics["model_tensor_inventory_sha256"] = serde_json::Value::String(inventory_hash);
        let path = output.join("model_stats.json");
        report::write_json_atomic(&path, &statistics)?;
        artifacts.push(report::artifact_entry(
            &output,
            &path,
            "application/json",
            None,
            None,
            None,
            "observational tensor and state statistics",
        )?);
        model_statistics = Some(statistics);
    }

    let mut rollout_reports = Vec::new();
    let mut primary_rollout: Option<rollout::RolloutRun> = None;
    if !request.frozen_rollouts.is_empty() {
        let full = step::Intervention::full();
        let run = rollout::run(
            &origin,
            "frozen_full",
            &full,
            &request.frozen_rollouts,
            request.analysis_stride,
            false,
            request.analysis_seed,
            false,
            None,
            |offset, total, record| {
                terminal::progress(
                    request.terminal,
                    "frozen_rollout",
                    "frozen_full",
                    offset,
                    total,
                    Some(record),
                )
            },
        )?;
        export_rollout(
            &output,
            Path::new("rollouts"),
            &run,
            !request.no_audio,
            &mut artifacts,
        )?;
        rollout_reports.push(serde_json::to_value(&run.summary)?);
        primary_rollout = Some(run);
    }

    let mut ablation_report = None;
    if let Some(horizon) = request.dynamics_ablation {
        let runs = ablation::run_suite(
            &origin,
            horizon,
            request.analysis_stride,
            &request.ablations,
            request.analysis_seed,
            |condition, offset, total| {
                terminal::progress(request.terminal, "ablation", condition, offset, total, None)
            },
            |name, run| {
                export_rollout(
                    &output,
                    &PathBuf::from("ablations").join(name),
                    run,
                    !request.no_audio,
                    &mut artifacts,
                )
            },
        )?;
        let suite_path = output.join("ablations/summary.json");
        report::write_json_atomic(&suite_path, &runs.suite)?;
        artifacts.push(report::artifact_entry(
            &output,
            &suite_path,
            "application/json",
            None,
            Some(0),
            Some(horizon),
            "typed cloned-state causal comparisons",
        )?);
        export_rollout(
            &output,
            Path::new("ablations/full"),
            &runs.baseline,
            !request.no_audio,
            &mut artifacts,
        )?;
        ablation_report = Some(serde_json::to_value(&runs.suite)?);
    }

    let mut perturbation_report = None;
    if let Some(horizon) = request.perturbation_analysis {
        let runs = perturbation::run_suite(
            &origin,
            horizon,
            request.analysis_stride,
            &request.perturbations,
            &request.perturbation_scales,
            request.analysis_seed,
            |condition, offset, total| {
                terminal::progress(
                    request.terminal,
                    "perturbation",
                    condition,
                    offset,
                    total,
                    None,
                )
            },
            |name, run| {
                export_rollout(
                    &output,
                    &PathBuf::from("perturbations").join(name),
                    run,
                    !request.no_audio,
                    &mut artifacts,
                )
            },
        )?;
        let suite_path = output.join("perturbations/summary.json");
        report::write_json_atomic(&suite_path, &runs.suite)?;
        artifacts.push(report::artifact_entry(
            &output,
            &suite_path,
            "application/json",
            None,
            Some(0),
            Some(horizon),
            "paired latent recovery and phenotype robustness analysis",
        )?);
        export_rollout(
            &output,
            Path::new("perturbations/baseline"),
            &runs.baseline,
            !request.no_audio,
            &mut artifacts,
        )?;
        perturbation_report = Some(serde_json::to_value(&runs.suite)?);
    }

    let mut benchmark_report = None;
    if request.benchmark {
        println_unless_quiet(
            request.terminal,
            "[benchmark] generating deterministic references",
        );
        let signals = benchmark::generate(request.analysis_seed);
        let generated = primary_rollout
            .as_ref()
            .map(|run| (run.all_left.as_slice(), run.all_right.as_slice()));
        let suite = benchmark::report(&signals, generated);
        let definitions_path = output.join("benchmark/definitions.json");
        report::write_json_atomic(&definitions_path, &suite.definitions)?;
        artifacts.push(report::artifact_entry(
            &output,
            &definitions_path,
            "application/json",
            None,
            None,
            None,
            "deterministic procedural definitions",
        )?);
        if !request.no_audio {
            for signal in &signals {
                let path = output
                    .join("benchmark/references")
                    .join(format!("{}.wav", signal.definition.id));
                report::write_wav_atomic(&path, &signal.left, &signal.right)?;
                artifacts.push(report::artifact_entry(
                    &output,
                    &path,
                    "audio/wav",
                    Some(&signal.definition.id),
                    None,
                    None,
                    "procedural reference; never entered training",
                )?);
            }
        }
        let path = output.join("benchmark/summary.json");
        report::write_json_atomic(&path, &suite)?;
        artifacts.push(report::artifact_entry(
            &output,
            &path,
            "application/json",
            None,
            None,
            None,
            "descriptor coverage; not reconstruction",
        )?);
        benchmark_report = Some(serde_json::to_value(suite)?);
    }

    let mut control_report = None;
    let mut control_runs_for_separability = None;
    if request.trained_init_control {
        let horizon = request
            .frozen_rollouts
            .iter()
            .copied()
            .max()
            .unwrap_or(64)
            .min(64);
        let controls = controls::run(
            &origin,
            horizon,
            request.analysis_stride.min(horizon).max(1),
            request.analysis_seed,
            |condition, offset, total| {
                terminal::progress(
                    request.terminal,
                    "trained_init_control",
                    condition,
                    offset,
                    total,
                    None,
                )
            },
        )?;
        let path = output.join("controls/trained_initialization.json");
        report::write_json_atomic(&path, &controls.report)?;
        artifacts.push(report::artifact_entry(
            &output,
            &path,
            "application/json",
            None,
            Some(0),
            Some(horizon),
            "active-depth-matched frozen controls",
        )?);
        for (name, run) in &controls.runs {
            export_rollout(
                &output,
                &PathBuf::from("controls").join(name),
                run,
                !request.no_audio,
                &mut artifacts,
            )?;
        }
        control_report = Some(serde_json::to_value(&controls.report)?);
        control_runs_for_separability = Some(controls.runs);
    }

    let target_switch_report = request
        .target_feedback_switch
        .map(attribution::target_switch_report);
    if let Some(target_switch) = &target_switch_report {
        let path = output.join("target_feedback_switch.json");
        report::write_json_atomic(&path, target_switch)?;
        artifacts.push(report::artifact_entry(
            &output,
            &path,
            "application/json",
            None,
            None,
            None,
            "capability boundary and first-effect semantics",
        )?);
        warnings.push(target_switch.status.clone());
    }

    let attribution_report = if request.render_attribution {
        if primary_rollout.is_none() {
            let run = rollout::run(
                &origin,
                "attribution_full",
                &step::Intervention::full(),
                &[1],
                1,
                false,
                request.analysis_seed,
                false,
                None,
                |_, _, _| {},
            )?;
            export_rollout(
                &output,
                Path::new("attribution/full"),
                &run,
                !request.no_audio,
                &mut artifacts,
            )?;
            primary_rollout = Some(run);
        }
        let attribution = attribution::report(!request.no_audio, !request.no_audio);
        let path = output.join("attribution/summary.json");
        report::write_json_atomic(&path, &attribution)?;
        artifacts.push(report::artifact_entry(
            &output,
            &path,
            "application/json",
            None,
            None,
            None,
            "full-path attribution boundary; no unvalidated stems",
        )?);
        Some(serde_json::to_value(attribution)?)
    } else {
        None
    };

    let separability_report = if request.separability_analysis {
        let mut generated_runs = Vec::new();
        if primary_rollout.is_none() && control_runs_for_separability.is_none() {
            for index in 0..3u64 {
                let name = format!("fresh_world_seed_{}", request.analysis_seed + index);
                let run = rollout::run(
                    &origin,
                    &name,
                    &step::Intervention::full(),
                    &[64],
                    request.analysis_stride.min(64),
                    false,
                    request.analysis_seed + index,
                    true,
                    None,
                    |offset, total, record| {
                        terminal::progress(
                            request.terminal,
                            "separability",
                            &name,
                            offset,
                            total,
                            Some(record),
                        )
                    },
                )?;
                export_rollout(
                    &output,
                    &PathBuf::from("separability").join(&name),
                    &run,
                    !request.no_audio,
                    &mut artifacts,
                )?;
                generated_runs.push((name, run));
            }
        }
        let references: Vec<(&str, &rollout::RolloutRun)> =
            if let Some(runs) = control_runs_for_separability.as_ref() {
                runs.iter()
                    .map(|(name, run)| (name.as_str(), run))
                    .collect()
            } else if !generated_runs.is_empty() {
                generated_runs
                    .iter()
                    .map(|(name, run)| (name.as_str(), run))
                    .collect()
            } else {
                primary_rollout
                    .as_ref()
                    .map(|run| vec![(run.summary.condition.as_str(), run)])
                    .unwrap_or_default()
            };
        let separability = separability::analyze(&references);
        let path = output.join("separability/summary.json");
        report::write_json_atomic(&path, &separability)?;
        artifacts.push(report::artifact_entry(
            &output,
            &path,
            "application/json",
            None,
            None,
            None,
            "transparent state and audio distance matrix",
        )?);
        Some(serde_json::to_value(separability)?)
    } else {
        None
    };

    warnings.sort();
    warnings.dedup();
    let canonical_paths: Vec<PathBuf> = origin
        .canonical_before
        .iter()
        .map(|identity| PathBuf::from(&identity.path))
        .collect();
    let canonical_after = super::provenance::identify_files(&canonical_paths)?;
    let unchanged = super::provenance::unchanged(&origin.canonical_before, &canonical_after);
    let non_mutation = NonMutationReport {
        canonical_before: origin.canonical_before.clone(),
        canonical_after,
        unchanged,
        canonical_world_written: false,
        canonical_checkpoint_bytes_unchanged: unchanged,
    };
    if !unchanged {
        anyhow::bail!("canonical input bytes or modification times changed during analysis");
    }

    let artifact_index_path = output.join("artifact_index.json");
    report::write_json_atomic(&artifact_index_path, &artifacts)?;
    let report_value = serde_json::json!({
        "schema": AnalysisSchema {name:"titan_audio_analysis", version:1, modality:"audio"},
        "analysis": {
            "id": analysis_id,
            "weights_frozen": true,
            "backward_passes": 0,
            "optimizer_constructed": false,
            "optimizer_steps": 0,
            "analysis_seed": request.analysis_seed,
            "stride_chunks": request.analysis_stride,
            "threads": effective_threads,
            "elapsed_seconds": started.elapsed().as_secs_f64(),
            "output_directory": output,
        },
        "identity": {
            "build_commit": super::BUILD_COMMIT,
            "model": origin.paths.model,
            "world": origin.paths.world,
            "checkpoint_step": origin.world.as_ref().map(|world| world.global_step),
            "checkpoint_set_consistency": if origin.optimizer.checkpoint_consistent_with_world == Some(true) {"consistent"} else {"warning"},
            "morph_physical_depth": origin.architecture.morph_blocks,
            "morph_active_depth": origin.world.as_ref().map(|world| world.active_depth),
        },
        "configuration": {
            "conditioning": "coarse_target_error_feedback_or_explicit_target_independent_ablation",
            "direct_reference_input": false,
            "morph_policy": request.morph_policy,
            "forcing_protocol": "common_exogenous_forcing_by_absolute_step",
        },
        "corpus_provenance": origin.corpus,
        "model_statistics": model_statistics,
        "rollouts": rollout_reports,
        "ablations": ablation_report,
        "perturbations": perturbation_report,
        "benchmarks": benchmark_report,
        "trained_initialization_controls": control_report,
        "separability": separability_report,
        "target_feedback_switch": target_switch_report,
        "attribution": attribution_report,
        "artifacts": artifacts,
        "warnings": warnings,
        "interpretation_constraints": [
            "persistent dynamics are not evidence of life",
            "divergence is not strong emergence",
            "phenotype robustness is not latent recovery or self-healing",
            "benchmark proximity is not semantic understanding",
            "recurrence candidates are not cognition, concepts, or proof of an attractor",
            "host-controller interventions are not learned-neural ablations"
        ],
        "non_mutation": non_mutation,
    });
    let final_path = output.join("analysis_report.json");
    report::write_json_atomic(&final_path, &report_value)?;
    std::fs::remove_file(incomplete_marker)?;
    terminal::completion(request.terminal, &output, unchanged, warnings.len());
    Ok(())
}

fn validate_output_root(output: &Path, origin: &load::LoadedOrigin) -> Result<()> {
    if output == origin.paths.base_dir || output == origin.paths.wav_dir {
        anyhow::bail!("analysis output must be a dedicated subdirectory, not a canonical root");
    }
    if output.starts_with(&origin.paths.wav_dir) {
        anyhow::bail!("analysis output may not be placed inside OLD_WAVS");
    }
    for identity in &origin.canonical_before {
        if output == Path::new(&identity.path) {
            anyhow::bail!(
                "analysis output collides with canonical input {}",
                identity.path
            );
        }
    }
    Ok(())
}

fn export_rollout(
    root: &Path,
    relative: &Path,
    run: &rollout::RolloutRun,
    audio: bool,
    artifacts: &mut Vec<ArtifactEntry>,
) -> Result<()> {
    let directory = root.join(relative);
    let summary_path = directory.join("summary.json");
    report::write_json_atomic(&summary_path, &run.summary)?;
    artifacts.push(report::artifact_entry(
        root,
        &summary_path,
        "application/json",
        Some(&run.summary.condition),
        Some(0),
        Some(run.summary.horizon_chunks),
        "frozen rollout summary",
    )?);
    let csv_path = directory.join("trace.csv");
    report::write_step_csv_atomic(&csv_path, &run.summary.sampled_steps)?;
    artifacts.push(report::artifact_entry(
        root,
        &csv_path,
        "text/csv",
        Some(&run.summary.condition),
        Some(0),
        Some(run.summary.horizon_chunks),
        "stride-sampled operational metrics",
    )?);
    if audio && !run.all_left.is_empty() {
        let raw_path = directory.join("raw_renderer.wav");
        report::write_wav_atomic(&raw_path, &run.all_raw_left, &run.all_raw_right)?;
        artifacts.push(report::artifact_entry(
            root,
            &raw_path,
            "audio/wav",
            Some(&run.summary.condition),
            Some(0),
            Some(run.summary.horizon_chunks),
            "full learned renderer after analysis tanh observation; before boost, saturation, and DC block",
        )?);
        let wav_path = directory.join("post_dsp.wav");
        report::write_wav_atomic(&wav_path, &run.all_left, &run.all_right)?;
        artifacts.push(report::artifact_entry(
            root,
            &wav_path,
            "audio/wav",
            Some(&run.summary.condition),
            Some(0),
            Some(run.summary.horizon_chunks),
            "post-saturation stateful DC-block; no final mastering normalization",
        )?);
        let spectrogram_path = directory.join("spectrogram.png");
        report::write_spectrogram_png(&spectrogram_path, &run.all_left, &run.all_right)?;
        artifacts.push(report::artifact_entry(
            root,
            &spectrogram_path,
            "image/png",
            Some(&run.summary.condition),
            Some(0),
            Some(run.summary.horizon_chunks),
            "fixed -80..0 dB analysis spectrogram",
        )?);
    }
    if let Some(frame) = run.frames.last() {
        let micro_path = directory.join(format!("micro_atlas_{:06}.png", frame.offset));
        report::write_state_atlas_png(
            &micro_path,
            &frame.micro,
            super::CA_CHANNELS,
            super::GRID_H,
            super::GRID_W,
        )?;
        artifacts.push(report::artifact_entry(
            root,
            &micro_path,
            "image/png",
            Some(&run.summary.condition),
            Some(frame.offset),
            Some(frame.offset),
            "fixed [-1,1] micro-state atlas",
        )?);
        let macro_path = directory.join(format!("macro_atlas_{:06}.png", frame.offset));
        report::write_state_atlas_png(
            &macro_path,
            &frame.macro_t,
            super::CA_CHANNELS,
            super::MACRO_H,
            super::MACRO_W,
        )?;
        artifacts.push(report::artifact_entry(
            root,
            &macro_path,
            "image/png",
            Some(&run.summary.condition),
            Some(frame.offset),
            Some(frame.offset),
            "fixed [-1,1] macro-state atlas",
        )?);
    }
    Ok(())
}

fn println_unless_quiet(mode: TerminalMode, message: &str) {
    if mode != TerminalMode::Quiet {
        println!("{message}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_arguments_do_not_activate_analysis() {
        for argument in ["--duration", "--fresh-world", "--run-tag", "--model"] {
            assert!(!is_analysis_flag(argument));
        }
    }

    #[test]
    fn every_instrumentation_selector_activates_early_dispatch() {
        for argument in [
            "--analysis-only",
            "--model-stats",
            "--frozen-rollout",
            "--dynamics-ablation",
            "--perturbation-analysis",
            "--benchmark",
        ] {
            assert!(is_analysis_flag(argument));
        }
    }
}
