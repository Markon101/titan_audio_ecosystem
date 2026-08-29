use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum MorphPolicy {
    Fixed,
    Ecology,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TerminalMode {
    Compact,
    Rich,
    Quiet,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct AnalysisRequest {
    pub analysis_only: bool,
    pub help: bool,
    pub base_dir: PathBuf,
    pub model_path: Option<PathBuf>,
    pub state_path: Option<PathBuf>,
    pub corpus_manifest: Option<PathBuf>,
    pub run_tag: Option<String>,
    pub analysis_dir: Option<PathBuf>,
    pub analysis_tag: Option<String>,
    pub analysis_seed: u64,
    pub analysis_stride: usize,
    pub threads: usize,
    pub model_stats: bool,
    pub frozen_rollouts: Vec<usize>,
    pub morph_policy: MorphPolicy,
    pub dynamics_ablation: Option<usize>,
    pub ablations: Vec<String>,
    pub perturbation_analysis: Option<usize>,
    pub perturbations: Vec<String>,
    pub perturbation_scales: Vec<f32>,
    pub benchmark: bool,
    pub trained_init_control: bool,
    pub separability_analysis: bool,
    pub target_feedback_switch: Option<usize>,
    pub render_attribution: bool,
    pub expensive: bool,
    pub no_audio: bool,
    pub terminal: TerminalMode,
    pub fast_provenance: bool,
    pub overwrite: bool,
    pub invocation: Vec<String>,
}

impl Default for AnalysisRequest {
    fn default() -> Self {
        Self {
            analysis_only: false,
            help: false,
            base_dir: PathBuf::from("/sdcard/Download"),
            model_path: None,
            state_path: None,
            corpus_manifest: None,
            run_tag: None,
            analysis_dir: None,
            analysis_tag: None,
            analysis_seed: 0xA11A_51A5,
            analysis_stride: 16,
            threads: 1,
            model_stats: false,
            frozen_rollouts: Vec::new(),
            morph_policy: MorphPolicy::Fixed,
            dynamics_ablation: None,
            ablations: Vec::new(),
            perturbation_analysis: None,
            perturbations: Vec::new(),
            perturbation_scales: vec![0.0001, 0.001, 0.01, 0.1],
            benchmark: false,
            trained_init_control: false,
            separability_analysis: false,
            target_feedback_switch: None,
            render_attribution: false,
            expensive: false,
            no_audio: false,
            terminal: TerminalMode::Rich,
            fast_provenance: false,
            overwrite: false,
            invocation: Vec::new(),
        }
    }
}

impl AnalysisRequest {
    pub(crate) fn requested_analyses(&self) -> Vec<&'static str> {
        let mut names = Vec::new();
        if self.model_stats {
            names.push("model_stats");
        }
        if !self.frozen_rollouts.is_empty() {
            names.push("frozen_rollout");
        }
        if self.dynamics_ablation.is_some() {
            names.push("dynamics_ablation");
        }
        if self.perturbation_analysis.is_some() {
            names.push("perturbation_analysis");
        }
        if self.benchmark {
            names.push("synthetic_benchmark");
        }
        if self.trained_init_control {
            names.push("trained_init_control");
        }
        if self.separability_analysis {
            names.push("separability_analysis");
        }
        if self.target_feedback_switch.is_some() {
            names.push("target_feedback_switch");
        }
        if self.render_attribution {
            names.push("render_attribution");
        }
        names
    }
}

pub(crate) fn parse(args: &[String]) -> Result<AnalysisRequest> {
    let mut request = AnalysisRequest {
        invocation: args.to_vec(),
        ..AnalysisRequest::default()
    };
    let mut positional_base_seen = false;
    let mut index = 1usize;
    while index < args.len() {
        let flag = args[index].as_str();
        match flag {
            "--analysis-only" => {
                request.analysis_only = true;
                index += 1;
            }
            "--analysis-help" => {
                request.help = true;
                index += 1;
            }
            "--base-dir" | "-b" => {
                request.base_dir = PathBuf::from(value(args, &mut index, flag)?);
            }
            "--model" => request.model_path = Some(PathBuf::from(value(args, &mut index, flag)?)),
            "--state" => request.state_path = Some(PathBuf::from(value(args, &mut index, flag)?)),
            "--corpus-manifest" => {
                request.corpus_manifest = Some(PathBuf::from(value(args, &mut index, flag)?));
            }
            "--run-tag" => {
                let tag = value(args, &mut index, flag)?;
                super::super::artifacts::validate_run_tag(&tag)?;
                request.run_tag = Some(tag);
            }
            "--analysis-dir" => {
                request.analysis_dir = Some(PathBuf::from(value(args, &mut index, flag)?));
            }
            "--analysis-tag" => {
                let tag = value(args, &mut index, flag)?;
                validate_analysis_tag(&tag)?;
                request.analysis_tag = Some(tag);
            }
            "--analysis-seed" => request.analysis_seed = parse_value(args, &mut index, flag)?,
            "--analysis-stride" => request.analysis_stride = parse_value(args, &mut index, flag)?,
            "--threads" | "-t" => request.threads = parse_value(args, &mut index, flag)?,
            "--model-stats" => {
                request.model_stats = true;
                index += 1;
            }
            "--frozen-rollout" => {
                request.frozen_rollouts = parse_list(&value(args, &mut index, flag)?, flag)?;
            }
            "--analysis-morph-policy" => {
                request.morph_policy = match value(args, &mut index, flag)?.as_str() {
                    "fixed" => MorphPolicy::Fixed,
                    "ecology" => MorphPolicy::Ecology,
                    other => anyhow::bail!("unsupported --analysis-morph-policy {other:?}"),
                };
            }
            "--dynamics-ablation" => {
                request.dynamics_ablation = Some(parse_value(args, &mut index, flag)?);
            }
            "--ablation" => request.ablations.push(value(args, &mut index, flag)?),
            "--perturbation-analysis" => {
                request.perturbation_analysis = Some(parse_value(args, &mut index, flag)?);
            }
            "--perturbation" => request.perturbations.push(value(args, &mut index, flag)?),
            "--perturbation-scales" => {
                request.perturbation_scales = parse_list(&value(args, &mut index, flag)?, flag)?;
            }
            "--benchmark" => {
                request.benchmark = true;
                index += 1;
            }
            "--trained-init-control" => {
                request.trained_init_control = true;
                index += 1;
            }
            "--separability-analysis" => {
                request.separability_analysis = true;
                index += 1;
            }
            "--target-feedback-switch" => {
                request.target_feedback_switch = Some(parse_value(args, &mut index, flag)?);
            }
            "--render-attribution" => {
                request.render_attribution = true;
                index += 1;
            }
            "--analysis-expensive" => {
                request.expensive = true;
                index += 1;
            }
            "--analysis-no-audio" => {
                request.no_audio = true;
                index += 1;
            }
            "--analysis-terminal" => {
                request.terminal = match value(args, &mut index, flag)?.as_str() {
                    "compact" => TerminalMode::Compact,
                    "rich" => TerminalMode::Rich,
                    "quiet" => TerminalMode::Quiet,
                    other => anyhow::bail!("unsupported --analysis-terminal {other:?}"),
                };
            }
            "--analysis-fast-provenance" => {
                request.fast_provenance = true;
                index += 1;
            }
            "--analysis-overwrite" => {
                request.overwrite = true;
                index += 1;
            }
            "--help" | "-h" => {
                request.help = true;
                index += 1;
            }
            forbidden if is_forbidden_training_flag(forbidden) => {
                anyhow::bail!("{forbidden} is incompatible with --analysis-only")
            }
            positional if !positional.starts_with('-') && !positional_base_seen => {
                request.base_dir = PathBuf::from(positional);
                positional_base_seen = true;
                index += 1;
            }
            unknown => anyhow::bail!("unknown analysis parameter: {unknown}"),
        }
    }
    if request.help {
        return Ok(request);
    }
    if !request.analysis_only {
        anyhow::bail!("scientific instrumentation flags require --analysis-only");
    }
    if request.analysis_stride == 0 {
        anyhow::bail!("--analysis-stride must be at least 1");
    }
    if request.threads == 0 {
        anyhow::bail!("--threads must be at least 1 in analysis mode");
    }
    if request.perturbation_scales.is_empty()
        || request
            .perturbation_scales
            .iter()
            .any(|scale| !scale.is_finite() || *scale < 0.0)
    {
        anyhow::bail!("perturbation scales must be finite and non-negative");
    }
    if request.requested_analyses().is_empty() {
        request.model_stats = true;
        request.frozen_rollouts = vec![256];
    }
    let maximum_horizon = if request.expensive { 16_384 } else { 4_096 };
    for horizon in request
        .frozen_rollouts
        .iter()
        .copied()
        .chain(request.dynamics_ablation)
        .chain(request.perturbation_analysis)
    {
        if horizon == 0 || horizon > maximum_horizon {
            anyhow::bail!(
                "analysis horizon {horizon} is outside 1..={maximum_horizon}; use --analysis-expensive for the extended limit"
            );
        }
    }
    if request.morph_policy == MorphPolicy::Ecology {
        anyhow::bail!(
            "clone-only ecology morph decisions require target-loss parity hooks that are not safely available in Phase 1; use --analysis-morph-policy fixed"
        );
    }
    Ok(request)
}

fn is_forbidden_training_flag(flag: &str) -> bool {
    matches!(
        flag,
        "--lr"
            | "-l"
            | "--duration"
            | "-d"
            | "--bptt"
            | "-w"
            | "--core-update-every"
            | "--seed"
            | "-s"
            | "--import-model"
            | "--refresh-corpus-manifest"
            | "--rebuild-corpus-manifest"
            | "--prune-missing-manifest-entries"
            | "--prune-missing"
            | "--manifest-only"
            | "--freeze-morph"
            | "--morph-layers"
            | "--morph-blocks"
            | "--morph-width"
            | "--motif-capacity"
            | "--morph-depth"
            | "--max-morph-depth"
            | "--fresh-world"
            | "--fresh-decoder"
            | "--fresh-model"
            | "--fresh"
            | "-f"
    )
}

fn value(args: &[String], index: &mut usize, flag: &str) -> Result<String> {
    let next = index.saturating_add(1);
    let value = args
        .get(next)
        .with_context(|| format!("missing value for {flag}"))?
        .clone();
    *index += 2;
    Ok(value)
}

fn parse_value<T>(args: &[String], index: &mut usize, flag: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    value(args, index, flag)?
        .parse::<T>()
        .with_context(|| format!("invalid value for {flag}"))
}

fn parse_list<T>(text: &str, flag: &str) -> Result<Vec<T>>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    let values: Vec<T> = text
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<T>()
                .with_context(|| format!("invalid value {part:?} in {flag}"))
        })
        .collect::<Result<_>>()?;
    if values.is_empty() {
        anyhow::bail!("{flag} requires a non-empty comma-separated list");
    }
    Ok(values)
}

fn validate_analysis_tag(tag: &str) -> Result<()> {
    if tag.is_empty()
        || tag.len() > 64
        || !tag
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
    {
        anyhow::bail!("analysis tag must be 1..64 ASCII letters, digits, '-' or '_'");
    }
    Ok(())
}

pub(crate) fn print_help() {
    println!(
        "TITAN Audio v9 scientific instrumentation\n\n\
Usage: titan --analysis-only [read-only inputs] [analyses]\n\n\
Read-only inputs:\n\
  --base-dir DIR              Checkpoint/corpus root\n\
  --model PATH                Exact v9 model checkpoint\n\
  --state PATH                Exact v9 world checkpoint\n\
  --corpus-manifest PATH      Existing manifest; never created or repaired\n\
  --run-tag NAME              Select tagged checkpoint companions\n\
  --analysis-dir PATH         Dedicated sidecar output directory\n\
  --analysis-tag NAME         Deterministic human-readable analysis prefix\n\
  --analysis-seed N           Independent perturbation/benchmark seed\n\
  --analysis-stride N         Observation stride in chunks (default 16)\n\
  --threads N                 Analysis CPU threads (default 1)\n\n\
Analyses (multiple may be selected):\n\
  --model-stats\n\
  --frozen-rollout 64,256,1024\n\
  --dynamics-ablation N [--ablation NAME ...]\n\
  --perturbation-analysis N [--perturbation NAME ...]\n\
  --perturbation-scales 0.0001,0.001,0.01,0.1\n\
  --benchmark\n\
  --trained-init-control\n\
  --separability-analysis\n\
  --target-feedback-switch N\n\
  --render-attribution\n\n\
Output/performance:\n\
  --analysis-no-audio\n\
  --analysis-terminal compact|rich|quiet\n\
  --analysis-expensive\n\
  --analysis-fast-provenance\n\
  --analysis-overwrite\n\n\
Analysis never constructs an optimizer, never runs backward, and never writes canonical checkpoints.\n\
Audio has no direct reference input; target experiments are explicitly labeled error-feedback probes."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(items: &[&str]) -> Vec<String> {
        items.iter().map(|item| (*item).to_string()).collect()
    }

    #[test]
    fn analysis_defaults_expand_explicitly() -> Result<()> {
        let request = parse(&strings(&["titan", "--analysis-only", "--base-dir", "/x"]))?;
        assert!(request.model_stats);
        assert_eq!(request.frozen_rollouts, vec![256]);
        Ok(())
    }

    #[test]
    fn training_mutations_are_rejected() {
        let error = parse(&strings(&[
            "titan",
            "--analysis-only",
            "--fresh-world",
            "--model-stats",
        ]))
        .unwrap_err();
        assert!(error.to_string().contains("incompatible"));
    }

    #[test]
    fn analysis_flags_require_analysis_only() {
        let error = parse(&strings(&["titan", "--model-stats"])).unwrap_err();
        assert!(error.to_string().contains("require --analysis-only"));
    }
}
