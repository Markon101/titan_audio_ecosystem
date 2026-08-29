use super::load::LoadedOrigin;
use super::rollout::{self, RolloutRun};
use super::state::AnalysisWorld;
use super::step::Intervention;
use anyhow::Result;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub(crate) struct TrainedInitializationReport {
    pub primary_comparison: String,
    pub conditions: Vec<serde_json::Value>,
    pub comparisons: Vec<serde_json::Value>,
    pub active_depth_confound_controlled: bool,
    pub optimizer_steps: usize,
    pub interpretation_constraints: Vec<String>,
}

pub(crate) struct ControlRuns {
    pub report: TrainedInitializationReport,
    pub runs: Vec<(String, RolloutRun)>,
}

pub(crate) fn run(
    origin: &LoadedOrigin,
    horizon: usize,
    stride: usize,
    analysis_seed: u64,
    progress: impl Fn(&str, usize, usize),
) -> Result<ControlRuns> {
    let full = Intervention::full();
    let trained_saved = rollout::run(
        origin,
        "trained_saved_world",
        &full,
        &[horizon],
        stride,
        false,
        analysis_seed,
        false,
        None,
        |offset, total, _| progress("trained_saved_world", offset, total),
    )?;
    let trained_fresh = rollout::run(
        origin,
        "trained_fresh_world",
        &full,
        &[horizon],
        stride,
        false,
        analysis_seed,
        true,
        None,
        |offset, total, _| progress("trained_fresh_world", offset, total),
    )?;
    let initialized_matched = rollout::run(
        origin,
        "init_depth_matched_fresh_world",
        &full,
        &[horizon],
        stride,
        true,
        analysis_seed,
        true,
        None,
        |offset, total, _| progress("init_depth_matched_fresh_world", offset, total),
    )?;
    let native_transform = |world: &mut AnalysisWorld| {
        world.bundle.model.set_depth(1);
        world.micro = world.bundle.model.project_manifold(&world.micro)?;
        world.macro_t = world.bundle.model.project_manifold(&world.macro_t)?;
        Ok(())
    };
    let initialized_native = rollout::run(
        origin,
        "init_native_depth_fresh_world",
        &full,
        &[horizon],
        stride,
        true,
        analysis_seed,
        true,
        Some(&native_transform),
        |offset, total, _| progress("init_native_depth_fresh_world", offset, total),
    )?;
    let runs = vec![
        ("trained_saved_world".to_string(), trained_saved),
        ("trained_fresh_world".to_string(), trained_fresh),
        (
            "init_depth_matched_fresh_world".to_string(),
            initialized_matched,
        ),
        (
            "init_native_depth_fresh_world".to_string(),
            initialized_native,
        ),
    ];
    let conditions = runs
        .iter()
        .map(|(name, run)| {
            serde_json::json!({
                "name": name,
                "weights": if name.starts_with("init_") {"deterministic_v9_initialization"} else {"checkpoint"},
                "world": if name.ends_with("saved_world") {"saved_clone"} else {"deterministic_fresh_analysis_world"},
                "physical_depth": origin.architecture.morph_blocks,
                "active_depth": run.summary.sampled_steps.first().map(|step| step.morph_active_depth),
                "analysis_seed": analysis_seed,
                "optimizer_steps": 0,
            })
        })
        .collect();
    let comparisons = vec![
        named_comparison(&runs[1], &runs[2], "primary_learned_weights_effect"),
        named_comparison(&runs[0], &runs[1], "carried_world_effect"),
        named_comparison(&runs[2], &runs[3], "active_depth_confound_demonstration"),
    ];
    Ok(ControlRuns {
        report: TrainedInitializationReport {
            primary_comparison:
                "trained_fresh_world versus init_depth_matched_fresh_world".to_string(),
            conditions,
            comparisons,
            active_depth_confound_controlled: true,
            optimizer_steps: 0,
            interpretation_constraints: vec![
                "trained_saved_world versus trained_fresh_world measures carried ecology, not training alone"
                    .to_string(),
                "init_native_depth_fresh_world is descriptive only because active capacity differs"
                    .to_string(),
                "all comparisons use frozen weights and zero optimizer steps".to_string(),
            ],
        },
        runs,
    })
}

fn named_comparison(
    left: &(String, RolloutRun),
    right: &(String, RolloutRun),
    estimand: &str,
) -> serde_json::Value {
    serde_json::json!({
        "estimand": estimand,
        "left": left.0,
        "right": right.0,
        "result": rollout::comparison(&left.1, &right.1),
    })
}
