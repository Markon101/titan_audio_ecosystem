use super::load::LoadedOrigin;
use super::rollout::{self, RolloutRun};
use super::step::Intervention;
use anyhow::Result;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub(crate) struct AblationCondition {
    pub name: String,
    pub subsystem_class: String,
    pub exact_disabled_behavior: String,
    pub comparison: serde_json::Value,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct AblationSuite {
    pub baseline: String,
    pub horizon_chunks: usize,
    pub forcing_protocol: String,
    pub controller_protocol: String,
    pub conditions: Vec<AblationCondition>,
    pub deferred_without_production_forward_hooks: Vec<String>,
    pub warnings: Vec<String>,
}

pub(crate) struct AblationRuns {
    pub suite: AblationSuite,
    pub baseline: RolloutRun,
}

pub(crate) fn run_suite(
    origin: &LoadedOrigin,
    horizon: usize,
    stride: usize,
    requested: &[String],
    analysis_seed: u64,
    progress: impl Fn(&str, usize, usize),
    mut export: impl FnMut(&str, &RolloutRun) -> Result<()>,
) -> Result<AblationRuns> {
    let baseline = rollout::run(
        origin,
        "ablation_full",
        &Intervention::full(),
        &[horizon],
        stride,
        false,
        analysis_seed,
        false,
        None,
        |offset, total, _| progress("ablation_full", offset, total),
    )?;
    let names = if requested.is_empty() {
        vec![
            "micro_nca_hold".to_string(),
            "macro_nca_hold".to_string(),
            "gru_hold".to_string(),
            "episodic_read_zero".to_string(),
            "planner_model_zero".to_string(),
            "bandit_zero".to_string(),
            "motif_recall_disabled".to_string(),
            "radiation_disabled".to_string(),
            "structured_shear_disabled".to_string(),
            "micro_kick_disabled".to_string(),
            "potential_gains_identity".to_string(),
            "target_independent_feedback_ablation".to_string(),
            "arbiter_bypass".to_string(),
        ]
    } else {
        requested.to_vec()
    };
    let mut conditions = Vec::new();
    for name in names {
        let intervention = Intervention::parse(&name)?;
        let run = rollout::run(
            origin,
            &name,
            &intervention,
            &[horizon],
            stride,
            false,
            analysis_seed,
            false,
            None,
            |offset, total, _| progress(&name, offset, total),
        )?;
        conditions.push(AblationCondition {
            name: name.clone(),
            subsystem_class: intervention.subsystem_class().to_string(),
            exact_disabled_behavior: definition(&name).to_string(),
            comparison: rollout::comparison(&baseline, &run),
        });
        export(&name, &run)?;
    }
    Ok(AblationRuns {
        suite: AblationSuite {
            baseline: baseline.summary.condition.clone(),
            horizon_chunks: horizon,
            forcing_protocol:
                "force-macro, radiation, and kick directions are deterministic functions of checkpoint RNG identity, absolute step, and stream tag"
                    .to_string(),
            controller_protocol: "closed_loop_with_separate_matched_controller_rng".to_string(),
            conditions,
            deferred_without_production_forward_hooks: vec![
                "morphic_bypass".to_string(),
                "memory_to_micro_zero".to_string(),
                "temporal_decoder_neutral".to_string(),
                "carrier_fm_family".to_string(),
                "auxiliary_modal_family".to_string(),
                "regional_scan_partial_family".to_string(),
                "learned_gated_excitation_family".to_string(),
                "wavefolder_family".to_string(),
                "haas_width_path".to_string(),
                "saturation_or_dc_blocker".to_string(),
            ],
            warnings: vec![
                "Transition holds are applied to the cloned next state; the intervention can affect audio from the following chunk onward."
                    .to_string(),
                "Host controller and forcing interventions are not learned-neural ablations."
                    .to_string(),
                "Unsafe renderer hooks were not inserted into the production forward path; unsupported names fail rather than silently approximating a stem."
                    .to_string(),
            ],
        },
        baseline,
    })
}

fn definition(name: &str) -> &'static str {
    match name {
        "micro_nca_hold" => "retain the prior projected micro field instead of committing the learned micro CA next state; subsequent declared forcing remains active",
        "macro_nca_hold" => "retain the prior macro field instead of committing the learned macro CA next state at eligible updates",
        "gru_hold" => "retain the prior 512-value recurrent state instead of committing the GRU next state",
        "episodic_read_zero" => "replace the learned 64-value episodic readout with zeros while leaving slot storage intact",
        "episodic_empty" => "prevent episodic read and snapshot effects in the cloned condition",
        "planner_model_zero" => "replace cached learned world-model action scores with zeros while leaving bandit and controller logic active",
        "bandit_zero" => "zero and hold model-free action values while leaving learned planner scores active",
        "motif_recall_disabled" => "disable motif availability and recall while retaining observational motif state",
        "radiation_disabled" => "draw the radiation hazard but do not apply the heavy-tailed micro-state event",
        "structured_shear_disabled" => "set clone-only structured macro shear amplitude to zero",
        "micro_kick_disabled" => "set clone-only Gaussian micro kick amplitude to zero",
        "potential_gains_identity" => "replace clone-only potential micro/macro amplitude gains with identity",
        "target_independent_feedback_ablation" => "do not sample or apply corpus target-error feedback; direct forward remains target-independent",
        "arbiter_bypass" => "training-only null: the frozen runner has no objective mixing, backward pass, or optimizer path",
        _ => "typed cloned-state intervention",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_and_neural_ablation_definitions_do_not_conflate_types() {
        assert!(definition("gru_hold").contains("GRU"));
        assert!(definition("radiation_disabled").contains("hazard"));
        assert_ne!(
            Intervention::parse("gru_hold").unwrap().subsystem_class(),
            Intervention::parse("radiation_disabled")
                .unwrap()
                .subsystem_class()
        );
    }
}
