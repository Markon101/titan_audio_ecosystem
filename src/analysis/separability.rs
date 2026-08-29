use super::metrics;
use super::rollout::RolloutRun;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub(crate) struct PairwiseDistance {
    pub left: String,
    pub right: String,
    pub state_distance: metrics::StateDistance,
    pub audio_distance: metrics::AudioDistance,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ConditionDrift {
    pub condition: String,
    pub state_drift: metrics::StateDistance,
    pub audio_drift: metrics::AudioDistance,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct SeparabilityReport {
    pub conditions: Vec<String>,
    pub pairwise: Vec<PairwiseDistance>,
    pub within_condition_drift: Vec<ConditionDrift>,
    pub mean_between_state_distance: Option<f32>,
    pub mean_within_state_drift: Option<f32>,
    pub between_within_ratio: Option<f32>,
    pub operational_classification: String,
    pub interpretation_constraints: Vec<String>,
}

pub(crate) fn analyze(runs: &[(&str, &RolloutRun)]) -> SeparabilityReport {
    let mut pairwise = Vec::new();
    for left_index in 0..runs.len() {
        for right_index in left_index + 1..runs.len() {
            let (left_name, left_run) = runs[left_index];
            let (right_name, right_run) = runs[right_index];
            let Some(left) = left_run.frames.last() else {
                continue;
            };
            let Some(right) = right_run.frames.last() else {
                continue;
            };
            pairwise.push(PairwiseDistance {
                left: left_name.to_string(),
                right: right_name.to_string(),
                state_distance: metrics::state_distance(
                    &left.micro,
                    &left.macro_t,
                    &left.hidden,
                    &right.micro,
                    &right.macro_t,
                    &right.hidden,
                ),
                audio_distance: metrics::audio_distance(
                    &left.left,
                    &left.right,
                    &right.left,
                    &right.right,
                ),
            });
        }
    }
    let within_condition_drift: Vec<ConditionDrift> = runs
        .iter()
        .filter_map(|(name, run)| {
            let first = run.frames.iter().find(|frame| frame.offset > 0)?;
            let last = run.frames.last()?;
            Some(ConditionDrift {
                condition: (*name).to_string(),
                state_drift: metrics::state_distance(
                    &last.micro,
                    &last.macro_t,
                    &last.hidden,
                    &first.micro,
                    &first.macro_t,
                    &first.hidden,
                ),
                audio_drift: metrics::audio_distance(
                    &last.left,
                    &last.right,
                    &first.left,
                    &first.right,
                ),
            })
        })
        .collect();
    let between_values: Vec<f32> = pairwise
        .iter()
        .filter_map(|distance| distance.state_distance.combined_relative_l2)
        .collect();
    let within_values: Vec<f32> = within_condition_drift
        .iter()
        .filter_map(|drift| drift.state_drift.combined_relative_l2)
        .collect();
    let mean = |values: &[f32]| {
        (!values.is_empty()).then(|| values.iter().sum::<f32>() / values.len() as f32)
    };
    let mean_between = mean(&between_values);
    let mean_within = mean(&within_values);
    let ratio = mean_between
        .zip(mean_within)
        .and_then(|(between, within)| (within > 1e-9).then_some(between / within));
    let classification = match (mean_between, mean_within) {
        (Some(between), Some(within)) if between < 0.1 && within < 0.1 => {
            "generic_stable_phenotype_candidate"
        }
        (Some(between), Some(within)) if between < within * 0.5 => "shared_drifting_family",
        (Some(between), Some(within)) if between > within * 2.0 && within < 0.5 => {
            "differentiated_stable_phenotype_families"
        }
        (Some(_), Some(_)) => "differentiated_or_drifting_trajectories",
        _ => "insufficient_valid_conditions",
    };
    SeparabilityReport {
        conditions: runs.iter().map(|(name, _)| (*name).to_string()).collect(),
        pairwise,
        within_condition_drift,
        mean_between_state_distance: mean_between,
        mean_within_state_drift: mean_within,
        between_within_ratio: ratio,
        operational_classification: classification.to_string(),
        interpretation_constraints: vec![
            "descriptor clusters are operational regimes, not concepts".to_string(),
            "finite trajectories cannot distinguish separate basins from slow mixing".to_string(),
            "waveform metrics are interpreted only for sample-aligned common-origin conditions"
                .to_string(),
        ],
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn classification_vocabulary_is_conservative() {
        let allowed = [
            "generic_stable_phenotype_candidate",
            "shared_drifting_family",
            "differentiated_stable_phenotype_families",
            "differentiated_or_drifting_trajectories",
            "insufficient_valid_conditions",
        ];
        assert!(allowed.iter().all(|label| !label.contains("concept")));
    }
}
