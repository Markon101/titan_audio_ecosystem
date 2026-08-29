use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub(crate) struct AttributionReport {
    pub baseline_raw_renderer_exported: bool,
    pub baseline_post_dsp_exported: bool,
    pub exact_additive_stems: Vec<String>,
    pub intervention_renders: Vec<String>,
    pub deferred_interventions: Vec<String>,
    pub sum_to_full_validation: String,
    pub explanation: String,
}

pub(crate) fn report(raw: bool, post: bool) -> AttributionReport {
    AttributionReport {
        baseline_raw_renderer_exported: raw,
        baseline_post_dsp_exported: post,
        exact_additive_stems: Vec::new(),
        intervention_renders: Vec::new(),
        deferred_interventions: vec![
            "carrier_fm_leave_one_family_out".to_string(),
            "auxiliary_modal_leave_one_family_out".to_string(),
            "regional_scan_leave_one_family_out".to_string(),
            "learned_gated_excitation_leave_one_family_out".to_string(),
            "wavefolder_intervention".to_string(),
            "haas_width_intervention".to_string(),
        ],
        sum_to_full_validation:
            "no component is called a stem because no current isolated component passed an exact sum-to-full test"
                .to_string(),
        explanation:
            "Phase 1 exports the full raw renderer and full post-DSP paths. It does not insert branches into the protected production forward equations merely to manufacture attribution artifacts."
                .to_string(),
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct TargetSwitchReport {
    pub supported: bool,
    pub direct_reference_input: bool,
    pub conditioning: String,
    pub requested_switch_chunk: usize,
    pub first_possible_effect_chunk: usize,
    pub identical_current_chunk_required: bool,
    pub status: String,
    pub interpretation_constraints: Vec<String>,
}

pub(crate) fn target_switch_report(requested_switch_chunk: usize) -> TargetSwitchReport {
    TargetSwitchReport {
        supported: false,
        direct_reference_input: false,
        conditioning: "target_error_feedback".to_string(),
        requested_switch_chunk,
        first_possible_effect_chunk: requested_switch_chunk.saturating_add(1),
        identical_current_chunk_required: true,
        status: "deferred: current target-error feature calculation is embedded in the protected training objective; an exact A/B feedback tape cannot be extracted without a production-parity hook"
            .to_string(),
        interpretation_constraints: vec![
            "Audio model.forward has no target argument".to_string(),
            "a future exact feedback-tape experiment would measure host ecological prehistory, not reconstruction or reference memory"
                .to_string(),
            "no target identity is injected into the current frozen forward chunk".to_string(),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attribution_never_claims_unvalidated_stems() {
        let report = report(true, true);
        assert!(report.exact_additive_stems.is_empty());
        assert!(report.sum_to_full_validation.contains("no component"));
    }

    #[test]
    fn target_switch_is_not_misreported_as_direct_conditioning() {
        let report = target_switch_report(10);
        assert!(!report.direct_reference_input);
        assert_eq!(report.first_possible_effect_chunk, 11);
    }
}
