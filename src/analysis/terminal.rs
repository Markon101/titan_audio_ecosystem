use super::config::TerminalMode;
use super::load::LoadedOrigin;
use super::step::StepRecord;
use std::path::Path;

pub(crate) fn banner(
    mode: TerminalMode,
    analysis_id: &str,
    output: &Path,
    origin: &LoadedOrigin,
    analyses: &[&str],
    analysis_seed: u64,
) {
    if mode == TerminalMode::Quiet {
        return;
    }
    println!("=== TITAN AUDIO SCIENTIFIC INSTRUMENTATION v1 ===");
    println!("Analysis: {analysis_id} · {:?}", analyses);
    println!("FROZEN WEIGHTS · OPTIMIZER STEPS = 0 · BACKWARD PASSES = 0");
    println!(
        "Checkpoint: {} · world step {} · model {}",
        origin.paths.model.display(),
        origin
            .world
            .as_ref()
            .map(|world| world.global_step)
            .unwrap_or(0),
        origin
            .canonical_before
            .iter()
            .find(|identity| identity.path == origin.paths.model.display().to_string())
            .and_then(|identity| identity.sha256.as_deref())
            .unwrap_or("unavailable")
    );
    println!(
        "Morphology: L{:02}/{} x {} · analysis RNG {} · world RNG used only as immutable seed material",
        origin
            .world
            .as_ref()
            .map(|world| world.active_depth)
            .unwrap_or(1),
        origin.architecture.morph_blocks,
        origin.architecture.morph_width,
        analysis_seed,
    );
    println!(
        "Conditioning: no direct reference input; target effects, when present, are post-forward coarse error feedback"
    );
    println!("Artifacts: {}", output.display());
}

pub(crate) fn progress(
    mode: TerminalMode,
    analysis: &str,
    condition: &str,
    offset: usize,
    total: usize,
    record: Option<&StepRecord>,
) {
    if mode == TerminalMode::Quiet {
        return;
    }
    let should_print = mode == TerminalMode::Rich
        || offset == total
        || offset == 1
        || offset.is_multiple_of((total / 8).max(1));
    if !should_print {
        return;
    }
    if let Some(record) = record {
        println!(
            "[{analysis}] {condition} {offset}/{total} · state rms μ/m/h {:.3}/{:.3}/{:.3} · audio rms {:.4} peak {:.4} · action {}{}",
            record.micro_rms,
            record.macro_rms,
            record.recurrent_rms,
            record.audio_post_dsp.rms_mid,
            record.audio_post_dsp.peak,
            record.action,
            if record.target_error_feedback.is_some() {
                " · TARGET-ERROR-FEEDBACK CONDITIONED"
            } else {
                " · TARGET-INDEPENDENT"
            }
        );
    } else {
        println!("[{analysis}] {condition} {offset}/{total}");
    }
}

pub(crate) fn completion(mode: TerminalMode, output: &Path, unchanged: bool, warnings: usize) {
    if mode == TerminalMode::Quiet {
        return;
    }
    println!("=== ANALYSIS COMPLETE ===");
    println!("Canonical checkpoint bytes unchanged: {unchanged}");
    println!(
        "Warnings: {warnings} · report: {}",
        output.join("analysis_report.json").display()
    );
    println!("Weights remained frozen; optimizer steps = 0.");
}

#[cfg(test)]
pub(crate) const TERMINAL_VOCABULARY: &[&str] = &[
    "state contracting",
    "state distance persisting",
    "output robust / state not recovered",
    "approximate recurrence candidate",
    "target-error-feedback conditioned",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automatic_terminal_vocabulary_avoids_prohibited_claims() {
        let prohibited = [
            "self-healed",
            "life",
            "cognition",
            "concept",
            "strange attractor",
        ];
        for label in TERMINAL_VOCABULARY {
            assert!(prohibited.iter().all(|word| !label.contains(word)));
        }
    }
}
