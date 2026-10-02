#!/usr/bin/env python3
"""Freeze exact-parent host-clamp comparisons after both local panels finish."""
import hashlib
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = ROOT / "analysis/frozen_v10_20261002/runs"
SCHEDULE = ROOT / "analysis/matched_exposure_20261001/schedule_20261002.json"
PAIRS = {
    "coarse_hold_control_energy_replay": "coarse_hold",
    "gru_hold_control_energy_replay": "gru_hold",
    "morphic_upper_bypass_control_energy_replay": "morphic_upper_bypass",
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text())


def main():
    schedule = load(SCHEDULE)
    result = {"schema": 1, "source_parent_step": 58107,
              "schedule_sha256": sha(SCHEDULE), "panels": {}}
    for warmup in (0, 128):
        new = HERE / "runs" / f"w{warmup}_256"
        old = PREVIOUS / f"panel_exact_rng_w{warmup}_256"
        report = load(new / "analysis_report.json")
        suite = load(new / "ablations/summary.json")
        old_suite = load(old / "ablations/summary.json")
        readout = load(new / "audio_readout.json")
        prior = load(old / "audio_readout.json")
        if not (report["non_mutation"]["unchanged"] and
                report["analysis"]["optimizer_steps"] == 0 and
                suite["no_op_clone_exact"]):
            raise RuntimeError(f"parent/no-op/optimizer gate failed: warmup {warmup}")
        baseline_hashes = {}
        for name in ("raw_renderer.wav", "post_dsp.wav"):
            new_hash = sha(new / "ablations/full" / name)
            prior_hash = sha(old / "ablations/full" / name)
            if new_hash != prior_hash:
                raise RuntimeError(f"unmodified baseline changed at warmup {warmup}: {name}")
            baseline_hashes[name] = new_hash
        for no_op in ("none", "none_control_energy_replay"):
            values = readout["conditions"][no_op]
            summary = load(new / "ablations" / no_op / "summary.json")
            baseline = load(new / "ablations/full/summary.json")
            if (values["waveform_l2_full"] != 0.0 or
                    not values["full_world_fingerprint_equal"] or
                    summary["final_world_fingerprint"] != baseline["final_world_fingerprint"]):
                raise RuntimeError(f"{no_op} failed at warmup {warmup}")
        rows = 0
        for condition in readout["conditions"]:
            summary = load(new / "ablations" / condition / "summary.json")
            for row in summary["sampled_steps"]:
                step = row["absolute_step"]
                episode = next((item for item in schedule["episodes"]
                                if item["start_step"] <= step < item["start_step"] + item["chunks"]), None)
                if episode is None or row["target_file"] != f"slot_{episode['slot']:02}.wav" or (
                    row["target_frame"] != episode["source_frame"] +
                    (step - episode["start_step"]) * 4096
                ):
                    raise RuntimeError(f"wrong target at step {step}")
                rows += 1
        arms = {}
        for new_name, old_name in PAIRS.items():
            values = readout["conditions"][new_name]
            previous = prior["conditions"][old_name]
            new_comparison = next(item for item in suite["conditions"] if item["name"] == new_name)
            old_comparison = next(item for item in old_suite["conditions"] if item["name"] == old_name)
            def median_audio(comparison, metric):
                return statistics.median(point["audio_distance"][metric]
                                         for point in comparison["comparison"]["points"]
                                         if point["offset"] > 0)
            summary = load(new / "ablations" / new_name / "summary.json")
            horizon = summary["horizons"][-1]
            arms[new_name] = {
                "post_dsp_wav_sha256": sha(new / "ablations" / new_name / "post_dsp.wav"),
                "waveform_l2_full": values["waveform_l2_full"],
                "waveform_l2_first_64_chunks": values["waveform_l2_first_64_chunks"],
                "waveform_l2_last_64_chunks": values["waveform_l2_last_64_chunks"],
                "prior_closed_loop_waveform_l2_full": previous["waveform_l2_full"],
                "open_to_closed_loop_l2_ratio": values["waveform_l2_full"] /
                    max(1e-12, previous["waveform_l2_full"]),
                "spectral_distribution_log_rms": values["spectral_distribution_log_rms"],
                "sampled_multiresolution_log_spectral_median": median_audio(
                    new_comparison, "multiresolution_log_spectral"),
                "prior_closed_loop_sampled_multiresolution_log_spectral_median": median_audio(
                    old_comparison, "multiresolution_log_spectral"),
                "sampled_temporal_envelope_median": median_audio(new_comparison, "temporal_envelope"),
                "envelope_correlation": values["envelope_correlation"],
                "sampled_action_changes": values["sampled_action_changes"],
                "final_coarse_state_distance": values["final_coarse_state_distance"],
                "bounded": horizon["bounded"],
                "nonfinite_detected": horizon["nonfinite_detected"],
                "audio_clipping_fraction": horizon["audio"]["clipping_fraction"],
            }
        result["panels"][f"warmup_{warmup}"] = {
            "report_sha256": sha(new / "analysis_report.json"),
            "previous_baseline_audio_byte_identical": True,
            "baseline_audio_sha256": baseline_hashes,
            "ordinary_no_op_exact": True,
            "control_energy_replay_no_op_exact": True,
            "parent_unchanged": True,
            "optimizer_steps": 0,
            "sampled_train_target_rows_verified": rows,
            "arms": arms,
        }
    pilot = HERE / "runs/suno_quietcut_pilot"
    quiet_validation = load(pilot / "control_validation.json")
    old_validation = load(PREVIOUS / "suno_exact_rng_pilot/control_validation.json")
    result["manual_suno_pilot"] = {
        "anonymous_wav_sha256": {
            path.stem: sha(path) for path in sorted((pilot / "uploads").glob("*.wav"))
        },
        "private_answer_key_sha256": sha(pilot / "private/answer_key.json"),
        "generation_outputs_recorded": 0,
        "time_fraction_moved": quiet_validation["time_fraction_moved"],
        "time_interior_block_correlation": quiet_validation["time_median_interior_block_correlation"],
        "quiet_cut_boundary_jump_ratio": quiet_validation["time_boundary_jump_ratio"],
        "fixed_cut_boundary_jump_ratio": old_validation["time_boundary_jump_ratio"],
        "quiet_cut_average_log_spectral_rms": quiet_validation["time_average_log_spectral_rms"],
        "fixed_cut_average_log_spectral_rms": old_validation["time_average_log_spectral_rms"],
        "phase_surrogate_preprocessing": quiet_validation["phase_surrogate_preprocessing"],
    }
    (HERE / "RESULTS.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"panels": list(result["panels"]),
                      "baseline_parity": True, "no_op_replay": True}))


if __name__ == "__main__":
    main()
