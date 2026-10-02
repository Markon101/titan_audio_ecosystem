#!/usr/bin/env python3
"""Freeze a compact, source-linked readout from local ignored analysis runs."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    schedule_path = ROOT / "analysis/matched_exposure_20261001/schedule_20261002.json"
    schedule = json.loads(schedule_path.read_text())
    result = {"schema": 1, "parent_step": 58107,
              "recovery_sha256": sha(HERE / "recovery.json"),
              "fixed_target_schedule_sha256": sha(schedule_path), "panels": {}}
    for name in ("panel_w0_256", "panel_exact_rng_w0_256", "panel_exact_rng_w128_256"):
        base = HERE / "runs" / name
        report_path = base / "analysis_report.json"
        report = json.loads(report_path.read_text())
        suite = json.loads((base / "ablations/summary.json").read_text())
        readout = json.loads((base / "audio_readout.json").read_text())
        if not (report["non_mutation"]["unchanged"] and suite["no_op_clone_exact"]
                and report["analysis"]["optimizer_steps"] == 0):
            raise RuntimeError(f"frozen gate failed: {name}")
        conditions = {}
        target_rows_verified = 0
        for condition, values in readout["conditions"].items():
            summary = json.loads((base / "ablations" / condition / "summary.json").read_text())
            horizon = summary["horizons"][-1]
            for row in summary["sampled_steps"]:
                step = row["absolute_step"]
                episode = next((item for item in schedule["episodes"]
                                if item["start_step"] <= step < item["start_step"] + item["chunks"]), None)
                if episode is None:
                    raise RuntimeError(f"uncovered panel target at step {step}")
                if row["target_file"] != f"slot_{episode['slot']:02}.wav" or row["target_frame"] != (
                    episode["source_frame"] + (step - episode["start_step"]) * 4096
                ):
                    raise RuntimeError(f"wrong panel target file/frame at step {step}")
                target_rows_verified += 1
            conditions[condition] = {
                "post_dsp_wav_sha256": sha(base / "ablations" / condition / "post_dsp.wav"),
                "waveform_l2_full": values["waveform_l2_full"],
                "waveform_l2_first_64_chunks": values["waveform_l2_first_64_chunks"],
                "waveform_l2_last_64_chunks": values["waveform_l2_last_64_chunks"],
                "spectral_distribution_log_rms": values["spectral_distribution_log_rms"],
                "envelope_correlation": values["envelope_correlation"],
                "sampled_action_changes": values["sampled_action_changes"],
                "full_world_fingerprint_equal": values["full_world_fingerprint_equal"],
                "final_coarse_state_distance": values["final_coarse_state_distance"],
                "bounded": horizon["bounded"],
                "nonfinite_detected": horizon["nonfinite_detected"],
                "audio_clipping_fraction": horizon["audio"]["clipping_fraction"],
                "final_coarse_rms": values["final_coarse_rms"],
            }
        result["panels"][name] = {
            "report_sha256": sha(report_path),
            "checkpoint_step": report["identity"]["checkpoint_step"],
            "warmup_chunks": json.loads((base / "ablations/full/summary.json").read_text())["warmup_chunks"],
            "horizon_chunks": suite["horizon_chunks"],
            "no_op_clone_exact": True,
            "parent_unchanged": True,
            "optimizer_steps": 0,
            "sampled_training_target_rows_verified": target_rows_verified,
            "conditions": conditions,
        }
    pilot = HERE / "runs/suno_exact_rng_pilot"
    pilot_key = json.loads((pilot / "private/answer_key.json").read_text())
    pilot_validation = json.loads((pilot / "control_validation.json").read_text())
    result["manual_suno_pilot"] = {
        "anonymous_wav_sha256": {path.stem: sha(path) for path in sorted((pilot / "uploads").glob("*.wav"))},
        "private_answer_key_sha256": sha(pilot / "private/answer_key.json"),
        "duration_seconds": pilot_key["duration_seconds"],
        "generation_outputs_recorded": 0,
        "time_boundary_jump_ratio": pilot_validation["time_boundary_jump_ratio"],
        "phase_null_validation": pilot_validation["phase_surrogate_preprocessing"],
    }
    (HERE / "RESULTS.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"panels": list(result["panels"]),
                      "pilot_wavs": len(result["manual_suno_pilot"]["anonymous_wav_sha256"])}))


if __name__ == "__main__":
    main()
