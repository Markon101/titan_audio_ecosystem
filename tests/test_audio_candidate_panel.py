"""Integrity tests for generic blinded TITAN WAV candidate packages."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audio_candidate_panel as candidate_panel  # noqa: E402


def make_wav(path: Path, amplitude: int, frames: int = 8192) -> None:
    samples = candidate_panel.panel.array("h")
    for index in range(frames):
        sign = 1 if index % 48 < 24 else -1
        samples.extend((sign * amplitude, sign * amplitude // 3))
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(48_000)
        writer.writeframes(candidate_panel.panel.pcm_bytes(samples))


class CandidatePanelTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.executable = self.root / "titan"
        self.executable.write_bytes(b"declared executable bytes")
        self.arms = []
        for index, name in enumerate(("legacy", "msfield", "control")):
            wav = self.root / f"{name}.wav"
            checkpoint = self.root / f"{name}.safetensors"
            world = self.root / f"{name}.bin"
            metadata = self.root / f"{name}.json"
            make_wav(wav, 2500 + 700 * index)
            checkpoint.write_bytes(f"weights-{name}".encode())
            world.write_bytes(f"state-{name}".encode())
            metadata.write_text(json.dumps({
                "run_id": f"run-{name}", "build": {"git_commit": "abc", "git_dirty": True},
                "run": {"start_global_step": 0, "end_global_step": 2,
                        "completed_chunks": 2},
                "field": {"chunk_size": 4096},
                "outputs": {"audio": str(wav), "model": str(checkpoint),
                            "world": str(world)},
                "corpus": {"wav_dir": "/example/OLD_WAVS"},
            }))
            self.arms.append({"id": name, "wav": str(wav), "start_frame": 512,
                              "checkpoint": str(checkpoint), "world": str(world),
                              "executable": str(self.executable),
                              "run_metadata": str(metadata),
                              "processing_note": "raw run output"})

    def config(self, names=("legacy", "msfield"), clip_frames=4096) -> Path:
        config = self.root / f"config_{len(names)}_{clip_frames}.json"
        arms = [arm for arm in self.arms if arm["id"] in names]
        config.write_text(json.dumps({"seed": 19, "clip_frames": clip_frames,
                                      "candidates": arms}))
        return config

    def test_two_arm_panel_is_deterministic_and_preserves_inputs(self) -> None:
        config = self.config()
        inputs = [config, self.executable, *self.root.glob("*.wav"), *self.root.glob("*.safetensors"),
                  *self.root.glob("*.bin"), *self.root.glob("*.json")]
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
        first = candidate_panel.build_panel(candidate_panel.load_config(config), self.root / "out1")
        second = candidate_panel.build_panel(candidate_panel.load_config(config), self.root / "out2")
        self.assertEqual([x["candidate_id"] for x in first["key"]],
                         [x["candidate_id"] for x in second["key"]])
        self.assertEqual([x["blind_audio"]["sha256"] for x in first["key"]],
                         [x["blind_audio"]["sha256"] for x in second["key"]])
        self.assertLess(first["maximum_level_difference_db"], 0.01)
        self.assertFalse(first["strict_state_or_weight_equivalence"])
        self.assertFalse((self.root / "out1" / "BLIND" / "key.json").exists())
        self.assertTrue((self.root / "out1" / "key.json").exists())
        self.assertFalse((self.root / "out1" / "INCOMPLETE").exists())
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
        for item in first["key"]:
            self.assertEqual(item["source_interval_frames"], [512, 4608])
            self.assertTrue(all(item["run_declaration"]["declared_files_match_reported_outputs"].values()))
            self.assertEqual(item["declared_executable"]["sha256"],
                             hashlib.sha256(self.executable.read_bytes()).hexdigest())
            self.assertLessEqual(item["descriptive_pcm"]["clipped_frames"], 0)
        template = json.loads((self.root / "out1" / "downstream_receipt_template.json").read_text())
        self.assertEqual(len(template["attempts"]), 2)
        self.assertIn("suno_model_version", template["attempts"][0])

    def test_three_arm_panel_and_source_bounds(self) -> None:
        config = self.config(("legacy", "msfield", "control"))
        receipt = candidate_panel.build_panel(candidate_panel.load_config(config), self.root / "three")
        self.assertEqual({x["anonymous_id"] for x in receipt["key"]}, {"A", "B", "C"})
        too_long = self.config(("legacy", "msfield"), clip_frames=8193)
        with self.assertRaisesRegex(ValueError, "extends beyond"):
            candidate_panel.load_config(too_long)

    def test_metadata_mismatch_rejected_before_creating_output(self) -> None:
        config = self.config()
        metadata = self.root / "legacy.json"
        value = json.loads(metadata.read_text())
        value["outputs"]["audio"] = str(self.root / "msfield.wav")
        metadata.write_text(json.dumps(value))
        output = self.root / "bad"
        with self.assertRaisesRegex(ValueError, "differs from run metadata output"):
            candidate_panel.build_panel(candidate_panel.load_config(config), output)
        self.assertFalse(output.exists())

    def test_existing_output_is_never_overwritten(self) -> None:
        config = self.config()
        output = self.root / "existing"
        output.mkdir()
        sentinel = output / "keep.txt"
        sentinel.write_text("keep")
        with self.assertRaises(FileExistsError):
            candidate_panel.build_panel(candidate_panel.load_config(config), output)
        self.assertEqual(sentinel.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
