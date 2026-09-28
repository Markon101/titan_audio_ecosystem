"""Focused integrity tests for the frozen same-world audio package."""

import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import matched_eval_package as matched  # noqa: E402


def make_wav(path: Path, frames: int, amplitude: int) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(48_000)
        samples = matched.panel.array("h")
        for index in range(frames):
            sign = 1 if index % 32 < 16 else -1
            samples.extend((sign * amplitude, sign * amplitude // 2))
        writer.writeframes(matched.panel.pcm_bytes(samples))


class MatchedEvalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base = self.root / "base"
        self.base.mkdir()
        self.corpus = self.root / "corpus"
        self.corpus.mkdir()
        make_wav(self.corpus / "target.wav", 8192, 2000)
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text('{"entries":[{"file":"target.wav","role":"train","family":"x"}]}')
        self.binary = self.root / "titan"
        self.binary.write_bytes(b"test executable")
        self.world = self.root / "world.bin"
        self.world.write_bytes(b"common world")
        self.models = [self.root / "model_0.safetensors", self.root / "model_1.safetensors"]
        for index, model in enumerate(self.models):
            model.write_bytes(f"model {index}".encode())
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps({
            "binary": str(self.binary), "base_dir": str(self.base),
            "corpus_manifest": str(self.manifest), "corpus_dir": str(self.corpus),
            "common_world": str(self.world), "seed": 23, "chunks": 2,
            "candidates": [{"id": "parent", "model": str(self.models[0])},
                           {"id": "child", "model": str(self.models[1])}],
        }))

    def fake_analysis(self, config, output, candidate, *, different_schedule=False,
                      optimizer_constructed=False) -> None:
        directory = output / "analyses" / candidate["id"]
        (directory / "rollouts").mkdir(parents=True)
        index = next(i for i, item in enumerate(config["candidates"])
                     if item["id"] == candidate["id"])
        make_wav(directory / "rollouts" / "post_dsp.wav", 8192, 2000 + 500 * index)
        (directory / "analysis_report.json").write_text(json.dumps({"analysis": {
            "weights_frozen": True, "optimizer_constructed": optimizer_constructed,
            "optimizer_steps": 0, "backward_passes": 0, "analysis_seed": 23},
            "identity": {"build_commit": "test-build"}}))
        (directory / "provenance.json").write_text(json.dumps({
            "analysis_configuration": {
                "frozen_rollouts": [2], "analysis_stride": 1,
                "model_path": str(candidate["model"]),
                "state_path": str(config["common_world"])},
            "paths": {"corpus_manifest": str(config["corpus_manifest"]),
                      "wav_dir": str(config["eval_base"] / "OLD_WAVS")},
            "corpus": {"fast_hash_mode": False,
                       "name_role_candidate_order": ["target.wav"],
                       "entries_in_manifest_order": [{"file": "target.wav", "present": True,
                           "sha256": matched.identity(config["corpus_dir"] / "target.wav")["sha256"]}],
                       "aggregate_identity_sha256": "same-corpus-hash"},
            "checkpoint_files": [{"path": str(config["common_world"]),
                                  "sha256": matched.identity(config["common_world"])["sha256"]}],
            "world": {"global_step": 1234},
        }))
        with (directory / "rollouts" / "trace.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["rollout_offset", "absolute_step",
                                                       "target_file", "target_frame"])
            writer.writeheader()
            for offset in (1, 2):
                writer.writerow({"rollout_offset": offset, "absolute_step": 1234 + offset,
                                 "target_file": "target.wav",
                                 "target_frame": 10 + offset + (index if different_schedule else 0)})

    def test_blind_package_is_deterministic_and_preserves_sources(self) -> None:
        config = matched.load_config(self.config_path)
        hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in [self.world, self.manifest, *self.models]}
        def fake(config, output, candidate):
            self.fake_analysis(config, output, candidate)
        with patch.object(matched, "run_analysis", side_effect=fake):
            first = matched.package(config, self.root / "out1")
            second = matched.package(config, self.root / "out2")
        self.assertEqual(first["target_schedule_sha256"], second["target_schedule_sha256"])
        self.assertEqual([x["candidate_id"] for x in first["key"]],
                         [x["candidate_id"] for x in second["key"]])
        self.assertEqual(first["level_matching"], second["level_matching"])
        self.assertEqual([x["blind_audio"]["sha256"] for x in first["key"]],
                         [x["blind_audio"]["sha256"] for x in second["key"]])
        self.assertLess(first["level_matching"]["maximum_pair_difference_db"], 0.01)
        self.assertEqual(first["corpus_wavs_verified_after"], 1)
        self.assertFalse((self.root / "out1" / "INCOMPLETE").exists())
        self.assertEqual(sorted(x.name for x in (self.root / "out1" / "BLIND").iterdir()
                                if x.suffix == ".wav"), ["A.wav", "B.wav"])
        self.assertEqual(hashes, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in hashes})
        template = json.loads((self.root / "out1" / "downstream_receipt_template.json").read_text())
        self.assertEqual(len(template["attempts"]), 2)
        self.assertIn("suno_custom_model", template["attempts"][0])

    def test_mismatched_schedule_is_rejected_with_incomplete_marker(self) -> None:
        config = matched.load_config(self.config_path)
        def fake(config, output, candidate):
            self.fake_analysis(config, output, candidate, different_schedule=True)
        output = self.root / "mismatch"
        with patch.object(matched, "run_analysis", side_effect=fake):
            with self.assertRaisesRegex(ValueError, "target file/frame schedules differ"):
                matched.package(config, output)
        self.assertTrue((output / "INCOMPLETE").exists())
        self.assertFalse((output / "key.json").exists())

    def test_optimizer_activity_is_rejected(self) -> None:
        config = matched.load_config(self.config_path)
        def fake(config, output, candidate):
            self.fake_analysis(config, output, candidate, optimizer_constructed=True)
        with patch.object(matched, "run_analysis", side_effect=fake):
            with self.assertRaisesRegex(ValueError, "optimizer-free"):
                matched.package(config, self.root / "optimizer")

    def test_refuses_existing_output_before_invoking_analysis(self) -> None:
        output = self.root / "existing"
        output.mkdir()
        sentinel = output / "sentinel"
        sentinel.write_text("keep")
        with patch.object(matched, "run_analysis") as runner:
            with self.assertRaises(FileExistsError):
                matched.package(matched.load_config(self.config_path), output)
            runner.assert_not_called()
        self.assertEqual(sentinel.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
