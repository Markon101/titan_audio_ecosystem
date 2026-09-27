import argparse
from array import array
import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prime_package as prime  # noqa: E402


def make_wav(path: Path, frames: int = 96_000) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(48_000)
        samples = array("h")
        for index in range(frames):
            samples.extend((int(10_000 * (index % 100) / 100), int(8_000 * (index % 79) / 79)))
        writer.writeframes(prime.pcm16_bytes(samples))


def request(source: Path, output: Path, **overrides) -> argparse.Namespace:
    fields = dict(
        source_wav=source,
        output_dir=output,
        clip_seconds=1.0,
        seed=17,
        metadata_json=None,
        corpus_manifest=None,
        corpus_wav_dir=source.parent,
        legacy_prime=None,
        legacy_prompt=None,
    )
    fields.update(overrides)
    return argparse.Namespace(**fields)


class PrimePackageTests(unittest.TestCase):
    def test_deterministic_clips_and_read_only_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            make_wav(source)
            before = prime.file_identity(source)
            first = prime.package(request(source, root / "first"))
            second = prime.package(request(source, root / "second"))
            self.assertEqual(before, prime.file_identity(source))
            self.assertEqual(first["source_wav"], second["source_wav"])
            self.assertEqual(
                (root / "first" / "receipt.json").read_bytes(),
                (root / "second" / "receipt.json").read_bytes(),
            )
            self.assertEqual(
                [(clip["selection_labels"], clip["start_frame"], clip["wav"]["sha256"]) for clip in first["clips"]],
                [(clip["selection_labels"], clip["start_frame"], clip["wav"]["sha256"]) for clip in second["clips"]],
            )
            self.assertEqual(first["clips"][0]["start_frame"], 0)
            self.assertEqual(first["clips"][0]["end_frame_exclusive"], 48_000)
            self.assertEqual(first["clips"][0]["signal_checks"]["clipped_frames"], 0)
            self.assertTrue((root / "first" / "receipt.json").is_file())
            with self.assertRaises(FileExistsError):
                prime.package(request(source, root / "first"))

    def test_corpus_audit_reports_declared_and_effective_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_wav(root / "ordinary.wav", frames=4_096)
            make_wav(root / "titan_prime_example.wav", frames=4_096)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"entries": [
                {"file": "ordinary.wav", "family": "ordinary", "role": "train", "provenance": "user_corpus"},
                {"file": "titan_prime_example.wav", "family": "generated", "role": "validation", "provenance": "titan_generated_quarantine"},
            ]}), encoding="utf-8")
            audit = prime.corpus_audit(manifest, root)
            self.assertEqual(audit["declared_role_counts"], {"train": 1, "validation": 1})
            self.assertEqual(audit["present_name_role_candidates"], {"train": 1})
            self.assertTrue(audit["quarantine_role_conflicts"][0]["production_filename_filter_excludes"])

    def test_metadata_audio_mismatch_fails_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            other = root / "other.wav"
            make_wav(source)
            make_wav(other, frames=48_000)
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({"outputs": {"audio": str(other)}}), encoding="utf-8")
            output = root / "package"
            with self.assertRaisesRegex(ValueError, "do not match"):
                prime.package(request(source, output, metadata_json=metadata))
            self.assertFalse(output.exists())

    def test_candidate_clips_and_descriptor_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            cand_wav = root / "candidate.wav"
            make_wav(source, frames=48_000)
            make_wav(cand_wav, frames=48_000)
            output = root / "package"
            req = request(
                source,
                output,
                candidate_clip=[f"wide_prime={cand_wav}"],
                include_descriptor_prompt=True,
            )
            result = prime.package(req)
            self.assertTrue((output / "clips" / "wide_prime.wav").is_file())
            self.assertTrue((output / "prompts" / "audible_descriptors.txt").is_file())
            clip_ids = [c["id"] for c in result["clips"]]
            self.assertIn("wide_prime", clip_ids)
            prompt_ids = [p["id"] for p in result["prompts"]]
            self.assertIn("neutral", prompt_ids)
            self.assertIn("audible_descriptors", prompt_ids)
            descriptor = (output / "prompts" / "audible_descriptors.txt").read_text()
            self.assertIn("Spatial image: \n", descriptor)
            self.assertNotIn("clear stereo separation", descriptor)
            descriptor_receipt = next(p for p in result["prompts"] if p["id"] == "audible_descriptors")
            self.assertEqual(descriptor_receipt["status"], "blank_listener_template")
            self.assertIn("verify", descriptor_receipt["warning"])

    def test_invalid_candidate_labels_fail_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            make_wav(source, frames=4_096)
            bad_specs = (
                ([f"../escaped={source}"], "invalid candidate label"),
                ([f"clip_01={source}"], "reserved candidate label"),
                ([f"legacy_prime={source}"], "reserved candidate label"),
                ([f"wide={source}", f"WIDE={source}"], "duplicate candidate label"),
                ([f"{'a' * 49}={source}"], "invalid candidate label"),
            )
            for index, (specs, message) in enumerate(bad_specs):
                output = root / f"package_{index}"
                with self.subTest(specs=specs), self.assertRaisesRegex(ValueError, message):
                    prime.package(request(source, output, candidate_clip=specs))
                self.assertFalse(output.exists())
            self.assertFalse((root / "escaped.wav").exists())

    def test_provided_descriptors_remain_unverified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            descriptor = root / "descriptor.txt"
            make_wav(source, frames=4_096)
            descriptor.write_text("A listener-supplied description.\n", encoding="utf-8")
            result = prime.package(request(source, root / "package", descriptor_prompt=descriptor))
            entry = next(p for p in result["prompts"] if p["id"] == "audible_descriptors")
            self.assertEqual(entry["status"], "provided_unverified")
            self.assertEqual(entry["source"]["sha256"], prime.file_identity(descriptor)["sha256"])

    def test_metadata_run_frames_must_match_audio_even_when_path_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            make_wav(source, frames=96_000)
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({
                "outputs": {"audio": str(source)},
                "field": {"chunk_size": 4096},
                "run": {"completed_chunks": 112, "rendered_seconds": 112 * 4096 / 48_000},
            }), encoding="utf-8")
            output = root / "package"
            with self.assertRaisesRegex(ValueError, "frame count does not match"):
                prime.package(request(source, output, metadata_json=metadata))
            self.assertFalse(output.exists())

    def test_metadata_matching_run_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            make_wav(source, frames=2 * 4096)
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({
                "outputs": {"audio": str(source)},
                "field": {"chunk_size": 4096},
                "run": {"completed_chunks": 2, "rendered_seconds": 2 * 4096 / 48_000},
            }), encoding="utf-8")
            result = prime.package(request(source, root / "package", metadata_json=metadata))
            self.assertTrue(result["metadata_audio_bytes_match"])
            self.assertEqual(result["run_metadata_reported_run"]["completed_chunks"], 2)


if __name__ == "__main__":
    unittest.main()
