import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
import wave


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audio_listening_panel as panel  # noqa: E402


def make_wav(path: Path, frames: int, amplitude: int, period: int) -> None:
    samples = []
    for index in range(frames):
        value = round(amplitude * math.sin(2 * math.pi * index / period))
        samples.extend((value, round(value * 0.8)))
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(2)
        writer.setframerate(panel.RATE)
        writer.setsampwidth(2)
        writer.writeframes(panel.pcm_bytes(panel.array("h", samples)))


def make_config(root: Path) -> Path:
    pairs = []
    for index in range(3):
        raw = root / f"source_{index}_raw.wav"
        eq = root / f"source_{index}_eq.wav"
        make_wav(raw, 4096, 26_000 - index * 2_000, 64 + index * 8)
        make_wav(eq, 4096, 15_000 - index * 1_000, 64 + index * 8)
        pairs.append({"id": f"window_{index}", "raw": raw.name, "eq": eq.name})
    config = root / "pairs.json"
    config.write_text(json.dumps({"pairs": pairs}), encoding="utf-8")
    return config


def read_samples(path: Path):
    with wave.open(str(path), "rb") as reader:
        return panel.pcm_samples(reader.readframes(reader.getnframes()))


class AudioListeningPanelTests(unittest.TestCase):
    def test_matched_levels_deterministic_assignment_and_source_immutability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = make_config(root)
            original = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob("*.wav")}
            first = panel.build_panel(config, root / "first", 29)
            second = panel.build_panel(config, root / "second", 29)
            self.assertEqual((root / "first" / "key.json").read_bytes(),
                             (root / "second" / "key.json").read_bytes())
            self.assertEqual(len(first["files"]), 6)
            self.assertEqual({entry["file"] for entry in first["files"]},
                             {f"listen_{label}.wav" for label in panel.LABELS})
            self.assertEqual({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob("*.wav")}, original)
            guide = (root / "first" / "LISTEN.txt").read_text()
            for index in range(3):
                pair_id = f"window_{index}"
                self.assertNotIn(pair_id, guide)
                entries = [entry for entry in first["files"] if entry["pair_id"] == pair_id]
                self.assertEqual({entry["arm"] for entry in entries}, {"raw", "eq"})
                levels = [entry["output"]["rms_pcm"] for entry in entries]
                self.assertLess(abs(20 * math.log10(levels[0] / levels[1])), 0.01)
            for entry in first["files"]:
                wav = root / "first" / entry["file"]
                self.assertEqual(entry["source_after"], entry["source"])
                self.assertEqual(wav.read_bytes(), (root / "second" / entry["file"]).read_bytes())
                self.assertEqual(entry["output"]["sha256"], hashlib.sha256(wav.read_bytes()).hexdigest())
                self.assertLessEqual(entry["output"]["peak_pcm"], panel.HEADROOM_SAMPLE)
                source = read_samples(Path(entry["source"]["path"]))
                output = read_samples(wav)
                self.assertEqual(output[:16].tolist(),
                                 [round(sample * entry["gain"]) for sample in source[:16]])
            all_levels = [entry["output"]["rms_pcm"] for entry in first["files"]]
            self.assertLess(20 * math.log10(max(all_levels) / min(all_levels)), 0.01)
            self.assertEqual(first, second)

    def test_peak_headroom_can_lower_common_rms_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = make_config(root)
            raw = root / "source_0_raw.wav"
            samples = panel.array("h", [0] * (4096 * 2))
            for index in range(0, 4096, 128):
                samples[2 * index] = 32767
                samples[2 * index + 1] = 32767
            with wave.open(str(raw), "wb") as writer:
                writer.setnchannels(2)
                writer.setframerate(panel.RATE)
                writer.setsampwidth(2)
                writer.writeframes(panel.pcm_bytes(samples))
            key = panel.build_panel(config, root / "panel", 4)
            pair = next(item for item in key["pairs"] if item["id"] == "window_0")
            self.assertLess(pair["applied_target_rms_pcm"], pair["requested_target_rms_pcm"])
            for entry in key["files"]:
                self.assertLessEqual(entry["output"]["peak_pcm"], panel.HEADROOM_SAMPLE)

    def test_rejects_overlap_malformed_and_mismatched_frames_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = make_config(root)
            base = json.loads(config.read_text())
            cases = []
            duplicate_path = json.loads(json.dumps(base))
            duplicate_path["pairs"][0]["eq"] = duplicate_path["pairs"][0]["raw"]
            cases.append((duplicate_path, "source path overlap"))
            duplicate_id = json.loads(json.dumps(base))
            duplicate_id["pairs"][1]["id"] = duplicate_id["pairs"][0]["id"].upper()
            cases.append((duplicate_id, "duplicate pair id"))
            malformed = json.loads(json.dumps(base))
            malformed["pairs"][0]["quality"] = "unknown"
            cases.append((malformed, "requires exactly"))
            unequal = json.loads(json.dumps(base))
            short = root / "short.wav"
            make_wav(short, 1024, 12_000, 64)
            unequal["pairs"][0]["eq"] = short.name
            cases.append((unequal, "unequal frame counts"))
            for index, (value, message) in enumerate(cases):
                candidate = root / f"config_{index}.json"
                candidate.write_text(json.dumps(value), encoding="utf-8")
                output = root / f"out_{index}"
                with self.subTest(index=index), self.assertRaisesRegex(ValueError, message):
                    panel.build_panel(candidate, output, 5)
                self.assertFalse(output.exists())

    def test_existing_directory_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = make_config(root)
            output = root / "panel"
            output.mkdir()
            sentinel = output / "key.json"
            sentinel.write_text("original", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                panel.build_panel(config, output, 8)
            self.assertEqual(sentinel.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
