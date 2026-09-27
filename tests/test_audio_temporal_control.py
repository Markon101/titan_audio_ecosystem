import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audio_temporal_control as control  # noqa: E402


def make_blocks(path: Path, block_seconds: int = 1, blocks: int = 5) -> None:
    length = block_seconds * control.RATE
    groups = []
    for index in range(blocks):
        time = np.arange(length) / control.RATE
        amplitude = 0.1 + index * 0.03
        wave_l = amplitude * np.sin(2 * np.pi * (110 + index * 55) * time)
        wave_r = amplitude * np.sin(2 * np.pi * (165 + index * 55) * time)
        groups.append(np.column_stack((wave_l, wave_r)))
    control.write_pcm(path, np.concatenate(groups))


class TemporalControlTests(unittest.TestCase):
    def test_order_shuffle_and_provenance_are_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "B.wav"
            make_blocks(source)
            before = control.sha256(source)
            first = control.make_controls(source, root / "one", 1, 10, 99)
            second = control.make_controls(source, root / "two", 1, 10, 99)
            self.assertEqual(first, second)
            self.assertEqual(control.sha256(source), before)
            self.assertEqual(control.sha256(root / "one/B_original_exact.wav"), before)
            self.assertNotEqual(first["permutation"], list(range(5)))
            self.assertEqual(first["frames"], 5 * control.RATE)
            self.assertLess(first["output_rms_spread_db"], 0.01)
            one = control.read_pcm(root / "one/S_shuffled_4s.wav")
            ordered = control.read_pcm(root / "one/O_ordered_seams.wav")
            self.assertEqual(len(one), len(ordered))
            # Away from the 10 ms seams, the first shuffled block has the
            # waveform of its declared source block after one global gain.
            source_float = control.read_pcm(source)
            original_index = first["permutation"][0]
            a = one[1000:2000, 0]
            b = source_float[original_index * control.RATE + 1000:
                             original_index * control.RATE + 2000, 0]
            self.assertGreater(float(np.corrcoef(a, b)[0, 1]), 0.999)
            for record in first["outputs"]:
                self.assertEqual(control.sha256(root / "one" / record["file"]), record["sha256"])
                self.assertEqual((root / "one" / record["file"]).read_bytes(),
                                 (root / "two" / record["file"]).read_bytes())
            self.assertEqual(first, json.loads((root / "one/receipt.json").read_text()))

    def test_invalid_or_existing_outputs_fail_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "B.wav"
            make_blocks(source)
            output = root / "existing"
            output.mkdir()
            (output / "sentinel").write_text("keep")
            with self.assertRaises(ValueError):
                control.make_controls(source, output, 1, 10, 1)
            self.assertEqual((output / "sentinel").read_text(), "keep")
            with self.assertRaises(ValueError):
                control.make_controls(source, root / "missing", 3, 10, 1)
            self.assertFalse((root / "missing").exists())


if __name__ == "__main__":
    unittest.main()
