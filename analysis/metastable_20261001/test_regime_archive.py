import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from regime_archive import Archive, calibrate, read_capture


def trajectory():
    rng = np.random.default_rng(17)
    rows = []
    for i in range(144):
        regime = 0 if i < 48 else 1 if i < 96 else 0
        base = regime * 1.5
        views = {
            "field_coarse": (base + 0.025 * rng.normal(size=68)).tolist(),
            "field_fine": (base * 0.8 + 0.025 * rng.normal(size=68)).tolist(),
            "gru": (base * 0.5 + 0.025 * rng.normal(size=32)).tolist(),
            "morphic_l16": (base * 0.6 + 0.025 * rng.normal(size=32)).tolist(),
            "decoder_control": (base * 0.3 + 0.025 * rng.normal(size=41)).tolist(),
            "audio_behavior": (0.2 + base * 0.2 + 0.025 * rng.normal(size=42)).tolist(),
        }
        rows.append({"schema": 1, "global_step": i * 4, "sample_stride": 4, "views": views,
                     "health": 0.8, "stagnation": 0.2, "model_confidence": 0.95,
                     "motif_nearest_distance": 0.01, "motif_candidates": i,
                     "motif_rejected_similarity": i // 2, "motif_rejected_quality": i // 4,
                     "active_morph_depth": 16, "optimizer_updates": i // 8})
    return rows


class ArchiveTests(unittest.TestCase):
    def test_replay_resume_exact_and_persistent_regime(self):
        rows = trajectory()
        cal = calibrate(rows)
        whole = Archive(cal)
        expected = [whole.process(row) for row in rows]
        first = Archive(cal)
        prefix = [first.process(row) for row in rows[:73]]
        restored = Archive(cal, json.loads(json.dumps(first.state)))
        suffix = [restored.process(row) for row in rows[73:]]
        self.assertEqual(expected, prefix + suffix)
        self.assertEqual(whole.state, restored.state)
        self.assertGreaterEqual(len(whole.state["regimes"]), 2)
        self.assertGreater(whole.state["revisit_count"], 0)
        self.assertTrue(any(row["persistent_new_regime"] for row in expected))

    def test_probe_leakage_is_rejected(self):
        row = trajectory()[0]
        row["validation_best_spectral"] = 0.1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "forbidden"):
                list(read_capture([path]))


if __name__ == "__main__":
    unittest.main()
