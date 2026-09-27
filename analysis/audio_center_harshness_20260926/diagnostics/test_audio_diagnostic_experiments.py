"""Focused regression checks for diagnostic alignment and lineage gates."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from audio_diagnostic_experiments import find_exact_prime_alignment, validate_run_lineage


class DiagnosticIntegrityTests(unittest.TestCase):
    def test_prime_alignment_ignores_only_renderer_edge_fades(self):
        rng = np.random.default_rng(3)
        full = rng.integers(-32768, 32767, size=(10_000, 2), dtype=np.int16)
        prime = full[3000:8000].copy()
        prime[:2048] = 0
        prime[-2048:] = 0
        alignment = find_exact_prime_alignment(full, prime)
        self.assertEqual(alignment["start_frame"], 3000)
        self.assertEqual(alignment["end_frame_exclusive"], 8000)

    def test_overwritten_metadata_cannot_be_attributed_to_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "render_abcd.wav"
            with wave.open(str(audio), "wb") as writer:
                writer.setnchannels(2)
                writer.setsampwidth(2)
                writer.setframerate(48000)
                writer.writeframes(bytes(48000 * 4))
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({
                "audio_file_hash": "abcd", "run_id": "later",
                "outputs": {"audio": str(audio)},
                "run": {"completed_chunks": 2, "rendered_seconds": 8192 / 48000,
                        "start_global_step": 10, "end_global_step": 12},
                "telemetry": {"trace_stride_chunks": 10},
            }))
            trace = root / "trace.csv"
            trace.write_text("run_id,step\nlater,10\n")
            lineage = validate_run_lineage(audio, metadata, trace)
            self.assertEqual(lineage["status"], "unavailable")
            self.assertTrue(any("completed_chunks imply" in reason for reason in lineage["reasons"]))


if __name__ == "__main__":
    unittest.main()
