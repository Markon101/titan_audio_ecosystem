import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "audio_diffusion_probe", Path(__file__).resolve().parents[1] / "scripts/audio_diffusion_probe.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class AudioDiffusionProbeTests(unittest.TestCase):
    def test_analytic_heat_attenuation_and_zero_bypass(self):
        omega = 2 * np.pi * 6000 / probe.RATE
        sine = np.sin(np.arange(4800) * omega)
        source = np.column_stack((sine, sine * 0.5))
        result = probe.heat_diffusion(source, 0.1, 5)
        attenuation = (1 - 4 * 0.1 * np.sin(omega / 2) ** 2) ** 5
        np.testing.assert_allclose(result[5:-5], source[5:-5] * attenuation, atol=1e-12)
        np.testing.assert_array_equal(probe.heat_diffusion(source, 0, 5), source)
        np.testing.assert_array_equal(probe.heat_diffusion(source, 0.1, 0), source)
        self.assertTrue(np.all(np.isfinite(probe.heat_diffusion(source, 0.25, 5))))
        for nu in (-0.1, 0.250001, 0.5, float("nan")):
            with self.assertRaises(ValueError):
                probe.heat_diffusion(source, nu, 5)

    def test_side_gain_preserves_mid_and_mono(self):
        rng = np.random.default_rng(17)
        source = rng.normal(size=(200, 2))
        result = probe.side_gain(source, 2)
        np.testing.assert_allclose(result.sum(axis=1), source.sum(axis=1), atol=1e-14)
        np.testing.assert_allclose(result[:, 0] - result[:, 1],
                                   2 * (source[:, 0] - source[:, 1]), atol=1e-14)
        mono = np.repeat(source[:, :1], 2, axis=1)
        np.testing.assert_array_equal(probe.side_gain(mono, 2), mono)

    def test_deterministic_export_preserves_source_and_matches_rms(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "source.wav"
            time = np.arange(4800) / probe.RATE
            mid = 0.35 * np.sin(2 * np.pi * 3000 * time)
            side = 0.15 * np.sin(2 * np.pi * 10000 * time)
            samples = np.rint(np.column_stack((mid + side, mid - side)) * 32768).astype("<i2")
            with wave.open(str(path), "wb") as writer:
                writer.setnchannels(2)
                writer.setsampwidth(2)
                writer.setframerate(probe.RATE)
                writer.writeframes(samples.tobytes())
            before = path.read_bytes()
            first = probe.export(path, root / "first")
            second = probe.export(path, root / "second")
            self.assertEqual(first, second)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual((root / "first/source_original.wav").read_bytes(), before)
            self.assertEqual(first, json.loads((root / "first/receipt.json").read_text()))
            self.assertEqual(first["source"]["sha256_before"], first["source"]["sha256_after"])
            self.assertEqual(first["tool"]["script_sha256"], probe.sha256(Path(probe.__file__)))
            self.assertEqual(first["tool"]["numpy_version"], np.__version__)
            values = list(first["listening_pcm16"].values())
            self.assertEqual({value["condition"] for value in values}, set(probe.CONDITIONS))
            for value in values:
                self.assertEqual(probe.sha256(root / "first" / value["file"]), value["sha256"])
                self.assertLessEqual(value["metrics"]["peak"], probe.LISTENING_PEAK_LIMIT + 0.5 / 32768)
                self.assertEqual(value["metrics"]["at_or_over_full_scale_samples"], 0)
                self.assertEqual(value["metrics"]["pcm16_endpoint_samples"], 0)
            rms = [value["metrics"]["stereo_rms"] for value in values]
            self.assertLess(max(rms) - min(rms), 1e-5)
            with self.assertRaises(FileExistsError):
                probe.export(path, root / "first")

    def test_silence_metrics_remain_finite_json(self):
        silence = np.zeros((4800, 2))
        metrics = probe.metrics(silence, probe.envelope(silence))
        self.assertIsNone(metrics["lr_centered_correlation"])
        self.assertIsNone(metrics["side_over_mid_power_db"])
        self.assertIsNone(metrics["onset_envelope_change_proxy"])
        self.assertTrue(all(item["lr_total_fraction"] is None for item in metrics["spectrum"]["bands"]))
        json.dumps(metrics, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
