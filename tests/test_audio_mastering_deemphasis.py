import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "audio_mastering_deemphasis", Path(__file__).resolve().parents[1] / "scripts/audio_mastering_deemphasis.py"
)
deemphasis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deemphasis)


class AudioMasteringDeemphasisTests(unittest.TestCase):
    def test_biquad_peaking_filter_attenuates_center_frequency(self):
        f0 = 3500.0
        q = 1.0
        gain_db = -3.0
        b, a = deemphasis.biquad_peaking_eq_coeffs(f0, q, gain_db, deemphasis.RATE)

        # Feed 3.5 kHz sine wave
        t = np.arange(48000) / deemphasis.RATE
        sine_in = np.column_stack((np.sin(2 * np.pi * f0 * t), np.sin(2 * np.pi * f0 * t)))
        filtered = deemphasis.apply_biquad(sine_in, b, a)

        # Steady-state amplitude check (ignoring initial transient)
        steady_in = np.max(np.abs(sine_in[2000:]))
        steady_out = np.max(np.abs(filtered[2000:]))
        measured_gain_db = 20 * np.log10(steady_out / steady_in)
        self.assertAlmostEqual(measured_gain_db, gain_db, delta=0.2)

        # Feed 100 Hz sine wave (should have almost 0 dB attenuation)
        sine_100 = np.column_stack((np.sin(2 * np.pi * 100.0 * t), np.sin(2 * np.pi * 100.0 * t)))
        filtered_100 = deemphasis.apply_biquad(sine_100, b, a)
        steady_100_in = np.max(np.abs(sine_100[2000:]))
        steady_100_out = np.max(np.abs(filtered_100[2000:]))
        gain_100_db = 20 * np.log10(steady_100_out / steady_100_in)
        self.assertAlmostEqual(gain_100_db, 0.0, delta=0.1)

    def test_process_wav_end_to_end_produces_valid_receipt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            in_wav = temp_path / "test_input.wav"
            out_wav = temp_path / "test_output.wav"

            # Create synthetic test WAV
            t = np.arange(48000) / deemphasis.RATE
            test_sig = (
                0.3 * np.sin(2 * np.pi * 200 * t)
                + 0.5 * np.sin(2 * np.pi * 3500 * t)
                + 0.1 * np.sin(2 * np.pi * 8000 * t)
            )
            samples = np.rint(np.column_stack((test_sig, test_sig)) * 32768.0).astype("<i2")
            with wave.open(str(in_wav), "wb") as w:
                w.setnchannels(2)
                w.setsampwidth(2)
                w.setframerate(deemphasis.RATE)
                w.writeframes(samples.tobytes())

            receipt = deemphasis.process_wav(in_wav, out_wav, center_hz=3500.0, gain_db=-3.0)
            self.assertTrue(out_wav.is_file())
            self.assertTrue(out_wav.with_suffix(".receipt.json").is_file())
            self.assertLess(
                receipt["after_metrics"]["bands"]["2000-6000Hz"],
                receipt["before_metrics"]["bands"]["2000-6000Hz"],
            )
            self.assertLessEqual(receipt["after_metrics"]["peak_dbfs"], -1.0 + 1e-4)
            self.assertEqual(receipt["input_sha256"], receipt["input_sha256_after"])
            self.assertFalse(receipt["filter_parameters"]["rms_match_limited_by_peak"])
            self.assertAlmostEqual(receipt["filter_parameters"]["achieved_rms_delta_db"], 0.0, delta=0.02)
            with self.assertRaises(ValueError):
                deemphasis.process_wav(in_wav, in_wav)
            with self.assertRaises(ValueError):
                deemphasis.process_wav(in_wav, out_wav)


if __name__ == "__main__":
    unittest.main()
