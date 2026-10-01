import unittest

import numpy as np

from checkpoint_geometry import phase_randomize, spectral_metrics, spatial_summary


class GeometryTests(unittest.TestCase):
    def test_phase_randomized_null_preserves_2d_power(self):
        rng = np.random.default_rng(33)
        y, x = np.mgrid[:32, :32]
        field = np.sin(x / 4) + np.cos(y / 7) + 0.1 * rng.normal(size=(32, 32))
        surrogate = phase_randomize(field, np.random.default_rng(44))
        observed_power = np.abs(np.fft.fft2(field - field.mean()))
        null_power = np.abs(np.fft.fft2(surrogate - surrogate.mean()))
        np.testing.assert_allclose(observed_power, null_power, atol=1e-10)
        self.assertFalse(np.array_equal(field, surrogate))

    def test_matrix_null_and_spatial_shuffle_have_expected_reach(self):
        rng = np.random.default_rng(7)
        matrix = np.outer(rng.normal(size=64), rng.normal(size=64))
        observed = spectral_metrics(matrix)
        shuffled = spectral_metrics(rng.permutation(matrix.reshape(-1)).reshape(matrix.shape))
        self.assertLess(observed["stable_rank"], shuffled["stable_rank"])
        field = np.repeat(np.arange(16, dtype=float)[:, None], 16, axis=1)
        null = rng.permutation(field.reshape(-1)).reshape(field.shape)
        self.assertGreater(spatial_summary(field)["lag1_correlation"],
                           spatial_summary(null)["lag1_correlation"])


if __name__ == "__main__":
    unittest.main()
