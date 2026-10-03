import copy
import math
import unittest
import numpy as np
from predictive import Quantizer, cdf, forecast_indices, measure, projection, reward
from selection import choose


class PredictiveTests(unittest.TestCase):
    def test_cdf_and_normalized_discrete_code(self):
        x = np.arange(-8, 8.01, .05)
        expected = np.array([.5 * (1 + math.erf(v / np.sqrt(2))) for v in x])
        self.assertLess(np.max(np.abs(cdf(x) - expected)), 8e-8)
        q = Quantizer(np.arange(100)[:, None])
        for value in (-1e9, 0, 1e9):
            bits = q.gaussian_bits(np.arange(100)[:, None], np.full((100, 1), value), np.ones(1))
            self.assertTrue(np.isfinite(bits).all())
            self.assertTrue((bits >= 0).all())

    def test_constant_exact_zero_bits_and_reward(self):
        y = np.ones((1024, 3))
        result = measure(y)
        self.assertEqual(reward(result)['value'], 0)
        for item in result['horizons'].values():
            self.assertEqual(item['probes']['output_slow']['audit']['gain_bits'], 0)

    def test_horizon_purging_and_no_selection_fit_leak(self):
        ids, masks = forecast_indices(1024, 64, 128)
        for segment, start, end in [('train', 128, 409), ('calibration', 409, 614),
                                    ('selection', 614, 819), ('audit', 819, 1024)]:
            self.assertTrue((ids[masks[segment]] >= start).all())
            self.assertTrue((ids[masks[segment]] + 64 < end).all())
        rng = np.random.default_rng(7)
        y = rng.normal(size=(1024, 2))
        a = measure(y, horizons=(4,))
        changed = y.copy(); changed[819:] *= 100
        b = measure(changed, horizons=(4,))
        self.assertEqual(a['horizons']['4']['selected_simple_baseline'], b['horizons']['4']['selected_simple_baseline'])
        self.assertEqual(a['horizons']['4']['probes']['output_slow']['selection'],
                         b['horizons']['4']['probes']['output_slow']['selection'])

    def test_fixed_projection_nonmutation_reproducible(self):
        x = np.arange(500).reshape(100, 5).astype(float)
        before = x.copy()
        np.testing.assert_array_equal(projection(x, 4, 'gru'), projection(x, 4, 'gru'))
        np.testing.assert_array_equal(x, before)

    def test_awkward_period_and_iid_not_rewarded(self):
        rng = np.random.default_rng(2)
        repeated = rng.normal(size=(37, 3))[np.arange(1024) % 37]
        self.assertEqual(reward(measure(repeated))['value'], 0)
        self.assertEqual(reward(measure(rng.normal(size=(1024, 3))))['value'], 0)

    def test_complementary_state_and_shuffled_null(self):
        rng = np.random.default_rng(5)
        a, b = rng.normal(size=(2, 1536, 4))
        y = np.roll(a[:, :2] + b[:, :2], 4, axis=0) + .08 * rng.normal(size=(1536, 2))
        result = measure(y, {'gru': a, 'field_coarse': b}, horizons=(4,))
        joint = result['horizons']['4']['synergy']['gru+field_coarse']['audit']
        self.assertGreater(joint['joint_beyond_best_individual_bits'], 100)
        shuffled = measure(y, {'gru': a[rng.permutation(len(a))]}, horizons=(4,))
        actual = result['horizons']['4']['probes']['gru']['audit']['gain_bits']
        self.assertGreater(actual, shuffled['horizons']['4']['probes']['gru']['audit']['gain_bits'])

    def test_disabled_selector_exact_bypass_and_audit_invariance(self):
        self.assertEqual(choose({'invalid': object()}, 0)['chosen'], 'baseline')
        rng = np.random.default_rng(9)
        report = measure(rng.normal(size=(1024, 2)), horizons=(4,), affine_controls=True)
        bank = {name: {'coding': copy.deepcopy(report), 'selection_physical': {'eligible': True},
                       'selection_waveform_displacement': distance, 'naive_selection_score': -distance}
                for name, distance in [('baseline', 0), ('candidate', .01)]}
        expected = choose(bank, .02)
        for item in bank.values():
            item['coding']['horizons']['4']['probes']['output_slow']['audit']['gain_bits'] = 1e15
        self.assertEqual(expected, choose(bank, .02))
        self.assertEqual(expected['chosen'], 'baseline')

    def test_affine_loop_falsifies_v1_and_hardened_protocol_rejects(self):
        rng = np.random.default_rng(19)
        y = rng.normal(size=(1024, 1))
        for t in range(65, len(y)):
            y[t] = -.92*y[t-65] + .12*rng.normal()
        self.assertGreater(reward(measure(y))['value'], 0)
        self.assertEqual(reward(measure(y, affine_controls=True))['value'], 0)

    def test_short_history_support_is_explicit_not_nonfinite(self):
        y = np.random.default_rng(2).normal(size=(256, 3))
        report = measure(y, short_protocol=True, affine_controls=True)
        self.assertTrue(report['horizons']['1']['supported'])
        self.assertFalse(report['horizons']['64']['supported'])
        self.assertTrue(np.isfinite(report['horizons']['1']['probes']['output_slow']['audit']['gain_bits']))


if __name__ == '__main__':
    unittest.main()
