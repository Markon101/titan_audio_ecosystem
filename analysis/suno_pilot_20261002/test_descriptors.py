import unittest
import numpy as np
from describe_pilot import corr, js, continuity
from check_overlap import match, RATE


class DescriptorTests(unittest.TestCase):
    def test_correlations_and_distances_do_not_mutate_inputs(self):
        a=np.array([.1,.3,.7,.5]);b=np.array([.2,.4,.6,.8]);aa=a.copy();bb=b.copy()
        self.assertIsNotNone(corr(a,b));self.assertAlmostEqual(js(a,b),js(b,a))
        self.assertAlmostEqual(js(a,a),0.0)
        np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(b,bb)

    def test_audio_overlap_finds_known_shift_and_gain(self):
        source=np.random.default_rng(44).normal(0,.1,12*RATE)
        target=np.concatenate((np.zeros(RATE),source*.37+.04,np.zeros(RATE)))
        result=match(source,target,1)
        self.assertAlmostEqual(result['match_seconds'],2,places=8)
        self.assertGreater(result['absolute_correlation'],.999999)

    def test_shuffle_distinguishes_smooth_order_from_window_distribution(self):
        x=np.tile(np.linspace(0,1,120)[:,None],(1,28))
        result=continuity(x,44)
        self.assertLess(result['1']['observed_to_shuffle_ratio'],.01)
        self.assertEqual(result,continuity(x,44))


if __name__=='__main__':unittest.main()
