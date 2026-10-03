import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from transfer import FrozenBank,plain


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(11)
        a,b=rng.normal(size=(2,1024,4))
        cls.y=np.roll(a[:,:2]+b[:,:2],4,axis=0)+.08*rng.normal(size=(1024,2))
        cls.views={'gru':a,'field_coarse':b}
        cls.bank=FrozenBank.train(cls.y,cls.views,horizons=(4,))

    def test_saved_state_and_predictions_restore_exactly(self):
        with tempfile.TemporaryDirectory(dir='/data/data/com.termux/files/usr/tmp') as d:
            path=Path(d)/'probe.json'
            self.bank.save(path)
            restored=FrozenBank.load(path)
            ids,target,pred=self.bank.predictions(self.y,self.views,4)
            ri,rt,rp=restored.predictions(self.y,self.views,4)
            np.testing.assert_array_equal(ids,ri)
            np.testing.assert_array_equal(target,rt)
            for name in pred:np.testing.assert_array_equal(pred[name],rp[name])
            self.assertEqual(self.bank.score(self.y,self.views),restored.score(self.y,self.views))

    def test_no_test_refit_and_no_suffix_history_leak(self):
        before=json.dumps(plain(self.bank.state),sort_keys=True)
        altered=self.y.copy();altered[800:]+=100
        _,_,a=self.bank.predictions(self.y,self.views,4)
        _,_,b=self.bank.predictions(altered,self.views,4)
        for name in a:np.testing.assert_array_equal(a[name][:600],b[name][:600])
        self.bank.score(altered,self.views,adapt_sigma=True)
        self.assertEqual(before,json.dumps(plain(self.bank.state),sort_keys=True))

    def test_transfers_known_complementary_process(self):
        rng=np.random.default_rng(13)
        a,b=rng.normal(size=(2,768,4))
        y=np.roll(a[:,:2]+b[:,:2],4,axis=0)+.08*rng.normal(size=(768,2))
        result=self.bank.score(y,{'gru':a,'field_coarse':b})['horizons']['4']
        synergy=result['synergy']['gru+field_coarse']
        self.assertGreater(synergy['joint_beyond_best_individual_bits'],300)
        self.assertGreater(synergy['history_conditional_joint_minus_sum_bits'],300)
        null=self.bank.score(y,{'gru':a[rng.permutation(len(a))],
                               'field_coarse':b[rng.permutation(len(b))]})['horizons']['4']
        self.assertGreater(result['probes']['gru+field_coarse']['over_history_bits'],
                           null['probes']['gru+field_coarse']['over_history_bits']+500)

    def test_episode_history_does_not_wrap(self):
        altered=self.y.copy();altered[-128:]*=1000
        _,f=self.bank.features(self.y,self.views)
        _,g=self.bank.features(altered,self.views)
        np.testing.assert_array_equal(f['output_slow'][:128],g['output_slow'][:128])

    def test_short_adaptation_purges_horizon_and_is_finite(self):
        result=self.bank.score(self.y[:384],{k:v[:384] for k,v in self.views.items()},adapt_sigma=True)
        # Origins >=256, future strictly <384, horizon4 =>124 forecasts.
        self.assertEqual(result['horizons']['4']['forecasts'],124)
        self.assertTrue(np.isfinite(result['horizons']['4']['probes']['gru']['gain_bits']))

    def test_train_masks_preserve_source_calibration(self):
        model=self.bank.state['horizons']['4']
        self.assertEqual(model['train_rows'],277) # ids128..404, future<409
        self.assertEqual(model['calibration_rows'],201) # ids409..609, future<614


if __name__=='__main__':unittest.main()
