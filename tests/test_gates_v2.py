import json
from pathlib import Path
import unittest
import numpy as np
from seabed_dataset_v2.src.gates import noise_gate
from seabed_dataset_v2.src.quality import noise_metrics

CFG=json.loads((Path(__file__).resolve().parents[1]/'configs/first_round.json').read_text())

class TestNoiseGate(unittest.TestCase):
    def test_featureless_noise_is_rejected(self):
        rng=np.random.default_rng(29)
        clear=np.full((64,64,3),.3)
        a=.2+rng.normal(0,.02,clear.shape)
        b=.2+rng.normal(0,.02,clear.shape)
        result=noise_gate(noise_metrics(clear,a,b,np.ones((64,64))),CFG)
        self.assertFalse(result['passed'])
        self.assertIn('structure_noise_min_not_met',result['reasons'])

    def test_structured_low_noise_passes_and_threshold_is_effective(self):
        rng=np.random.default_rng(13)
        clear=np.repeat(np.linspace(.1,.8,64)[None,:,None],64,axis=0)
        clear=np.repeat(clear,3,axis=2)
        a=clear*.6+rng.normal(0,.0001,clear.shape)
        b=clear*.6+rng.normal(0,.0001,clear.shape)
        metrics=noise_metrics(clear,a,b,np.ones((64,64)))
        self.assertTrue(noise_gate(metrics,CFG)['passed'])
        strict={**CFG,'quality':{**CFG['quality'],'effect_noise_min':1e10}}
        self.assertFalse(noise_gate(metrics,strict)['passed'])

if __name__=='__main__':unittest.main()
