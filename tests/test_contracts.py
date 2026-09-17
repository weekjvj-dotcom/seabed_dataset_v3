import sys, unittest, copy, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.contracts import load_config,make_plan,canonical_hash,ensure_plan,atomic_json,complete_record,seal_record
class Contracts(unittest.TestCase):
    def test_pilot_grid(self):
        jobs=make_plan(load_config(ROOT/'configs/pilot.json'))
        self.assertEqual(len(jobs),36)
        self.assertEqual(len({j['layout_family'] for j in jobs}),3)
        self.assertEqual(len({j['sample_key'] for j in jobs}),36)
        self.assertEqual(len({(j['layout_seed'],j['view']['id']) for j in jobs}),6)
        for seed in {j['layout_seed'] for j in jobs}:
            self.assertEqual(len({j['split'] for j in jobs if j['layout_seed']==seed}),1)
    def test_validation(self):
        cfg=load_config(ROOT/'configs/pilot.json')
        for change in [lambda c:c['presets'][0]['absorption'].__setitem__(0,-1),lambda c:c['presets'][0].__setitem__('g',1),lambda c:c['depths_m'].__setitem__(0,1),lambda c:c['presets'].append(c['presets'][0])]:
            c=copy.deepcopy(cfg);change(c)
            with self.assertRaises(ValueError):make_plan(c)
    def test_family_cannot_cross_split(self):
        cfg=load_config(ROOT/'configs/pilot.json');cfg['layouts'].append({'seed':304,'split':'test'})
        with self.assertRaises(ValueError):make_plan(cfg)
    def test_identity(self):
        self.assertEqual(canonical_hash({'b':2,'a':1}),canonical_hash({'a':1,'b':2}))
        self.assertNotEqual(canonical_hash({'camera':1}),canonical_hash({'camera':2}))
    def test_close_depths_and_incomplete_success(self):
        cfg=load_config(ROOT/'configs/pilot.json');cfg['depths_m']=[5.000001,5.000002]
        jobs=make_plan(cfg);self.assertEqual(len(jobs),len({j['sample_key'] for j in jobs}))
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);(d/'data').write_bytes(b'hello');seal_record(d,['data'],{})
            self.assertFalse(complete_record(d,['data','missing']))
            with self.assertRaises(ValueError):ensure_plan(d,{'config_hash':'new'})
    def test_resume_gate(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);p={'config_hash':'a','source_hash':'s','jobs':[]}
            ensure_plan(d,p);ensure_plan(d,p)
            with self.assertRaises(ValueError):ensure_plan(d,dict(p,source_hash='changed'))
            (d/'data').write_bytes(b'hello');seal_record(d,['data'],{'reference_id':'r'})
            self.assertTrue(complete_record(d))
            (d/'data').write_bytes(b'broken');self.assertFalse(complete_record(d))
            (d/'data').unlink();self.assertFalse(complete_record(d))
if __name__=='__main__':unittest.main()
