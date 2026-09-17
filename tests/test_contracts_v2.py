import unittest,sys,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.contracts import derive_seed,make_plan,load_config
class V2Contracts(unittest.TestCase):
 def test_seed_streams(self):
  self.assertEqual(derive_seed(42,'geometry',0),derive_seed(42,'geometry',0))
  self.assertNotEqual(derive_seed(42,'geometry',0),derive_seed(42,'lighting',0))
 def test_grid(self):
  p=make_plan(load_config(ROOT/'configs/first_round.json'))
  self.assertEqual(len(p['pairs']),27);self.assertEqual(len(p['references']),9);self.assertEqual(len(p['geometry_states']),3)
 def test_attempt_does_not_shift_future_state(self):
  later=derive_seed(42,'fish',2,0)
  for i in range(3):derive_seed(42,'fish',1,i)
  self.assertEqual(later,derive_seed(42,'fish',2,0))
 def test_illegal_modes(self):
  c=load_config(ROOT/'configs/first_round.json');c['lighting_modes'].append('fake')
  with self.assertRaises(ValueError):make_plan(c)
if __name__=='__main__':unittest.main()
