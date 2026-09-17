"""Re-sealed malformed bundles must still fail semantic/format validation."""
import json,sys,shutil,tempfile,struct
from pathlib import Path
import unittest
try:
 import bpy
except ModuleNotFoundError:bpy=None
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.contracts import seal_record,atomic_json,SAMPLE_FILES
from src.storage import readable_bundle,rgb_exr_header_valid

@unittest.skipIf(bpy is None,'requires Blender image encoder')
class StorageV2(unittest.TestCase):
 def test_resealed_wrong_png_bits_and_schema_are_refused(self):
  calibration=ROOT/'reports/calibration_128_initial'
  report=json.loads((calibration/'result.json').read_text())
  render=next(x['stats'] for x in report['renders'] if x['mode']=='natural' and x['condition']=='mild')
  contract={k:render[k] for k in ('engine','resolution','samples','seed','denoising')}
  expected={'schema_version':2,'sample_id':'storage_fixture','render_contract':contract}
  folder=Path(tempfile.mkdtemp(prefix='storage_v2_',dir=ROOT/'reports'))
  for ext in ('png','exr'):shutil.copy2(calibration/'natural'/('mild.'+ext),folder/('degraded_rgb.'+ext))
  meta={**expected,'render':render,'rgb_quality':{'passed':True},'geometry_quality':{'passed':True},'state_sha256':'test','state_after_sha256':'test'};atomic_json(folder/'metadata.json',meta);seal_record(folder,SAMPLE_FILES,expected)
  self.assertTrue(readable_bundle(folder,'sample',expected))
  self.assertTrue(rgb_exr_header_valid(folder/'degraded_rgb.exr'))
  meta.pop('schema_version');atomic_json(folder/'metadata.json',meta);seal_record(folder,SAMPLE_FILES,expected)
  self.assertFalse(readable_bundle(folder,'sample',expected))
  meta['schema_version']=2;atomic_json(folder/'metadata.json',meta)
  scene=bpy.context.scene;scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB';scene.render.image_settings.color_depth='8'
  img=bpy.data.images.load(str(calibration/'natural/mild.exr'));img.save_render(str(folder/'degraded_rgb.png'),scene=scene);bpy.data.images.remove(img)
  self.assertEqual(struct.unpack('>IIBB',(folder/'degraded_rgb.png').read_bytes()[16:26])[2],8)
  seal_record(folder,SAMPLE_FILES,expected)
  self.assertFalse(readable_bundle(folder,'sample',expected))
if __name__=='__main__':unittest.main()
