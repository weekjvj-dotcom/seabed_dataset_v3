"""Regression for native .blend encoding after cache-only construction."""
import sys,tempfile,unittest
from pathlib import Path
try:import bpy
except ImportError:bpy=None

@unittest.skipIf(bpy is None,'requires actual Blender')
class CachedNativeEncoding(unittest.TestCase):
 def test_unrendered_scenes_save_rgb_png16(self):
  from seabed_dataset_v2.src.scene import build_layout,add_water_scenes,save_blend
  from seabed_dataset_v2.src.contracts import load_config
  root=Path(__file__).resolve().parents[1];cfg=load_config(root/'configs/smoke.json');h=build_layout(cfg,101);add_water_scenes(h,cfg,5)
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'cached.blend';save_blend(h,path,cfg)
   bpy.ops.wm.open_mainfile(filepath=str(path),load_ui=False,use_scripts=False)
   for scene in bpy.data.scenes:
    fmt=scene.render.image_settings
    self.assertEqual((fmt.file_format,fmt.color_mode,fmt.color_depth),('PNG','RGB','16'),scene.name)
if __name__=='__main__':unittest.main(argv=[sys.argv[0]])
