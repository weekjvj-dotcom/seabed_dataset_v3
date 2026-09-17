import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import bpy
from src.contracts import load_config,atomic_json
from src.scene import build_layout
from src.render import render_rgb
from src.labels import export_labels
cfg=load_config(ROOT/'configs/smoke.json');h=build_layout(cfg,101)
out=ROOT/'reports/visual/preflight';out.mkdir(parents=True,exist_ok=True)
render_rgb(h['clear'],out/'clear_rgb');meta=export_labels(h['clear'],out,h['ecology']['objects']);atomic_json(out/'assets.json',h['ecology']['summary'])
bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'scenes/clear_preflight.blend'),compress=True)
print('CLEAR_PREFLIGHT_COMPLETE',flush=True)
