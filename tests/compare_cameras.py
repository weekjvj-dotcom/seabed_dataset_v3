import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.contracts import load_config,atomic_json
from src.scene import build_layout,set_camera
from src.render import render_rgb
from src.labels import export_labels
cfg=load_config(ROOT/'configs/smoke.json');cfg['render']['resolution']=[384,384];cfg['render']['samples']=128
h=build_layout(cfg,101);out=ROOT/'reports/visual/cameras';out.mkdir(parents=True,exist_ok=True)
views=[{'id':'a','eye':[3.5,-3.5,4.4],'target':[0,.5,-.25],'lens_mm':30},{'id':'b','eye':[3,-4.2,4.4],'target':[0,.5,-.35],'lens_mm':28},{'id':'c','eye':[2.5,-3.5,4.4],'target':[0,1,-.2],'lens_mm':28}]
for view in views:
 set_camera(h['clear'],view);d=out/view['id'];d.mkdir(exist_ok=True);render_rgb(h['clear'],d/'clear_rgb');m=export_labels(h['clear'],d,h['ecology']['objects']);atomic_json(d/'camera_candidate.json',view)
 print('CAMERA_COMPLETE '+view['id'],flush=True)
