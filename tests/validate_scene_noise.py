"""Measure seed noise and water effect in the actual complete ecology scene."""
import sys,json,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.deps/py313'))
import numpy as np
from PIL import Image
from src.scene import build_layout,add_water_scenes
from src.render import render_rgb,read_render_rgb
from src.labels import export_labels
from src.contracts import load_config,atomic_json
cfg=load_config(ROOT/'configs/smoke.json');h=build_layout(cfg,101);s=h['clear'];out=ROOT/'reports/scene_noise';out.mkdir(parents=True,exist_ok=True)
s.cycles.samples=512;render_rgb(s,out/'clear_512');clear=read_render_rgb(out/'clear_512.exr');export_labels(s,out,h['ecology']['objects'])
sem=np.array(Image.open(out/'semantic_id.png'));valid=sem>0
for dy in [-1,0,1]:
 for dx in [-1,0,1]:valid &= sem==np.roll(np.roll(sem,dy,0),dx,1)
local=copy.deepcopy(cfg);local['presets']=local['presets'][:1];_,wet,_=add_water_scenes(h,local,5)[0]
images={};timings={}
for samples in [128,512]:
 for seed in [4401,9127]:
  wet.cycles.samples=samples;wet.cycles.seed=seed;key=f'water_{samples}_{seed}';stats=render_rgb(wet,out/key);images[(samples,seed)]=read_render_rgb(out/(key+'.exr'));timings[key]=stats['seconds']
noise={n:float(np.sqrt(np.mean((images[(n,4401)][valid]-images[(n,9127)][valid])**2))) for n in [128,512]}
effect=float(np.sqrt(np.mean((images[(512,4401)][valid]-clear[valid])**2)))
r={'status':'pass' if noise[512]<noise[128] and effect/noise[512]>5 else 'fail','full_ecology':True,'resolution':cfg['render']['resolution'],'interior_pixels':int(valid.sum()),'noise_rmse':noise,'noise_ratio_512_over_128':noise[512]/noise[128],'water_clear_rmse':effect,'effect_to_512_seed_noise_ratio':effect/noise[512],'effect_noise_threshold':5,'formal_samples_selected':512,'denoising':False,'timings_seconds':timings}
atomic_json(out/'validation.json',r);print(json.dumps(r,indent=2))
if r['status']!='pass':raise SystemExit(1)
