"""Render controlled complete-scene previews and raw quality evidence."""
import argparse,sys,json,copy,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent/'seabed_dataset/.deps/py313'))
import bpy,numpy as np
from src.contracts import load_config,atomic_json,source_hash
from src.scene import build_geometry,build_lighting_groups,save_native
from src.render import render_rgb,read_render_rgb
from src.labels import export_labels
from src.labels.formats import read_png,read_exr
from src.quality import geometry_metrics,geometry_decision,rgb_metrics,rgb_decision,noise_metrics
from src.lighting import lighting_metadata
from src.gates import geometry_gate,noise_gate
from src.pipeline import water_plan,runtime_signature

p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--state-index',type=int,default=0);p.add_argument('--attempt',type=int,default=0);p.add_argument('--out',required=True);p.add_argument('--samples',type=int,default=128);p.add_argument('--geometry-only',action='store_true');p.add_argument('--noise',action='store_true');p.add_argument('--noise-low-samples',type=int)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);cfg=load_config(a.config);cfg['render']['samples']=a.samples;out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
h=build_geometry(cfg,cfg['layouts'][0],a.state_index,a.attempt);labels=export_labels(h['base'],out/'geometry',h['active_objects'])
def arr(name):
 p=out/'geometry'/name
 return np.asarray((read_exr(p) if p.suffix=='.exr' else read_png(p))['pixels'])
depth=arr('depth_range_m.exr');sem=arr('semantic_id.png');inst=arr('instance_id.png');mask=arr('valid_mask.png');registry={str(x['instance_id']):x for x in h['geometry_plan']['instances']}
gm=geometry_metrics(depth,sem,inst,registry,cfg);decision=geometry_gate(gm,inst,registry,cfg);report={'stage':'calibration','source_sha256':source_hash(),'config':cfg,'runtime':runtime_signature(cfg),'geometry_plan':h['geometry_plan'],'geometry_quality':decision,'renders':[],'noise':{},'noise_convergence':{}};atomic_json(out/'result.json',report)
print('GEOMETRY_CALIBRATION '+json.dumps({'passed':decision['passed'],'reasons':decision['reasons'],'bands':{k:v['fraction_of_valid'] for k,v in gm['bands'].items()},'fish':gm['visible_fish_count']}),flush=True)
if not a.geometry_only:
 wp=water_plan(cfg,{'layout_seed':cfg['layouts'][0]['seed'],'state_index':a.state_index},a.attempt);report['water_plan']=wp
 groups=build_lighting_groups(h,cfg,f'cal_g{a.state_index}',wp['effective_presets'])
 for g in groups:
  md=out/g['mode'];md.mkdir(exist_ok=True);stats=render_rgb(g['clear'],md/'clear_rgb');clear=read_render_rgb(md/'clear_rgb.exr')
  report['renders'].append({'mode':g['mode'],'condition':'clear','stats':stats,'quality':rgb_decision(rgb_metrics(clear,mask,sem),cfg,g['mode']),'lighting':lighting_metadata(g['lighting'])})
  for v in g['variants']:
   stats=render_rgb(v['scene'],md/v['preset']['id']);rgb=read_render_rgb(md/(v['preset']['id']+'.exr'));q=rgb_decision(rgb_metrics(rgb,mask,sem),cfg,g['mode']);report['renders'].append({'mode':g['mode'],'condition':v['preset']['id'],'stats':stats,'quality':q})
   if a.noise and v['preset']['id']=='strong':
    seed=v['scene'].cycles.seed;v['scene'].cycles.seed=seed+9011;render_rgb(v['scene'],md/'strong_seed_b');b=read_render_rgb(md/'strong_seed_b.exr');report['noise'][g['mode']]=noise_gate(noise_metrics(clear,rgb,b,mask),cfg);v['scene'].cycles.seed=seed
    if a.noise_low_samples:
     if not 0<a.noise_low_samples<a.samples:raise ValueError('Noise comparison needs fewer positive samples')
     v['scene'].cycles.samples=a.noise_low_samples
     render_rgb(v['scene'],md/'strong_low_a');low_a=read_render_rgb(md/'strong_low_a.exr');v['scene'].cycles.seed=seed+9011
     render_rgb(v['scene'],md/'strong_low_b');low_b=read_render_rgb(md/'strong_low_b.exr')
     low=noise_metrics(clear,low_a,low_b,mask);high=report['noise'][g['mode']]['metrics']
     report['noise_convergence'][g['mode']]={'low_samples':a.noise_low_samples,'high_samples':a.samples,'low_noise':low,'noise_ratio_high_over_low':high['noise_rms']/low['noise_rms'],'passed':high['noise_rms']<low['noise_rms']}
     v['scene'].cycles.samples=a.samples;v['scene'].cycles.seed=seed
   atomic_json(out/'result.json',report)
  print('LIGHTING_CALIBRATION_COMPLETE '+g['mode'],flush=True)
 save_native(h,out/'calibration.blend',cfg,report['geometry_plan'])
report['status']='pass' if decision['passed'] and all(x['quality']['passed'] for x in report['renders']) and all(x['passed'] for x in report['noise'].values()) and all(x['passed'] for x in report['noise_convergence'].values()) else 'fail';atomic_json(out/'result.json',report)
