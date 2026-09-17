"""Diagnostic only: omit fish to measure the benthic camera composition."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src import randomization
randomization._select_fish=lambda *a,**k: []
from src.scene import build_geometry,build_lighting_groups
from src.contracts import load_config,atomic_json
from src.pipeline import label_arrays
from src.labels import export_labels
from src.quality import geometry_metrics
from src.render import render_rgb
cfg=load_config(ROOT/'configs/first_round.json');cfg['render']['samples']=16
out=ROOT/'reports/benthic_diagnostic'
for i in range(3):
 h=build_geometry(cfg,cfg['layouts'][0],i,0)
 folder=out/f'state{i}';folder.mkdir(parents=True,exist_ok=True)
 export_labels(h['base'],folder,h['active_objects'])
 d,s,n,m=label_arrays(folder)
 metrics=geometry_metrics(d,s,n,{str(x['instance_id']):x for x in h['geometry_plan']['instances']},cfg)
 atomic_json(folder/'result.json',{'camera':h['geometry_plan']['camera'],'metrics':metrics})
 if i==0:
  groups=build_lighting_groups(h,cfg,'diagnostic',cfg['presets']);render_rgb(groups[0]['clear'],folder/'benthic_clear')
 print('BENTHIC',i,metrics['bands'],flush=True)
