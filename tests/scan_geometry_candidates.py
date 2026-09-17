"""Measure bounded candidate availability; never promotes dataset artifacts."""
import sys,json,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.scene import build_geometry
from src.contracts import load_config,atomic_json,source_hash
from src.labels import export_labels
from src.pipeline import label_arrays
from src.quality import geometry_metrics
from src.gates import geometry_gate
cfg=load_config(Path(sys.argv[sys.argv.index('--')+2]) if '--' in sys.argv and len(sys.argv)>sys.argv.index('--')+2 else ROOT/'configs/first_round.json');out=Path(sys.argv[sys.argv.index('--')+1]) if '--' in sys.argv else ROOT/'reports/geometry_scan';rows=[]
for state in range(3):
 for attempt in range(cfg['max_attempts']):
  row={'state':state,'attempt':attempt}
  try:
   h=build_geometry(cfg,cfg['layouts'][0],state,attempt);folder=out/f'g{state}_a{attempt}'
   export_labels(h['base'],folder,h['active_objects']);d,s,i,m=label_arrays(folder)
   registry={str(x['instance_id']):x for x in h['geometry_plan']['instances']}
   metrics=geometry_metrics(d,s,i,registry,cfg);gate=geometry_gate(metrics,i,registry,cfg)
   row.update(passed=gate['passed'],quality=gate,plan=h['geometry_plan'])
  except Exception as exc:row.update(passed=False,error=str(exc))
  rows.append(row);atomic_json(out/'result.json',{'source_sha256':source_hash(),'candidates':rows})
  print('CANDIDATE_SCAN',state,attempt,row['passed'],row.get('error',row.get('quality',{}).get('reasons')),flush=True)
