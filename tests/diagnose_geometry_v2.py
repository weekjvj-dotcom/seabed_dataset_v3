"""Log actual candidate and mesh bounds without changing acceptance rules."""
import sys, json, collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import bpy
from mathutils import Vector
from src import randomization as r
from src.scene import build_geometry
from src.contracts import load_config, atomic_json,source_hash
cfg=load_config(ROOT/'configs/first_round.json')
out=ROOT/'reports/geometry_diagnostic_v3'
out.mkdir(exist_ok=True)
original=r._fish_candidate_ok
rows=[]
def vec(v):return [float(x) for x in v]
def bounds(b):return [vec(x) for x in b]
def actual(obj,matrix):
    points=[matrix@v.co for v in obj.data.vertices]
    return [[min(p[k] for p in points) for k in range(3)],[max(p[k] for p in points) for k in range(3)]]
def wrap(record,matrix,current,active,records,floor_z,seed_phase,*args,**kw):
    ok,detail=original(record,matrix,current,active,records,floor_z,seed_phase,*args,**kw)
    row={'fish':record['name'],'center':vec(matrix.translation),'ok':ok,'detail':detail,'proxy':bounds(r._aabb_from_matrix(record['object'],matrix)),'actual':actual(record['object'],matrix)}
    if detail.get('other'):
        other=next(x for x in records if x['name']==detail['other']);obj=other['object'];mat=current[other['name']]
        row['other_proxy']=bounds(r._aabb_from_matrix(obj,mat));row['other_actual']=actual(obj,mat)
    rows.append(row)
    return ok,detail
r._fish_candidate_ok=wrap
for state in range(3):
    rows.clear()
    try:
        h=build_geometry(cfg,cfg['layouts'][0],state,0)
        result={'passed':True,'plan':h['geometry_plan']}
    except Exception as exc:result={'passed':False,'error':str(exc)}
    result.update(source_sha256=source_hash(),state=state,checks=list(rows),counts=dict(collections.Counter(x['detail'].get('reason') for x in rows)))
    atomic_json(out/f'state{state}.json',result)
    print('DIAGNOSTIC',state,result['passed'],result.get('error'),result['counts'],flush=True)
