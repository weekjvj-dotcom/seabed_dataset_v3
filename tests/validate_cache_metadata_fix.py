"""Two fresh Blender processes must preserve every cached data/metadata byte."""
import sys,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.contracts import load_config,canonical_hash,source_hash,file_hash,atomic_json
cfgpath=ROOT/'reports/core_preview_metadata_fix_config.json';cfg=load_config(cfgpath);root=ROOT/'previews'/('config_'+canonical_hash(cfg)[:12]);out=ROOT/'reports/cache_metadata_fix';out.mkdir(exist_ok=True)
def hashes():
 return {str(p.relative_to(root)):file_hash(p) for sub in ('references','samples') for p in (root/sub).rglob('*') if p.is_file()}
def run(name):
 cmd=[sys.executable,str(ROOT/'run.py'),'--config',str(cfgpath),'--preview','--state-index','0','--resume'];start=time.time()
 with (out/(name+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 assert r.returncode==0,(name,r.returncode)
 return {'name':name,'seconds':time.time()-start,'exit_code':r.returncode,'log':str(out/(name+'.log'))}
steps=[run('initial')];before=hashes();event_count=len((root/'events.jsonl').read_text().splitlines());steps.append(run('reuse'));after=hashes();assert before==after,'Cached RGB/GT/metadata bytes changed'
events=[json.loads(x) for x in (root/'events.jsonl').read_text().splitlines()[event_count:]];counts={kind:sum(x['event']==kind for x in events) for kind in ('reference_reused','sample_reused','reference_rendered','sample_rendered')}
assert counts=={'reference_reused':3,'sample_reused':9,'reference_rendered':0,'sample_rendered':0},counts
result={'status':'pass','source_sha256':source_hash(),'config_sha256':canonical_hash(cfg),'root':str(root),'all_cached_files_byte_identical':True,'cached_file_count':len(before),'hashes':before,'events':counts,'steps':steps};atomic_json(ROOT/'reports/cache_reuse_metadata_fix.json',result);print('CACHE_REUSE_FIX_PASS '+json.dumps(counts))
