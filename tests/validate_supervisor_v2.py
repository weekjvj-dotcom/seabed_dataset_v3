"""Real subprocess exit injections, without causing Blender or GUI crashes."""
import json,sys,tempfile,subprocess,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.contracts import load_config,atomic_json
cfg=load_config(ROOT/'configs/first_round.json');out=ROOT/'reports/supervisor_faults';out.mkdir(exist_ok=True);results=[]
for scenario in ('candidate_exhausted','fatal','crash'):
 folder=Path(tempfile.mkdtemp(prefix=scenario+'_',dir=out));marker=folder/'calls.jsonl';script=folder/'worker_stub.py'
 stub='''#!{python}
import sys,json,os,signal
from pathlib import Path
state=int(sys.argv[sys.argv.index('--state-index')+1])
p=Path({marker!r})
with p.open('a') as f:f.write(json.dumps({{'state':state}})+'\\n')
mode={scenario!r}
if mode=='fatal':sys.exit(1)
if mode=='crash':os.kill(os.getpid(),signal.SIGTERM)
sys.exit(2 if state==0 else 0)
'''.format(python=sys.executable,marker=str(marker),scenario=scenario)
 script.write_text(stub);script.chmod(0o755)
 local=json.loads(json.dumps(cfg));local['output_root']=str(folder/'output');config=folder/'config.json';atomic_json(config,local)
 run=subprocess.run([sys.executable,str(ROOT/'run.py'),'--config',str(config),'--blender',str(script)],capture_output=True,text=True)
 (folder/'run.log').write_text(run.stdout+'\n'+run.stderr)
 calls=[json.loads(x)['state'] for x in marker.read_text().splitlines()]
 expected=[0,1,2] if scenario=='candidate_exhausted' else ([0,0,0] if scenario=='crash' else [0])
 assert calls==expected,(scenario,calls)
 assert run.returncode==(2 if scenario=='candidate_exhausted' else 1)
 results.append({'scenario':scenario,'calls':calls,'exit_code':run.returncode,'passed':True,'log':str(folder/'run.log')})
atomic_json(out/'result.json',{'status':'pass','method':'subprocess exit-status injection; actual Blender was not crashed','scenarios':results})
print(json.dumps(results))
