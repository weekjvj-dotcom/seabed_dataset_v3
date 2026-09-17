"""Exercise pipeline exception classes in Blender using explicit fault injection."""
import sys,json,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src import pipeline as p
from src.contracts import load_config,atomic_json,derive_seed
cfg=load_config(ROOT/'configs/first_round.json');out=ROOT/'reports/pipeline_faults';out.mkdir(exist_ok=True);results=[]
original=p.run_candidate
p.calibration_gate=lambda *args: {'test_only':'injected faults do not render images'}
for mode in ('recoverable','exhausted','fatal'):
 folder=Path(tempfile.mkdtemp(prefix=mode+'_',dir=out));local=json.loads(json.dumps(cfg));local['output_root']=str(folder/'output');config=folder/'config.json';atomic_json(config,local);attempts=[]
 def fault(config,plan,state,attempt,*args,**kwargs):
  attempts.append(attempt)
  if mode=='recoverable':
   if attempt==0:raise p.CandidateRejected('controlled_quality_failure',{'test_measurement':0})
   return {'manifest_rows':[]}
  if mode=='exhausted':raise ValueError('CANDIDATE_CONTROLLED_CLEARANCE: injected')
  raise ValueError('FATAL_CONTROLLED_FORMAT: injected')
 p.run_candidate=fault
 sys.argv=['blender','--disable-autoexec','--','--config',str(config),'--state-index','0']
 outcome='returned'
 try:p.main()
 except SystemExit as exc:outcome='exit_'+str(exc.code)
 except ValueError as exc:outcome='raised_'+str(exc)
 expected={'recoverable':([0,1],'returned'),'exhausted':([0,1,2],'exit_2'),'fatal':([0],'raised_FATAL_CONTROLLED_FORMAT: injected')}[mode]
 assert (attempts,outcome)==expected,(mode,attempts,outcome)
 results.append({'mode':mode,'attempts':attempts,'outcome':outcome,'passed':True,'root':str(folder)})
p.run_candidate=original
atomic_json(out/'result.json',{'status':'pass','method':'controlled CandidateRejected/ValueError injection in actual Blender Python; no RGB was generated','scenarios':results})
print('PIPELINE_FAULTS_PASS '+json.dumps(results),flush=True)
