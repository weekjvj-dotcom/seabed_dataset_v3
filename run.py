#!/usr/bin/env python3
"""Serial per-state supervisor for the v2 native dataset generator."""
import argparse,fcntl,json,subprocess,sys,time
from pathlib import Path
from src.contracts import ROOT,load_config,make_plan,source_hash,atomic_json,canonical_hash


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--dry-run',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--preview',action='store_true');p.add_argument('--state-index',type=int);p.add_argument('--layout-index',type=int);p.add_argument('--attempt',type=int);p.add_argument('--blender',default='/Applications/Blender.app/Contents/MacOS/Blender')
    a=p.parse_args();cfg=load_config(a.config);grid=make_plan(cfg)
    if a.dry_run:print(json.dumps({'config_hash':canonical_hash(cfg),'source_hash':source_hash(),**grid},indent=2));return 0
    output=Path(cfg['output_root']);output=output if output.is_absolute() else ROOT/output
    if a.preview:output=ROOT/'previews'/('config_'+canonical_hash(cfg)[:12])
    if a.layout_index is not None and not 0<=a.layout_index<len(cfg['layouts']):raise ValueError('Layout index is outside plan')
    if a.attempt is not None and not 0<=a.attempt<cfg['max_attempts']:raise ValueError('Attempt is outside allowed range')
    states=[x for x in grid['geometry_states'] if (a.state_index is None or x['state_index']==a.state_index) and (a.layout_index is None or x['layout_seed']==cfg['layouts'][a.layout_index]['seed'])]
    if not states:raise ValueError('Requested state index is outside plan')
    started=time.time();outcomes=[]
    run_report=ROOT/'reports/supervisor_runs'/f'{time.time_ns()}_{canonical_hash(cfg)[:12]}.json'
    with (ROOT/'.render.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Another v2 Blender worker holds the render lock')
        for state in states:
            cmd=[a.blender,'--background','--factory-startup','--disable-autoexec','--python-exit-code','1','--python',str(ROOT/'src/pipeline.py'),'--','--config',str(Path(a.config).resolve()),'--state-index',str(state['state_index']),'--layout-index',str(next(i for i,x in enumerate(cfg['layouts']) if x['seed']==state['layout_seed']))]
            if a.preview:cmd.append('--preview')
            if a.attempt is not None:cmd.extend(['--attempt',str(a.attempt)])
            for restart in range(3):
                result=subprocess.run(cmd);code=result.returncode
                if code in (0,1,2):break
                print('NATIVE_WORKER_CRASH '+state['state_key']+' restart='+str(restart),flush=True)
            outcomes.append({**state,'exit_code':code,'native_restarts':restart,'command':cmd})
            if code not in (0,2):
                print('FATAL_BATCH_STOP '+state['state_key'],flush=True)
                failure={'status':'failed','outcomes':outcomes,'seconds':time.time()-started,'source_sha256':source_hash(),'config_sha256':canonical_hash(cfg),'output_root':str(output)}
                atomic_json(run_report,failure)
                if output.exists():atomic_json(output/'supervisor.json',failure)
                return 1
    manifest=output/'manifest.jsonl';rows=[json.loads(x) for x in manifest.read_text().splitlines()] if manifest.exists() else []
    full=len(rows)==len(grid['pairs']);status='complete' if full else 'partial'
    report={'status':status,'pairs':len(rows),'planned_pairs':len(grid['pairs']),'references':len({x['reference_id'] for x in rows}),'states':len({x['state_key'] for x in rows}),'outcomes':outcomes,'seconds':time.time()-started,'source_sha256':source_hash(),'config_sha256':canonical_hash(cfg),'output_root':str(output)}
    if output.exists():atomic_json(output/'supervisor.json',report)
    atomic_json(run_report,report)
    print('SUPERVISOR_COMPLETE '+json.dumps(report),flush=True)
    return 0 if full or ((a.state_index is not None or a.layout_index is not None) and all(x['exit_code']==0 for x in outcomes)) else 2
if __name__=='__main__':sys.exit(main())
