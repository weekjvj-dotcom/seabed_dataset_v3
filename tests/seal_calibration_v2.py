"""Seal only completed, passing native calibration evidence for this source/config."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.contracts import load_config,canonical_hash,source_hash,file_hash,atomic_json
cfg=load_config(ROOT/'configs/first_round.json');caldir=(ROOT/cfg['calibration_gate']).parent;path=caldir/'result.json';r=json.loads(path.read_text())
assert r['status']=='pass' and r['source_sha256']==source_hash() and canonical_hash(r['config'])==canonical_hash(cfg)
assert r['geometry_quality']['passed'] and len(r['renders'])==12
modes={'natural','artificial','mixed'}
assert {(x['mode'],x['condition']) for x in r['renders']}=={(m,c) for m in modes for c in ('clear','mild','medium','strong')}
assert all(x['quality']['passed'] for x in r['renders']) and set(r['noise'])==modes and set(r['noise_convergence'])==modes
assert all(x['passed'] for x in r['noise'].values()) and all(x['passed'] for x in r['noise_convergence'].values())
scan=json.loads((ROOT/'reports/geometry_scan_pitch/result.json').read_text());first={x['state']:x for x in scan['candidates'] if x['attempt']==0}
assert set(first)=={0,1,2} and all(x['passed'] for x in first.values())
trial=json.loads((ROOT/'reports/camera_pitch_trial.json').read_text())
assert {k:v for k,v in trial.items() if k not in ('output_root','calibration_gate')}=={k:v for k,v in cfg.items() if k not in ('output_root','calibration_gate')}
native_log=ROOT/'reports/combined_native_metadata_fix.log'
assert '\nOK\n' in native_log.read_text()
assert json.loads((ROOT/'reports/core_preview_metadata_fix_validation.json').read_text())['status']=='pass'
assert json.loads((ROOT/'reports/cache_reuse_metadata_fix.json').read_text())['status']=='pass'
evidence=[str((caldir/'result.json').relative_to(ROOT)),str((caldir/'calibration.blend').relative_to(ROOT)),'reports/geometry_scan_pitch/result.json','reports/camera_pitch_trial.json','reports/camera_calibration_decision.json','reports/combined_native_metadata_fix.log','reports/core_preview_metadata_fix_validation.json','reports/cache_reuse_metadata_fix.json','reports/pipeline_faults/result.json','reports/supervisor_faults/result.json']
for p in sorted(caldir.rglob('*.exr')):evidence.append(str(p.relative_to(ROOT)))
gate={'status':'pass','scope':'Stage A native 512-sample three-mode calibration; all three geometry first candidates passed GT gates','source_sha256':source_hash(),'config_sha256':canonical_hash(cfg),'runtime_sha256':canonical_hash(r['runtime']),'lighting_modes':sorted(modes),'samples':cfg['render']['samples'],'noise':r['noise'],'noise_convergence':r['noise_convergence'],'first_candidate_geometry':{str(k):{'valid_fraction':v['quality']['metrics']['valid_fraction'],'bands':v['quality']['metrics']['bands'],'meaningful_visibility':v['quality']['meaningful_visibility']} for k,v in first.items()},'evidence':[{'path':p,'sha256':file_hash(ROOT/p)} for p in evidence],'limits':['synthetic RGB water coefficients','absolute cross-interface eta-squared radiance uncalibrated','procedural stylized assets','noise gate must also be checked on all final strong-lighting states']}
atomic_json(ROOT/cfg['calibration_gate'],gate);print('CALIBRATION_GATE_SEALED',canonical_hash(cfg),source_hash())
