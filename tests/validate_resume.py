"""Actual resume gates: exact cache reuse, numerical RGB rebuild, exact GT restore."""
import sys,json,subprocess,shutil,argparse,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.deps/py313'))
from src.contracts import load_config,file_hash,atomic_json
import OpenEXR,numpy as np

RGB_NRMSE_LIMIT=.001  # Same normalized-RMSE scale as the approved zero-medium check.

def image_hashes(root):
    return {str(p.relative_to(root)):file_hash(p) for p in root.rglob('*') if p.is_file() and p.suffix in ('.png','.exr')}

def numeric_rgb(previous,recovered):
    def read(path):
        with OpenEXR.File(str(path),separate_channels=True) as f:return np.stack([f.channels()[c].pixels for c in 'RGB'],-1)
    a,b=read(previous),read(recovered);assert a.shape==b.shape and np.isfinite(b).all()
    difference=b-a;rmse=float(np.sqrt(np.mean(difference*difference)));nrmse=rmse/max(float(np.sqrt(np.mean(a*a))),1e-12)
    result={'normalized_rmse':nrmse,'rmse':rmse,'max_abs_difference':float(np.abs(difference).max()),'limit':RGB_NRMSE_LIMIT,'byte_identical':file_hash(previous)==file_hash(recovered)}
    assert nrmse<=RGB_NRMSE_LIMIT,result
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=load_config(a.config);root=Path(cfg['output_root']);root=root if root.is_absolute() else ROOT/root
    report_dir=ROOT/'reports'/('resume_'+root.name);report_dir.mkdir(parents=True,exist_ok=True);steps=[]
    def run(label,config=a.config,expected=0):
        cmd=[sys.executable,str(ROOT/'run.py'),'--config',str(Path(config).resolve()),'--resume'];start=time.time();log=report_dir/(label+'.log')
        with log.open('w') as f:result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        row={'name':label,'command':cmd,'exit_code':result.returncode,'expected_exit':expected,'seconds':time.time()-start,'log':str(log)};steps.append(row);atomic_json(report_dir/'result.json',{'status':'running','steps':steps})
        if (result.returncode==0)!=(expected==0):raise RuntimeError('Unexpected exit '+label)
    def unchanged_except(before,allowed):
        after=image_hashes(root);assert set(before)==set(after)
        assert all(after[k]==v for k,v in before.items() if k not in allowed)
        return after
    baseline=image_hashes(root);atomic_json(report_dir/'initial_image_hashes.json',baseline)
    run('completed_reuse');assert image_hashes(root)==baseline;steps[-1]['all_image_hashes_unchanged']=True
    records=[json.loads(x) for x in (root/'manifest.jsonl').read_text().splitlines() if x]
    sample=root/records[0]['sample_path'];missing=sample/'degraded_rgb.exr';backup=report_dir/'missing_original.exr';shutil.copy2(missing,backup);missing.unlink()
    run('missing_exr_recovery');assert missing.is_file();baseline=unchanged_except(baseline,{str((sample/f).relative_to(root)) for f in ('degraded_rgb.exr','degraded_rgb.png')});steps[-1]['rgb_comparison']=numeric_rgb(backup,missing)
    reference=root/records[0]['reference_path'];corrupted=reference/'instance_id.png';shutil.copy2(corrupted,report_dir/'corrupt_original.png');shutil.copy2(reference/'clear_rgb.exr',report_dir/'clear_original.exr');corrupted.write_bytes(b'INTENTIONAL_RECOVERY_TEST_INVALID_PNG')
    run('corrupt_label_recovery');baseline=unchanged_except(baseline,{str((reference/f).relative_to(root)) for f in ('clear_rgb.exr','clear_rgb.png')});steps[-1]['gt_restored_byte_exactly']=file_hash(corrupted)==file_hash(report_dir/'corrupt_original.png');assert steps[-1]['gt_restored_byte_exactly'];steps[-1]['clear_comparison']=numeric_rgb(report_dir/'clear_original.exr',reference/'clear_rgb.exr')
    # A complete, valid bundle copied under another water job must be regenerated,
    # even though both samples share the same immutable reference and source hash.
    other=root/records[1]['sample_path'];shutil.copy2(other/'degraded_rgb.exr',report_dir/'wrong_job_original.exr')
    for filename in ('degraded_rgb.exr','degraded_rgb.png','metadata.json','success.json'):shutil.copy2(sample/filename,other/filename)
    run('wrong_job_bundle_recovery');baseline=unchanged_except(baseline,{str((other/f).relative_to(root)) for f in ('degraded_rgb.exr','degraded_rgb.png')});steps[-1]['rgb_comparison']=numeric_rgb(report_dir/'wrong_job_original.exr',other/'degraded_rgb.exr');assert json.loads((other/'metadata.json').read_text())['sample_id']==records[1]['sample_id']
    altered=json.loads(json.dumps(cfg));altered['presets'][0]['absorption'][0]+=.001;invalid=report_dir/'mismatching_config.json';atomic_json(invalid,altered)
    run('different_config_refused',invalid,expected=1);assert image_hashes(root)==baseline
    after=[json.loads(x) for x in (root/'manifest.jsonl').read_text().splitlines() if x];assert len(after)==len(records) and len({r['sample_id'] for r in after})==len(after)
    result={'status':'pass_with_non_bitwise_rgb_rerender','steps':steps,'pairs':len(after),'manifest_unique':True,'cached_files_unchanged_byte_exactly':True,'gt_restored_byte_exactly':True,'rgb_rerender_nrmse_limit':RGB_NRMSE_LIMIT,'earlier_strict_hash_failure':'reports/resume/initial_hash_failure_diagnosis.json','explanation':'RGB re-render may differ slightly on Metal. Cached files must retain their exact hash; regenerated RGB must pass numerical comparison and is sealed with its new hash. No geometry/config/water changes are permitted.'};atomic_json(report_dir/'result.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
