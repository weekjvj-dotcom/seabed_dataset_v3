"""Independent OpenEXR/Pillow validation of the accepted v2 dataset."""
import sys,json,struct,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent/'seabed_dataset/.deps/py313'))
import OpenEXR,numpy as np
from PIL import Image
from src.contracts import complete_record,REFERENCE_FILES,SAMPLE_FILES,canonical_hash,atomic_json


def exr(path):
    with OpenEXR.File(str(path),separate_channels=True) as f:
        result={k:c.pixels.copy() for k,c in f.channels().items()}
        assert all(a.dtype==np.float32 and np.isfinite(a).all() for a in result.values()),str(path)
        return result


def png(path,bits,color):
    raw=path.read_bytes();w,h,depth,ct=struct.unpack('>IIBB',raw[16:26]);assert raw[:8]==b'\x89PNG\r\n\x1a\n' and depth==bits and ct==color
    with Image.open(path) as im:im.load();a=np.array(im)
    assert a.shape[:2]==(h,w);return a


def validate(root,allow_partial=False):
    root=Path(root);plan=json.loads((root/'plan.json').read_text());rows=[json.loads(x) for x in (root/'manifest.jsonl').read_text().splitlines() if x];jobs={x['sample_id']:x for x in plan['pairs']}
    assert len({x['sample_id'] for x in rows})==len(rows)
    if not allow_partial:assert len(rows)==len(jobs)
    refs={};geos={};result_rows=[];runtime_hash=canonical_hash(plan['runtime']);dims=plan['config']['render']['resolution']
    for row in rows:
        expected=jobs[row['sample_id']]
        for k in ('state_key','layout_seed','layout_family','split','lighting_mode','water_id','seabed_depth_m'):assert row[k]==expected[k]
        sd=root/row['sample_path'];rd=root/row['reference_path'];assert complete_record(sd,SAMPLE_FILES)
        m=json.loads((sd/'metadata.json').read_text());assert m['source_sha256']==plan['source_hash'] and m['config_sha256']==plan['config_hash'] and m['runtime_sha256']==runtime_hash
        for k in ('sample_id','state_key','reference_id','geometry_id','reference_path','lighting_mode','seabed_depth_m','layout_seed','split'):assert m[k]==row[k]
        assert m['state_sha256']==m['state_after_sha256'] and m['rgb_quality']['passed'] and m['geometry_quality']['passed']
        assert canonical_hash(m['water_template'])==m['water_template_sha256']
        channels=exr(sd/'degraded_rgb.exr');assert set(channels)=={'R','G','B'};image=png(sd/'degraded_rgb.png',16,2)
        assert list(channels['R'].shape[::-1])==dims and list(image.shape[:2][::-1])==dims
        assert m['water']['enabled'] and m['water']['boundary']['closed']
        assert abs(m['water']['boundary']['surface_z']-row['seabed_depth_m'])<1e-5
        assert m['render']['engine']=='CYCLES' and not m['render']['denoising'] and m['render']['samples']==plan['config']['render']['samples']
        if row['reference_id'] not in refs:
            assert complete_record(rd,REFERENCE_FILES);r=json.loads((rd/'metadata.json').read_text());assert r['state_sha256']==m['state_sha256'] and r['lighting']==m['lighting'];assert not r['water_objects_in_scene']
            assert r['source_sha256']==plan['source_hash'] and r['runtime_sha256']==runtime_hash
            clear=exr(rd/'clear_rgb.exr');png(rd/'clear_rgb.png',16,2)
            dr=exr(rd/'depth_range_m.exr')['Y'];dz=exr(rd/'depth_camera_z_m.exr')['Y'];sem=png(rd/'semantic_id.png',16,0);inst=png(rd/'instance_id.png',16,0);mask=png(rd/'valid_mask.png',8,0);valid=mask==255
            assert set(np.unique(mask)).issubset({0,255}) and set(np.unique(sem)).issubset(set(range(6)))
            assert np.array_equal(valid,sem>0) and np.array_equal(valid,inst>0)
            assert np.all(dr[~valid]==0) and np.all(dz[~valid]==0) and np.all(dz[valid]>0)
            for a in [dr,dz,sem,inst,mask,clear['R']]:assert list(a.shape[::-1])==dims
            k=np.array(r['labels']['K']);u,v=np.meshgrid(np.arange(dims[0])+.5,np.arange(dims[1])+.5);ratio=np.sqrt(1+((u-k[0,2])/k[0,0])**2+((v-k[1,2])/k[1,1])**2)
            maxerr=float(np.max(abs(dr[valid]-dz[valid]*ratio[valid])/dr[valid]));assert maxerr<1e-5
            registry=r['full_instance_registry'];labelmap=r['labels']['instances']
            for iid in np.unique(inst[valid]):
                item=registry[str(iid)];assert np.all(sem[inst==iid]==item['semantic_id']);assert str(iid) in labelmap
            state_key=row['state_key'];geo_sha=canonical_hash(r['geometry_state'])
            if state_key in geos:assert geos[state_key]==geo_sha
            geos[state_key]=geo_sha
            refs[row['reference_id']]={'state_key':state_key,'lighting_mode':row['lighting_mode'],'range_z_max_relative_error':maxerr,'valid_fraction':float(valid.mean()),'geometry_quality':r['geometry_quality'],'visible_fish_count':r['geometry_quality']['metrics']['visible_fish_count']}
        result_rows.append({'sample_id':row['sample_id'],'lighting_mode':row['lighting_mode'],'water_id':row['water_id'],'mean_rgb_linear':[float(channels[c].mean()) for c in 'RGB']})
    if not allow_partial:
        assert len(refs)==len(plan['references']) and len(geos)==len(plan['geometry_states'])
        for key in geos:
            group=[r for r in rows if r['state_key']==key];assert len(group)==9 and len({r['reference_id'] for r in group})==3
    return {'status':'pass','pairs':len(rows),'references':len(refs),'geometry_states':len(geos),'source_sha256':plan['source_hash'],'reference_details':refs,'samples':result_rows,'independent_readers':{'OpenEXR':OpenEXR.__version__,'Pillow':Image.__version__},'allow_partial':allow_partial}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--report');p.add_argument('--allow-partial',action='store_true');a=p.parse_args();r=validate(a.root,a.allow_partial);atomic_json(Path(a.report) if a.report else Path(a.root)/'validation.json',r);print(json.dumps({k:r[k] for k in ('status','pairs','references','geometry_states')}))
