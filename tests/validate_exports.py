"""Independently decode every export with OpenEXR + Pillow; no bpy required."""
import sys,json,struct,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.deps/py313'))
import OpenEXR,numpy as np
from PIL import Image
from src.contracts import file_hash,atomic_json,REFERENCE_FILES,SAMPLE_FILES,canonical_hash

def exr(path):
    with OpenEXR.File(str(path),separate_channels=True) as f:
        channels={k:c.pixels.copy() for k,c in f.channels().items()}
        assert channels and all(v.dtype==np.float32 and np.isfinite(v).all() for v in channels.values()),str(path)
        # dtype from OpenEXR preserves HALF/FLOAT storage, and is checked above.
        return channels

def png(path,bits=None):
    raw=Path(path).read_bytes();assert raw[:8]==b'\x89PNG\r\n\x1a\n'
    w,h,depth,col=struct.unpack('>IIBB',raw[16:26])
    if bits is not None:assert depth==bits,(path,depth,bits)
    with Image.open(path) as im:im.load();arr=np.array(im)
    assert arr.shape[:2]==(h,w)
    return arr,{'width':w,'height':h,'bits':depth,'color_type':col}

def check_hashes(directory,required):
    record=json.loads((directory/'success.json').read_text());assert record['files']
    assert set(record['files'])==set(required)
    for name,r in record['files'].items():
        p=directory/name;assert p.is_file() and p.stat().st_size==r['bytes'] and file_hash(p)==r['sha256'],str(p)

def validate(root):
    root=Path(root);plan=json.loads((root/'plan.json').read_text());manifest=[json.loads(x) for x in (root/'manifest.jsonl').read_text().splitlines() if x]
    assert len(manifest)==len(plan['jobs']);assert len({r['sample_id'] for r in manifest})==len(manifest)
    assert {r['sample_id'] for r in manifest}=={j['sample_key'] for j in plan['jobs']}
    expected_wh=plan['config']['render']['resolution'];refs={};samples=[];splits={};water_templates={}
    jobs={j['sample_key']:j for j in plan['jobs']}
    for row in manifest:
        job=jobs[row['sample_id']]
        for field in ('layout_seed','layout_family','split','seabed_depth_m'):assert row[field]==job[field]
        assert row['preset']==job['preset']['id']
        assert row['sample_path']=='samples/'+row['sample_id'] and row['reference_path']=='references/'+row['reference_id']
        sd=root/row['sample_path'];rd=root/row['reference_path'];check_hashes(sd,SAMPLE_FILES)
        m=json.loads((sd/'metadata.json').read_text());assert m['reference_id']==row['reference_id']
        for field in ('layout_seed','layout_family','split','seabed_depth_m','view'):assert m[field]==job[field]
        assert m['sample_id']==row['sample_id'] and m['reference_path']==row['reference_path']
        assert m['state_sha256']==m['state_after_sha256']
        assert m['config_sha256']==plan['config_hash'] and m['source_sha256']==plan['source_hash']
        render=m['render'];assert render['engine']=='CYCLES' and render['rgb_pixels_generated_by']=='Cycles path tracing'
        assert render['exr']['space']=='Linear Rec.709' and not render['exr']['display_transform_applied']
        assert render['png']=={'bits':16,'display':'sRGB','view_transform':'Standard','look':'None','exposure':0,'gamma':1}
        assert render['samples']==plan['config']['render']['samples'] and not render['denoising']
        rgb=exr(sd/'degraded_rgb.exr');p,ph=png(sd/'degraded_rgb.png',16)
        assert set(rgb)=={'R','G','B'} and list(rgb['R'].shape[::-1])==expected_wh
        assert [ph['width'],ph['height']]==expected_wh
        assert ph['color_type']==2
        assert m['water']['enabled'] and m['water']['boundary']['closed']
        boundary=m['water']['boundary'];assert abs(boundary['surface_z']-job['seabed_depth_m'])<1e-5
        assert boundary['bottom_z']==-3 and boundary['extent_x']==[-70,70] and boundary['extent_y']==[-70,70]
        actual=m['water']['preset'];expected=job['preset']
        assert actual['id']==expected['id']
        for field in ('absorption','scatter','g','ior'):assert np.allclose(actual[field],expected[field],rtol=0,atol=1e-6)
        template_hash=canonical_hash(m['water_template']);assert template_hash==m['water_template_sha256']
        template_key=(row['layout_seed'],row['preset'])
        if template_key in water_templates:assert water_templates[template_key]==template_hash
        water_templates[template_key]=template_hash
        assert m['camera_submersion_m']>0
        assert abs(m['camera_submersion_m']-(job['seabed_depth_m']-job['view']['eye'][2]))<1e-5
        seed=str(row['layout_seed']);splits.setdefault(seed,set()).add(row['split'])
        samples.append({'sample_id':row['sample_id'],'mean_linear_rgb':[float(rgb[c].mean()) for c in 'RGB'],'passed':True})
        if row['reference_id'] in refs:continue
        check_hashes(rd,REFERENCE_FILES);rm=json.loads((rd/'metadata.json').read_text());labels=rm['labels'];assert rm['state_sha256']==m['state_sha256'] and not rm['water_objects_in_scene']
        assert rm['source_sha256']==plan['source_hash'] and rm['config_sha256']==plan['config_hash']
        assert rm['state']['materials'] and rm['render']['engine']=='CYCLES'
        target_rows=rm['state']['geometry'];assert {str(o['instance_id']) for o in target_rows}==set(labels['instances'])
        assert {o['name'] for o in target_rows}==set(labels['target_objects'])
        clear=exr(rd/'clear_rgb.exr');cp,ch=png(rd/'clear_rgb.png',16);assert set(clear)=={'R','G','B'}
        assert ch['color_type']==2
        dr=exr(rd/'depth_range_m.exr')['Y'];dz=exr(rd/'depth_camera_z_m.exr')['Y']
        sem,sh=png(rd/'semantic_id.png',16);inst,ih=png(rd/'instance_id.png',16);mask,mh=png(rd/'valid_mask.png',8);valid=mask==255
        assert sh['color_type']==ih['color_type']==mh['color_type']==0
        for a in (clear['R'],dr,dz,sem,inst,mask):assert list(a.shape[::-1])==expected_wh
        assert set(np.unique(sem)).issubset(set(range(6))) and set(np.unique(mask)).issubset({0,255})
        assert np.array_equal(sem>0,valid) and np.array_equal(inst>0,valid)
        assert np.all(dr[~valid]==0) and np.all(dz[~valid]==0) and np.all(dr[valid]>0) and np.all(dz[valid]>0)
        K=np.array(labels['K']);w,h=expected_wh;u,v=np.meshgrid(np.arange(w)+.5,np.arange(h)+.5)
        expected_ratio=np.sqrt(1+((u-K[0,2])/K[0,0])**2+((v-K[1,2])/K[1,1])**2)
        rel=np.abs(dr[valid]-dz[valid]*expected_ratio[valid])/dr[valid];assert float(rel.max())<1e-5
        imap=labels['instances'];assert set(map(str,np.unique(inst[valid]))).issubset(imap)
        visible=[]
        for iid in np.unique(inst[valid]):
            item=imap[str(iid)];pixels=inst==iid;assert np.all(sem[pixels]==item['semantic_id']);assert int(pixels.sum())==item['visible_pixel_count'];visible.append({'instance_id':int(iid),'semantic_id':item['semantic_id'],'morphology':item['asset_morphologies'],'pixels':int(pixels.sum())})
        counts={str(int(i)):int((sem==i).sum()) for i in np.unique(sem)}
        for c in range(1,6):assert counts.get(str(c),0)>0,('Missing visible class',row['reference_id'],c)
        refs[row['reference_id']]={'path':row['reference_path'],'resolution':expected_wh,'valid_fraction':float(valid.mean()),'range_minmax':[float(dr[valid].min()),float(dr[valid].max())],'depth_range_z_max_relative_error':float(rel.max()),'class_pixel_counts':counts,'visible_instances':visible,'passed':True}
    assert all(len(s)==1 for s in splits.values())
    expected_conditions={(j['layout_seed'],canonical_hash({k:v for k,v in j['view'].items() if k!='id'})) for j in plan['jobs']}
    assert len(refs)==len(expected_conditions)
    for reference_id in refs:
        related=[r for r in manifest if r['reference_id']==reference_id]
        assert len(related)>=len(plan['config']['depths_m'])*len(plan['config']['presets'])
    return {'status':'pass','pairs':len(manifest),'references':len(refs),'independent_readers':{'OpenEXR':OpenEXR.__version__,'Pillow':Image.__version__},'all_hashes_verified':True,'references_detail':refs,'samples':samples,'layout_splits':{k:list(v)[0] for k,v in splits.items()}}

def main():
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--report');a=p.parse_args()
    result=validate(a.root);path=Path(a.report) if a.report else Path(a.root)/'validation.json';atomic_json(path,result)
    print(json.dumps({'status':result['status'],'pairs':result['pairs'],'references':result['references'],'report':str(path)}))
if __name__=='__main__':main()
