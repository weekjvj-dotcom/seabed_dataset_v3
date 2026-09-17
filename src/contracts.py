"""Pure-Python configuration, deterministic plan and completed-artifact contracts."""
import hashlib
import json
import math
import os
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CLASSES={'0':'background','1':'sand','2':'rock','3':'coral','4':'grass','5':'fish'}
REFERENCE_FILES=('clear_rgb.exr','clear_rgb.png','depth_range_m.exr','depth_camera_z_m.exr','semantic_id.png','instance_id.png','valid_mask.png','metadata.json')
SAMPLE_FILES=('degraded_rgb.exr','degraded_rgb.png','metadata.json')
LAYOUT_FAMILIES=('curved_dual_ridge','left_concentrated_island','diagonal_scatter')

def layout_family_for_seed(seed):
    index={101:0,202:1,303:2}.get(seed,abs(int(seed))%3)
    return LAYOUT_FAMILIES[index]

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def source_hash():
    paths=sorted((ROOT/'src').rglob('*.py'))+[ROOT/'run.py']
    catalog=ROOT/'assets'/'asset_catalog.json'
    if not catalog.is_file():raise ValueError('v3 requires assets/asset_catalog.json before rendering')
    asset_root=ROOT/'assets'/'source'
    asset_paths=sorted(path for path in asset_root.rglob('*') if path.is_file())
    return canonical_hash({
        'source':{str(p.relative_to(ROOT)):file_hash(p) for p in paths},
        'asset_catalog':file_hash(catalog),
        'asset_files':{str(p.relative_to(ROOT)):file_hash(p) for p in asset_paths},
    })

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(temp,path)

def derive_seed(root_seed,*parts):
    """Stable subsystem seed, independent of Python hash randomization/call order."""
    return int(canonical_hash([int(root_seed),*parts])[:8],16) & 0x7fffffff

def load_config(path):
    cfg=json.loads(Path(path).read_text());validate_config(cfg);return cfg

def _number(v,lo,hi,name):
    if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not lo<=v<=hi:raise ValueError('Invalid '+name)

def _range(values,lo,hi,name):
    if not isinstance(values,list) or len(values)!=2:raise ValueError('Expected range '+name)
    for v in values:_number(v,lo,hi,name)
    if values[0]>values[1]:raise ValueError('Reversed range '+name)

def validate_config(cfg):
    if cfg.get('schema_version')!=2:raise ValueError('Expected v2 config')
    for name in ('root_seed','geometry_states','max_attempts'):
        v=cfg[name]
        if isinstance(v,bool) or not isinstance(v,int) or v< (0 if name=='root_seed' else 1):raise ValueError('Invalid '+name)
    if cfg['max_attempts']>3:raise ValueError('At most three geometry attempts are allowed')
    if not cfg['layouts'] or len({x['seed'] for x in cfg['layouts']})!=len(cfg['layouts']):raise ValueError('Duplicate/empty layouts')
    families={}
    for x in cfg['layouts']:
        if isinstance(x['seed'],bool) or not isinstance(x['seed'],int) or x['seed']<0 or x['split'] not in ('train','validation','test'):raise ValueError('Invalid layout')
        family=layout_family_for_seed(x['seed'])
        if family in families and families[family]!=x['split']:raise ValueError('Layout family crosses split')
        families[family]=x['split']
    if set(cfg['lighting_modes'])!={'natural','artificial','mixed'} or len(cfg['lighting_modes'])!=3:raise ValueError('Expected three unique lighting modes')
    if len(set(cfg['depths_m']))!=len(cfg['depths_m']) or not cfg['depths_m']:raise ValueError('Duplicate/empty depths')
    for d in cfg['depths_m']:_number(d,3,40,'depth')
    if [p['id'] for p in cfg['presets']]!=['mild','medium','strong']:raise ValueError('Expected ordered mild/medium/strong presets')
    for p in cfg['presets']:
        for key in ('absorption','scatter'):
            if len(p[key])!=3:raise ValueError('Expected RGB '+key)
            for v in p[key]:_number(v,0,10,key)
        _number(p['g'],-.99,.99,'g');_number(p['ior'],1,2,'ior')
    for key in ('absorption','scatter'):
        for channel in range(3):
            if any(cfg['presets'][i][key][channel]>cfg['presets'][i+1][key][channel] for i in (0,1)):raise ValueError('Water levels must be monotonic')
    r=cfg['render']
    if len(r['resolution'])!=2:raise ValueError('Expected width,height')
    for v in r['resolution']+[r['samples']]:
        if isinstance(v,bool) or not isinstance(v,int) or v<1:raise ValueError('Invalid render size/samples')
    _number(r['seed'],0,2**31-1,'render seed')
    if not isinstance(r['seed'],int):raise ValueError('Render seed must be an integer')
    for name in ('max_bounces','volume_bounces'):
        _number(r[name],0,128,name)
        if not isinstance(r[name],int):raise ValueError('Bounce counts must be integers')
    cs=cfg['camera_sampling'];_number(cs['lens_mm'],10,120,'lens')
    if 'pitch_down_deg' in cs:_range(cs['pitch_down_deg'],1,89,'pitch_down_deg')
    for key in ('height_m','x_range','y_range','look_y_range','look_z_range','roll_deg'):_range(cs[key],-50,50,key)
    if cs['height_m'][0]<=0:raise ValueError('Camera height must be positive')
    for key in ('fish_scale','small_scale'):_range(cfg['object_sampling'][key],.01,3,key)
    for key in ('fish_pitch_deg','fish_roll_deg','small_yaw_deg'):_range(cfg['object_sampling'][key],-180,180,key)
    for key in ('foreground_fish','midground_fish','background_fish'):
        v=cfg['object_sampling'][key];_range(v,0,20,key)
        if any(isinstance(n,bool) or not isinstance(n,int) for n in v):raise ValueError('Fish counts must be integers')
    ls=cfg['lighting_sampling']
    limits={'sky_strength':(0,5),'sun_elevation_deg':(1,89),'sun_rotation_deg':(-360,360),'spot_energy':(0,5000),'spot_full_angle_deg':(1,179),'spot_blend':(0,1),'temperature_k':(1000,20000)}
    for key,(lo,hi) in limits.items():_range(ls[key],lo,hi,key)
    _range(cfg.get('density_multiplier',[1,1]),0,10,'density_multiplier')
    q=cfg['quality']
    for name in ('valid_fraction_min','near_fraction_min','middle_fraction_min','far_fraction_min','clipped_fraction_max','display_dynamic_range_min'):_number(q[name],0,1,name)
    for name in ('band_ecology_pixels_min','instance_pixels_min'):
        _number(q[name],1,1000000,name)
        if not isinstance(q[name],int):raise ValueError('Pixel thresholds must be integers')
    for name in ('mean_linear_min','effect_noise_min','structure_noise_min'):_number(q[name],1e-12,1e6,name)
    if not cfg.get('output_root'):raise ValueError('output_root required')
    if cfg.get('purpose')!='engineering_validation':raise ValueError('First release is engineering_validation')
    return cfg

def depth_token(depth):
    d=float(depth);return str(int(d)) if d.is_integer() else repr(d).replace('.','p')

def sample_key(state_key,mode,depth,preset_id):
    return f'{state_key}_{mode}_d{depth_token(depth)}_{preset_id}'

def make_plan(cfg):
    validate_config(cfg);states=[];refs=[];pairs=[]
    for layout in cfg['layouts']:
        for index in range(cfg['geometry_states']):
            key=f"l{layout['seed']}_g{index:03d}"
            item={'state_key':key,'layout_seed':layout['seed'],'state_index':index,'layout_family':layout_family_for_seed(layout['seed']),'split':layout['split']}
            states.append(item)
            for mode in cfg['lighting_modes']:
                refs.append({**item,'lighting_mode':mode,'reference_key':key+'_'+mode})
                for d in cfg['depths_m']:
                    for p in cfg['presets']:pairs.append({**item,'lighting_mode':mode,'sample_id':sample_key(key,mode,d,p['id']),'seabed_depth_m':d,'water_id':p['id']})
    assert len({x['sample_id'] for x in pairs})==len(pairs)
    return {'schema_version':2,'geometry_states':states,'references':refs,'pairs':pairs}

def ensure_plan(root,plan):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);path=root/'plan.json'
    if path.exists():
        old=json.loads(path.read_text())
        if canonical_hash(old)!=canonical_hash(plan):raise ValueError('Existing output has a different config/source/plan; choose a new output_root')
    else:
        if any(root.iterdir()):raise ValueError('Nonempty output root without plan.json; refuse to adopt untracked artifacts')
        atomic_json(path,plan)


def calibration_gate(cfg,source,runtime):
    path=Path(cfg['calibration_gate'])
    path=path if path.is_absolute() else ROOT/path
    gate=json.loads(path.read_text())
    if gate['status']!='pass':raise ValueError('Calibration has not passed')
    if gate['source_sha256']!=source or gate['config_sha256']!=canonical_hash(cfg):raise ValueError('Calibration config/source signature mismatch')
    if gate['runtime_sha256']!=canonical_hash(runtime):raise ValueError('Calibration runtime signature mismatch')
    if set(gate['lighting_modes'])!={'natural','artificial','mixed'}:raise ValueError('Incomplete lighting calibration')
    for evidence in gate['evidence']:
        actual=ROOT/evidence['path']
        if file_hash(actual)!=evidence['sha256']:raise ValueError('Calibration evidence changed: '+str(actual))
    return {'path':str(path.relative_to(ROOT)),'sha256':file_hash(path)}


def validate_checkpoint(checkpoint,state,plan):
    if checkpoint['status']!='accepted':raise ValueError('Checkpoint is not accepted')
    if any(checkpoint[k]!=v for k,v in state.items()):raise ValueError('Checkpoint state identity mismatch')
    attempt=checkpoint['accepted_attempt']
    if type(attempt) is not int or not 0<=attempt<plan['config']['max_attempts']:raise ValueError('Invalid accepted attempt')
    checks={'source_sha256':plan['source_hash'],'config_sha256':plan['config_hash'],
            'runtime_sha256':canonical_hash(plan['runtime']),'calibration_gate':plan['calibration_gate']}
    if any(checkpoint[k]!=v for k,v in checks.items()):raise ValueError('Checkpoint provenance mismatch')
    jobs={x['sample_id']:x for x in plan['pairs'] if x['state_key']==state['state_key']}
    rows=checkpoint['manifest_rows']
    if len(rows)!=len(jobs) or {x['sample_id'] for x in rows}!=set(jobs):raise ValueError('Checkpoint job set mismatch')
    for row in rows:
        job=jobs[row['sample_id']]
        if any(row[k]!=v for k,v in job.items()):raise ValueError('Checkpoint job identity mismatch')
        if row['accepted_attempt']!=attempt:raise ValueError('Checkpoint row attempt mismatch')
    refs=checkpoint['reference_ids']
    if len(refs)!=len(plan['config']['lighting_modes']) or set(refs)!={x['reference_id'] for x in rows}:raise ValueError('Checkpoint reference set mismatch')
    return checkpoint

def seal_record(root,files,metadata):
    root=Path(root)
    record={'schema_version':2,'metadata':metadata,'files':{str(p):{'sha256':file_hash(root/p),'bytes':(root/p).stat().st_size} for p in sorted(files)}}
    atomic_json(root/'success.json',record);return record

def complete_record(root,expected_files=None):
    root=Path(root)
    try:
        record=json.loads((root/'success.json').read_text())
        if record['schema_version']!=2:return False
        if not record['files']:return False
        if expected_files is not None and set(record['files'])!=set(expected_files):return False
        for name,item in record['files'].items():
            p=(root/name).resolve()
            if not p.is_relative_to(root.resolve()) or not p.is_file() or p.stat().st_size!=item['bytes'] or file_hash(p)!=item['sha256']:return False
        return True
    except (OSError,ValueError,KeyError,TypeError):return False
