"""Atomic bundles, append-only event log and manifest assembly."""
import csv,json,os,time,shutil,struct
import numpy as np
from pathlib import Path
from .contracts import atomic_json,canonical_hash,complete_record,seal_record,REFERENCE_FILES,SAMPLE_FILES,validate_checkpoint


def event(root,kind,**fields):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    row={'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'event':kind,**fields}
    with (root/'events.jsonl').open('a') as f:f.write(json.dumps(row,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')
    return row


def readable_bundle(path,kind,expected=None):
    from .labels.formats import read_png,read_exr
    from .render import read_render_rgb
    path=Path(path);required=REFERENCE_FILES if kind=='reference' else SAMPLE_FILES
    if not complete_record(path,required):return False
    try:
        meta=json.loads((path/'metadata.json').read_text())
        if meta['schema_version']!=2:return False
        if not meta['rgb_quality']['passed'] or not meta['geometry_quality']['passed']:return False
        if kind=='sample' and meta['state_after_sha256']!=meta['state_sha256']:return False
        if kind=='reference' and (meta['water_objects_in_scene'] or meta['clear_definition']!='No seawater volume or air-water surface'):return False
        if expected and any(k not in meta or canonical_hash(meta[k])!=canonical_hash(v) for k,v in expected.items()):return False
        seal=json.loads((path/'success.json').read_text())
        if any(k not in meta or canonical_hash(meta[k])!=canonical_hash(v) for k,v in seal['metadata'].items()):return False
        render=meta['render'];contract=meta['render_contract']
        if any(render[k]!=v for k,v in contract.items()):return False
        if render['exr']['pixel_type']!='FLOAT32' or render['exr']['space']!='Linear Rec.709' or render['exr']['display_transform_applied']:return False
        if any(render['png'].get(k)!=v for k,v in {'bits':16,'display':'sRGB','view_transform':'Standard','look':'None','exposure':0,'gamma':1}.items()):return False
        width,height=meta['render']['resolution']
        labels={}
        for name in required:
            p=path/name
            if name.endswith('.png'):
                data=p.read_bytes()
                w,h,bits,color=struct.unpack('>IIBB',data[16:26])
                label=name in ('semantic_id.png','instance_id.png','valid_mask.png')
                if data[:8]!=b'\x89PNG\r\n\x1a\n' or data[8:16]!=b'\x00\x00\x00\rIHDR' or data[26:29]!=b'\0\0\0' or [w,h]!=[width,height] or bits!=(8 if name=='valid_mask.png' else 16) or color!=(0 if label else 2):return False
                if name in ('semantic_id.png','instance_id.png','valid_mask.png'):
                    im=read_png(p)
                    if [im['width'],im['height']]!=[width,height]:return False
                    labels[name]=np.asarray(im['pixels'])
                else:
                    rgb=read_render_rgb(p)
                    if rgb.shape!=(height,width,3) or not np.isfinite(rgb).all():return False
            elif name.endswith('.exr'):
                if name.startswith('depth_'):
                    im=read_exr(p)
                    if [im['width'],im['height']]!=[width,height] or im['channels']!=['Y'] or im['pixel_type']!='FLOAT':return False
                    labels[name]=np.asarray(im['pixels'])
                    if not np.isfinite(labels[name]).all():return False
                else:
                    if not rgb_exr_header_valid(p,(width,height)):return False
                    rgb=read_render_rgb(p)
                    if rgb.shape!=(height,width,3) or not np.isfinite(rgb).all():return False
        if kind=='reference':
            mask=labels['valid_mask.png'];sem=labels['semantic_id.png'];inst=labels['instance_id.png'];valid=mask==255
            if not set(np.unique(mask)).issubset({0,255}) or not set(np.unique(sem)).issubset(set(range(6))):return False
            if not np.array_equal(valid,sem>0) or not np.array_equal(valid,inst>0):return False
            for name in ('depth_range_m.exr','depth_camera_z_m.exr'):
                if np.any(labels[name][valid]<=0) or np.any(labels[name][~valid]!=0):return False
            registry=meta['full_instance_registry']
            for iid in np.unique(inst[valid]):
                if str(iid) not in registry or not np.all(sem[inst==iid]==registry[str(iid)]['semantic_id']):return False
        return True
    except (ValueError,OSError,RuntimeError,KeyError,TypeError,struct.error,IndexError):return False


def rgb_exr_header_valid(path,dimensions=None):
    """Read the channel header independently of Blender's image conversion."""
    data=Path(path).read_bytes()
    if data[:4]!=struct.pack('<I',20000630):return False
    version=struct.unpack_from('<I',data,4)[0]
    # Blender marks its scanline files as long-name capable (0x400).
    # Reject tiled/deep/multipart flags while accepting that native flag.
    if version & 0xff!=2 or version & ~(0xff|0x400):return False
    offset=8;channels=None;compression=None;window=None
    def string(index):
        end=data.index(b'\0',index)
        return data[index:end].decode('ascii'),end+1
    while data[offset]!=0:
        name,offset=string(offset);kind,offset=string(offset)
        size=struct.unpack_from('<I',data,offset)[0];offset+=4;end=offset+size
        if name=='channels':
            if kind!='chlist':return False
            channels={};pos=offset
            while pos<end and data[pos]:
                channel,pos=string(pos)
                if channel in channels:return False
                pixel_type=struct.unpack_from('<i',data,pos)[0];pos+=16;channels[channel]=pixel_type
            if pos!=end-1 or data[pos]!=0:return False
        elif name=='compression':compression=data[offset:end] if kind=='compression' else None
        elif name=='dataWindow':window=struct.unpack_from('<iiii',data,offset) if kind=='box2i' and size==16 else None
        offset=end
    return channels=={'R':2,'G':2,'B':2} and compression==b'\x03' and window is not None and (dimensions is None or window==(0,0,dimensions[0]-1,dimensions[1]-1))


def promote_directory(stage,target,backup_root):
    stage,target,backup_root=Path(stage),Path(target),Path(backup_root)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        backup_root.mkdir(parents=True,exist_ok=True)
        backup=backup_root/(target.name+'_'+str(time.time_ns()))
        os.replace(target,backup)
    os.replace(stage,target)


def rebuild_indexes(root,plan):
    root=Path(root);rows=[]
    for state in plan['geometry_states']:
        checkpoint=root/'states'/state['state_key']/'accepted.json'
        if checkpoint.exists():rows.extend(validate_checkpoint(json.loads(checkpoint.read_text()),state,plan)['manifest_rows'])
    if len({r['sample_id'] for r in rows})!=len(rows):raise RuntimeError('Duplicate sample ID in manifest')
    rows.sort(key=lambda x:x['sample_id']);path=root/'manifest.jsonl';tmp=path.with_suffix('.tmp')
    tmp.write_text(''.join(json.dumps(r,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n' for r in rows));os.replace(tmp,path)
    events=[]
    log=root/'events.jsonl'
    if log.exists():events=[json.loads(x) for x in log.read_text().splitlines() if x]
    columns=['time_utc','event','state_key','state_index','attempt_index','geometry_id','layout_family','split','sample_id','reference_id','lighting_mode','water_id','seed','water_seed','lighting_seed','visible_fish_count','fish_morphologies','depth_bands','mean_linear','rgb_quality','geometry_quality','artifact_files','candidate_evidence','render_success','depth_success','status','invalid_reason','render_seconds']
    temp=root/'generation_log.csv.tmp'
    with temp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader()
        for e in events:
            e=dict(e)
            for k,v in e.items():
                if isinstance(v,(list,dict)):e[k]=json.dumps(v,sort_keys=True,ensure_ascii=False)
            writer.writerow(e)
    os.replace(temp,root/'generation_log.csv')
    return rows
