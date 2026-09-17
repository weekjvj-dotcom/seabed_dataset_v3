"""One serial Blender worker for one deterministic geometry state."""
import argparse,json,sys,time,platform,traceback,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent))
import bpy,numpy as np
from random import Random
from src.contracts import load_config,make_plan,canonical_hash,derive_seed,source_hash,file_hash,atomic_json,ensure_plan,seal_record,REFERENCE_FILES,SAMPLE_FILES,sample_key,calibration_gate,validate_checkpoint
from src.scene import build_geometry,build_lighting_groups,save_native
from src.state import state_digest,geometry_digest,assert_clear_scene,water_template
from src.lighting import lighting_metadata
from src.render import configure,select_device,render_rgb,read_render_rgb
from src.water import water_metadata
from src.labels import export_labels
from src.labels.formats import read_png,read_exr
from src.quality import geometry_metrics,geometry_decision,rgb_metrics,rgb_decision
from src.gates import geometry_gate
from src.storage import event,readable_bundle,promote_directory,rebuild_indexes


class CandidateRejected(Exception):
    def __init__(self,reason,evidence=None):
        super().__init__(reason);self.evidence=evidence


def runtime_signature(cfg):
    configure(bpy.context.scene,cfg);device=select_device(bpy.context.scene)
    ocio=Path(bpy.utils.resource_path('LOCAL'))/'datafiles/colormanagement/config.ocio'
    if not ocio.is_file():raise RuntimeError('Native OCIO config was not found')
    disabled='--disable-autoexec' in sys.argv
    if not disabled:raise RuntimeError('Pipeline requires --disable-autoexec')
    return {'blender':bpy.app.version_string,'build_hash':bpy.app.build_hash.decode(),'python':platform.python_version(),'os':platform.mac_ver()[0],'device':device,'working_space':'Linear Rec.709','display':'Standard/sRGB','ocio_config_sha256':file_hash(ocio),'autoexec_disabled_cli':disabled,'user_preference_scripts_auto_execute':bpy.context.preferences.filepaths.use_scripts_auto_execute}


def water_plan(cfg,state,attempt):
    seed=derive_seed(cfg['root_seed'],'water',state['layout_seed'],state['state_index'],attempt)
    rng=Random(seed);lo,hi=cfg.get('density_multiplier',[1,1]);k=rng.uniform(lo,hi)
    variants=[]
    for base in cfg['presets']:
        p=dict(base);p['absorption']=[k*x for x in base['absorption']];p['scatter']=[k*x for x in base['scatter']];variants.append(p)
    return {'seed':seed,'density_multiplier':k,'base_presets':cfg['presets'],'effective_presets':variants}


def label_arrays(folder):
    return (np.asarray(read_exr(folder/'depth_range_m.exr')['pixels'],dtype=np.float32),np.asarray(read_png(folder/'semantic_id.png')['pixels'],dtype=np.uint16),np.asarray(read_png(folder/'instance_id.png')['pixels'],dtype=np.uint16),np.asarray(read_png(folder/'valid_mask.png')['pixels'],dtype=np.uint8))


def assert_unchanged(scene,objects,digest):
    current,_=state_digest(scene,objects)
    if current!=digest:raise RuntimeError('FATAL_PAIR_STATE_MISMATCH: '+scene.name)


def run_candidate(cfg,plan,state,attempt,root,identity,accepted=None,preview=False):
    started=time.time();key=state['state_key'];folder=root/'attempts'/key/f'a{attempt:02d}'
    if folder.exists():
        archived=root/'rejected'/key/(f'a{attempt:02d}_previous_'+str(time.time_ns()));archived.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(folder),str(archived))
    folder.mkdir(parents=True,exist_ok=True)
    event(root,'candidate_started',state_key=key,state_index=state['state_index'],attempt_index=attempt,status='running')
    handle=build_geometry(cfg,{'seed':state['layout_seed'],'split':state['split']},state['state_index'],attempt)
    pool=handle['objects'];active=handle['active_objects'];geo_plan=handle['geometry_plan'];registry={str(x['instance_id']):x for x in geo_plan['instances']}
    geo_sha,geo_snapshot=geometry_digest(handle['base'],pool);geometry_id='geo_'+geo_sha[:24]
    geometa=export_labels(handle['base'],folder/'geometry',active);depth,sem,inst,mask=label_arrays(folder/'geometry')
    gm=geometry_metrics(depth,sem,inst,registry,cfg);gd=geometry_gate(gm,inst,registry,cfg)
    atomic_json(folder/'geometry_plan.json',geo_plan);atomic_json(folder/'geometry_quality.json',gd)
    if not gd['passed']:raise CandidateRejected('geometry:'+','.join(gd['reasons']),gd)
    wp=water_plan(cfg,state,attempt)
    if accepted:
        for name,current in (('geometry_id',geometry_id),('geometry_plan',geo_plan),('lighting_plan',handle['lighting_plan']),('water_plan',wp)):
            if canonical_hash(accepted[name])!=canonical_hash(current):raise RuntimeError('FATAL_ACCEPTED_STATE_REPLAY_MISMATCH: '+name)
    groups=build_lighting_groups(handle,cfg,key,wp['effective_presets']);new_refs=[];new_samples=[];rows=[];ref_ids=[]
    for group in groups:
        mode=group['mode'];clear=group['clear'];rig_meta=lighting_metadata(group['lighting']);rig_meta['clearance']=group['lighting']['clearance'];assert_clear_scene(clear,pool)
        digest,scene_snapshot=state_digest(clear,pool);ref_id='ref_'+digest[:24];ref_ids.append(ref_id)
        for variant in group['variants']:assert_unchanged(variant['scene'],pool,digest)
        common={**identity,'state_key':key,'state_index':state['state_index'],'attempt_index':attempt,'layout_seed':state['layout_seed'],'layout_family':state['layout_family'],'split':state['split'],'geometry_id':geometry_id,'reference_id':ref_id,'state_sha256':digest,'lighting_mode':mode,'lighting':rig_meta,'geometry_quality':gd,'render_contract':{'engine':'CYCLES','resolution':cfg['render']['resolution'],'samples':cfg['render']['samples'],'seed':cfg['render']['seed'],'denoising':False}}
        ref_target=root/'references'/ref_id;ref_stage=folder/'references'/ref_id
        if accepted and readable_bundle(ref_target,'reference',common):
            refmeta=json.loads((ref_target/'metadata.json').read_text());ref_stats=refmeta['render'];event(root,'reference_reused',state_key=key,reference_id=ref_id,lighting_mode=mode,status='reused')
        else:
            ref_stage.mkdir(parents=True,exist_ok=True);ref_stats=render_rgb(clear,ref_stage/'clear_rgb');assert_unchanged(clear,pool,digest)
            for f in ['depth_range_m.exr','depth_camera_z_m.exr','semantic_id.png','instance_id.png','valid_mask.png']:shutil.copy2(folder/'geometry'/f,ref_stage/f)
            metrics=rgb_metrics(read_render_rgb(ref_stage/'clear_rgb.exr'),mask,sem);decision=rgb_decision(metrics,cfg,mode)
            atomic_json(ref_stage/'quality.json',decision)
            if not decision['passed']:raise CandidateRejected(mode+'_clear:'+','.join(decision['reasons']),decision)
            refmeta={**common,'state':scene_snapshot,'geometry_state':geo_snapshot,'geometry_plan':geo_plan,'full_instance_registry':registry,'labels':geometa,'geometry_quality':gd,'rgb_quality':decision,'render':ref_stats,'clear_definition':'No seawater volume or air-water surface','water_objects_in_scene':[o.name for o in clear.objects if o.get('is_water')],'asset_summary':handle['ecology']['summary']}
            atomic_json(ref_stage/'metadata.json',refmeta);seal_record(ref_stage,REFERENCE_FILES,common)
            if not readable_bundle(ref_stage,'reference',common):raise RuntimeError('FATAL_REFERENCE_EXPORT_INVALID')
            new_refs.append((ref_stage,ref_target));event(root,'reference_rendered',state_key=key,reference_id=ref_id,lighting_mode=mode,render_success=True,depth_success=True,status='candidate',render_seconds=ref_stats['seconds'])
        for variant in group['variants']:
            preset=variant['preset'];depth_m=variant['depth'];wet=variant['scene'];wm=water_metadata(variant['water']);template=water_template(variant['water'])
            sid=sample_key(key,mode,depth_m,preset['id']);target=root/'samples'/sid;stage=folder/'samples'/sid
            expected={**common,'sample_id':sid,'reference_path':str(ref_target.relative_to(root)),'seabed_depth_m':depth_m,'camera_submersion_m':depth_m-handle['camera'].matrix_world.translation.z,'water':wm,'water_template_sha256':canonical_hash(template)}
            assert_unchanged(wet,pool,digest)
            if accepted and readable_bundle(target,'sample',expected):
                meta=json.loads((target/'metadata.json').read_text());stats=meta['render'];decision=meta['rgb_quality'];event(root,'sample_reused',state_key=key,sample_id=sid,reference_id=ref_id,lighting_mode=mode,water_id=preset['id'],status='reused')
            else:
                stage.mkdir(parents=True,exist_ok=True);stats=render_rgb(wet,stage/'degraded_rgb');assert_unchanged(wet,pool,digest)
                metrics=rgb_metrics(read_render_rgb(stage/'degraded_rgb.exr'),mask,sem);decision=rgb_decision(metrics,cfg,mode)
                atomic_json(stage/'quality.json',decision)
                if not decision['passed']:raise CandidateRejected(mode+'_'+preset['id']+':'+','.join(decision['reasons']),decision)
                meta={**expected,'water_plan':wp,'water_template':template,'geometry_quality':gd,'rgb_quality':decision,'render':stats,'state_after_sha256':digest,'visible_fish_count':gm['visible_fish_count']}
                atomic_json(stage/'metadata.json',meta);seal=seal_record(stage,SAMPLE_FILES,expected)
                if not readable_bundle(stage,'sample',expected):raise RuntimeError('FATAL_SAMPLE_EXPORT_INVALID')
                new_samples.append((stage,target));event(root,'sample_rendered',state_key=key,state_index=state['state_index'],attempt_index=attempt,geometry_id=geometry_id,layout_family=state['layout_family'],split=state['split'],sample_id=sid,reference_id=ref_id,lighting_mode=mode,water_id=preset['id'],seed=geo_plan['seed_streams'],water_seed=wp['seed'],lighting_seed=handle['lighting_plan'].get('seed_streams'),visible_fish_count=gm['visible_fish_count'],fish_morphologies=gm['fish_morphologies'],depth_bands=gm['bands'],mean_linear=metrics['mean_linear'],rgb_quality=decision,artifact_files=seal['files'],render_success=True,depth_success=True,status='candidate',render_seconds=stats['seconds'])
            rows.append({'sample_id':sid,'state_key':key,'state_index':state['state_index'],'geometry_id':geometry_id,'reference_id':ref_id,'reference_path':str(ref_target.relative_to(root)),'sample_path':str(target.relative_to(root)),'layout_seed':state['layout_seed'],'layout_family':state['layout_family'],'split':state['split'],'lighting_mode':mode,'water_id':preset['id'],'seabed_depth_m':depth_m,'accepted_attempt':attempt})
            print('CANDIDATE_PAIR_COMPLETE '+sid,flush=True)
    native_target=root/'scenes'/f'{key}.blend';native_target.parent.mkdir(exist_ok=True)
    if native_target.exists():shutil.copy2(native_target,folder/'previous_native.blend')
    # Save directly to its final location.  V3 external PBR images are stored
    # relative to this native blend, so saving to an attempt folder and copying
    # the result would invalidate an otherwise correct relative asset path.
    save_native(handle,native_target,cfg,{'provenance':identity,'geometry_id':geometry_id,'reference_ids':ref_ids,'geometry_plan':geo_plan,'lighting_plan':handle['lighting_plan'],'water_plan':wp})
    for stage,target in new_refs+new_samples:promote_directory(stage,target,root/'rejected'/'repaired_bundles')
    checkpoint={**identity,**state,'status':'accepted','accepted_attempt':attempt,'geometry_id':geometry_id,'geometry_plan':geo_plan,'lighting_plan':handle['lighting_plan'],'water_plan':wp,'geometry_quality':gd,'reference_ids':ref_ids,'manifest_rows':rows,'blend':str(native_target.relative_to(root)),'blend_sha256':file_hash(native_target),'blend_bytes':native_target.stat().st_size,'seconds':time.time()-started}
    atomic_json(root/'states'/key/'accepted.json',checkpoint);event(root,'state_accepted',state_key=key,state_index=state['state_index'],attempt_index=attempt,status='accepted',visible_fish_count=gm['visible_fish_count'])
    rebuild_indexes(root,plan);return checkpoint


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--state-index',type=int,required=True);p.add_argument('--layout-index',type=int,default=0);p.add_argument('--preview',action='store_true');p.add_argument('--attempt',type=int)
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:]);cfg=load_config(args.config);grid=make_plan(cfg);runtime=runtime_signature(cfg)
    root=Path(cfg['output_root']);root=root if root.is_absolute() else ROOT/root
    if args.preview:root=ROOT/'previews'/('config_'+canonical_hash(cfg)[:12])
    source=source_hash();baseline=json.loads((ROOT/'reports/baseline_provenance.json').read_text())
    gate=None if args.preview else calibration_gate(cfg,source,runtime)
    plan={**grid,'config':cfg,'config_hash':canonical_hash(cfg),'source_hash':source,'runtime':runtime,'calibration_gate':gate,'baseline_source_sha256':baseline['baseline_source_sha256'],'baseline_blend_sha256':baseline['baseline_blend_sha256']}
    ensure_plan(root,plan)
    identity={'schema_version':2,'source_sha256':source,'config_sha256':plan['config_hash'],'runtime_signature':runtime,'runtime_sha256':canonical_hash(runtime),'calibration_gate':gate,'baseline_source_sha256':baseline['baseline_source_sha256'],'baseline_blend_sha256':baseline['baseline_blend_sha256']}
    layout=cfg['layouts'][args.layout_index];state=next(x for x in grid['geometry_states'] if x['layout_seed']==layout['seed'] and x['state_index']==args.state_index)
    cp=root/'states'/state['state_key']/'accepted.json';accepted=json.loads(cp.read_text()) if cp.exists() else None
    if accepted:validate_checkpoint(accepted,state,plan)
    if accepted and any(canonical_hash(accepted[k])!=canonical_hash(v) for k,v in identity.items()):raise RuntimeError('FATAL_ACCEPTED_CHECKPOINT_IDENTITY_MISMATCH')
    attempts=[accepted['accepted_attempt']] if accepted else ([args.attempt] if args.attempt is not None else range(cfg['max_attempts']))
    for attempt in attempts:
        try:
            result=run_candidate(cfg,plan,state,attempt,root,identity,accepted,args.preview);print('STATE_COMPLETE '+json.dumps({'state_key':state['state_key'],'accepted_attempt':attempt,'pairs':len(result['manifest_rows']),'root':str(root)}),flush=True);return
        except CandidateRejected as exc:
            event(root,'candidate_rejected',state_key=state['state_key'],state_index=state['state_index'],attempt_index=attempt,status='rejected',invalid_reason=str(exc),candidate_evidence=exc.evidence);rebuild_indexes(root,plan)
            if accepted:raise RuntimeError('Accepted-state repair failed quality; do not resample') from exc
        except ValueError as exc:
            # Explicit candidate geometry failures are recoverable. Schema,
            # pairing, export and I/O failures are not silently skipped.
            if not str(exc).startswith(('CANDIDATE_','camera placement','fish placement','No active target')):raise
            event(root,'candidate_rejected',state_key=state['state_key'],attempt_index=attempt,status='rejected',invalid_reason=str(exc));rebuild_indexes(root,plan)
            if accepted:raise RuntimeError('Accepted-state repair produced invalid geometry; do not resample') from exc
    event(root,'state_failed',state_key=state['state_key'],status='failed',invalid_reason='max_attempts_exhausted');rebuild_indexes(root,plan);raise SystemExit(2)
if __name__=='__main__':main()
