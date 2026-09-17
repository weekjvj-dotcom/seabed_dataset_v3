"""Build one randomized geometry state and immutable native lighting variants."""
import bpy,json,os
from mathutils import Vector
from pathlib import Path
from .assets import build_ecosystem
from .randomization import prepare_layout,apply_geometry_state
from .lighting import sample_lighting,build_lighting
from .render import configure,select_device
from .water import build_water
from .contracts import derive_seed
from .state import geometry_constraints


def validate_light_positions(handle, rig, cfg):
    """Conservative point-to-target AABB clearance for camera-mounted sources."""
    rows=[]
    minimum=.05
    surface=min(cfg['depths_m'])
    for light in rig['objects']:
        position=light.matrix_world.translation.copy()
        if not (-69<position.x<69 and -69<position.y<69 and -2.9<position.z<surface-.1):
            raise ValueError('CANDIDATE_LIGHT_WATER_BOUNDS: '+light.name)
        distances=[]
        for obj in handle['active_objects']:
            if int(obj.get('semantic_id',0))==1:continue
            points=[obj.matrix_world@Vector(v) for v in obj.bound_box]
            lo=[min(v[i] for v in points) for i in range(3)]
            hi=[max(v[i] for v in points) for i in range(3)]
            distance=sum(max(lo[i]-position[i],0,position[i]-hi[i])**2 for i in range(3))**.5
            distances.append((distance,obj.name))
        distance,nearest=min(distances)
        if distance<minimum:raise ValueError('CANDIDATE_LIGHT_CLEARANCE: '+light.name+' '+nearest)
        rows.append({'name':light.name,'position_m':list(position),'nearest_target':nearest,
                     'target_aabb_distance_m':distance,'surface_clearance_m':surface-position.z})
    return {'passed':True,'method':'point to active target world AABB; water bounds',
            'target_clearance_min_m':minimum,'lights':rows}


def build_geometry(cfg,layout,state_index,attempt_index):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    base=bpy.context.scene;base.name=f"L{layout['seed']}_G{state_index:03d}_Geometry"
    configure(base,cfg)
    ecosystem=build_ecosystem(base,layout['seed'],0.0)
    camera=bpy.data.objects.new('Camera_Main',bpy.data.cameras.new('Camera_Main_Pinhole'));base.collection.objects.link(camera);base.camera=camera
    camera.data.type='PERSP';camera.data.sensor_fit='HORIZONTAL';camera.data.sensor_width=36;camera.data.clip_start=.01;camera.data.clip_end=150;camera.data.dof.use_dof=False
    world=bpy.data.worlds.new('Geometry_Check_Black');world.use_nodes=True;world.node_tree.nodes.get('Background').inputs['Strength'].default_value=0;base.world=world
    prepare_layout(base,ecosystem)
    local=dict(cfg,_layout_seed=layout['seed'])
    plan=apply_geometry_state(base,ecosystem,camera,local,state_index,attempt_index)
    bpy.context.view_layer.update()
    active=[o for o in ecosystem['objects'] if o.name in set(plan['visible_object_names'])]
    if not active:raise ValueError('No active target objects')
    try:geometry_constraints(base,active,min(cfg['depths_m']))
    except ValueError as exc:raise ValueError('CANDIDATE_WATER_OR_CAMERA_BOUNDS: '+str(exc)) from exc
    device=select_device(base)
    return {'base':base,'ecology':ecosystem,'camera':camera,'geometry_plan':plan,'active_objects':active,'objects':ecosystem['objects'],'device':device,'lighting_plan':sample_lighting(local,state_index,attempt_index),'groups':[]}


def build_lighting_groups(handle,cfg,state_key,water_presets):
    groups=[]
    for mode in cfg['lighting_modes']:
        clear=configure(bpy.data.scenes.new(f'{state_key}_{mode}_Clear'),cfg)
        clear.collection.children.link(handle['ecology']['collection']);clear.collection.objects.link(handle['camera']);clear.camera=handle['camera'];clear.cycles.device=handle['base'].cycles.device
        rig=build_lighting(clear,handle['camera'],handle['lighting_plan'],mode)
        clear.view_layers[0].update()
        rig['clearance']=validate_light_positions(handle,rig,cfg)
        clear.world=rig['world'];water_scenes=[]
        for depth in cfg['depths_m']:
            for p in water_presets:
                wet=configure(bpy.data.scenes.new(f"{state_key}_{mode}_D{depth:g}_{p['id']}"),cfg)
                wet.collection.children.link(handle['ecology']['collection']);wet.collection.objects.link(handle['camera']);wet.camera=handle['camera'];wet.world=clear.world;wet.cycles.device=clear.cycles.device
                if rig.get('collection'):wet.collection.children.link(rig['collection'])
                else:
                    for light in rig['objects']:wet.collection.objects.link(light)
                w=build_water(wet,depth,p,extent=70,bottom_z=-3)
                wet['lighting_mode']=mode;wet['seabed_depth_m']=depth;wet['camera_submersion_m']=depth-handle['camera'].matrix_world.translation.z
                water_scenes.append({'preset':p,'depth':depth,'scene':wet,'water':w})
        groups.append({'mode':mode,'clear':clear,'lighting':rig,'variants':water_scenes})
    handle['groups']=groups
    return groups


def save_native(handle,path,cfg,state_metadata):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    # The geometry-check-only scene is useful internally but is not an extra
    # unlit user-facing render option in the delivered native file.
    if handle.get('base') and handle['base'].name in bpy.data.scenes:
        base=handle['base'];bpy.context.window.scene=handle['groups'][0]['clear'];bpy.data.scenes.remove(base);handle['base']=None
    active=next(g for g in handle['groups'] if g['mode']=='mixed')['variants'][1]['scene']
    bpy.context.window.scene=active
    for s in bpy.data.scenes:s.render.filepath='//manual_'+s.name+'.png'
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=='VIEW_3D':area.spaces.active.region_3d.view_perspective='CAMERA'
    doc=bpy.data.texts.get('START_HERE_Lighting_and_Data') or bpy.data.texts.new('START_HERE_Lighting_and_Data')
    doc.clear();doc.write('Native Cycles v2: choose natural/artificial/mixed and Clear/mild/medium/strong in Scene selector; F12. No project Python or autoexec required. Clear excludes BOTH seawater and air-water surface. Each lighting mode has its own matching clear. Lights_Left/Right are camera-mounted native Spot lights: energy, temperature, cone and blend are editable. Generated pairs stay frozen; edits invalidate existing dataset references. All RGBs use the same fixed color management.\n\n'+json.dumps({'config':cfg,'state':state_metadata},ensure_ascii=False,indent=2))
    # Asset images are intentionally external during v3 development. Set every
    # audited PBR path relative to the *final* native blend before saving, and
    # disable Blender's automatic second remap from the unsaved worker file.
    # Without this, an otherwise valid preview can render in-process from the
    # loaded image cache but reopen with paths one parent directory too high.
    project_root=Path(__file__).resolve().parents[1]
    for image in bpy.data.images:
        relative=image.get('source_relative_path') if hasattr(image,'get') else None
        if not relative:
            continue
        source=(project_root/str(relative)).resolve()
        if not source.is_file() or not source.is_relative_to(project_root.resolve()):
            raise RuntimeError('Native save refused missing/outside-v3 image: '+str(relative))
        path_from_blend=os.path.relpath(source,start=path.parent.resolve()).replace(os.sep,'/')
        image.filepath='//'+path_from_blend
        image.filepath_raw='//'+path_from_blend
    result=bpy.ops.wm.save_as_mainfile(filepath=str(path),compress=True,relative_remap=False)
    if result!={'FINISHED'} or not path.exists() or path.stat().st_size<1024:raise RuntimeError('Native blend save failed')
