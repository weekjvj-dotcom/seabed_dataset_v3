"""Hash the actual evaluated, water-independent state before and after every output."""
import hashlib
from array import array
from pathlib import Path
import bpy
from .contracts import canonical_hash
from .labels import camera_metadata

NODE_PROPS=('operation','blend_type','noise_dimensions','normalize','noise_type','voronoi_dimensions','feature','distance','gradient_type','distribution','space','vector_type','wave_type','bands_direction','rings_direction','wave_profile','sky_type','sun_disc','sun_size','sun_intensity','sun_elevation','sun_rotation','altitude','air_density','aerosol_density','ozone_density','sun_direction','turbidity','ground_albedo','interpolation','extension','projection','phase')
OBJECT_FLAGS=('hide_render','visible_camera','visible_diffuse','visible_glossy','visible_transmission','visible_volume_scatter','visible_shadow','is_holdout','is_shadow_catcher')
RENDER_FLAGS=('film_transparent','use_compositing','use_sequencer','use_motion_blur','use_border','use_crop_to_border','border_min_x','border_max_x','border_min_y','border_max_y','pixel_aspect_x','pixel_aspect_y','dither_intensity','filter_size','film_transparent_glass')
CYCLES_FLAGS=('device','samples','seed','use_animated_seed','use_adaptive_sampling','use_denoising','max_bounces','diffuse_bounces','glossy_bounces','transmission_bounces','volume_bounces','transparent_max_bounces','caustics_reflective','caustics_refractive','use_light_tree','blur_glossy','sample_clamp_direct','sample_clamp_indirect','volume_step_rate','volume_max_steps','pixel_filter_type','filter_width','use_pixel_jitter','use_fast_gi','film_exposure')
ROOT=Path(__file__).resolve().parents[1]

def props(obj,names):
    return {name:scalar(getattr(obj,name)) for name in names if hasattr(obj,name)}

def collection_paths(scene,obj):
    result=[]
    def visit(collection,path):
        current=path+[{'name':collection.name if collection!=scene.collection else 'SCENE_ROOT','hide_render':collection.hide_render}]
        if collection.objects.get(obj.name)==obj:result.append(current)
        for child in collection.children:visit(child,current)
    visit(scene.collection,[])
    return result

def assert_clear_scene(scene,objects):
    targets=set(objects)
    actual={o for o in scene.objects if o.type not in ('CAMERA','LIGHT') and not (o.type=='EMPTY' and o.get('is_data_helper'))}
    if actual!=targets:raise ValueError('Clear scene has missing targets or additional geometry')
    for owner in [scene.world]+list({m for o in objects for m in o.data.materials}):
        for node in owner.node_tree.nodes:
            if node.bl_idname in ('ShaderNodeOutputWorld','ShaderNodeOutputMaterial') and node.inputs['Volume'].is_linked:
                raise ValueError('Clear target/world contains a volume shader')
    if any(o.get('is_water') for o in scene.objects):raise ValueError('Clear scene contains water')

def geometry_constraints(scene,objects,surface_z):
    from mathutils import Vector
    camera=scene.camera.matrix_world.translation
    if not -3<camera.z<surface_z:raise ValueError('Camera is outside water bounds')
    for obj in objects:
        corners=[obj.matrix_world@Vector(v) for v in obj.bound_box]
        lo=[min(p[i] for p in corners) for i in range(3)]
        hi=[max(p[i] for p in corners) for i in range(3)]
        if lo[2]<=-3 or hi[2]>=surface_z or max(abs(x) for x in lo[:2]+hi[:2])>=70:
            raise ValueError('Target crosses water boundary: '+obj.name)
        if all(lo[i]<=camera[i]<=hi[i] for i in range(3)):
            raise ValueError('Camera overlaps target bounds; choose a safely separated viewpoint: '+obj.name)

def scalar(value):
    if isinstance(value,(str,int,float,bool)) or value is None:return value
    try:return [scalar(x) for x in value]
    except TypeError:return str(value)


def _file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def image_state(image):
    """Return stable, provenance-aware image state without hashing temp paths.

    V3 permits two kinds of images: project-relative CC0 PBR files and packed
    images embedded in an audited GLB.  The latter is tied to the owning mesh's
    source SHA in ``immutable_state``; it intentionally does not include any
    process-specific unpack path in the digest.
    """
    relative=image.get('source_relative_path') if hasattr(image,'get') else None
    packed=image.packed_file is not None
    row={'name':image.name,'packed':packed,'colorspace':image.colorspace_settings.name,
         'size_px':[int(image.size[0]),int(image.size[1])],'file_format':image.file_format,
         'source_relative_path':str(relative) if relative else None}
    if relative:
        path=(ROOT/str(relative)).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
            raise ValueError('External image source is missing or outside v3 root: '+str(relative))
        row['source_sha256']=_file_hash(path)
    elif not packed:
        raise ValueError('Unpacked image lacks v3 source_relative_path: '+image.name)
    return row

def nodes_state(tree,seen=None):
    if tree is None:return None
    seen=set() if seen is None else set(seen)
    if tree.name in seen:raise ValueError('Recursive node group')
    seen.add(tree.name);nodes=[]
    for n in sorted(tree.nodes,key=lambda x:x.name):
        state={'name':n.name,'type':n.bl_idname,'mute':n.mute,'inputs':[(i.identifier,scalar(i.default_value)) for i in n.inputs if hasattr(i,'default_value')]}
        state['properties']={p:scalar(getattr(n,p)) for p in NODE_PROPS if hasattr(n,p)}
        if hasattr(n,'color_ramp'):state['ramp']={'interpolation':n.color_ramp.interpolation,'elements':[(e.position,list(e.color)) for e in n.color_ramp.elements]}
        if getattr(n,'node_tree',None):state['group']=nodes_state(n.node_tree,seen)
        if getattr(n,'image',None):state['image']=image_state(n.image)
        nodes.append(state)
    return {'nodes':nodes,'links':sorted((l.from_node.name,l.from_socket.identifier,l.to_node.name,l.to_socket.identifier) for l in tree.links)}

def water_template(handle):
    """Canonical native graph and box topology, with only top height normalized."""
    def graph(tree):
        ns=list(tree.nodes);index={n:i for i,n in enumerate(ns)}
        rows=[]
        for n in ns:
            r={'type':n.bl_idname,'mute':n.mute,'properties':props(n,NODE_PROPS),'inputs':[(i.identifier,scalar(i.default_value)) for i in n.inputs if hasattr(i,'default_value')]}
            if getattr(n,'node_tree',None):r['group']=graph(n.node_tree)
            rows.append(r)
        return {'nodes':rows,'links':sorted((index[l.from_node],l.from_socket.identifier,index[l.to_node],l.to_socket.identifier) for l in tree.links)}
    rows=[]
    for obj in handle['objects']:
        points=[obj.matrix_world@v.co for v in obj.data.vertices]
        bottom=min(p.z for p in points);top=max(p.z for p in points)
        if len(points)!=8 or len(obj.data.polygons)!=6:raise ValueError('Water boundary must remain the approved box template')
        if any(min(abs(p.z-bottom),abs(p.z-top))>1e-6 for p in points):raise ValueError('Unexpected internal water vertices')
        rows.append({'vertices_xy_and_top_flag':[(p.x,p.y,1 if abs(p.z-top)<1e-6 else 0) for p in points],'bottom_z':bottom,'faces':[(list(p.vertices),p.material_index) for p in obj.data.polygons],'materials':[graph(m.node_tree) for m in obj.data.materials],'visibility':props(obj,OBJECT_FLAGS)})
    return rows

def _array_hash(elements,prop,count,typecode):
    data=array(typecode,[0])*count;elements.foreach_get(prop,data)
    return hashlib.sha256(data.tobytes()).hexdigest()

def immutable_state(scene,objects):
    with bpy.context.temp_override(scene=scene,view_layer=scene.view_layers[0]):
        deps=bpy.context.evaluated_depsgraph_get();deps.update();rows=[]
        for obj in sorted(objects,key=lambda x:x.name):
            if obj.get('is_water'):raise ValueError('Water leaked into immutable targets')
            if obj.animation_data or len(obj.modifiers):raise ValueError('v2 requires evaluated static meshes without live modifiers')
            ev=obj.evaluated_get(deps);mesh=ev.to_mesh()
            try:
                rows.append({'name':obj.name,'matrix':[list(r) for r in ev.matrix_world],'visibility':props(obj,OBJECT_FLAGS),'collections':collection_paths(scene,obj),'semantic_id':obj['semantic_id'],'instance_id':obj['instance_id'],'asset_uid':obj.get('asset_uid'),'asset_source':obj.get('asset_source'),'external_source_relative_path':obj.get('external_source_relative_path'),'external_source_sha256':obj.get('external_source_sha256'),'external_license':obj.get('external_license'),'vertices':_array_hash(mesh.vertices,'co',len(mesh.vertices)*3,'f'),'loops':_array_hash(mesh.loops,'vertex_index',len(mesh.loops),'i'),'face_materials':_array_hash(mesh.polygons,'material_index',len(mesh.polygons),'i'),'smooth':_array_hash(mesh.polygons,'use_smooth',len(mesh.polygons),'b'),'material_slots':[m.name for m in obj.data.materials]})
            finally:ev.to_mesh_clear()
    materials={m.name:m for o in objects for m in o.data.materials}
    r=scene.render;c=scene.cycles;v=scene.view_settings
    return {'geometry':rows,'scene_root_hidden':scene.collection.hide_render,'materials':{name:{'nodes':nodes_state(m.node_tree),'properties':props(m,('use_nodes','diffuse_color','roughness','metallic','use_backface_culling','surface_render_method','displacement_method'))} for name,m in sorted(materials.items())},'camera':camera_metadata(scene),'world':nodes_state(scene.world.node_tree),'lights':[{'name':o.name,'matrix':[list(r) for r in o.matrix_world],'visibility':props(o,OBJECT_FLAGS),'collections':collection_paths(scene,o),'properties':props(o.data,('type','energy','color','shape','size','size_y','angle','shadow_soft_size','use_shadow','use_temperature','temperature','temperature_color','exposure','normalize','spot_size','spot_blend','use_square','use_soft_falloff','use_custom_distance','cutoff_distance','diffuse_factor','specular_factor','transmission_factor','volume_factor','use_nodes')),'nodes':nodes_state(o.data.node_tree) if o.data.use_nodes else None} for o in scene.objects if o.type=='LIGHT'],'frame':scene.frame_current,'render':{'engine':r.engine,'resolution':[r.resolution_x,r.resolution_y,r.resolution_percentage],'cycles':props(c,CYCLES_FLAGS),'flags':props(r,RENDER_FLAGS),'samples':c.samples,'seed':c.seed,'max_bounces':c.max_bounces,'volume_bounces':c.volume_bounces,'transmission_bounces':c.transmission_bounces,'caustics_refractive':c.caustics_refractive,'denoising':c.use_denoising,'adaptive':c.use_adaptive_sampling,'compositor':r.use_compositing,'motion_blur':r.use_motion_blur,'view_transform':v.view_transform,'look':v.look,'exposure':v.exposure,'gamma':v.gamma,'display_device':scene.display_settings.display_device,'working_space':'Linear Rec.709','units_scale':scene.unit_settings.scale_length}}

def state_digest(scene,objects):
    state=immutable_state(scene,objects)
    state['render']['output_encoding']=props(scene.render.image_settings,('file_format','color_mode','color_depth','compression','exr_codec','color_management'))
    return canonical_hash(state),state


def geometry_digest(scene,objects):
    full=immutable_state(scene,objects)
    geometry={'geometry':full['geometry'],'materials':full['materials'],'camera':full['camera'],'frame':full['frame'],'resolution':full['render']['resolution'],'units_scale':full['render']['units_scale']}
    return canonical_hash(geometry),geometry
