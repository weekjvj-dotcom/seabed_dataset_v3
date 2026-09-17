"""Native three-mode illumination isolation, independent of complex geometry."""
import sys,json,copy,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import bpy,numpy as np
from mathutils import Vector
from src.contracts import load_config,atomic_json
from src.render import configure,select_device,render_rgb,read_render_rgb
from src.lighting import sample_lighting,build_lighting,lighting_metadata
from src.water import build_water
cfg=load_config(ROOT/'configs/first_round.json');cfg['render'].update(resolution=[96,96],samples=256)
out=ROOT/'reports/light_isolation';out.mkdir(exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
base=bpy.context.scene;configure(base,cfg);select_device(base)
geo=bpy.data.collections.new('CALIBRATION_TARGETS');base.collection.children.link(geo)
cam=bpy.data.objects.new('CalibrationCamera',bpy.data.cameras.new('CalibrationCamera'));geo.objects.link(cam);cam.location=(0,-4,2);cam.rotation_euler=(Vector((0,0,0))-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.lens=40;cam.data.sensor_width=36;cam.data.sensor_fit='HORIZONTAL';base.camera=cam
mesh=bpy.data.meshes.new('DiffusePlane');mesh.from_pydata([(-8,-8,0),(8,-8,0),(8,8,0),(-8,8,0)],[],[(0,1,2,3)]);mesh.update();obj=bpy.data.objects.new('NeutralTarget',mesh);geo.objects.link(obj)
mat=bpy.data.materials.new('DiffuseWhite');mat.use_nodes=True;n=mat.node_tree.nodes;n.clear();d=n.new('ShaderNodeBsdfDiffuse');d.inputs['Color'].default_value=(.5,.5,.5,1);o=n.new('ShaderNodeOutputMaterial');mat.node_tree.links.new(d.outputs[0],o.inputs['Surface']);obj.data.materials.append(mat)
bpy.context.view_layer.update();plan=sample_lighting(cfg,0,0);images={};results=[]
for mode in cfg['lighting_modes']:
 s=configure(bpy.data.scenes.new(mode),cfg);s.collection.children.link(geo);s.camera=cam;s.cycles.device=base.cycles.device
 rig=build_lighting(s,cam,plan,mode);s.world=rig['world'];stats=render_rgb(s,out/(mode+'_clear'));images[mode]=read_render_rgb(out/(mode+'_clear.exr'))
 w=build_water(s,5,cfg['presets'][1],extent=50,bottom_z=-3);wetstats=render_rgb(s,out/(mode+'_medium'))
 results.append({'mode':mode,'lighting':lighting_metadata(rig),'clear':stats,'water':wetstats})
roi=(slice(25,85),slice(15,81));add=images['natural']+images['artificial'];error=float(np.sqrt(np.mean((images['mixed'][roi]-add[roi])**2))/max(np.sqrt(np.mean(add[roi]**2)),1e-10))
report={'status':'rendered','mode_additivity_nrmse':error,'interpretation':'Linear no-water lighting superposition diagnostic; MC noise included','renders':results};atomic_json(out/'result.json',report);print(json.dumps({'status':'rendered','additivity_nrmse':error,'means':{k:v.mean(axis=(0,1)).tolist() for k,v in images.items()}}))
