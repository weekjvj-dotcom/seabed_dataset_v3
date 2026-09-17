"""Compare actual rendered flat-ID radiance against exported integer labels."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.deps/py313'))
import bpy,numpy as np,OpenEXR
from mathutils import Vector
from PIL import Image
from src.render import configure,select_device,render_rgb,read_render_rgb
from src.labels import export_labels
from src.contracts import load_config,atomic_json
bpy.ops.wm.read_factory_settings(use_empty=True);s=bpy.context.scene
cfg=load_config(ROOT/'configs/smoke.json');cfg['render'].update(resolution=[96,64],samples=64);configure(s,cfg);select_device(s)
s.render.pixel_aspect_x=1.25;s.render.pixel_aspect_y=1
w=bpy.data.worlds.new('Black');w.use_nodes=True;w.node_tree.nodes.get('Background').inputs['Strength'].default_value=0;s.world=w
cam=bpy.data.objects.new('Rotated shifted test camera',bpy.data.cameras.new('Independent projection probe'));s.collection.objects.link(cam);s.camera=cam
cam.location=(3,-5,4);cam.rotation_euler=(Vector((0,1,.6))-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.lens=40;cam.data.sensor_fit='VERTICAL';cam.data.sensor_height=24;cam.data.shift_x=.07;cam.data.shift_y=-.04;cam.data.dof.use_dof=False
palette={0:(0,0,0),1:(.15,.25,.65),2:(.8,.1,.15),3:(.1,.7,.15),5:(.8,.7,.1)}
objects=[]
for sem,location,scale in [(1,(0,1,-.2),(9,9,.2)),(2,(-1,0,.7),(.7,.7,.7)),(3,(1.1,1.2,1.0),(.7,.7,1)),(5,(-.2,3,1.5),(.65,.3,.4))]:
 bpy.ops.mesh.primitive_cube_add(size=2,location=location);o=bpy.context.object;o.scale=scale;o['semantic_id']=sem;o['instance_id']=100+sem;o['asset_morphology']='analytic_alignment_probe';objects.append(o)
 m=bpy.data.materials.new('Known flat ID '+str(sem));m.use_nodes=True;n=m.node_tree.nodes;n.clear();e=n.new('ShaderNodeEmission');e.inputs['Color'].default_value=(*palette[sem],1);out=n.new('ShaderNodeOutputMaterial');m.node_tree.links.new(e.outputs[0],out.inputs['Surface']);o.data.materials.append(m)
bpy.context.view_layer.update();out=ROOT/'reports/label_alignment';out.mkdir(parents=True,exist_ok=True)
render_rgb(s,out/'rendered_flat_ids');meta=export_labels(s,out,objects)
rgb=read_render_rgb(out/'rendered_flat_ids.exr');sem=np.array(Image.open(out/'semantic_id.png'));inst=np.array(Image.open(out/'instance_id.png'))
interior=np.ones(sem.shape,dtype=bool);interior[[0,-1],:]=False;interior[:,[0,-1]]=False
for dv in [-1,0,1]:
 for du in [-1,0,1]:interior &= sem==np.roll(np.roll(sem,dv,0),du,1)
colors=np.array(list(palette.values()));ids=np.array(list(palette));pred=ids[np.argmin(((rgb[:,:,None,:]-colors[None,None,:,:])**2).sum(axis=-1),axis=-1)]
accuracy=float((pred[interior]==sem[interior]).mean());max_color_error=float(np.abs(rgb[interior]-np.array([palette[int(x)] for x in sem[interior]])).max())
with OpenEXR.File(str(out/'depth_range_m.exr'),separate_channels=True) as f:dr=f.channels()['Y'].pixels.copy();assert dr.dtype==np.float32
result={'status':'pass' if accuracy==1.0 and max_color_error<.02 else 'fail','interior_pixels':int(interior.sum()),'semantic_accuracy_vs_cycles_render':accuracy,'max_linear_color_error':max_color_error,'camera':'Rotated+translated, VERTICAL sensor fit, nonunit pixel aspect, nonzero shift','K':meta['K'],'depth_read_by':'OpenEXR3','visible_classes':np.unique(sem).tolist(),'integer_bit_depth':16}
atomic_json(out/'validation.json',result);print(json.dumps(result))
if result['status']!='pass':raise SystemExit(1)
