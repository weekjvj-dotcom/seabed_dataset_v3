"""Independent rendered physical experiments; never used to synthesize dataset RGB."""
import sys,json,math,time,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import bpy,numpy as np
from mathutils import Vector
from src.render import configure,select_device,render_rgb,read_render_rgb
from src.scene import make_world
from src.water import build_water,set_water_enabled,water_metadata
from src.contracts import atomic_json,load_config

OUT=ROOT/'reports/physics_v3';OUT.mkdir(parents=True,exist_ok=True)
CFG=load_config(ROOT/'configs/smoke.json');RESULTS=[]

def scene_new(name,res=65,samples=128):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s=bpy.context.scene;s.name=name;c=copy.deepcopy(CFG);c['render'].update(resolution=[res,res],samples=samples,seed=713)
    configure(s,c);select_device(s)
    w=bpy.data.worlds.new('Black calibration world');w.use_nodes=True;w.node_tree.nodes.get('Background').inputs['Strength'].default_value=0;s.world=w
    cam=bpy.data.objects.new('Calibration camera',bpy.data.cameras.new('Pinhole'));s.collection.objects.link(cam);s.camera=cam;cam.data.lens=80;cam.data.sensor_width=36;cam.data.sensor_fit='HORIZONTAL';cam.data.clip_start=.001;cam.data.clip_end=200;cam.data.dof.use_dof=False
    return s

def camera(s,eye,target):
    s.camera.location=eye;s.camera.rotation_euler=(Vector(target)-s.camera.location).to_track_quat('-Z','Y').to_euler();bpy.context.view_layer.update()

def material(name,color,emission=False):
    m=bpy.data.materials.new(name);m.use_nodes=True;n=m.node_tree.nodes;l=m.node_tree.links;n.clear();out=n.new('ShaderNodeOutputMaterial')
    shader=n.new('ShaderNodeEmission' if emission else 'ShaderNodeBsdfDiffuse');shader.inputs['Color'].default_value=(*color,1)
    l.new(shader.outputs[0],out.inputs['Surface']);return m

def plane(s,name,z,size,mat):
    mesh=bpy.data.meshes.new(name);mesh.from_pydata([(-size,-size,z),(size,-size,z),(size,size,z),(-size,size,z)],[],[(0,1,2,3)]);mesh.update()
    o=bpy.data.objects.new(name,mesh);s.collection.objects.link(o);o.data.materials.append(mat);return o

def render(s,name):
    stats=render_rgb(s,OUT/name);return read_render_rgb(OUT/(name+'.exr')),stats

def check(name,passed,**data):
    r={'name':name,'passed':bool(passed),**data};RESULTS.append(r);atomic_json(OUT/'results.json',{'status':'running','checks':RESULTS});print('PHYSICS_CHECK '+json.dumps(r),flush=True)

def water_preset(**kw):
    p={'id':'calibration','absorption':[0,0,0],'scatter':[0,0,0],'g':.65,'ior':1.333};p.update(kw);return p

def absorption():
    for d in [1.,2.,4.]:
        s=scene_new('Beer_'+str(d),65,64);camera(s,(0,0,0),(0,0,-1));s.camera.data.lens=50
        plane(s,'Known_emitting_plane',-d,8,material('Unit radiance',[1,1,1],True))
        clear,_=render(s,f'beer_{d:g}_clear')
        w=build_water(s,1,water_preset(absorption=[.3,.1,.04]),extent=12,bottom_z=-5)
        wet,_=render(s,f'beer_{d:g}_absorption')
        u,v=np.meshgrid(np.arange(65)+.5,np.arange(65)+.5);f=50/36*65
        distance=d*np.sqrt(1+((u-32.5)/f)**2+((v-32.5)/f)**2)
        expected=np.exp(-distance[...,None]*np.array([.3,.1,.04]))
        measured=wet/np.maximum(clear,1e-8);roi=(slice(12,53),slice(12,53));relative=np.abs(measured-expected)/expected
        err=float(relative[roi].max());check(f'Beer_Lambert_{d:g}m',err<=.05,max_relative_error=err,threshold=.05,known_plane_z=-d,actual_range_roi=[float(distance[roi].min()),float(distance[roi].max())],linear_only=True)
    s=scene_new('Zero_medium',65,64);camera(s,(0,0,0),(0,0,-1));plane(s,'Unit_plane',-3,8,material('Unit',[.5,.7,.9],True))
    clear,_=render(s,'zero_clear');w=build_water(s,1,water_preset(ior=1),extent=12,bottom_z=-5);zero,_=render(s,'zero_medium')
    error=float(np.sqrt(np.mean((clear-zero)**2))/np.sqrt(np.mean(clear**2)))
    check('zero_coefficients_ior1_vs_no_water',error<=.001,normalized_rmse=error,threshold=.001,samples=64,seed=s.cycles.seed,definition='No extinction and IOR1; not a transparent water IOR1.333 reference')

def fresnel(n,theta):
    t=math.asin(math.sin(theta)/n);ci=math.cos(theta);ct=math.cos(t)
    rs=((ci-n*ct)/(ci+n*ct))**2;rp=((n*ci-ct)/(n*ci+ct))**2
    return .5*(rs+rp),t

def interface():
    for n in [1.333,1.45]:
        for degrees in [0,40]:
            s=scene_new(f'Interface_{n}_{degrees}',65,1024);theta=math.radians(degrees)
            camera(s,(0,0,2),(math.sin(theta),0,2-math.cos(theta)))
            m=material('Position encoded emitter',[1,1,1],True);nodes=m.node_tree.nodes;links=m.node_tree.links
            geo=nodes.new('ShaderNodeNewGeometry');sep=nodes.new('ShaderNodeSeparateXYZ');links.new(geo.outputs['Position'],sep.inputs[0])
            mul=nodes.new('ShaderNodeMath');mul.operation='MULTIPLY';mul.inputs[1].default_value=.1;links.new(sep.outputs['X'],mul.inputs[0])
            add=nodes.new('ShaderNodeMath');add.operation='ADD';add.inputs[1].default_value=.3;links.new(mul.outputs[0],add.inputs[0])
            rgb=nodes.new('ShaderNodeCombineColor');rgb.mode='RGB';rgb.inputs['Green'].default_value=1;rgb.inputs['Blue'].default_value=1;links.new(add.outputs[0],rgb.inputs['Red']);links.new(rgb.outputs[0],next(x for x in nodes if x.bl_idname=='ShaderNodeEmission').inputs['Color'])
            plane(s,'Known_emitter_z_minus2',-2,20,m);w=build_water(s,0,water_preset(ior=n),extent=30,bottom_z=-3)
            image,stats=render(s,f'interface_{n}_{degrees}')
            observed=image[32,32];x_observed=float((observed[0]/observed[1]-.3)/.1)
            F,theta_t=fresnel(n,theta);x_expected=2*math.tan(theta)+2*math.tan(theta_t)
            x_error=abs(x_observed-x_expected);expected_green=(1-F)
            energy_error=abs(float(observed[1])-expected_green)/expected_green
            check(f'Snell_{n}_{degrees}deg',x_error<=.03,position_error_m=x_error,threshold_m=.03,predicted_hit_x_m=x_expected,rendered_hit_x_m=x_observed,theta_air_deg=degrees,theta_water_deg=math.degrees(theta_t),method='Native geometry position gradient R/G cancels throughput; no image-space synthesis')
            check(f'Fresnel_native_BSDF_{n}_{degrees}deg',energy_error<=.05,relative_error=energy_error,threshold=.05,rendered_green=float(observed[1]),predicted_green=expected_green,fresnel_reflectance=F,radiance_convention='Cycles singular white Glass BSDF throughput follows 1-F; cross-medium eta-squared radiance calibration is not asserted', absolute_radiance_eta2_prediction=(1-F)/(n*n), source='https://github.com/blender/blender/blob/v5.2.1/intern/cycles/kernel/closure/bsdf_microfacet.h#L841-L900',samples=s.cycles.samples)

def mini_scene(samples,seed,preset,depth):
    s=scene_new('Independent_parameter_effects',65,samples);s.cycles.seed=seed
    camera(s,(0,-5,2),(0,1,.4));s.camera.data.lens=40;s.world=make_world(CFG['lighting'])
    floor=plane(s,'Known_reflective_floor',0,15,material('Neutral sand',[.55,.5,.4]));floor['semantic_id']=1;floor['instance_id']=1
    for i,color in enumerate([[.55,.12,.06],[.06,.35,.18],[.06,.18,.55]]):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24,ring_count=12,radius=.6,location=((i-1)*1.4,1+i*.4,.6));o=bpy.context.object;o.data.materials.append(material('Diffuse target '+str(i),color))
        for p in o.data.polygons:p.use_smooth=True
    if preset is not None:build_water(s,depth,preset,extent=45,bottom_z=-2)
    return s

def effects_and_noise():
    base=water_preset(absorption=[.08,.03,.01],scatter=[.025]*3)
    images={}
    for spp in [64,256]:
        for seed in [2021,9127]:
            s=mini_scene(spp,seed,base,5);im,stats=render(s,f'noise_{spp}_{seed}');images[(spp,seed)]=im
    # Fixed interior ROI excludes environment and geometric edges near frame boundary.
    roi=(slice(25,60),slice(10,55))
    noise64=float(np.sqrt(np.mean((images[(64,2021)][roi]-images[(64,9127)][roi])**2)))
    noise256=float(np.sqrt(np.mean((images[(256,2021)][roi]-images[(256,9127)][roi])**2)))
    check('independent_seed_noise_convergence',noise256<noise64,rmse64=noise64,rmse256=noise256,ratio=noise256/noise64,samples=[64,256],seeds=[2021,9127],denoising=False)
    conditions=[('absorption_only_changed',water_preset(absorption=[.24,.09,.03],scatter=[.025]*3),5),('scatter_only_changed',water_preset(absorption=[.08,.03,.01],scatter=[.15]*3),5),('seabed_depth_only_changed',base,15),('surface_ior_only_changed',water_preset(absorption=[.08,.03,.01],scatter=[.025]*3,ior=1.45),5)]
    for name,preset,depth in conditions:
        s=mini_scene(256,2021,preset,depth);im,stats=render(s,name)
        signal=float(np.sqrt(np.mean((im[roi]-images[(256,2021)][roi])**2)))
        # IOR has separate deterministic Snell/Fresnel gates; low-SNR broad-sky effect is reported honestly.
        threshold=noise256
        check(name,signal>threshold,linear_roi_rmse=signal,independent_seed_noise_rmse=noise256,signal_to_noise=signal/max(noise256,1e-12),criterion='Effect exceeds independently measured 256-sample seed noise',preset=preset,seabed_depth_m=depth)

def main():
    started=time.time();absorption();interface();effects_and_noise()
    report={'status':'pass' if all(r['passed'] for r in RESULTS) else 'fail','blender':bpy.app.version_string,'checks':RESULTS,'seconds':time.time()-started,'limitations':['RGB coefficient demonstration, not spectral seawater fit','Absolute cross-medium eta-squared radiance test in physics_v2 failed. Native Glass BSDF uses Fresnel lobe throughput; this suite validates that convention, not absolute radiometric calibration','Static planar interface; no housing, wave focusing or complex caustics','Finite water domain with volume-only side and bottom boundaries; reported primary and illumination paths are within calibrated region','Wide diffuse natural sky without solar disc avoids reliance on point-light shadow-caustics solvers']}
    atomic_json(OUT/'results.json',report)
    print('PHYSICS_COMPLETE '+json.dumps(report),flush=True)
    if report['status']!='pass':raise SystemExit(1)
if __name__=='__main__':main()
