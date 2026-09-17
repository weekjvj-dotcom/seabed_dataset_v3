"""Cycles rendering and unmodified Render Result encoding. No RGB synthesis."""
import time
from pathlib import Path
import bpy
import numpy as np

def configure(scene,cfg):
    scene.render.engine='CYCLES';r=scene.render;c=scene.cycles
    r.resolution_x,r.resolution_y=cfg['render']['resolution'];r.resolution_percentage=100
    r.pixel_aspect_x=r.pixel_aspect_y=1;r.film_transparent=False;r.use_compositing=False;r.use_sequencer=False;r.use_motion_blur=False;r.dither_intensity=0
    fmt=r.image_settings;fmt.file_format='PNG';fmt.color_mode='RGB';fmt.color_depth='16';fmt.compression=20;fmt.exr_codec='ZIP'
    c.samples=cfg['render']['samples'];c.seed=cfg['render']['seed'];c.use_animated_seed=False;c.use_adaptive_sampling=False;c.use_denoising=False
    c.max_bounces=cfg['render']['max_bounces'];c.volume_bounces=cfg['render']['volume_bounces'];c.transmission_bounces=8;c.diffuse_bounces=4;c.glossy_bounces=4;c.transparent_max_bounces=8
    c.caustics_reflective=True;c.caustics_refractive=True;c.blur_glossy=0;c.sample_clamp_direct=0;c.sample_clamp_indirect=0;c.use_light_tree=True
    scene.view_settings.view_transform='Standard';scene.view_settings.look='None';scene.view_settings.exposure=0;scene.view_settings.gamma=1
    scene.display_settings.display_device='sRGB';scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1
    scene.frame_set(1)
    return scene

def select_device(scene):
    prefs=bpy.context.preferences.addons['cycles'].preferences
    result={'selected':'CPU','reason':'No usable Metal device','devices':[]}
    scene.cycles.device='CPU'
    try:
        prefs.compute_device_type='METAL';prefs.get_devices();gpu=[d for d in prefs.devices if d.type=='METAL']
        result['devices']=[{'name':d.name,'type':d.type} for d in prefs.devices]
        if gpu:
            for d in prefs.devices:d.use=d in gpu
            scene.cycles.device='GPU';result.update(selected='METAL',reason='Actual Cycles Metal device enumerated')
    except Exception as e:result['reason']=repr(e)
    return result

def read_render_rgb(path):
    img=bpy.data.images.load(str(Path(path).resolve()),check_existing=False)
    try:
        width,height=map(int,img.size)
        values=np.empty(width*height*4,dtype=np.float32);img.pixels.foreach_get(values)
        return values.reshape(height,width,4)[::-1,:,:3].copy()
    finally:bpy.data.images.remove(img)

def render_rgb(scene,stem):
    stem=Path(stem);stem.parent.mkdir(parents=True,exist_ok=True)
    bpy.context.window.scene=scene;bpy.context.view_layer.update();start=time.perf_counter()
    bpy.ops.render.render(scene=scene.name,write_still=False)
    result=bpy.data.images['Render Result'];fmt=scene.render.image_settings
    fmt.color_mode='RGB';fmt.file_format='OPEN_EXR';fmt.color_depth='32';fmt.exr_codec='ZIP'
    result.save_render(str(Path(str(stem)+'.exr')),scene=scene)
    fmt.file_format='PNG';fmt.color_depth='16';fmt.compression=20
    result.save_render(str(Path(str(stem)+'.png')),scene=scene)
    rgb=read_render_rgb(Path(str(stem)+'.exr'))
    if not np.isfinite(rgb).all():raise ValueError('Nonfinite rendered RGB')
    return {'seconds':time.perf_counter()-start,'resolution':[rgb.shape[1],rgb.shape[0]],'linear_rgb_mean':rgb.mean(axis=(0,1)).tolist(),'min':float(rgb.min()),'max':float(rgb.max()),'engine':'CYCLES','device':scene.cycles.device,'samples':scene.cycles.samples,'seed':scene.cycles.seed,'denoising':False,'exr':{'pixel_type':'FLOAT32','compression':'ZIP','space':'Linear Rec.709','display_transform_applied':False},'png':{'bits':16,'display':'sRGB','view_transform':'Standard','look':'None','exposure':0,'gamma':1},'rgb_pixels_generated_by':'Cycles path tracing'}
