"""Render saved scenes using Blender CLI with no project Python or autoexec."""
import argparse,json,sys,subprocess,time,struct
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent/'seabed_dataset/.deps/py313'))
from src.contracts import atomic_json,file_hash
from PIL import Image
import numpy as np
argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args(argv);root=Path(a.root).resolve();out=ROOT/'reports/native_cli_v2';out.mkdir(exist_ok=True)
rows=[json.loads(x) for x in (root/'manifest.jsonl').read_text().splitlines()]
states=sorted({r['state_key'] for r in rows});targets=[]
for index,state in enumerate(states):
 for mode in (('natural','artificial','mixed') if index==0 else ('mixed',)):
  targets.append(next(r for r in rows if r['state_key']==state and r['lighting_mode']==mode and r['water_id']=='medium'))
results=[]
for row in targets:
 state=row['state_key'];cp=json.loads((root/'states'/state/'accepted.json').read_text());blend=root/cp['blend']
 assert file_hash(blend)==cp['blend_sha256'] and blend.stat().st_size==cp['blend_bytes']
 scene=state+'_'+row['lighting_mode']+'_D5_medium';stem=out/(scene+'_');log=out/(scene+'.log')
 cmd=['/Applications/Blender.app/Contents/MacOS/Blender','--background','--factory-startup','--disable-autoexec',str(blend),'--scene',scene,'-o',str(stem),'-F','PNG','-f','1','--','--cycles-device','METAL']
 start=time.time()
 with log.open('w') as f:run=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 assert run.returncode==0,(scene,run.returncode,str(log))
 native=Path(str(stem)+'0001.png');b=native.read_bytes();w,h,bits,color=struct.unpack('>IIBB',b[16:26]);assert (w,h,bits,color)==(256,256,16,2)
 expected=root/row['sample_path']/'degraded_rgb.png'
 with Image.open(native) as im:x=np.asarray(im.convert('RGB'),dtype=np.float32)
 with Image.open(expected) as im:y=np.asarray(im.convert('RGB'),dtype=np.float32)
 rmse=float(np.sqrt(np.mean((x-y)**2)));assert rmse<=1.0,(scene,rmse)
 result={'state_key':state,'scene':scene,'sample_id':row['sample_id'],'exit_code':0,'project_python_invoked':False,'autoexec_disabled':True,'command':cmd,'seconds':time.time()-start,'png':{'width':w,'height':h,'bits':bits,'color_type':color},'display8_rmse':rmse,'display8_rmse_limit':1.0,'image_sha256':file_hash(native),'blend_sha256':file_hash(blend),'blend':str(blend),'log':str(log)}
 results.append(result);atomic_json(out/'result.json',{'status':'running','checks':results});print('NATIVE_SCENE_PASS '+scene,flush=True)
atomic_json(out/'result.json',{'status':'pass','checks':results,'distinct_native_files':len({x['blend'] for x in results}),'lighting_modes':sorted({r['lighting_mode'] for r in targets}),'method':'native CLI without --python/--python-expr; full saved 512-sample scenes; PNG is compared at Pillow RGB8 display precision'})
