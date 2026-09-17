"""Representative full-complexity views for every pilot layout, before pilot rendering."""
import sys,json,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.scene import build_layout,set_camera,add_water_scenes
from src.render import render_rgb
from src.labels import export_labels
from src.contracts import load_config,atomic_json,source_hash
cfg=load_config(ROOT/'configs/pilot.json');cfg['render'].update(resolution=[384,384],samples=128);cfg['presets']=cfg['presets'][:1]
out=ROOT/'reports/composition';out.mkdir(parents=True,exist_ok=True);rows=[]
for layout in cfg['layouts']:
 h=build_layout(cfg,layout['seed']);variants=add_water_scenes(h,cfg,5)
 for vi,view in enumerate(cfg['views']):
  set_camera(h['clear'],view);folder=out/f"l{layout['seed']}_{view['id']}";folder.mkdir(exist_ok=True)
  render_rgb(h['clear'],folder/'clear_rgb');labels=export_labels(h['clear'],folder,h['ecology']['objects'])
  if vi==0:render_rgb(variants[0][1],folder/'degraded_rgb')
  row={'layout_seed':layout['seed'],'view':view,'folder':str(folder),'classes':labels['semantic_pixel_counts'],'morphologies_visible':sorted({m for x in labels['instances'].values() if x['visible_pixel_count']>=16 for m in x['asset_morphologies']}),'visible_instances_over_16_pixels':sum(x['visible_pixel_count']>=16 for x in labels['instances'].values()),'asset_summary':h['ecology']['summary']}
  rows.append(row);atomic_json(out/'execution.json',{'status':'rendering','source_hash':source_hash(),'views':rows});print('COMPOSITION_COMPLETE '+row['folder'],flush=True)
atomic_json(out/'execution.json',{'status':'rendered_pending_visual_review','source_hash':source_hash(),'views':rows})
