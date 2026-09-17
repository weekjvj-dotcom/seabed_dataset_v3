"""Stage-by-stage diagnostic for the v3 template-cache builder.

Run only with a saved v3 blend loaded.  This script does not render and writes
an append-only progress trace so a native Blender crash still leaves the last
completed Python stage.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import sys

import bpy  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from run_500_scenes import (
    configure_template_asset_cache,
    build_geometry_cached,
    water_plan,
    _window_and_view_layer,
    render_rgb,
)
from src.contracts import derive_seed, load_config
from src.labels import export_labels
from src.scene import build_lighting_groups


TRACE = ROOT / "reports" / "template_cache_diagnosis.jsonl"


def mark(stage: str, **fields) -> None:
    TRACE.parent.mkdir(parents=True, exist_ok=True)
    with TRACE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": time.time(), "stage": stage, **fields}, ensure_ascii=False) + "\n")
        handle.flush()


def main() -> None:
    cfg = load_config(ROOT / "configs" / "v3_500_scenes.json")
    mark("config_loaded")
    configure_template_asset_cache()
    mark("cache_configured", keys=["Coral_Branching_01", "Coral_Brain_02", "Fish_Disk_06"])
    layout = {"seed": derive_seed(cfg["root_seed"], "scene_layout", 1), "split": "train"}
    mark("before_build_geometry", layout=layout)
    handle = build_geometry_cached(cfg, layout, 0, 0)
    mark("after_build_geometry", objects=len(handle["objects"]), scenes=len(bpy.data.scenes))
    mark("before_export_labels")
    export_labels(handle["base"], ROOT / "reports" / "template_cache_diagnosis_labels", handle["active_objects"])
    mark("after_export_labels")
    water = water_plan(cfg, {"layout_seed": layout["seed"], "state_index": 0}, 0)
    single_cfg = dict(cfg)
    single_cfg["lighting_modes"] = ["natural"]
    mark("before_lighting_groups")
    groups = build_lighting_groups(handle, single_cfg, "diagnostic", water["effective_presets"])
    mark("after_lighting_groups", groups=len(groups), variants=len(groups[0]["variants"]))
    mark("before_render")
    render_rgb(groups[0]["clear"], ROOT / "reports" / "template_cache_diagnosis_clear")
    mark("after_render")
    print("TEMPLATE_CACHE_DIAGNOSIS_OK", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        mark("python_exception")
        raise
