"""Report scene_0001 object bounds after a cached v3 geometry attempt."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy  # type: ignore
from mathutils import Vector  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
from run_500_scenes import build_geometry_in_place, configure_template_asset_cache  # noqa: E402
from src.contracts import derive_seed, load_config  # noqa: E402


def bounds(obj):
    points = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return {"min": [float(min(point[i] for point in points)) for i in range(3)], "max": [float(max(point[i] for point in points)) for i in range(3)]}


def main():
    cfg = load_config(ROOT / "configs" / "v3_500_scenes.json")
    configure_template_asset_cache()
    layout = {"seed": derive_seed(cfg["root_seed"], "scene_layout", 1), "split": "train"}
    report = {"layout": layout}
    try:
        handle = build_geometry_in_place(cfg, layout, 0, 0)
        report["status"] = "built"
        report["camera"] = list(handle["camera"].matrix_world.translation)
        report["objects"] = {obj.name: bounds(obj) for obj in handle["objects"] if obj.name in ("Coral_Branching_01", "Coral_Brain_02", "Sandbed_Undulating_Path")}
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        report["objects"] = {obj.name: bounds(obj) for obj in bpy.data.objects if obj.name in ("Coral_Branching_01", "Coral_Brain_02", "Sandbed_Undulating_Path")}
    (ROOT / "reports" / "scene1_bounds.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
