"""Create a minimal, read-only-derived scan cache from a v3 native blend."""

from __future__ import annotations

import json
from pathlib import Path

import bpy  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "v3_first_round" / "scenes" / "l101_g000.blend"
TARGET = ROOT / "reports" / "v3_scan_cache.blend"
SCAN_NAMES = ("Coral_Branching_01", "Coral_Brain_02", "Fish_Disk_06")


def main() -> None:
    scans = []
    for name in SCAN_NAMES:
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "MESH" or obj.data is None or "CC0ScanMesh" not in obj.data.name:
            raise RuntimeError("missing verified scan object: " + name)
        scans.append(obj)
    keep = bpy.data.scenes[0]
    for scene in list(bpy.data.scenes):
        if scene != keep:
            bpy.data.scenes.remove(scene)
    collection = bpy.data.collections.get("V3_Scan_Cache") or bpy.data.collections.new("V3_Scan_Cache")
    if collection.name not in {child.name for child in keep.collection.children}:
        keep.collection.children.link(collection)
    scan_set = set(scans)
    for obj in list(bpy.data.objects):
        if obj not in scan_set:
            bpy.data.objects.remove(obj, do_unlink=True)
    for obj in scans:
        for parent in list(obj.users_collection):
            parent.objects.unlink(obj)
        collection.objects.link(obj)
    for child in list(keep.collection.children):
        if child != collection:
            keep.collection.children.unlink(child)
            if child.users == 0:
                bpy.data.collections.remove(child)
    keep.name = "V3_Scan_Cache"
    keep.camera = None
    bpy.context.window.scene = keep
    bpy.ops.wm.save_as_mainfile(filepath=str(TARGET), compress=True, relative_remap=False)
    if not TARGET.is_file() or TARGET.stat().st_size < 1024:
        raise RuntimeError("scan cache save failed")
    (ROOT / "reports" / "v3_scan_cache_provenance.json").write_text(
        json.dumps(
            {
                "source": str(SOURCE),
                "source_sha256": "a26542f2add8a2a499f53a6516bc145dd6a1b96c672940dc970720d5cf4d93a1",
                "target": str(TARGET),
                "kept_objects": list(SCAN_NAMES),
                "purpose": "batch template only; original v3 blend and first-round outputs are unchanged",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("V3_SCAN_CACHE_CREATED", TARGET, flush=True)


if __name__ == "__main__":
    main()
