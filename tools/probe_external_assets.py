#!/usr/bin/env python3
"""Inspect the downloaded Smithsonian GLB assets before v3 integration.

This is deliberately an asset-only probe: it starts from an empty Blender
file, imports each source GLB independently, records evaluated mesh and image
metadata, then removes the imported data.  It never opens or edits v2/v3
dataset scenes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import bpy  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
ASSETS = (
    (
        "smithsonian:acropora-cervicornis-usnm-1171477:low-100k",
        ROOT / "assets/source/smithsonian/acropora_cervicornis_usnm_1171477_100k_2048.glb",
    ),
    (
        "smithsonian:diploria-labyrinthiformis-usnm-74947:low-100k",
        ROOT / "assets/source/smithsonian/diploria_labyrinthiformis_usnm_74947_100k_2048.glb",
    ),
    (
        "smithsonian:diodon-hystrix-usnm-195928:low-100k",
        ROOT / "assets/source/smithsonian/diodon_hystrix_usnm_195928_100k_2048.glb",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scene_object_pointers() -> set[int]:
    return {object_.as_pointer() for object_ in bpy.data.objects}


def image_rows(materials: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for material in materials:
        tree = getattr(material, "node_tree", None)
        for node in getattr(tree, "nodes", ()):
            image = getattr(node, "image", None)
            if image is None or image.as_pointer() in seen:
                continue
            seen.add(image.as_pointer())
            rows.append(
                {
                    "name": str(image.name),
                    "filepath": str(image.filepath),
                    "packed": image.packed_file is not None,
                    "colorspace": str(image.colorspace_settings.name),
                    "size_px": [int(image.size[0]), int(image.size[1])],
                    "file_format": str(image.file_format),
                }
            )
    return sorted(rows, key=lambda row: row["name"])


def inspect(asset_uid: str, path: Path) -> dict[str, Any]:
    before = scene_object_pointers()
    bpy.ops.import_scene.gltf(filepath=str(path))
    bpy.context.view_layer.update()
    imported = [object_ for object_ in bpy.data.objects if object_.as_pointer() not in before]
    mesh_rows: list[dict[str, Any]] = []
    materials: list[Any] = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for object_ in imported:
        if object_.type != "MESH":
            continue
        evaluated = object_.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
        try:
            mesh.calc_loop_triangles()
            vertices = [object_.matrix_world @ vertex.co for vertex in mesh.vertices]
            mesh_rows.append(
                {
                    "name": str(object_.name),
                    "vertices": int(len(mesh.vertices)),
                    "triangles": int(len(mesh.loop_triangles)),
                    "material_slots": [str(slot.material.name) if slot.material else None for slot in object_.material_slots],
                    "bounds_m": {
                        "min": [float(min(vertex[index] for vertex in vertices)) for index in range(3)],
                        "max": [float(max(vertex[index] for vertex in vertices)) for index in range(3)],
                    }
                    if vertices
                    else None,
                }
            )
        finally:
            evaluated.to_mesh_clear()
        for slot in object_.material_slots:
            if slot.material is not None:
                materials.append(slot.material)
    result = {
        "asset_uid": asset_uid,
        "source_relative_path": str(path.relative_to(ROOT)),
        "source_sha256": sha256(path),
        "source_bytes": int(path.stat().st_size),
        "imported_object_count": int(len(imported)),
        "mesh_count": int(len(mesh_rows)),
        "evaluated_triangles": int(sum(row["triangles"] for row in mesh_rows)),
        "meshes": sorted(mesh_rows, key=lambda row: row["name"]),
        "images": image_rows(materials),
    }
    for object_ in imported:
        bpy.data.objects.remove(object_, do_unlink=True)
    return result


def main() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    rows = []
    for asset_uid, path in ASSETS:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(inspect(asset_uid, path))
    output = ROOT / "reports/external_asset_probe.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "purpose": "asset-only import audit before v3 scene integration",
        "blender": bpy.app.version_string,
        "assets": rows,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
