"""Deterministic integration of the audited CC0 Smithsonian scan assets.

The source GLBs remain in ``assets/source/``.  This module imports one mesh at
build time, replaces a pre-existing procedural asset while preserving its
stable instance ID, and records the source identity on the resulting object
and mesh.  The replacement happens before ``prepare_layout`` so the existing
label and camera pipeline continues to work with the evaluated scanned mesh.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, MutableSequence

import bpy  # type: ignore
from mathutils import Vector  # type: ignore


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ScanReplacement:
    """One pre-audited source GLB and the procedural object it replaces."""

    target_name: str
    asset_uid: str
    source_relative_path: str
    source_sha256: str
    source_url: str
    morphology: str
    uniform_scale: float
    yaw_radians: float
    fish: bool = False


REPLACEMENTS: tuple[ScanReplacement, ...] = (
    ScanReplacement(
        target_name="Coral_Branching_01",
        asset_uid="smithsonian:acropora-cervicornis-usnm-1171477:low-100k",
        source_relative_path="assets/source/smithsonian/acropora_cervicornis_usnm_1171477_100k_2048.glb",
        source_sha256="c8e1189cbd23cfab84a8fdd6b2f410b4569df4b371fb8542c09d07cd1b5fb249",
        source_url="https://3d.si.edu/object/3d/acropora-cervicornis%3A31d1de41-2c76-46bd-8484-afb32e969431",
        morphology="scanned_branching_acropora",
        uniform_scale=10.8,
        yaw_radians=0.31,
    ),
    ScanReplacement(
        target_name="Coral_Brain_02",
        asset_uid="smithsonian:diploria-labyrinthiformis-usnm-74947:low-100k",
        source_relative_path="assets/source/smithsonian/diploria_labyrinthiformis_usnm_74947_100k_2048.glb",
        source_sha256="c55301c7ae8d2b829f855b69f6943a4ead345e727860fa77ce1617b43496af03",
        source_url="https://3d.si.edu/object/3d/diploria-labyrinthiformis%3A87738412-3acd-45d1-bff4-3ab67093470e",
        morphology="scanned_brain_diploria",
        uniform_scale=12.2,
        yaw_radians=-0.42,
    ),
    ScanReplacement(
        target_name="Fish_Disk_06",
        asset_uid="smithsonian:diodon-hystrix-usnm-195928:low-100k",
        source_relative_path="assets/source/smithsonian/diodon_hystrix_usnm_195928_100k_2048.glb",
        source_sha256="38ed855046ee516f56c29e1497790f2acaa4cbd159781734436b7bb62beb8b3f",
        source_url="https://3d.si.edu/object/3d/diodon-hystrix%3Ae0bdb234-d604-41fd-8f47-27368159c476",
        morphology="scanned_porcupinefish",
        uniform_scale=6.7,
        yaw_radians=0.0,
        fish=True,
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _world_bounds(object_: bpy.types.Object) -> tuple[Vector, Vector]:
    points = [object_.matrix_world @ Vector(corner) for corner in object_.bound_box]
    return (
        Vector((min(point.x for point in points), min(point.y for point in points), min(point.z for point in points))),
        Vector((max(point.x for point in points), max(point.y for point in points), max(point.z for point in points))),
    )


def _remove_imported(imported: list[bpy.types.Object], keep_mesh: bpy.types.Mesh | None = None) -> None:
    for object_ in imported:
        mesh = object_.data if object_.type == "MESH" else None
        bpy.data.objects.remove(object_, do_unlink=True)
        if mesh is not None and mesh != keep_mesh and mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def _import_single_mesh(path: Path) -> tuple[bpy.types.Object, list[bpy.types.Object]]:
    before = {object_.as_pointer() for object_ in bpy.data.objects}
    result = bpy.ops.import_scene.gltf(filepath=str(path))
    if result != {"FINISHED"}:
        raise RuntimeError(f"GLB import failed for {path.name}: {result}")
    imported = [object_ for object_ in bpy.data.objects if object_.as_pointer() not in before]
    meshes = [object_ for object_ in imported if object_.type == "MESH"]
    if len(meshes) != 1:
        _remove_imported(imported)
        raise RuntimeError(f"Expected exactly one imported mesh in {path.name}, found {len(meshes)}")
    return meshes[0], imported


def _replace_mesh(target: bpy.types.Object, spec: ScanReplacement) -> None:
    path = ROOT / spec.source_relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha = _sha256(path)
    if actual_sha != spec.source_sha256:
        raise RuntimeError(
            f"External asset SHA mismatch for {path.name}: expected {spec.source_sha256}, got {actual_sha}"
        )
    previous_bounds = _world_bounds(target)
    centre = (previous_bounds[0] + previous_bounds[1]) * 0.5
    base_z = float(previous_bounds[0].z)
    imported, imported_objects = _import_single_mesh(path)
    new_mesh = imported.data
    old_mesh = target.data
    target.data = new_mesh
    target.rotation_mode = "XYZ"
    target.rotation_euler = (0.0, 0.0, float(spec.yaw_radians))
    target.scale = (float(spec.uniform_scale),) * 3
    bpy.context.view_layer.update()
    scaled_bounds = _world_bounds(target)
    target.location = Vector((float(centre.x), float(centre.y), base_z - float(scaled_bounds[0].z)))
    target.data.name = f"{target.name}_CC0ScanMesh"
    target.data["asset_source"] = "cc0_imported"
    target.data["asset_uid"] = spec.asset_uid
    target.data["source_relative_path"] = spec.source_relative_path
    target.data["source_sha256"] = actual_sha
    target["asset_source"] = "cc0_imported"
    target["asset_uid"] = spec.asset_uid
    target["asset_morphology"] = spec.morphology
    target["external_source_relative_path"] = spec.source_relative_path
    target["external_source_sha256"] = actual_sha
    target["external_source_url"] = spec.source_url
    target["external_license"] = "CC0"
    target["external_import_scale"] = float(spec.uniform_scale)
    target["external_import_yaw_rad"] = float(spec.yaw_radians)
    target["external_images_packed_in_glb"] = True
    if spec.fish:
        yaw = float(spec.yaw_radians)
        target["asset_forward"] = (float(math.cos(yaw)), float(math.sin(yaw)), 0.0)
        target["asset_up"] = (0.0, 0.0, 1.0)
    for material in target.data.materials:
        if material is None:
            continue
        material["asset_source"] = "cc0_imported"
        material["asset_uid"] = spec.asset_uid
        material["external_license"] = "CC0"
        material["source_relative_path"] = spec.source_relative_path
    _remove_imported(imported_objects, keep_mesh=new_mesh)
    if old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)
    bpy.context.view_layer.update()


def apply_scanned_replacements(
    objects: MutableSequence[bpy.types.Object],
    instances: MutableSequence[dict[str, Any]],
    morphology_counts: Any,
) -> list[dict[str, Any]]:
    """Replace three named v2 placeholders and return JSON-safe provenance."""

    by_name = {object_.name: object_ for object_ in objects}
    rows: list[dict[str, Any]] = []
    for spec in REPLACEMENTS:
        target = by_name.get(spec.target_name)
        if target is None:
            raise RuntimeError(f"Missing procedural replacement target {spec.target_name}")
        previous_morphology = str(target.get("asset_morphology", "unknown"))
        _replace_mesh(target, spec)
        if previous_morphology in morphology_counts:
            morphology_counts[previous_morphology] -= 1
            if morphology_counts[previous_morphology] <= 0:
                del morphology_counts[previous_morphology]
        morphology_counts[spec.morphology] += 1
        instance_id = int(target["instance_id"])
        for instance in instances:
            if int(instance.get("instance_id", -1)) == instance_id:
                instance["name"] = target.name
                instance["morphology"] = spec.morphology
                instance["asset_uid"] = spec.asset_uid
                instance["source"] = "cc0_imported"
                break
        else:
            raise RuntimeError(f"Missing instance record for {target.name}")
        rows.append(
            {
                "asset_uid": spec.asset_uid,
                "target_name": target.name,
                "instance_id": instance_id,
                "semantic_id": int(target["semantic_id"]),
                "morphology": spec.morphology,
                "source_relative_path": spec.source_relative_path,
                "source_sha256": spec.source_sha256,
                "source_url": spec.source_url,
                "license": "CC0",
                "uniform_scale": float(spec.uniform_scale),
                "yaw_radians": float(spec.yaw_radians),
                "images_packed": True,
            }
        )
    return rows


__all__ = ["REPLACEMENTS", "apply_scanned_replacements"]
