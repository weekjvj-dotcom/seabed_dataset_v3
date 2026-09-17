"""Assembly of the complete deterministic coral-reef ecosystem.

The public entry point is :func:`build_ecosystem`.  It creates one owned
collection containing static mesh objects for sand, rock, coral, seagrass, and
fish.  Every target object carries semantic and instance IDs so the scene can
later feed the label exporter without reconstructing object identity.
"""

from __future__ import annotations

from collections import Counter
from math import cos, pi, sin
from random import Random
from typing import Any, Sequence

import bpy  # type: ignore

from ..contracts import layout_family_for_seed
from .geometry import (
    GeometryData,
    append_leaf_blade,
    append_plate,
    append_rock_lump,
    append_sandbed,
    append_thick_triangle,
    append_tube,
    append_uv_sphere,
    make_mesh_object,
    mesh_triangle_count,
    sandbed_height,
    transform_geometry,
)
from .external import apply_scanned_replacements
from .materials import build_material_library


SEMANTIC_BACKGROUND = 0
SEMANTIC_SAND = 1
SEMANTIC_ROCK = 2
SEMANTIC_CORAL = 3
SEMANTIC_GRASS = 4
SEMANTIC_FISH = 5
ASSET_SOURCE = "original_programmatic"
TRIANGLE_BUDGET = 1_500_000


def _clear_owned_collection(collection: bpy.types.Collection) -> None:
    """Clear only objects already inside this builder-owned collection."""

    # Assets may be grouped into optional foreground/midground/background
    # child collections by downstream scene builders. Clear those children as
    # well so rebuilding a scene never leaves stale geometry or IDs behind.
    for child in list(collection.children):
        _clear_owned_collection(child)
        collection.children.unlink(child)
        if child.users == 0:
            bpy.data.collections.remove(child)
    for obj in list(collection.objects):
        mesh = obj.data if obj.type == "MESH" else None
        collection.objects.unlink(obj)
        if obj.users == 0:
            bpy.data.objects.remove(obj)
            if mesh is not None and mesh.users == 0:
                bpy.data.meshes.remove(mesh)


def _new_collection_for_scene(scene: bpy.types.Scene, base_name: str) -> bpy.types.Collection:
    suffix = 1
    name = base_name
    while bpy.data.collections.get(name) is not None:
        suffix += 1
        name = f"{base_name}_{suffix:02d}"
    collection = bpy.data.collections.new(name)
    scene.collection.children.link(collection)
    return collection


def _ensure_collection(scene: bpy.types.Scene, seed: int) -> bpy.types.Collection:
    name = "Seabed_Ecosystem"
    owned_children = [
        child for child in scene.collection.children if child.get("scene_role") == "complex_procedural_seabed_ecology"
    ]
    if any(
        any(
            candidate_scene is not scene
            and candidate_scene.collection.children.get(child.name) == child
            for candidate_scene in bpy.data.scenes
        )
        for child in owned_children
    ):
        raise ValueError("shared ecosystem scene cannot be rebuilt; use a fresh scene")
    current_owned = owned_children[0] if owned_children else None
    if current_owned is not None:
        _clear_owned_collection(current_owned)
        current_owned["asset_source"] = ASSET_SOURCE
        current_owned["scene_role"] = "complex_procedural_seabed_ecology"
        return current_owned
    collection = bpy.data.collections.get(name)
    if collection is not None and collection.get("asset_source") != ASSET_SOURCE:
        # Respect a pre-existing collection owned by another scene builder.
        # Only collections marked by this module are safe to rebuild in place.
        return _new_collection_for_scene(scene, f"{name}_Procedural")
    if collection is not None:
        current_scene_owns = scene.collection.children.get(collection.name) == collection
        other_scene_owns = any(
            candidate_scene is not scene
            and candidate_scene.collection.children.get(collection.name) == collection
            for candidate_scene in bpy.data.scenes
        )
        if not current_scene_owns or other_scene_owns:
            return _new_collection_for_scene(scene, f"{name}_Seed{int(seed)}")
    if collection is None:
        collection = bpy.data.collections.new(name)
        scene.collection.children.link(collection)
    elif collection.name not in {child.name for child in scene.collection.children}:
        scene.collection.children.link(collection)
    _clear_owned_collection(collection)
    collection["asset_source"] = ASSET_SOURCE
    collection["scene_role"] = "complex_procedural_seabed_ecology"
    return collection


def _register(
    collection: bpy.types.Collection,
    objects: list[bpy.types.Object],
    instances: list[dict[str, Any]],
    morphology_counts: Counter[str],
    *,
    name: str,
    geometry: GeometryData,
    materials: Sequence[bpy.types.Material],
    semantic_id: int,
    instance_id: int,
    morphology: str,
    layer: str = "midground",
    active: bool = True,
    asset_uid: str | None = None,
    base_forward: Sequence[float] | None = None,
    base_up: Sequence[float] | None = None,
) -> bpy.types.Object:
    uid = asset_uid or f"seabed-v2:asset-{int(instance_id):04d}"
    obj = make_mesh_object(
        collection,
        name,
        geometry,
        materials,
        semantic_id=semantic_id,
        instance_id=instance_id,
        morphology=morphology,
    )
    objects.append(obj)
    obj["asset_uid"] = uid
    obj["asset_layer"] = str(layer)
    obj["asset_pool_active"] = bool(active)
    obj["asset_forward"] = list(base_forward or (0.0, 0.0, 1.0))
    obj["asset_up"] = list(base_up or (0.0, 1.0, 0.0))
    obj.hide_render = not bool(active)
    obj["visible_camera"] = bool(active)
    instances.append(
        {
            "instance_id": int(instance_id),
            "semantic_id": int(semantic_id),
            "name": obj.name,
            "morphology": str(morphology),
            "objects": [obj.name],
            "asset_uid": uid,
            "layer": str(layer),
            "pool_active": bool(active),
        }
    )
    morphology_counts[morphology] += 1
    return obj


def _coral_materials(materials: dict[str, bpy.types.Material]) -> list[bpy.types.Material]:
    return [
        materials["coral_teal"],
        materials["coral_red"],
        materials["coral_purple"],
        materials["coral_gold"],
        materials["coral_groove"],
        materials["coral_polyp"],
        materials["coral_tip"],
        materials["coral_table_ochre"],
        materials["coral_table_edge"],
    ]


def _layout_profile(seed: int) -> tuple[str, str]:
    """Return a stable layout family for pilot splits and arbitrary seeds."""

    styles = {
        "curved_dual_ridge": "弯曲双礁脊",
        "left_concentrated_island": "偏左集中礁岛",
        "diagonal_scatter": "斜向错落群落",
    }
    family = layout_family_for_seed(seed)
    return family, styles[family]


def _layout_xy(x: float, y: float, layout_family: str, intensity: float = 1.0) -> tuple[float, float]:
    """Apply a visible, deterministic family-level deformation to a point."""

    x = float(x)
    y = float(y)
    if layout_family == "curved_dual_ridge":
        x += intensity * (0.72 * sin(0.36 * (y + 1.0)) + 0.16 * (1.0 if x >= 0.0 else -1.0) * cos(0.22 * y))
        y += intensity * 0.16 * sin(0.30 * x)
    elif layout_family == "left_concentrated_island":
        x = x * (1.0 - 0.13 * intensity) - 0.95 * intensity + 0.22 * intensity * sin(0.44 * y)
        y += intensity * 0.24 * cos(0.25 * x)
    else:
        x += intensity * (0.23 * (y - 4.0) + 0.20 * sin(0.46 * y))
        y += intensity * (0.15 * x + 0.12 * sin(0.31 * x))
    return (max(-6.6, min(6.6, x)), y)


def _build_branching_coral(
    rng: Random,
    center: Sequence[float],
    height: float,
    spread: float,
    variant: int,
) -> GeometryData:
    """Build a multi-level 3-D branching coral with visible polyps."""

    geometry = GeometryData()
    base_index = variant % 4
    base = tuple(float(v) for v in center)
    # Two or three low crowns share one broad base. Each crown has four arms,
    # two short child branches and one second-order twig per arm: 32-48 tips
    # per colony while avoiding a tall, single tree-like trunk.
    append_uv_sphere(geometry, (base[0], base[1], base[2] + height * 0.15), (spread * 0.66, spread * 0.56, height * 0.14), segments=14, rings=6, material_index=base_index)
    crown_count = 2 + variant % 2
    for crown in range(crown_count):
        crown_angle = 2.0 * pi * crown / crown_count + rng.uniform(-0.22, 0.22)
        crown_radius = spread * rng.uniform(0.15, 0.30)
        crown_base = (
            base[0] + cos(crown_angle) * crown_radius,
            base[1] + sin(crown_angle) * crown_radius,
            base[2] + height * 0.12,
        )
        crown_top = (
            crown_base[0],
            crown_base[1],
            base[2] + height * rng.uniform(0.35, 0.45),
        )
        append_tube(geometry, [crown_base, crown_top], [spread * 0.15, spread * 0.075], sides=8, material_index=base_index)
        append_uv_sphere(geometry, crown_base, (spread * 0.24, spread * 0.20, height * 0.12), segments=10, rings=5, material_index=base_index)
        for branch in range(4):
            angle = 2.0 * pi * branch / 4.0 + crown_angle * 0.37 + rng.uniform(-0.22, 0.22)
            radial = spread * rng.uniform(0.34, 0.58)
            first = (
                crown_top[0] + cos(angle) * radial * 0.28,
                crown_top[1] + sin(angle) * radial * 0.28,
                base[2] + height * rng.uniform(0.38, 0.53),
            )
            second = (
                crown_top[0] + cos(angle) * radial * 0.66,
                crown_top[1] + sin(angle) * radial * 0.66,
                base[2] + height * rng.uniform(0.55, 0.70),
            )
            endpoint = (
                crown_top[0] + cos(angle) * radial,
                crown_top[1] + sin(angle) * radial,
                base[2] + height * rng.uniform(0.72, 0.90),
            )
            append_tube(geometry, [crown_top, first, second, endpoint], [spread * 0.085, spread * 0.070, spread * 0.045, spread * 0.024], sides=7, material_index=base_index)
            append_uv_sphere(geometry, endpoint, (spread * 0.055, spread * 0.055, spread * 0.065), segments=7, rings=4, material_index=6)
            for child in range(2):
                child_angle = angle + (-0.48 if child == 0 else 0.48) + rng.uniform(-0.15, 0.15)
                child_length = radial * rng.uniform(0.34, 0.55)
                child_end = (
                    endpoint[0] + cos(child_angle) * child_length,
                    endpoint[1] + sin(child_angle) * child_length,
                    endpoint[2] + height * rng.uniform(0.045, 0.15),
                )
                append_tube(geometry, [second, endpoint, child_end], [spread * 0.046, spread * 0.026, spread * 0.010], sides=6, material_index=base_index)
                append_uv_sphere(geometry, child_end, (spread * 0.035, spread * 0.035, spread * 0.044), segments=6, rings=4, material_index=6)
                if child == 0:
                    twig_angle = child_angle + rng.uniform(-0.48, 0.48)
                    twig_end = (
                        child_end[0] + cos(twig_angle) * child_length * 0.44,
                        child_end[1] + sin(twig_angle) * child_length * 0.44,
                        child_end[2] + height * rng.uniform(0.025, 0.075),
                    )
                    append_tube(geometry, [child_end, twig_end], [spread * 0.014, spread * 0.005], sides=5, material_index=base_index)
                    append_uv_sphere(geometry, twig_end, (spread * 0.021, spread * 0.021, spread * 0.026), segments=6, rings=3, material_index=6)
    # Polyps around the base collar break up the trunk-to-sand transition.
    for polyp in range(7):
        angle = 2.0 * pi * polyp / 7.0 + 0.17 * sin(polyp)
        radius = spread * rng.uniform(0.17, 0.29)
        point = (base[0] + cos(angle) * radius, base[1] + sin(angle) * radius, base[2] + height * 0.12)
        append_uv_sphere(geometry, point, (spread * 0.042, spread * 0.042, spread * 0.052), segments=6, rings=4, material_index=5)
    return geometry


def _build_brain_coral(
    rng: Random,
    center: Sequence[float],
    radius: float,
    height: float,
    variant: int,
) -> GeometryData:
    """Build a thick lobed brain/block coral with inset maze-like grooves."""

    geometry = GeometryData()
    base_index = variant % 4
    base = tuple(float(v) for v in center)
    # A broad lower lobe joins the upper lobes into one rounded colony instead
    # of leaving several disconnected ball-like pieces.
    append_uv_sphere(
        geometry,
        (base[0], base[1], base[2] + height * 0.35),
        (radius * 1.08, radius * 0.93, height * 0.35),
        segments=18,
        rings=10,
        material_index=base_index,
    )
    lobes = 3 + (variant % 2)
    for lobe in range(lobes):
        angle = 2.0 * pi * lobe / lobes + rng.uniform(-0.23, 0.23)
        offset = radius * rng.uniform(0.08, 0.24)
        lobe_center = (
            base[0] + cos(angle) * offset,
            base[1] + sin(angle) * offset,
            base[2] + height * rng.uniform(0.52, 0.62),
        )
        append_uv_sphere(
            geometry,
            lobe_center,
            (radius * rng.uniform(0.55, 0.74), radius * rng.uniform(0.50, 0.69), height * rng.uniform(0.34, 0.48)),
            segments=18,
            rings=9,
            material_index=base_index,
        )
    # Dense, irregular grooves are sunk into the surface. Two crossing families
    # create the characteristic brain-like folded topology from the camera.
    for groove in range(10):
        phase = groove * 0.92 + rng.uniform(-0.10, 0.10)
        points: list[tuple[float, float, float]] = []
        for sample in range(13):
            t = sample / 12.0
            x = base[0] + (t - 0.5) * radius * 1.84
            y = base[1] + sin(t * pi * 2.7 + phase) * radius * (0.10 + groove * 0.010)
            dome = max(0.0, 1.0 - ((x - base[0]) / (radius * 1.10)) ** 2 - ((y - base[1]) / (radius * 0.92)) ** 2)
            # The groove center is a shallow inset on the dome, with roughly
            # half the tube radius inside the lobe mesh so it cannot float.
            z = base[2] + height * (0.75 + 0.12 * dome) - 0.042 + 0.008 * sin(sample + phase)
            points.append((x, y, z))
        append_tube(geometry, points, radius * 0.032, sides=5, material_index=4, cap=True)
    for groove in range(5):
        phase = groove * 1.22 + rng.uniform(-0.16, 0.16)
        points = []
        for sample in range(11):
            t = sample / 10.0
            y = base[1] + (t - 0.5) * radius * 1.54
            x = base[0] + sin(t * pi * 2.1 + phase) * radius * 0.25
            dome = max(0.0, 1.0 - ((x - base[0]) / (radius * 1.10)) ** 2 - ((y - base[1]) / (radius * 0.92)) ** 2)
            z = base[2] + height * (0.76 + 0.10 * dome) - 0.040
            points.append((x, y, z))
        append_tube(geometry, points, radius * 0.030, sides=5, material_index=4, cap=True)
    # Small coral polyps are sparse on brain ridges and keep the surface tactile.
    for polyp in range(8):
        angle = 2.0 * pi * polyp / 8.0 + rng.uniform(-0.18, 0.18)
        point = (
            base[0] + cos(angle) * radius * rng.uniform(0.25, 0.78),
            base[1] + sin(angle) * radius * rng.uniform(0.20, 0.72),
            base[2] + height * rng.uniform(0.72, 0.94),
        )
        append_uv_sphere(geometry, point, (radius * 0.035, radius * 0.035, radius * 0.045), segments=6, rings=4, material_index=5)
    return geometry


def _build_table_or_fan_coral(
    rng: Random,
    center: Sequence[float],
    height: float,
    radius: float,
    variant: int,
    morphology: str,
) -> GeometryData:
    """Build layered table plates or a thick, ribbed sea-fan silhouette."""

    geometry = GeometryData()
    base_index = variant % 4
    table_base_index = 7 if morphology == "table" else base_index
    base = tuple(float(v) for v in center)
    append_uv_sphere(geometry, (base[0], base[1], base[2] + radius * 0.18), (radius * 0.74, radius * 0.62, radius * 0.18), segments=16, rings=6, material_index=table_base_index)
    append_tube(geometry, [base, (base[0], base[1], base[2] + height * 0.34), (base[0], base[1], base[2] + height * 0.68)], [radius * 0.24, radius * 0.16, radius * 0.075], sides=8, material_index=table_base_index)
    for support in range(3):
        support_angle = support * (2.0 * pi / 3.0) + rng.uniform(-0.16, 0.16)
        support_base = (base[0] + cos(support_angle) * radius * 0.32, base[1] + sin(support_angle) * radius * 0.27, base[2] + radius * 0.16)
        support_top = (base[0] + cos(support_angle) * radius * 0.19, base[1] + sin(support_angle) * radius * 0.16, base[2] + height * 0.48)
        append_tube(geometry, [support_base, support_top], [radius * 0.11, radius * 0.055], sides=7, material_index=table_base_index)

    if morphology == "table":
        levels = 3
        for level in range(levels):
            fraction = 0.48 + level * 0.18
            offset_angle = rng.uniform(0.0, 2.0 * pi)
            eccentric = radius * (0.15 + 0.06 * level)
            plate_center = (
                base[0] + cos(offset_angle) * eccentric,
                base[1] + sin(offset_angle) * eccentric * 0.72,
                base[2] + height * fraction,
            )
            rx = radius * (0.92 - level * 0.13) * rng.uniform(0.90, 1.06)
            ry = radius * (0.66 - level * 0.07) * rng.uniform(0.92, 1.06)
            append_plate(geometry, plate_center, rx, ry, radius * 0.075, segments=28, material_index=7, rim_material_index=8, phase=rng.uniform(0.0, pi))
            # Small offset rosette petals break the circular-disk silhouette.
            for petal in range(3):
                petal_angle = offset_angle + petal * (2.0 * pi / 3.0) + rng.uniform(-0.12, 0.12)
                petal_center = (
                    plate_center[0] + cos(petal_angle) * rx * 0.38,
                    plate_center[1] + sin(petal_angle) * ry * 0.34,
                    plate_center[2] + radius * 0.035,
                )
                append_plate(geometry, petal_center, rx * 0.25, ry * 0.17, radius * 0.042, segments=16, material_index=7, rim_material_index=8, phase=petal_angle)
            for polyp in range(10):
                angle = 2.0 * pi * polyp / 8.0 + 0.13 * level
                edge = (rx * cos(angle), ry * sin(angle), radius * 0.085)
                append_uv_sphere(geometry, (plate_center[0] + edge[0], plate_center[1] + edge[1], plate_center[2] + edge[2]), (radius * 0.026, radius * 0.026, radius * 0.035), segments=6, rings=4, material_index=8)
    else:
        # A porous fan is a connected mesh of fine arched ribs, with open holes
        # between them instead of opaque triangular panels.
        fan_width = radius * rng.uniform(1.55, 1.90)
        fan_height = height * rng.uniform(0.68, 0.88)
        fan_base_z = base[2] + height * 0.56
        for rib in range(8):
            t = rib / 7.0 - 0.5
            top_fraction = 0.80 - abs(t) * 0.24
            points = []
            for sample in range(6):
                u = sample / 5.0
                x = base[0] + t * fan_width * 0.95 + sin(u * pi + rib) * radius * 0.045
                y = base[1] + sin(rib * 0.81 + u * pi * 1.5) * radius * 0.10
                z = fan_base_z + fan_height * top_fraction * u
                points.append((x, y, z))
            append_tube(geometry, points, [radius * 0.034, radius * 0.027, radius * 0.021, radius * 0.016, radius * 0.011, radius * 0.006], sides=5, material_index=base_index)
            append_uv_sphere(geometry, points[-1], (radius * 0.018, radius * 0.018, radius * 0.024), segments=6, rings=3, material_index=6)
        for row in range(5):
            row_fraction = row / 4.0
            points = []
            for sample in range(13):
                u = sample / 12.0 - 0.5
                x = base[0] + u * fan_width * (0.98 - row_fraction * 0.10)
                y = base[1] + sin(sample * 0.75 + row * 0.70) * radius * 0.08
                z = fan_base_z + fan_height * row_fraction * (0.83 + 0.08 * cos(u * pi))
                points.append((x, y, z))
            append_tube(geometry, points, radius * 0.014, sides=5, material_index=base_index)
        append_tube(geometry, [base, (base[0], base[1], base[2] + height * 0.50), (base[0], base[1], fan_base_z)], [radius * 0.20, radius * 0.11, radius * 0.07], sides=8, material_index=base_index)
    return geometry


def _build_grass_clump(rng: Random, center: Sequence[float], height_scale: float) -> GeometryData:
    """Build a dense cluster of independently curved seagrass blades."""

    geometry = GeometryData()
    base = tuple(float(v) for v in center)
    blade_count = rng.randint(10, 15)
    for blade in range(blade_count):
        angle = 2.0 * pi * blade / blade_count + rng.uniform(-0.35, 0.35)
        height = height_scale * rng.uniform(0.68, 1.18)
        lean = rng.uniform(0.10, 0.34)
        points: list[tuple[float, float, float]] = []
        for sample in range(6):
            t = sample / 5.0
            curl = sin(t * pi * rng.uniform(0.68, 1.25) + angle) * lean * 0.28 * t
            radial = lean * t * t
            points.append(
                (
                    base[0] + cos(angle) * radial + cos(angle + pi * 0.5) * curl,
                    base[1] + sin(angle) * radial + sin(angle + pi * 0.5) * curl,
                    base[2] + height * t,
                )
            )
        blade_width = rng.uniform(0.065, 0.105)
        widths = [blade_width * (1.0 - 0.82 * (index / 5.0)) for index in range(6)]
        append_leaf_blade(geometry, points, widths, thickness=0.008, material_index=0)
    return geometry


def _build_fish(
    rng: Random,
    kind: str,
    center: Sequence[float],
    scale: float,
    yaw: float,
    pitch: float,
    body_key: str,
) -> GeometryData:
    """Build one fish as a single multi-material static mesh assembly."""

    local = GeometryData()
    if kind == "fusiform_fish":
        half_length = 0.64 * scale
        body_width = 0.21 * scale
        body_height = 0.18 * scale
        fin_span = 0.38 * scale
    else:
        # Disk fish are deliberately sideways thin, but broad in profile:
        # body width 0.10-0.16 m, height 0.35-0.45 m, half-length 0.40-0.55 m.
        body_width = max(0.10, min(0.16, 0.14 * scale))
        body_height = max(0.35, min(0.45, 0.40 * scale))
        half_length = max(0.40, min(0.55, 0.46 * scale))
        fin_span = 0.52 * scale

    append_uv_sphere(local, (0.0, 0.0, 0.0), (body_width, body_height, half_length), segments=18, rings=10, material_index=0)
    # Tail, dorsal, anal, and paired pectoral fins are closed solids.
    tail_root = half_length * 0.58
    fork_length = max(0.30 * scale, min(0.45 * scale, 0.38 * scale))
    append_tube(local, [(0.0, 0.0, tail_root), (0.0, 0.0, half_length * 0.86), (0.0, 0.0, half_length * 1.08)], [body_width * 0.45, body_width * 0.25, body_width * 0.10], sides=7, material_index=0)
    append_thick_triangle(local, [(-body_width * 0.34, 0.0, tail_root), (body_width * 0.08, fin_span * 0.70, tail_root + scale * 0.06), (body_width * 0.02, fin_span * 0.18, tail_root + fork_length)], scale * 0.030, material_index=3)
    append_thick_triangle(local, [(-body_width * 0.34, 0.0, tail_root), (-body_width * 0.08, -fin_span * 0.70, tail_root + scale * 0.06), (-body_width * 0.02, -fin_span * 0.18, tail_root + fork_length)], scale * 0.030, material_index=3)
    append_thick_triangle(local, [(-body_width * 0.20, body_height * 0.66, -scale * 0.01), (0.0, body_height * 1.48, scale * 0.13), (body_width * 0.26, body_height * 0.64, scale * 0.20)], scale * 0.032, material_index=4)
    append_thick_triangle(local, [(-body_width * 0.20, -body_height * 0.66, scale * 0.04), (0.0, -body_height * 1.36, scale * 0.17), (body_width * 0.26, -body_height * 0.62, scale * 0.20)], scale * 0.032, material_index=3)
    append_thick_triangle(local, [(body_width * 0.60, 0.0, -scale * 0.03), (body_width * 1.65, -body_height * 0.24, scale * 0.16), (body_width * 0.72, body_height * 0.22, scale * 0.22)], scale * 0.028, material_index=4)
    append_thick_triangle(local, [(-body_width * 0.60, 0.0, -scale * 0.03), (-body_width * 1.65, body_height * 0.24, scale * 0.16), (-body_width * 0.72, -body_height * 0.22, scale * 0.22)], scale * 0.028, material_index=3)

    # Eyes and mouth: same mesh object and therefore the fish instance ID.
    eye_z = -half_length * 0.73
    for side in (-1.0, 1.0):
        append_uv_sphere(local, (side * body_width * 0.83, body_height * 0.19, eye_z), (scale * 0.040, scale * 0.040, scale * 0.040), segments=8, rings=5, material_index=5)
        append_uv_sphere(local, (side * body_width * 0.84, body_height * 0.19, eye_z - scale * 0.024), (scale * 0.014, scale * 0.014, scale * 0.014), segments=6, rings=4, material_index=6)
    append_uv_sphere(local, (0.0, -body_height * 0.035, eye_z - scale * 0.035), (scale * 0.030, scale * 0.020, scale * 0.022), segments=7, rings=4, material_index=6)

    # Thin bands follow the elliptical body surface instead of using protruding
    # spheres. The x position is evaluated from the local cross-section, so the
    # decals remain flush for both fish morphologies.
    def append_surface_band(side: float, z_center: float, z_half: float, material_index: int) -> None:
        rows = 4
        row_indices: list[tuple[int, int]] = []
        for row in range(rows):
            y = -body_height * 0.62 + body_height * 1.24 * row / (rows - 1)
            profile = max(0.24, 1.0 - (y / max(body_height, 1.0e-6)) ** 2) ** 0.5
            x = side * (body_width * profile + 0.0025)
            left = local.add_vertex((x, y, z_center - z_half))
            right = local.add_vertex((x, y, z_center + z_half))
            row_indices.append((left, right))
        for row in range(rows - 1):
            if side > 0.0:
                local.add_face((row_indices[row][0], row_indices[row + 1][0], row_indices[row + 1][1], row_indices[row][1]), material_index)
            else:
                local.add_face((row_indices[row][0], row_indices[row][1], row_indices[row + 1][1], row_indices[row + 1][0]), material_index)

    stripe_count = 3 if kind == "fusiform_fish" else 4
    for stripe in range(stripe_count):
        z = -half_length * 0.34 + stripe * (half_length * 0.22)
        for side in (-1.0, 1.0):
            append_surface_band(side, z, scale * (0.038 if kind == "fusiform_fish" else 0.050), 2)
    # Small alternating flush scale marks sit between the main bands.
    for scale_mark in range(6):
        z = -half_length * 0.40 + scale_mark * half_length * 0.15
        side = -1.0 if scale_mark % 2 == 0 else 1.0
        append_surface_band(side, z, scale * 0.014, 1)
    return transform_geometry(local, center, yaw, pitch)


def build_ecosystem(scene: bpy.types.Scene, seed: int, floor_z: float = 0.0) -> dict[str, Any]:
    """Build and return a complex, deterministic static reef ecosystem.

    Parameters
    ----------
    scene:
        Blender scene whose root collection receives the owned ecosystem
        collection.
    seed:
        Integer layout seed.  All positional and morphological choices derive
        from this value; the resulting scene has no animated random state.
    floor_z:
        World Z offset for the sand bed and all benthic assets.
    """

    if scene is None:
        raise ValueError("scene is required")
    seed = int(seed)
    floor_z = float(floor_z)
    rng = Random(seed)
    layout_family, layout_style = _layout_profile(seed)
    sand_phase = seed * 0.017

    def benthic_z(x: float, y: float, clearance: float = 0.035) -> float:
        return sandbed_height(x, y, floor_z=floor_z, seed_phase=sand_phase) + clearance

    collection = _ensure_collection(scene, seed)
    collection["seed"] = seed
    collection["floor_z"] = floor_z
    collection["layout_family"] = layout_family
    collection["layout_style"] = layout_style
    collection["layer_hierarchy"] = "ENVIRONMENT/FOREGROUND/MIDGROUND/BACKGROUND"
    collection["layout_notes"] = "中心双层珊瑚礁岛；前景锚点位于开放通道两侧；岩礁沿珊瑚脊排列；内区 x[-16,16],y[-10,24] 保持高密度；15m以上保留独立珊瑚/岩礁/海草轮廓；seed 101/202/303 使用不同布局族"
    collection["semantic_ids"] = "0=background;1=sand;2=rock;3=coral;4=grass;5=fish"

    material_library = build_material_library()
    coral_material_library = _coral_materials(material_library)
    objects: list[bpy.types.Object] = []
    instances: list[dict[str, Any]] = []
    morphology_counts: Counter[str] = Counter()
    semantic_counts: Counter[int] = Counter()
    next_instance_id = 1

    def register_asset(**kwargs: Any) -> bpy.types.Object:
        nonlocal next_instance_id
        # Layer metadata is intentionally explicit and stable. It is used by
        # the per-state sampler as a creative anchor; measured near/mid/far
        # bins still come from the camera and geometry labels.
        if "layer" not in kwargs:
            kwargs["layer"] = "midground"
        obj = _register(
            collection,
            objects,
            instances,
            morphology_counts,
            instance_id=next_instance_id,
            **kwargs,
        )
        semantic_counts[int(kwargs["semantic_id"])] += 1
        next_instance_id += 1
        return obj

    # A 32 m by 34 m surface has low-frequency relief, fine ripples, and a
    # visibly lighter winding path through the central open sand area.
    sand = GeometryData()
    append_sandbed(sand, floor_z=floor_z, seed_phase=sand_phase, nx=65, ny=69, material_index=0, path_material_index=1)
    register_asset(
        name="Sandbed_Undulating_Path",
        geometry=sand,
        materials=[material_library["sand"], material_library["sand_path"]],
        semantic_id=SEMANTIC_SAND,
        morphology="sandbed",
        layer="environment",
    )

    # Edge reefs are composed of multiple staggered lumps; dark crevice ribs
    # preserve readable gaps when viewed from the main camera.
    rock_positions = [
        (-1.95, -2.55), (1.95, -2.35), (-4.9, 1.1), (4.8, 1.5),
        (-5.1, 4.0), (5.0, 4.3), (-4.6, 6.8), (4.7, 7.1),
        (-3.7, 9.4), (3.8, 9.8), (-5.5, 11.0), (5.4, 11.2),
        (-2.0, 12.6), (2.7, 13.1),
    ]
    for rock_index, (x, y) in enumerate(rock_positions):
        x, y = _layout_xy(x, y, layout_family, intensity=0.82)
        rock_base_z = benthic_z(x, y, clearance=0.02)
        geometry = GeometryData()
        cluster_radius = rng.uniform(0.85, 1.38)
        lumps = 2 + rock_index % 3
        for lump in range(lumps):
            angle = 2.0 * pi * lump / lumps + rng.uniform(-0.25, 0.25)
            scale_x = cluster_radius * rng.uniform(0.55, 0.88)
            scale_y = cluster_radius * rng.uniform(0.45, 0.80)
            scale_z = cluster_radius * rng.uniform(0.40, 0.73)
            lump_center = (
                x + cos(angle) * cluster_radius * rng.uniform(0.16, 0.52),
                y + sin(angle) * cluster_radius * rng.uniform(0.16, 0.52),
                rock_base_z + scale_z * 0.72 + rng.uniform(0.02, 0.07),
            )
            append_rock_lump(
                geometry,
                lump_center,
                (scale_x, scale_y, scale_z),
                seed_phase=seed * 0.013 + rock_index * 0.71 + lump,
                segments=13,
                rings=6,
                material_index=0,
                dark_material_index=1,
            )
        # A set of narrow, dark crack surfaces makes the cluster's negative
        # space legible without adding a second target instance.
        for crevice in range(2 + rock_index % 2):
            angle = rng.uniform(0.0, 2.0 * pi)
            p0 = (x + cos(angle) * cluster_radius * 0.28, y + sin(angle) * cluster_radius * 0.28, rock_base_z + cluster_radius * 0.22)
            p1 = (x - cos(angle) * cluster_radius * 0.40, y - sin(angle) * cluster_radius * 0.40, rock_base_z + cluster_radius * rng.uniform(0.30, 0.52))
            append_tube(geometry, [p0, p1], [cluster_radius * 0.055, cluster_radius * 0.022], sides=5, material_index=1)
        register_asset(
            name=f"Rock_ReefCluster_{rock_index + 1:02d}",
            geometry=geometry,
            materials=[material_library["rock"], material_library["rock_crevice"]],
            semantic_id=SEMANTIC_ROCK,
            morphology="rock_cluster",
            layer="foreground" if rock_index < 2 else ("midground" if rock_index < 10 else "background"),
        )

    # Thirty coral groups in four explicit morphologies.  Positions form dense
    # left/right layers while preserving the center sand path and front/mid/back
    # depth cues requested by the dataset design.
    branching_specs = [
        (-1.05, -1.55, 1.35, 0.95), (1.25, -1.35, 1.60, 1.02), (-0.95, -0.45, 1.20, 0.92), (4.1, 0.45, 1.45, 0.98),
        (-4.8, 1.8, 1.35, 0.90), (0.8, 1.9, 1.55, 1.00), (3.4, 3.1, 1.65, 0.95), (-3.8, 4.0, 1.25, 0.96),
        (-1.0, 5.2, 1.40, 0.98), (4.4, 5.8, 1.50, 0.94),
    ]
    brain_specs = [
        (-4.1, -0.3, 0.90, 0.82), (1.1, 0.1, 0.98, 0.90), (-4.7, 3.2, 0.82, 0.80), (2.0, 3.8, 0.90, 0.86),
        (-2.4, 6.2, 1.00, 0.94), (4.6, 6.9, 0.87, 0.86), (-4.2, 8.6, 0.90, 0.90), (1.0, 8.9, 0.88, 0.92),
        (-2.7, 10.5, 0.82, 0.82), (4.0, 10.8, 0.88, 0.88),
    ]
    table_specs = [
        (-2.4, -0.2, 1.50, 0.95), (2.1, 0.5, 1.55, 0.98), (-2.1, 3.2, 1.65, 0.92), (2.5, 4.8, 1.50, 0.90),
        (-0.5, 6.8, 1.60, 0.98), (3.1, 8.1, 1.70, 0.90),
    ]
    fan_specs = [
        (-4.6, 5.8, 1.70, 0.90), (0.0, 8.8, 1.65, 0.90), (4.4, 9.5, 1.85, 0.90), (-1.6, 10.9, 1.70, 0.90),
    ]

    for coral_index, (x, y, height, spread) in enumerate(branching_specs):
        x, y = _layout_xy(x, y, layout_family, intensity=0.95)
        x_j = x + rng.uniform(-0.18, 0.18)
        y_j = y + rng.uniform(-0.16, 0.16)
        register_asset(
            name=f"Coral_Branching_{coral_index + 1:02d}",
            geometry=_build_branching_coral(rng, (x_j, y_j, benthic_z(x_j, y_j)), height, spread, coral_index),
            materials=coral_material_library,
            semantic_id=SEMANTIC_CORAL,
            morphology="branching",
            layer="foreground" if coral_index < 3 else "midground",
        )
    for coral_index, (x, y, radius, height) in enumerate(brain_specs):
        x, y = _layout_xy(x, y, layout_family, intensity=0.95)
        x_j = x + rng.uniform(-0.18, 0.18)
        y_j = y + rng.uniform(-0.18, 0.18)
        register_asset(
            name=f"Coral_Brain_{coral_index + 1:02d}",
            geometry=_build_brain_coral(rng, (x_j, y_j, benthic_z(x_j, y_j)), radius, height, coral_index + 1),
            materials=coral_material_library,
            semantic_id=SEMANTIC_CORAL,
            morphology="brain",
            layer="foreground" if coral_index < 2 else ("midground" if coral_index < 8 else "background"),
        )
    for coral_index, (x, y, height, radius) in enumerate(table_specs):
        x, y = _layout_xy(x, y, layout_family, intensity=0.95)
        x_j = x + rng.uniform(-0.16, 0.16)
        y_j = y + rng.uniform(-0.16, 0.16)
        register_asset(
            name=f"Coral_Table_{coral_index + 1:02d}",
            geometry=_build_table_or_fan_coral(rng, (x_j, y_j, benthic_z(x_j, y_j)), height, radius, coral_index + 2, "table"),
            materials=coral_material_library,
            semantic_id=SEMANTIC_CORAL,
            morphology="table",
            layer="foreground" if coral_index < 2 else "midground",
        )
    for coral_index, (x, y, height, radius) in enumerate(fan_specs):
        x, y = _layout_xy(x, y, layout_family, intensity=0.95)
        x_j = x + rng.uniform(-0.16, 0.16)
        y_j = y + rng.uniform(-0.16, 0.16)
        register_asset(
            name=f"Coral_Fan_{coral_index + 1:02d}",
            geometry=_build_table_or_fan_coral(rng, (x_j, y_j, benthic_z(x_j, y_j)), height, radius, coral_index + 3, "fan"),
            materials=coral_material_library,
            semantic_id=SEMANTIC_CORAL,
            morphology="fan",
            layer="midground" if coral_index < 2 else "background",
        )

    grass_specs = [
        (-5.4, -1.3, 1.30), (4.7, -1.0, 1.44), (-5.2, 2.6, 1.48), (5.0, 2.9, 1.34),
        (-5.6, 6.0, 1.38), (5.3, 6.3, 1.55), (-4.8, 9.5, 1.50), (4.9, 9.8, 1.42),
        (-2.7, 11.7, 1.32), (2.9, 12.0, 1.48), (-5.9, 11.6, 1.46), (5.8, 11.9, 1.40),
    ]
    for grass_index, (x, y, h) in enumerate(grass_specs):
        x, y = _layout_xy(x, y, layout_family, intensity=0.92)
        x_j = x + rng.uniform(-0.18, 0.18)
        y_j = y + rng.uniform(-0.18, 0.18)
        register_asset(
            name=f"Seagrass_Clump_{grass_index + 1:02d}",
            geometry=_build_grass_clump(rng, (x_j, y_j, benthic_z(x_j, y_j, clearance=0.025)), h),
            materials=[material_library["grass"], material_library["grass_tip"]],
            semantic_id=SEMANTIC_GRASS,
            morphology="seagrass_clump",
            layer="foreground" if grass_index < 4 else ("midground" if grass_index < 9 else "background"),
        )

    fish_specs = [
        # Foreground trio: large and turned side-on to the main camera.
        ("disk_fish", -2.9, -1.4, 1.70, 1.15, 0.45, 0.02),
        ("fusiform_fish", 2.8, -1.8, 1.60, 1.10, -0.75, -0.03),
        ("disk_fish", 0.4, -0.5, 2.20, 1.00, 2.50, 0.01),
        # Eight midwater fish form two loose schools inside y=1..6.
        ("fusiform_fish", -3.9, 1.3, 2.00, 0.92, 0.25, 0.04),
        ("fusiform_fish", 3.6, 1.7, 2.35, 0.88, -0.40, -0.02),
        ("disk_fish", -1.8, 2.5, 1.85, 0.86, 0.15, 0.05),
        ("fusiform_fish", 1.7, 3.1, 2.55, 0.90, 2.50, 0.01),
        ("fusiform_fish", -3.4, 4.2, 2.75, 0.85, -2.40, -0.05),
        ("disk_fish", 0.2, 4.4, 2.40, 0.85, 2.80, 0.02),
        ("fusiform_fish", 3.9, 4.9, 2.05, 0.82, 1.60, -0.04),
        ("disk_fish", -0.9, 5.7, 2.90, 0.80, -1.80, 0.03),
        # Smaller background fish add depth without reintroducing wide empty
        # foreground sand or putting all silhouettes on one row.
        ("fusiform_fish", -4.8, 7.8, 3.00, 0.68, 0.30, 0.00),
        ("disk_fish", 4.7, 8.1, 3.20, 0.70, -0.50, 0.02),
        ("fusiform_fish", -3.6, 9.2, 3.15, 0.64, 2.70, -0.01),
        ("fusiform_fish", 3.2, 9.6, 3.35, 0.65, -2.20, 0.03),
        ("disk_fish", -1.1, 10.4, 3.25, 0.62, 1.40, -0.02),
        ("fusiform_fish", 4.8, 11.3, 3.45, 0.60, -1.00, 0.04),
        ("disk_fish", -4.4, 12.2, 3.15, 0.58, 0.20, 0.00),
        ("fusiform_fish", 1.5, 13.4, 3.55, 0.60, 2.80, -0.02),
        ("disk_fish", 4.9, 14.0, 3.45, 0.56, -2.60, 0.01),
    ]
    body_keys = ["fish_blue", "fish_orange", "fish_yellow", "fish_teal"]
    for fish_index, (kind, x, y, z, scale, yaw_base, pitch_base) in enumerate(fish_specs):
        x, y = _layout_xy(x, y, layout_family, intensity=0.88)
        center = (x + rng.uniform(-0.18, 0.18), y + rng.uniform(-0.16, 0.16), floor_z + z + rng.uniform(-0.05, 0.05))
        yaw = yaw_base + rng.uniform(-0.12, 0.12)
        pitch = pitch_base + rng.uniform(-0.035, 0.035)
        body_key = body_keys[fish_index % len(body_keys)]
        fish_geometry = _build_fish(rng, kind, center, scale, yaw, pitch, body_key)
        alt_key = body_keys[(fish_index + 1) % len(body_keys)]
        register_asset(
            name=f"Fish_{kind.replace('_fish', '').title()}_{fish_index + 1:02d}",
            geometry=fish_geometry,
            materials=[material_library[body_key], material_library[alt_key], material_library["fish_stripe"], material_library["fish_fin"], material_library["fish_fin_gold"], material_library["fish_eye"], material_library["fish_mouth"]],
            semantic_id=SEMANTIC_FISH,
            morphology=kind,
            layer="foreground" if fish_index < 3 else ("midground" if fish_index < 11 else "background"),
            base_forward=(cos(pitch) * cos(yaw), cos(pitch) * sin(yaw), sin(pitch)),
            base_up=(-sin(pitch) * cos(yaw), -sin(pitch) * sin(yaw), cos(pitch)),
        )

    # A small, render-hidden background silhouette pool extends the scene
    # beyond the 15 m label band.  These are real reef/coral/grass meshes,
    # built with the same detailed generators as the foreground assets; the
    # state sampler activates a deterministic subset when its camera can see
    # them.  IDs 1..77 remain untouched, so hiding a background candidate
    # never renumbers the original asset pool.
    background_specs = [
        # Keep the silhouettes 15 m+ from the sampled camera while allowing
        # the 28 mm square frame's downward pitch to see their upper outline.
        ("rock", "Background_RockReef_01", -4.2, 15.8, 2.35, 1.40),
        ("coral_branching", "Background_CoralOutline_01", -1.3, 17.2, 2.55, 1.35),
        ("coral_fan", "Background_CoralOutline_02", 3.1, 16.4, 2.60, 1.42),
        ("grass", "Background_SeagrassOutline_01", 3.5, 18.0, 2.10, 0.0),
        ("grass", "Background_SeagrassOutline_02", -3.4, 18.8, 2.25, 0.0),
        ("rock", "Background_RockReef_02", 1.4, 19.6, 2.10, 1.22),
    ]
    for bg_index, (kind, name, x, y, height, spread) in enumerate(background_specs):
        x_bg, y_bg = _layout_xy(x, y, layout_family, intensity=0.32)
        z_bg = benthic_z(x_bg, y_bg, clearance=0.03 if kind != "grass" else 0.025)
        if kind == "rock":
            bg_geometry = GeometryData()
            for lump in range(3):
                angle = 2.0 * pi * lump / 3.0 + rng.uniform(-0.18, 0.18)
                lump_scale = spread * rng.uniform(0.55, 0.86)
                append_rock_lump(
                    bg_geometry,
                    (
                        x_bg + cos(angle) * spread * 0.22,
                        y_bg + sin(angle) * spread * 0.22,
                        z_bg + lump_scale * 0.67,
                    ),
                    (lump_scale * 0.82, lump_scale * 0.68, lump_scale * 0.72),
                    seed_phase=seed * 0.019 + bg_index * 0.73 + lump,
                    segments=11,
                    rings=5,
                    material_index=0,
                    dark_material_index=1,
                )
            bg_materials = [material_library["rock"], material_library["rock_crevice"]]
            bg_morphology = "background_rock_reef"
            bg_semantic = SEMANTIC_ROCK
        elif kind == "coral_branching":
            bg_geometry = _build_branching_coral(rng, (x_bg, y_bg, z_bg), height, spread, bg_index + 2)
            bg_materials = coral_material_library
            bg_morphology = "background_branching"
            bg_semantic = SEMANTIC_CORAL
        elif kind == "coral_fan":
            bg_geometry = _build_table_or_fan_coral(rng, (x_bg, y_bg, z_bg), height, spread, bg_index + 2, "fan")
            bg_materials = coral_material_library
            bg_morphology = "background_fan"
            bg_semantic = SEMANTIC_CORAL
        else:
            bg_geometry = _build_grass_clump(rng, (x_bg, y_bg, z_bg), height)
            bg_materials = [material_library["grass"], material_library["grass_tip"]]
            bg_morphology = "background_seagrass"
            bg_semantic = SEMANTIC_GRASS
        register_asset(
            name=name,
            geometry=bg_geometry,
            materials=bg_materials,
            semantic_id=bg_semantic,
            morphology=bg_morphology,
            layer="background",
            active=False,
        )

    # Replace a small, named subset of the original v2 asset pool with audited
    # CC0 scans.  Keeping the object names and instance IDs stable lets the
    # existing state/label contracts observe real geometry rather than a photo
    # card or an unlabelled imported mesh.
    external_assets = apply_scanned_replacements(objects, instances, morphology_counts)
    triangle_count = mesh_triangle_count(objects)
    if triangle_count > TRIANGLE_BUDGET:
        raise ValueError(
            f"v3 evaluated triangle budget exceeded: {triangle_count} > {TRIANGLE_BUDGET}"
        )
    summary = {
        "morphology_counts": dict(sorted(morphology_counts.items())),
        "semantic_counts": {str(key): int(value) for key, value in sorted(semantic_counts.items())},
        "triangle_count": int(triangle_count),
        "mesh_budget": TRIANGLE_BUDGET,
        "asset_source": "mixed_original_programmatic_and_cc0_imported",
        "external_assets": external_assets,
        "object_count": len(objects),
        "instance_count": len(instances),
        "original_pool_object_count": 77,
        "original_pool_instance_ids": [int(i) for i in range(1, 78)],
        "background_pool_instance_ids": [int(i) for i in range(78, len(instances) + 1)],
        "background_anchor_count": len(background_specs),
        "coral_group_count": int(semantic_counts[SEMANTIC_CORAL]),
        "fish_count": int(semantic_counts[SEMANTIC_FISH]),
        "static_geometry": True,
        "seed": seed,
        "floor_z": floor_z,
        "layout_family": layout_family,
        "layout_style": layout_style,
        "layout_notes": "中心群落与岩礁脊按布局族变形；前景锚点位于通道两侧；内区 x[-16,16], y[-10,24] 保持高密度，外环扩展至 x[-40,40], y[-50,50]，并保留15m以上背景生态轮廓",
        "sand_extent": {"x": [-40.0, 40.0], "y": [-50.0, 50.0], "inner_x": [-16.0, 16.0], "inner_y": [-10.0, 24.0]},
    }
    collection["summary_asset_count"] = len(objects)
    collection["summary_triangle_count"] = int(triangle_count)
    collection["mesh_budget_limit"] = TRIANGLE_BUDGET
    return {
        "collection": collection,
        "objects": objects,
        "instances": instances,
        "summary": summary,
    }


__all__ = [
    "ASSET_SOURCE",
    "SEMANTIC_BACKGROUND",
    "SEMANTIC_CORAL",
    "SEMANTIC_FISH",
    "SEMANTIC_GRASS",
    "SEMANTIC_ROCK",
    "SEMANTIC_SAND",
    "build_ecosystem",
]
