"""Deterministic camera and geometry-state randomisation for seabed v2.

The base ecosystem is deliberately built once.  ``prepare_layout`` converts
the old world-baked meshes into local, pivot-centred assets while checking that
their evaluated world vertices do not move.  ``apply_geometry_state`` then
resets every object to that frozen base and samples a camera, a bounded set of
fish poses, and small benthic perturbations from independent seed streams.

The module is imported by Blender-side code.  It keeps Blender handles only in
an explicitly private cache attached to the ecosystem; returned plans contain
plain Python values suitable for JSON serialisation.
"""

from __future__ import annotations

import hashlib
import json
import math
from random import Random
from typing import Any, Iterable, Mapping, Sequence

try:  # pragma: no cover - Blender supplies these modules at runtime.
    import bpy  # type: ignore
    from mathutils import Matrix, Vector  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - compile/static imports only.
    bpy = None  # type: ignore
    Matrix = Vector = None  # type: ignore

from . import contracts as _contracts

try:  # Keep contract/default inspection importable outside Blender.
    from .assets.geometry import sandbed_height
except ModuleNotFoundError:  # pragma: no cover - runtime functions require Blender.
    sandbed_height = None  # type: ignore


DEFAULT_CAMERA_SAMPLING: dict[str, Any] = {
    "lens_mm": 28.0,
    "height_m": [1.8, 2.3],
    "x_range": [-0.45, 0.45],
    "y_range": [-3.8, -3.1],
    "look_y_range": [0.0, 0.7],
    "look_z_range": [-0.6, -0.3],
    "roll_deg": [-3.0, 3.0],
}

DEFAULT_OBJECT_SAMPLING: dict[str, Any] = {
    "fish_scale": [0.75, 1.25],
    "fish_pitch_deg": [-15.0, 15.0],
    "fish_roll_deg": [-8.0, 8.0],
    "foreground_fish": [0, 2],
    "midground_fish": [4, 8],
    "background_fish": [3, 6],
    "small_scale": [0.9, 1.1],
    "small_yaw_deg": [-20.0, 20.0],
}

_SEMANTIC_SAND = 1
_SEMANTIC_ROCK = 2
_SEMANTIC_CORAL = 3
_SEMANTIC_GRASS = 4
_SEMANTIC_FISH = 5
_ORIGINAL_POOL_MAX = 77
_EPSILON = 1.0e-7


def _require_blender() -> None:
    if bpy is None or Matrix is None or Vector is None or sandbed_height is None:
        raise RuntimeError("seabed geometry randomisation requires Blender's bpy/mathutils modules")


def _prop(obj: Any, key: str, default: Any = None) -> Any:
    getter = getattr(obj, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    try:
        return obj[key]
    except (KeyError, TypeError, IndexError):
        return default


def _as_float(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return number if math.isfinite(number) else float(fallback)


def _range(mapping: Mapping[str, Any], key: str, fallback: Sequence[float]) -> tuple[float, float]:
    value = mapping.get(key, fallback)
    try:
        lo, hi = float(value[0]), float(value[1])
    except (TypeError, ValueError, IndexError):
        lo, hi = float(fallback[0]), float(fallback[1])
    if not math.isfinite(lo) or not math.isfinite(hi):
        lo, hi = float(fallback[0]), float(fallback[1])
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _stable_seed(root_seed: int, *parts: Any) -> int:
    """Call the parent contract's stable seed helper, with a safe fallback."""

    derive = getattr(_contracts, "derive_seed", None)
    if callable(derive):
        return int(derive(int(root_seed), *parts))
    payload = json.dumps(
        {"root": int(root_seed), "parts": [str(part) if not isinstance(part, (int, float, bool)) else part for part in parts]},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def _vec3(value: Any, fallback: Sequence[float]) -> Vector:
    try:
        candidate = Vector((float(value[0]), float(value[1]), float(value[2])))
        if candidate.length > _EPSILON and all(math.isfinite(float(component)) for component in candidate):
            return candidate
    except (TypeError, ValueError, IndexError):
        pass
    return Vector((float(fallback[0]), float(fallback[1]), float(fallback[2])))


def _unit(value: Vector, fallback: Vector) -> Vector:
    vector = value.copy()
    if vector.length <= _EPSILON:
        return fallback.copy()
    vector.normalize()
    return vector


def _orthonormal(forward: Vector, up: Vector) -> tuple[Vector, Vector, Vector]:
    """Return right, up, forward columns for a right-handed local frame."""

    f = _unit(forward, Vector((0.0, 0.0, 1.0)))
    u = up - f * up.dot(f)
    if u.length <= _EPSILON:
        candidate = Vector((0.0, 1.0, 0.0)) if abs(f.z) > 0.75 else Vector((0.0, 0.0, 1.0))
        u = candidate - f * candidate.dot(f)
    u.normalize()
    r = u.cross(f)
    if r.length <= _EPSILON:
        r = Vector((1.0, 0.0, 0.0))
    else:
        r.normalize()
    u = f.cross(r)
    u.normalize()
    return r, u, f


def _basis_matrix(pivot: Vector, right: Vector, up: Vector, forward: Vector) -> Matrix:
    matrix = Matrix.Identity(4)
    for row in range(3):
        matrix[row][0] = right[row]
        matrix[row][1] = up[row]
        matrix[row][2] = forward[row]
    matrix.translation = pivot
    return matrix


def _matrix_rows(matrix: Matrix) -> list[list[float]]:
    return [[float(matrix[row][column]) for column in range(4)] for row in range(4)]


def _tuple3(value: Vector) -> list[float]:
    return [float(value.x), float(value.y), float(value.z)]


def _semantic(obj: Any) -> int:
    try:
        return int(_prop(obj, "semantic_id", -1))
    except (TypeError, ValueError):
        return -1


def _instance_id(obj: Any) -> int:
    try:
        return int(_prop(obj, "instance_id", 0))
    except (TypeError, ValueError):
        return 0


def _asset_uid(obj: Any, instance_id: int) -> str:
    uid = _prop(obj, "asset_uid", None)
    return str(uid) if uid else f"seabed-v2:asset-{int(instance_id):04d}"


def _morphology(obj: Any) -> str:
    return str(_prop(obj, "asset_morphology", "unknown"))


def _layer(obj: Any, y: float, instance_id: int) -> str:
    stored = _prop(obj, "asset_layer", None)
    if stored:
        return str(stored)
    if instance_id >= 78 or y >= 18.0:
        return "background"
    if y < 1.0:
        return "foreground"
    if y < 8.0:
        return "midground"
    return "background"


def _active_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "off", "none", "null"}
    return bool(value)


def _world_vertices(obj: Any, matrix: Matrix) -> list[Vector]:
    return [matrix @ vertex.co for vertex in obj.data.vertices]


def _aabb_from_matrix(obj: Any, matrix: Matrix) -> tuple[Vector, Vector]:
    points = [matrix @ Vector(corner) for corner in obj.bound_box]
    return (
        Vector((min(point.x for point in points), min(point.y for point in points), min(point.z for point in points))),
        Vector((max(point.x for point in points), max(point.y for point in points), max(point.z for point in points))),
    )


def _aabb_overlap(first: tuple[Vector, Vector], second: tuple[Vector, Vector], margin: float = 0.0) -> bool:
    first_lo, first_hi = first
    second_lo, second_hi = second
    return all(
        float(first_lo[axis]) <= float(second_hi[axis]) + margin
        and float(second_lo[axis]) <= float(first_hi[axis]) + margin
        for axis in range(3)
    )


def _obb_from_matrix(obj: Any, matrix: Matrix) -> tuple[Vector, tuple[Vector, Vector, Vector], Vector]:
    """Return an oriented box proxy from the post-pivot local mesh bounds."""

    corners = [Vector(corner) for corner in obj.bound_box]
    local_lo = Vector((min(point.x for point in corners), min(point.y for point in corners), min(point.z for point in corners)))
    local_hi = Vector((max(point.x for point in corners), max(point.y for point in corners), max(point.z for point in corners)))
    local_center = (local_lo + local_hi) * 0.5
    local_half = (local_hi - local_lo) * 0.5
    rotation = matrix.to_3x3()
    axes: list[Vector] = []
    half = []
    for axis in range(3):
        column = Vector((rotation[0][axis], rotation[1][axis], rotation[2][axis]))
        length = float(column.length)
        if length <= _EPSILON:
            column = Vector((1.0, 0.0, 0.0)) if axis == 0 else (Vector((0.0, 1.0, 0.0)) if axis == 1 else Vector((0.0, 0.0, 1.0)))
            length = 1.0
        else:
            column.normalize()
        axes.append(column)
        half.append(float(local_half[axis]) * length)
    return matrix @ local_center, (axes[0], axes[1], axes[2]), Vector((half[0], half[1], half[2]))


def _obb_sphere_radius(obb: tuple[Vector, tuple[Vector, Vector, Vector], Vector]) -> float:
    return float(obb[2].length)


def _obb_separated(first: tuple[Vector, tuple[Vector, Vector, Vector], Vector], second: tuple[Vector, tuple[Vector, Vector, Vector], Vector], margin: float = 0.0) -> bool:
    """Return true when two oriented boxes are separated by SAT.

    The 15 separating-axis tests are a tighter geometry proxy than comparing
    world AABBs after every fish heading change. The requested safety margin is
    added to every projection, so the check never weakens the clearance rule.
    """

    center_a, axes_a, half_a = first
    center_b, axes_b, half_b = second
    rotation = [[abs(float(axes_a[i].dot(axes_b[j]))) + 1.0e-8 for j in range(3)] for i in range(3)]
    delta = center_b - center_a
    translation_a = [float(delta.dot(axes_a[i])) for i in range(3)]
    translation_b = [float(delta.dot(axes_b[j])) for j in range(3)]
    for i in range(3):
        radius_a = float(half_a[i])
        radius_b = sum(float(half_b[j]) * rotation[i][j] for j in range(3))
        if abs(translation_a[i]) > radius_a + radius_b + margin:
            return True
    for j in range(3):
        radius_a = sum(float(half_a[i]) * rotation[i][j] for i in range(3))
        radius_b = float(half_b[j])
        if abs(translation_b[j]) > radius_a + radius_b + margin:
            return True
    for i in range(3):
        for j in range(3):
            cross_length = math.sqrt(max(0.0, 1.0 - float(axes_a[i].dot(axes_b[j])) ** 2))
            if cross_length <= 1.0e-7:
                continue
            radius_a = float(half_a[(i + 1) % 3]) * rotation[(i + 2) % 3][j] + float(half_a[(i + 2) % 3]) * rotation[(i + 1) % 3][j]
            radius_b = float(half_b[(j + 1) % 3]) * rotation[i][(j + 2) % 3] + float(half_b[(j + 2) % 3]) * rotation[i][(j + 1) % 3]
            projection = abs(translation_a[(i + 2) % 3] * float(axes_a[(i + 1) % 3].dot(axes_b[j])) - translation_a[(i + 1) % 3] * float(axes_a[(i + 2) % 3].dot(axes_b[j])))
            if projection > radius_a + radius_b + margin * cross_length:
                return True
    return False


def _proxy_overlap(first_obj: Any, first_matrix: Matrix, second_obj: Any, second_matrix: Matrix, margin: float) -> bool:
    """Use a sphere broad phase followed by a tight oriented-box proxy."""

    first = _obb_from_matrix(first_obj, first_matrix)
    second = _obb_from_matrix(second_obj, second_matrix)
    centre_distance = float((second[0] - first[0]).length)
    if centre_distance > _obb_sphere_radius(first) + _obb_sphere_radius(second) + margin:
        return False
    return not _obb_separated(first, second, margin)


def _point_aabb_distance(point: Vector, bounds: tuple[Vector, Vector]) -> float:
    lo, hi = bounds
    distance_sq = 0.0
    for axis in range(3):
        value = float(point[axis])
        if value < lo[axis]:
            distance_sq += (float(lo[axis]) - value) ** 2
        elif value > hi[axis]:
            distance_sq += (value - float(hi[axis])) ** 2
    return math.sqrt(distance_sq)


def _record_output(record: Mapping[str, Any], active: bool, actual: Matrix) -> dict[str, Any]:
    row = {
        "instance_id": int(record["instance_id"]),
        "semantic_id": int(record["semantic_id"]),
        "name": str(record["name"]),
        "morphology": str(record["morphology"]),
        "active": bool(active),
        "base_matrix": _matrix_rows(record["base_matrix"]),
        "actual_matrix": _matrix_rows(actual),
        "asset_uid": str(record["asset_uid"]),
        "pose_space": "fish" if int(record["semantic_id"]) == _SEMANTIC_FISH else "benthic",
    }
    if "pivot" in record:
        row["pivot"] = _tuple3(record["pivot"])
    if "forward" in record:
        row["forward"] = _tuple3(record["forward"])
    if "up" in record:
        row["up"] = _tuple3(record["up"])
        rotation = actual.to_3x3()
        row["actual_forward"] = _tuple3(rotation @ Vector((0.0, 0.0, 1.0)))
        row["actual_up"] = _tuple3(rotation @ Vector((0.0, 1.0, 0.0)))
    if "layer" in record:
        row["layer"] = str(record["layer"])
    return row


def _json_pose_value(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_json_pose_value(item) for item in value]
    return str(value)


def _prepare_cache(scene: Any, ecosystem: Mapping[str, Any]) -> dict[str, Any] | None:
    private = ecosystem.get("_layout_private") if isinstance(ecosystem, dict) else None
    if isinstance(private, dict) and private.get("prepared_scene") is scene and private.get("prepared"):
        return private
    return None


def prepare_layout(scene: Any, ecosystem: dict[str, Any]) -> dict[str, Any]:
    """Recenter world-baked mesh assets and return a JSON-safe base layout.

    Each mesh is converted exactly once.  Fish use their stored generated
    heading/up vectors as the local basis; benthic assets use the stable world
    axes.  The object's origin is moved to a geometry-derived pivot, and every
    vertex is checked before/after the conversion with a 1e-5 m limit.
    """

    _require_blender()
    if scene is None or not isinstance(ecosystem, dict):
        raise ValueError("prepare_layout requires a scene and ecosystem dictionary")
    cached = _prepare_cache(scene, ecosystem)
    if cached is not None:
        return json.loads(json.dumps(cached["public_layout"], allow_nan=False))

    objects = list(ecosystem.get("objects") or [])
    instances_by_name = {str(row.get("name")): row for row in ecosystem.get("instances") or [] if isinstance(row, dict)}
    records: list[dict[str, Any]] = []
    max_error = 0.0
    seen_ids: set[int] = set()
    for obj in objects:
        if getattr(obj, "type", None) != "MESH":
            raise TypeError(f"ecosystem asset {getattr(obj, 'name', '<unnamed>')} is not a MESH")
        instance_id = _instance_id(obj)
        semantic_id = _semantic(obj)
        if instance_id <= 0 or semantic_id <= 0:
            raise ValueError(f"asset {getattr(obj, 'name', '<unnamed>')} lacks valid semantic/instance IDs")
        if instance_id in seen_ids:
            raise ValueError(f"duplicate ecosystem instance_id {instance_id}")
        seen_ids.add(instance_id)
        original_matrix = obj.matrix_world.copy()
        before = _world_vertices(obj, original_matrix)
        if not before:
            raise ValueError(f"asset {obj.name} has no vertices")
        min_z = min(float(point.z) for point in before)
        if semantic_id in (_SEMANTIC_FISH,):
            pivot = sum(before, Vector((0.0, 0.0, 0.0))) / len(before)
        else:
            # Grounded assets rotate around their footprint centre at the
            # seabed contact plane; the sandbed uses the same deterministic
            # pivot but remains semantically an environment target.
            pivot = Vector((sum(float(point.x) for point in before) / len(before), sum(float(point.y) for point in before) / len(before), min_z))

        stored_forward = _vec3(_prop(obj, "asset_forward", (0.0, 0.0, 1.0)), (0.0, 0.0, 1.0))
        stored_up = _vec3(_prop(obj, "asset_up", (0.0, 1.0, 0.0)), (0.0, 1.0, 0.0))
        if semantic_id == _SEMANTIC_FISH:
            right, up, forward = _orthonormal(stored_forward, stored_up)
        else:
            # Benthic meshes are generated in world axes. Keep that basis
            # stable while still exposing forward/up metadata for the plan.
            right, up, forward = _orthonormal(Vector((0.0, 0.0, 1.0)), Vector((0.0, 1.0, 0.0)))

        new_local: list[Vector] = []
        for point in before:
            delta = point - pivot
            new_local.append(Vector((delta.dot(right), delta.dot(up), delta.dot(forward))))
        for vertex, local in zip(obj.data.vertices, new_local):
            vertex.co = local
        base_matrix = _basis_matrix(pivot, right, up, forward)
        obj.matrix_world = base_matrix
        obj.data.update()
        after = _world_vertices(obj, base_matrix)
        error = max((before[index] - after[index]).length for index in range(len(before)))
        max_error = max(max_error, float(error))
        if error > 1.0e-5:
            raise RuntimeError(f"local pivot conversion moved {obj.name} by {float(error):.9g} m")

        row = instances_by_name.get(str(obj.name), {})
        active = _active_value(_prop(obj, "asset_pool_active", row.get("pool_active", not obj.hide_render)))
        y_pivot = float(pivot.y)
        layer = _layer(obj, y_pivot, instance_id)
        record = {
            "object": obj,
            "name": str(obj.name),
            "instance_id": instance_id,
            "semantic_id": semantic_id,
            "morphology": _morphology(obj),
            "asset_uid": _asset_uid(obj, instance_id),
            "layer": layer,
            "pool_active": active,
            "base_matrix": base_matrix.copy(),
            "pivot": pivot.copy(),
            "forward": forward.copy(),
            "up": up.copy(),
            "local_vertices": tuple(vertex.co.copy() for vertex in obj.data.vertices),
            "original_world_vertex_count": len(before),
            "original_world_error_m": float(error),
            "grounded": semantic_id in (_SEMANTIC_SAND, _SEMANTIC_ROCK, _SEMANTIC_CORAL, _SEMANTIC_GRASS),
        }
        records.append(record)
        obj["asset_uid"] = record["asset_uid"]
        obj["asset_layer"] = layer
        obj["local_pivot"] = _tuple3(pivot)
        obj["local_forward"] = _tuple3(forward)
        obj["local_up"] = _tuple3(up)
        obj["layout_prepared"] = True

    # Force Blender's object bounds to observe the newly assigned local mesh
    # coordinates before proxy checks use ``Object.bound_box``. This update is
    # read/evaluate-only and does not start a render.
    try:
        view_layers = getattr(scene, "view_layers", ())
        if view_layers and callable(getattr(view_layers[0], "update", None)):
            view_layers[0].update()
    except (AttributeError, RuntimeError, TypeError):
        pass
    records.sort(key=lambda item: int(item["instance_id"]))
    if sorted(seen_ids)[: len(range(1, min(77, len(seen_ids)) + 1))] != list(range(1, min(77, len(seen_ids)) + 1)):
        raise ValueError("ecosystem instance IDs must preserve the original pool order")
    summary = ecosystem.get("summary") or {}
    layout_family = str(summary.get("layout_family", _prop(ecosystem.get("collection"), "layout_family", "unknown")))
    seed_value = int(summary.get("seed", ecosystem.get("seed", _prop(ecosystem.get("collection"), "seed", 0))))
    public_instances = [
        _record_output(record, bool(record["pool_active"]), record["base_matrix"])
        for record in records
    ]
    public_layout = {
        "layout_family": layout_family,
        "seed": seed_value,
        "asset_count": len(records),
        "original_pool_instance_ids": [int(record["instance_id"]) for record in records if int(record["instance_id"]) <= _ORIGINAL_POOL_MAX],
        "background_pool_instance_ids": [int(record["instance_id"]) for record in records if int(record["instance_id"]) > _ORIGINAL_POOL_MAX],
        "world_vertex_max_error_m": float(max_error),
        "instances": public_instances,
    }
    private = {
        "prepared": True,
        "prepared_scene": scene,
        "records": records,
        "public_layout": public_layout,
        "max_world_vertex_error_m": float(max_error),
    }
    ecosystem["_layout_private"] = private
    return json.loads(json.dumps(public_layout, allow_nan=False))


def _camera_pose(camera: Any, eye: Vector, target: Vector, lens_mm: float, roll_deg: float) -> None:
    # Construct the complete world matrix in one step. Assigning only
    # location/rotation_euler leaves Blender's cached matrix_world stale until
    # a dependency-graph update; repeated same-state calls could then choose a
    # different camera basis (fallback versus native) by a few millimetres.
    rotation = (target - eye).to_track_quat("-Z", "Y").to_matrix().to_4x4()
    if abs(roll_deg) > _EPSILON:
        # Post-multiply so the roll is around the camera's local forward axis.
        rotation = rotation @ Matrix.Rotation(math.radians(float(roll_deg)), 4, "Z")
    world_matrix = rotation.copy()
    world_matrix.translation = eye
    camera.matrix_world = world_matrix
    if getattr(camera, "data", None) is not None:
        camera.data.lens = float(lens_mm)
        camera.data.sensor_fit = "HORIZONTAL"
        camera.data.sensor_width = 36.0
        camera.data.clip_start = 0.01
        camera.data.clip_end = 200.0
        if getattr(camera.data, "dof", None) is not None:
            camera.data.dof.use_dof = False


def _camera_target_from_xy(
    eye: Vector,
    target_x: float,
    target_y: float,
    camera_sampling: Mapping[str, Any],
    rng: Random,
    floor_z: float,
    seed_phase: float,
) -> tuple[Vector, float]:
    """Sample target Z and return the actual downward pitch in degrees.

    ``pitch_down_deg`` is an optional replacement for the old target-Z offset.
    When present, the target XY values remain untouched and target Z is solved
    from the horizontal eye-to-target distance, so every candidate has a
    controlled downward angle. Without it, the historical look-Z range remains
    the exact fallback for old configs.
    """

    horizontal_distance = math.hypot(float(target_x) - float(eye.x), float(target_y) - float(eye.y))
    if horizontal_distance <= _EPSILON:
        raise ValueError("camera target XY must differ from sampled eye XY")
    pitch_range = camera_sampling.get("pitch_down_deg")
    if pitch_range is not None:
        pitch_lo, pitch_hi = _range(camera_sampling, "pitch_down_deg", (30.5, 31.5))
        pitch_down = rng.uniform(pitch_lo, pitch_hi)
        target_z = float(eye.z) - math.tan(math.radians(float(pitch_down))) * horizontal_distance
        actual = math.degrees(math.atan2(float(eye.z) - target_z, horizontal_distance))
        return Vector((float(target_x), float(target_y), target_z)), float(actual)
    target_ground = sandbed_height(float(target_x), float(target_y), floor_z=floor_z, seed_phase=seed_phase)
    target_z = target_ground + rng.uniform(look_z_lo, look_z_hi)
    actual = math.degrees(math.atan2(float(eye.z) - target_z, horizontal_distance))
    return Vector((float(target_x), float(target_y), float(target_z))), float(actual)


def _camera_clearance(eye: Vector, records: Iterable[Mapping[str, Any]], matrices: Mapping[str, Matrix], active: Mapping[str, bool], margin: float = 0.12) -> dict[str, Any]:
    minimum = float("inf")
    closest = None
    tested = 0
    for record in records:
        name = str(record["name"])
        if not active.get(name, False) or int(record["semantic_id"]) == _SEMANTIC_SAND:
            continue
        bounds = _aabb_from_matrix(record["object"], matrices[name])
        distance = _point_aabb_distance(eye, bounds)
        tested += 1
        if distance < minimum:
            minimum = distance
            closest = name
        if distance < margin:
            return {
                "passed": False,
                "method": "world_aabb_proxy_with_margin",
                "margin_m": float(margin),
                "min_distance_m": float(distance),
                "closest_object": closest,
                "tested_objects": tested,
            }
    if not math.isfinite(minimum):
        minimum = 1.0e9
    return {
        "passed": True,
        "method": "world_aabb_proxy_with_margin",
        "margin_m": float(margin),
        "min_distance_m": float(minimum),
        "closest_object": closest,
        "tested_objects": tested,
    }


def _terrain_gap(record: Mapping[str, Any], matrix: Matrix, floor_z: float, seed_phase: float) -> float:
    minimum = float("inf")
    for point in _world_vertices(record["object"], matrix):
        terrain = sandbed_height(float(point.x), float(point.y), floor_z=floor_z, seed_phase=seed_phase)
        minimum = min(minimum, float(point.z) - terrain)
    return float(minimum if math.isfinite(minimum) else 0.0)


def _surface_gap(record: Mapping[str, Any], matrix: Matrix, surface_z: float) -> float:
    """Return the minimum water-surface clearance of a transformed asset."""

    maximum = float("inf")
    for point in _world_vertices(record["object"], matrix):
        maximum = min(maximum, float(surface_z) - float(point.z))
    return float(maximum if math.isfinite(maximum) else 0.0)


def _ground_matrix(record: Mapping[str, Any], matrix: Matrix, floor_z: float, seed_phase: float, clearance: float = 0.018) -> tuple[Matrix, float]:
    matrix = matrix.copy()
    gap = _terrain_gap(record, matrix, floor_z, seed_phase)
    shift = max(0.0, float(clearance) - gap)
    if shift:
        matrix.translation.z += shift
    return matrix, float(shift)


def _relative_pose(
    base: Matrix,
    center: Vector,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    scale: float = 1.0,
    *,
    pose_space: str = "benthic",
) -> Matrix:
    """Build a pivot-centred pose in the asset's canonical local frame.

    Benthic meshes use local ``Z`` as the up axis, so their small yaw remains
    a ground-plane rotation. Fish are canonicalised with ``X=right``,
    ``Y=up``, ``Z=forward``; their heading therefore rotates around local Y,
    pitch around local X, and roll around local Z.
    """

    local_center = base.inverted_safe() @ center
    translation = Matrix.Translation(local_center)
    if pose_space == "fish":
        rotation = Matrix.Rotation(math.radians(float(yaw_deg)), 4, "Y")
        if abs(pitch_deg) > _EPSILON:
            rotation = rotation @ Matrix.Rotation(math.radians(float(pitch_deg)), 4, "X")
        if abs(roll_deg) > _EPSILON:
            rotation = rotation @ Matrix.Rotation(math.radians(float(roll_deg)), 4, "Z")
    elif pose_space == "benthic":
        rotation = Matrix.Rotation(math.radians(float(yaw_deg)), 4, "Z")
        if abs(pitch_deg) > _EPSILON:
            rotation = rotation @ Matrix.Rotation(math.radians(float(pitch_deg)), 4, "X")
        if abs(roll_deg) > _EPSILON:
            rotation = rotation @ Matrix.Rotation(math.radians(float(roll_deg)), 4, "Y")
    else:
        raise ValueError(f"unknown pose_space {pose_space!r}")
    scaling = Matrix.Diagonal((float(scale), float(scale), float(scale), 1.0))
    return base @ translation @ rotation @ scaling


def _camera_basis(eye: Vector, target: Vector, camera: Any | None = None) -> tuple[Vector, Vector, Vector]:
    if camera is not None and getattr(camera, "matrix_world", None) is not None:
        try:
            if (camera.matrix_world.translation - eye).length > 1.0e-4:
                raise RuntimeError("camera matrix has not updated to sampled eye")
            rotation = camera.matrix_world.to_3x3()
            right = _unit(rotation @ Vector((1.0, 0.0, 0.0)), Vector((1.0, 0.0, 0.0)))
            up = _unit(rotation @ Vector((0.0, 1.0, 0.0)), Vector((0.0, 0.0, 1.0)))
            forward = _unit(-(rotation @ Vector((0.0, 0.0, 1.0))), _unit(target - eye, Vector((0.0, 1.0, -0.4))))
            return right, up, forward
        except (AttributeError, TypeError, RuntimeError):
            raise
    forward = _unit(target - eye, Vector((0.0, 1.0, -0.4)))
    world_up = Vector((0.0, 0.0, 1.0))
    right = forward.cross(world_up)
    if right.length <= _EPSILON:
        right = Vector((1.0, 0.0, 0.0))
    else:
        right.normalize()
    up = right.cross(forward)
    up.normalize()
    return right, up, forward


def _camera_frustum_tangents(camera: Any, cfg: Mapping[str, Any]) -> tuple[float, float]:
    """Return horizontal/vertical half-angle tangents for the active camera."""

    data = getattr(camera, "data", None)
    lens = _as_float(getattr(data, "lens", 28.0), 28.0) if data is not None else 28.0
    sensor_width = _as_float(getattr(data, "sensor_width", 36.0), 36.0) if data is not None else 36.0
    resolution = (cfg.get("render") or {}).get("resolution", [1, 1])
    try:
        aspect = float(resolution[1]) / max(float(resolution[0]), 1.0)
    except (TypeError, ValueError, IndexError):
        aspect = 1.0
    if not math.isfinite(aspect) or aspect <= 0.0:
        aspect = 1.0
    return float(sensor_width / max(2.0 * lens, _EPSILON)), float(sensor_width * aspect / max(2.0 * lens, _EPSILON))


def _projection_proxy(
    record: Mapping[str, Any],
    matrix: Matrix,
    eye: Vector,
    right: Vector,
    screen_up: Vector,
    forward: Vector,
    tan_x: float,
    tan_y: float,
    margin: float = 0.96,
) -> tuple[bool, dict[str, Any]]:
    """Check actual fish vertices against the camera frustum.

    The check accepts partial edge cropping but requires at least one full
    body/fin proxy point in front of the image and keeps the object centre away
    from the extreme boundary. This is stronger than checking only a centre
    point and remains cheap enough for bounded candidate retries.
    """

    points = _world_vertices(record["object"], matrix)
    if not points:
        return False, {"reason": "empty_geometry"}
    ratios: list[tuple[float, float]] = []
    for point in points:
        delta = point - eye
        depth = float(delta.dot(forward))
        if depth <= _EPSILON:
            continue
        ratios.append((float(delta.dot(right)) / depth, float(delta.dot(screen_up)) / depth))
    if not ratios:
        return False, {"reason": "behind_camera"}
    min_u = min(value[0] for value in ratios)
    max_u = max(value[0] for value in ratios)
    min_v = min(value[1] for value in ratios)
    max_v = max(value[1] for value in ratios)
    visible = max_u >= -tan_x and min_u <= tan_x and max_v >= -tan_y and min_v <= tan_y
    centre = matrix.translation - eye
    centre_depth = float(centre.dot(forward))
    centre_u = float(centre.dot(right)) / centre_depth if centre_depth > _EPSILON else float("inf")
    centre_v = float(centre.dot(screen_up)) / centre_depth if centre_depth > _EPSILON else float("inf")
    centre_inside = abs(centre_u) <= tan_x * margin and abs(centre_v) <= tan_y * margin
    detail = {
        "reason": None if visible and centre_inside else ("frustum_overlap" if visible else "outside_frustum"),
        "method": "all_fish_vertices_projected_to_camera_frustum",
        "visible_overlap": bool(visible),
        "centre_inside": bool(centre_inside),
        "centre_depth_m": float(centre_depth),
        "centre_u_over_depth": float(centre_u),
        "centre_v_over_depth": float(centre_v),
        "u_bounds": [float(min_u), float(max_u)],
        "v_bounds": [float(min_v), float(max_v)],
        "frustum_tangent": [float(tan_x), float(tan_y)],
    }
    return bool(visible and centre_inside), detail


def _safe_frustum_v_interval(
    eye: Vector,
    base: Vector,
    screen_up: Vector,
    d: float,
    tan_y: float,
    v_limit: float,
    terrain_extent: float,
    surface_extent: float,
    surface_z: float,
    floor_z: float,
    seed_phase: float,
    clearance: float = 0.075,
) -> tuple[float, float] | None:
    """Find a safe vertical ``v`` interval on one fixed camera-depth plane.

    ``base`` already contains the eye, forward-depth, and horizontal ``u``
    terms. Varying ``v`` therefore moves only along the camera image plane at
    constant nominal depth. The scan keeps the fish's lower/upper vertical
    extents above the seeded terrain and below the water surface, while the
    final candidate checks still use the exact transformed vertices.
    """

    upper = max(0.05, min(float(v_limit), 0.94))
    lower = -upper
    sample_count = 97
    samples = [lower + (upper - lower) * index / (sample_count - 1) for index in range(sample_count)]
    valid: list[bool] = []
    for value in samples:
        point = base + screen_up * (float(value) * float(d) * float(tan_y))
        terrain = sandbed_height(float(point.x), float(point.y), floor_z=floor_z, seed_phase=seed_phase)
        terrain_clearance = float(point.z) - float(terrain) - float(terrain_extent) - float(clearance)
        surface_clearance = float(surface_z) - float(point.z) - float(surface_extent) - float(clearance)
        valid.append(terrain_clearance >= 0.0 and surface_clearance >= 0.0)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_valid in enumerate(valid + [False]):
        if is_valid and start is None:
            start = index
        elif not is_valid and start is not None:
            runs.append((start, index - 1))
            start = None
    if not runs:
        return None
    first, last = max(runs, key=lambda pair: pair[1] - pair[0])
    if last - first < 2:
        return None
    step = (upper - lower) / (sample_count - 1)
    interval = (samples[first] + step * 0.05, samples[last] - step * 0.05)
    if interval[1] - interval[0] <= 0.01:
        return None
    return float(interval[0]), float(interval[1])


def _count_inclusive(rng: Random, value: Any, fallback: Sequence[int], forced_cycle: int | None = None) -> int:
    lo, hi = _range(value if isinstance(value, Mapping) else {"v": value}, "v", fallback)
    lo_i, hi_i = int(round(lo)), int(round(hi))
    lo_i, hi_i = min(lo_i, hi_i), max(lo_i, hi_i)
    if forced_cycle is not None and hi_i >= lo_i:
        return lo_i + int(forced_cycle) % (hi_i - lo_i + 1)
    return rng.randint(lo_i, hi_i)


def _select_fish(records: list[dict[str, Any]], counts: tuple[int, int, int], rng: Random) -> list[dict[str, Any]]:
    fishes = [record for record in records if int(record["semantic_id"]) == _SEMANTIC_FISH]
    groups = {
        "foreground": [record for record in fishes if record.get("layer") == "foreground"],
        "midground": [record for record in fishes if record.get("layer") == "midground"],
        "background": [record for record in fishes if record.get("layer") == "background"],
    }
    selection: list[dict[str, Any]] = []
    for layer, count in zip(("foreground", "midground", "background"), counts):
        if count < 0 or count > len(groups[layer]):
            raise ValueError(
                f"CANDIDATE_FISH_POOL: {layer} fish pool has {len(groups[layer])} assets but needs {count}"
            )
        selection.extend(rng.sample(groups[layer], count))
    morphologies = {str(record["morphology"]) for record in selection}
    if len(morphologies) < 2:
        alternatives: list[dict[str, Any]] = []
        for index in range(len(selection) - 1, -1, -1):
            layer = str(selection[index].get("layer", "midground"))
            candidates = [
                record for record in groups.get(layer, [])
                if str(record["morphology"]) not in morphologies and record not in selection
            ]
            if candidates:
                alternatives = candidates
                selection[index] = candidates[0]
                break
        if not alternatives:
            alternatives = [record for record in fishes if str(record["morphology"]) not in morphologies and record not in selection]
        if not alternatives or not selection:
            raise ValueError("CANDIDATE_FISH_MORPHOLOGY: selected fish do not cover two morphologies")
    return selection


def _sample_fish_pose(rng: Random, object_sampling: Mapping[str, Any], yaw_center: float | None = None) -> dict[str, Any]:
    scale_lo, scale_hi = _range(object_sampling, "fish_scale", DEFAULT_OBJECT_SAMPLING["fish_scale"])
    pitch_lo, pitch_hi = _range(object_sampling, "fish_pitch_deg", DEFAULT_OBJECT_SAMPLING["fish_pitch_deg"])
    roll_lo, roll_hi = _range(object_sampling, "fish_roll_deg", DEFAULT_OBJECT_SAMPLING["fish_roll_deg"])
    if yaw_center is None:
        yaw = rng.uniform(-180.0, 180.0)
    else:
        # Keep one selected fish approximately facing the camera so a state is
        # never composed entirely of edge-on silhouettes. Other fish retain a
        # broad heading distribution for natural variation.
        yaw = rng.uniform(float(yaw_center) - 35.0, float(yaw_center) + 35.0)
    pitch = rng.uniform(pitch_lo, pitch_hi)
    roll = rng.uniform(roll_lo, roll_hi)
    scale = rng.uniform(scale_lo, scale_hi)
    return {"yaw_deg": float(yaw), "pitch_deg": float(pitch), "roll_deg": float(roll), "scale": float(scale)}


def _fish_matrix_from_pose(record: Mapping[str, Any], center: Vector, pose: Mapping[str, Any]) -> Matrix:
    return _relative_pose(
        record["base_matrix"],
        center,
        yaw_deg=float(pose["yaw_deg"]),
        pitch_deg=float(pose["pitch_deg"]),
        roll_deg=float(pose["roll_deg"]),
        scale=float(pose["scale"]),
        pose_space="fish",
    )


def _fish_vertical_extents(record: Mapping[str, Any], matrix: Matrix) -> tuple[float, float]:
    """Return world-Z lower/upper extents for one sampled fish pose."""

    centre_z = float(matrix.translation.z)
    minimum = float("inf")
    maximum = float("-inf")
    for point in _world_vertices(record["object"], matrix):
        minimum = min(minimum, float(point.z) - centre_z)
        maximum = max(maximum, float(point.z) - centre_z)
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        return 0.45, 0.45
    return max(0.0, -minimum), max(0.0, maximum)


def _fish_candidate_ok(
    record: Mapping[str, Any],
    matrix: Matrix,
    current: Mapping[str, Matrix],
    active: Mapping[str, bool],
    records: Iterable[Mapping[str, Any]],
    floor_z: float,
    seed_phase: float,
    margin: float = 0.045,
) -> tuple[bool, dict[str, Any]]:
    gap = _terrain_gap(record, matrix, floor_z, seed_phase)
    if gap < 0.075:
        return False, {"reason": "terrain_clearance", "min_gap_m": float(gap)}
    checked_benthic = 0
    for other in records:
        name = str(other["name"])
        if name == str(record["name"]) or not active.get(name, False):
            continue
        semantic = int(other["semantic_id"])
        if semantic == _SEMANTIC_SAND:
            continue
        checked_benthic += 1
        if _proxy_overlap(record["object"], matrix, other["object"], current[name], margin):
            return False, {
                "reason": "object_proxy_collision",
                "other": name,
                "method": "body_fin_oriented_bbox_15_axis_proxy",
                "margin_m": float(margin),
                "checked_objects": checked_benthic,
            }
    return True, {
        "reason": None,
        "method": "body_fin_oriented_bbox_15_axis_proxy_plus_terrain_vertices",
        "terrain_min_gap_m": float(gap),
        "checked_objects": checked_benthic,
    }


def apply_geometry_state(scene: Any, ecosystem: dict[str, Any], camera: Any, cfg: Mapping[str, Any], state_index: int, attempt_index: int = 0) -> dict[str, Any]:
    """Reset and apply one deterministic geometry/camera state.

    ``camera`` must be created by the caller.  Every subsystem derives a seed
    directly from root/layout/state/attempt; no mutable RNG is shared with a
    prior sample.  A failed camera or fish candidate raises an identifiable
    ``ValueError`` after bounded retries, leaving the caller free to reject the
    whole state.
    """

    _require_blender()
    if scene is None or camera is None or not isinstance(ecosystem, dict):
        raise ValueError("apply_geometry_state requires scene, ecosystem, and caller-created camera")
    if not isinstance(state_index, int) or state_index < 0:
        raise ValueError("state_index must be a non-negative integer")
    if not isinstance(attempt_index, int) or attempt_index < 0:
        raise ValueError("attempt_index must be a non-negative integer")
    public_layout = prepare_layout(scene, ecosystem)
    private = ecosystem.get("_layout_private")
    if not isinstance(private, dict) or not private.get("prepared"):
        raise RuntimeError("prepare_layout did not create a private state cache")
    records = list(private["records"])
    summary = ecosystem.get("summary") or {}
    layout_seed = int(summary.get("seed", ecosystem.get("seed", _prop(ecosystem.get("collection"), "seed", 0))))
    root_seed = int(cfg.get("root_seed", layout_seed))
    layout_family = str(summary.get("layout_family", _prop(ecosystem.get("collection"), "layout_family", "unknown")))
    seed_streams = {
        "root": int(root_seed),
        "layout": _stable_seed(root_seed, "layout", layout_seed),
        # Include layout_seed in every subsystem stream. This gives the
        # hierarchy root -> layout -> state/attempt while keeping camera, fish,
        # and benthic object RNGs independent of one another.
        "camera": _stable_seed(root_seed, "camera", layout_seed, state_index, attempt_index),
        "fish": _stable_seed(root_seed, "fish", layout_seed, state_index, attempt_index),
        "objects": _stable_seed(root_seed, "objects", layout_seed, state_index, attempt_index),
    }
    camera_rng = Random(seed_streams["camera"])
    fish_rng = Random(seed_streams["fish"])
    objects_rng = Random(seed_streams["objects"])

    camera_sampling = dict(DEFAULT_CAMERA_SAMPLING)
    camera_sampling.update(dict(cfg.get("camera_sampling") or {}))
    object_sampling = dict(DEFAULT_OBJECT_SAMPLING)
    object_sampling.update(dict(cfg.get("object_sampling") or {}))
    floor_z = float(summary.get("floor_z", _prop(ecosystem.get("collection"), "floor_z", 0.0)))
    seed_phase = layout_seed * 0.017
    surface_z = float(min(cfg.get("depths_m") or [5.0]))

    # Every call starts from the exact prepared baseline, including hidden
    # pool records.  The camera is deliberately independent of object RNG.
    matrices: dict[str, Matrix] = {}
    active: dict[str, bool] = {}
    for record in records:
        obj = record["object"]
        matrix = record["base_matrix"].copy()
        obj.matrix_world = matrix
        pool_active = bool(record["pool_active"])
        is_background_pool = int(record["instance_id"]) > _ORIGINAL_POOL_MAX or str(record["morphology"]).startswith("background_")
        is_fish = int(record["semantic_id"]) == _SEMANTIC_FISH
        active_value = pool_active and not is_background_pool and not is_fish
        matrices[str(record["name"])] = matrix
        active[str(record["name"])] = bool(active_value)
        obj.hide_render = not active_value
        obj["active_geometry_state"] = bool(active_value)
        obj["state_index"] = int(state_index)
        obj["state_attempt_index"] = int(attempt_index)
        obj["visible_camera"] = bool(active_value)

    # Activate all required background semantic types and vary only the extra
    # silhouettes. Their far y positions make them genuine ecosystem geometry
    # in the 15 m+ band rather than a background colour card.
    background = [record for record in records if int(record["instance_id"]) > _ORIGINAL_POOL_MAX or str(record["morphology"]).startswith("background_")]
    required_background: list[dict[str, Any]] = []
    for semantic in (_SEMANTIC_ROCK, _SEMANTIC_CORAL, _SEMANTIC_GRASS):
        candidates = [record for record in background if int(record["semantic_id"]) == semantic]
        if candidates:
            required_background.append(candidates[0])
    extras = [record for record in background if record not in required_background]
    extra_count = state_index % (len(extras) + 1) if extras else 0
    background_active = required_background + extras[:extra_count]
    for record in background_active:
        name = str(record["name"])
        active[name] = True
        record["object"].hide_render = False
        record["object"]["active_geometry_state"] = True
        record["object"]["visible_camera"] = True

    # Small benthic perturbations stay local to their prepared pivot. Large
    # table/fan/rock anchors receive only a restrained adjustment so the open
    # channel and base reef hierarchy remain recognisable.
    for record in records:
        semantic = int(record["semantic_id"])
        name = str(record["name"])
        if semantic not in (_SEMANTIC_ROCK, _SEMANTIC_CORAL, _SEMANTIC_GRASS):
            continue
        if not active.get(name, False):
            continue
        morphology = str(record["morphology"])
        large = morphology in {"table", "fan", "rock_cluster"} or morphology.startswith("background_")
        offset_limit = 0.055 if large else 0.2
        yaw_limit = 6.0 if large else float(_range(object_sampling, "small_yaw_deg", DEFAULT_OBJECT_SAMPLING["small_yaw_deg"])[1])
        scale_range = (0.96, 1.04) if large else _range(object_sampling, "small_scale", DEFAULT_OBJECT_SAMPLING["small_scale"])
        center = record["base_matrix"].translation + Vector((objects_rng.uniform(-offset_limit, offset_limit), objects_rng.uniform(-offset_limit, offset_limit), 0.0))
        scale = objects_rng.uniform(*scale_range)
        yaw = objects_rng.uniform(-yaw_limit, yaw_limit)
        matrix = _relative_pose(record["base_matrix"], center, yaw_deg=yaw, scale=scale)
        if bool(record.get("grounded")):
            matrix, _ = _ground_matrix(record, matrix, floor_z, seed_phase)
        matrices[name] = matrix
        record["object"].matrix_world = matrix

    # Hide a bounded, deterministic subset of optional mid/background benthic
    # pool members. Foreground anchors are kept visible to protect the near
    # band and the requested morphology coverage.
    optional = [record for record in records if int(record["semantic_id"]) in (_SEMANTIC_CORAL, _SEMANTIC_GRASS) and active.get(str(record["name"]), False) and record["layer"] != "foreground" and int(record["instance_id"]) <= _ORIGINAL_POOL_MAX]
    for record in optional:
        if objects_rng.random() < 0.055:
            name = str(record["name"])
            active[name] = False
            record["object"].hide_render = True
            record["object"]["active_geometry_state"] = False
            record["object"]["visible_camera"] = False

    # Sample a camera-safe pose. Geometry proxy checks include every active
    # object bound, so a camera is never accepted solely from its centre point
    # relative to the scene origin.
    height_lo, height_hi = _range(camera_sampling, "height_m", DEFAULT_CAMERA_SAMPLING["height_m"])
    x_lo, x_hi = _range(camera_sampling, "x_range", DEFAULT_CAMERA_SAMPLING["x_range"])
    y_lo, y_hi = _range(camera_sampling, "y_range", DEFAULT_CAMERA_SAMPLING["y_range"])
    look_y_lo, look_y_hi = _range(camera_sampling, "look_y_range", DEFAULT_CAMERA_SAMPLING["look_y_range"])
    look_z_lo, look_z_hi = _range(camera_sampling, "look_z_range", DEFAULT_CAMERA_SAMPLING["look_z_range"])
    roll_lo, roll_hi = _range(camera_sampling, "roll_deg", DEFAULT_CAMERA_SAMPLING["roll_deg"])
    lens = _as_float(camera_sampling.get("lens_mm", 28.0), 28.0)
    camera_error: dict[str, Any] | None = None
    eye = target = None
    camera_roll = 0.0
    camera_checks: dict[str, Any] | None = None
    for camera_attempt in range(24):
        x = camera_rng.uniform(x_lo, x_hi)
        y = camera_rng.uniform(y_lo, y_hi)
        seabed = sandbed_height(x, y, floor_z=floor_z, seed_phase=seed_phase)
        eye_candidate = Vector((x, y, seabed + camera_rng.uniform(height_lo, height_hi)))
        target_x = x + camera_rng.uniform(-0.28, 0.28)
        target_y = camera_rng.uniform(look_y_lo, look_y_hi)
        target_candidate, pitch_down_actual = _camera_target_from_xy(
            eye_candidate,
            target_x,
            target_y,
            camera_sampling,
            camera_rng,
            floor_z,
            seed_phase,
        )
        roll = camera_rng.uniform(roll_lo, roll_hi)
        _camera_pose(camera, eye_candidate, target_candidate, lens, roll)
        check = _camera_clearance(eye_candidate, records, matrices, active)
        if check["passed"] and eye_candidate.z < surface_z - 0.25:
            eye, target = eye_candidate, target_candidate
            camera_roll = float(roll)
            camera_checks = check
            camera_checks["camera_attempt"] = int(camera_attempt)
            camera_checks["height_above_seabed_m"] = float(eye.z - sandbed_height(float(eye.x), float(eye.y), floor_z=floor_z, seed_phase=seed_phase))
            camera_checks["pitch_down_actual_deg"] = float(pitch_down_actual)
            camera_checks["world_z"] = float(eye.z)
            camera_checks["surface_clearance_m"] = float(surface_z - eye.z)
            break
        camera_error = {
            "camera_attempt": camera_attempt,
            "check": check,
            "eye": _tuple3(eye_candidate),
            "pitch_down_actual_deg": float(pitch_down_actual),
        }
    else:
        raise ValueError(f"CANDIDATE_CAMERA_CLEARANCE: camera placement failed after 24 attempts ({json.dumps(camera_error, sort_keys=True)})")

    assert eye is not None and target is not None and camera_checks is not None
    right, screen_up, forward = _camera_basis(eye, target, camera)
    tan_x, tan_y = _camera_frustum_tangents(camera, cfg)
    fish_records = _select_fish(records, (
        _count_inclusive(fish_rng, object_sampling.get("foreground_fish"), DEFAULT_OBJECT_SAMPLING["foreground_fish"], state_index + attempt_index),
        _count_inclusive(fish_rng, object_sampling.get("midground_fish"), DEFAULT_OBJECT_SAMPLING["midground_fish"], state_index * 3 + attempt_index),
        _count_inclusive(fish_rng, object_sampling.get("background_fish"), DEFAULT_OBJECT_SAMPLING["background_fish"], state_index * 5 + attempt_index),
    ), fish_rng)
    fish_set = {str(record["name"]) for record in fish_records}
    fish_counts_by_layer: dict[str, int] = {"foreground": 0, "midground": 0, "background": 0}
    fish_checks: list[dict[str, Any]] = []
    accepted_fish_matrices: dict[str, Matrix] = {}
    fish_pose_metadata: dict[str, dict[str, Any]] = {}
    camera_clearance_recheck: dict[str, Any] | None = None
    for record in records:
        if int(record["semantic_id"]) != _SEMANTIC_FISH:
            continue
        name = str(record["name"])
        if name not in fish_set:
            active[name] = False
            record["object"].hide_render = True
            record["object"]["active_geometry_state"] = False
            record["object"]["visible_camera"] = False
            continue
        layer = record["layer"] if record["layer"] in fish_counts_by_layer else "midground"
        if layer == "foreground":
            distance_lo, distance_hi, screen_margin = 1.50, 3.00, 0.92
        elif layer == "background":
            # Long foreground-facing fish in the far pool need a modestly
            # closer depth so their full vertical silhouette fits beneath the
            # 5 m surface while remaining in the requested far GT band.
            distance_lo, distance_hi, screen_margin = 8.5, 13.5, 0.94
        else:
            distance_lo, distance_hi, screen_margin = 4.2, 8.0, 0.92
        accepted = False
        last_failure: dict[str, Any] | None = None
        # A low-discrepancy horizontal sequence explores both sides of the
        # view cone deterministically. Pure random u samples happened to land
        # inside the same near-reef proxies for all 28 candidates in one
        # state, even though the opposite side of the channel was clear.
        u_phase = fish_rng.random()
        for candidate_attempt in range(28):
            distance = fish_rng.uniform(distance_lo, distance_hi)
            u_unit = ((u_phase + candidate_attempt * 0.6180339887498949) % 1.0) * 2.0 - 1.0
            image_u = u_unit * screen_margin * 0.86
            base = eye + forward * distance + right * (image_u * distance * tan_x)
            yaw_center = None
            if not accepted_fish_matrices:
                local_to_camera = record["base_matrix"].inverted_safe().to_3x3() @ (eye - base).normalized()
                yaw_center = math.degrees(math.atan2(float(local_to_camera.x), float(local_to_camera.z)))
            # Sample the orientation before solving the vertical interval. The
            # fish's actual local pitch/roll changes its world-Z footprint, so
            # using its largest Euclidean radius here would incorrectly reject
            # slender fish in the rear water column.
            pose = _sample_fish_pose(fish_rng, object_sampling, yaw_center)
            provisional = _fish_matrix_from_pose(record, base, pose)
            terrain_extent, surface_extent = _fish_vertical_extents(record, provisional)
            # Solve the legal vertical image-coordinate interval on this fixed
            # forward-depth plane. The interval accounts for terrain and water
            # surface using this orientation's actual vertical extents; no
            # later world-Z lift is allowed to change the requested depth.
            v_interval = _safe_frustum_v_interval(
                eye,
                base,
                screen_up,
                distance,
                tan_y,
                screen_margin,
                float(terrain_extent),
                float(surface_extent),
                surface_z,
                floor_z,
                seed_phase,
                clearance=0.075,
            )
            if v_interval is None:
                last_failure = {
                    "candidate_attempt": candidate_attempt,
                    "reason": "no_safe_frustum_v_interval",
                    "nominal_distance_m": float(distance),
                    "screen_u": float(image_u),
                }
                continue
            # Reef tops occupy the lower part of the available water column.
            # Stratify candidates toward the upper portion of the *legal*
            # interval so foreground/midground fish can use open water while
            # retaining deterministic variation across attempts and states.
            v_bias = 0.48 if layer == "foreground" else (0.40 if layer == "midground" else 0.30)
            v_lower = v_interval[0] + (v_interval[1] - v_interval[0]) * v_bias
            image_v = fish_rng.uniform(v_lower, v_interval[1])
            center = base + screen_up * (image_v * distance * tan_y)
            matrix = _fish_matrix_from_pose(record, center, pose)
            if yaw_center is not None:
                pose["orientation_role"] = "camera_facing"
            pose["forward_depth_nominal_m"] = float(distance)
            pose["v_interval"] = [float(v_interval[0]), float(v_interval[1])]
            pose["vertical_extent_m"] = [float(terrain_extent), float(surface_extent)]
            water_gap = _surface_gap(record, matrix, surface_z)
            if water_gap < 0.075:
                last_failure = {
                    "candidate_attempt": candidate_attempt,
                    "reason": "water_surface_clearance",
                    "min_surface_gap_m": float(water_gap),
                    "surface_z_m": float(surface_z),
                }
                continue
            pose["min_surface_gap_m"] = float(water_gap)
            projection_ok, projection_detail = _projection_proxy(record, matrix, eye, right, screen_up, forward, tan_x, tan_y, margin=0.96)
            if not projection_ok:
                last_failure = {"candidate_attempt": candidate_attempt, **projection_detail}
                continue
            projected_depth = float((matrix.translation - eye).dot(forward))
            if abs(projected_depth - distance) > 1.0e-4 or projected_depth < distance_lo - 0.05 or projected_depth > distance_hi + 0.05:
                last_failure = {
                    "candidate_attempt": candidate_attempt,
                    "reason": "fixed_forward_depth_drift",
                    "nominal_distance_m": float(distance),
                    "projected_depth_m": projected_depth,
                }
                continue
            candidate_active = dict(active)
            candidate_matrices = dict(matrices)
            candidate_active[name] = True
            candidate_matrices[name] = matrix
            for prior_name, prior_matrix in accepted_fish_matrices.items():
                candidate_matrices[prior_name] = prior_matrix
            ok, detail = _fish_candidate_ok(record, matrix, candidate_matrices, candidate_active, records, floor_z, seed_phase)
            if ok:
                accepted = True
                active[name] = True
                matrices[name] = matrix
                accepted_fish_matrices[name] = matrix
                fish_pose_metadata[name] = pose
                record["object"].matrix_world = matrix
                record["object"].hide_render = False
                record["object"]["active_geometry_state"] = True
                record["object"]["visible_camera"] = True
                fish_counts_by_layer[layer] += 1
                fish_checks.append({"name": name, "accepted_attempt": candidate_attempt, "projection": projection_detail, **detail})
                break
            last_failure = {"candidate_attempt": candidate_attempt, **detail}
        if not accepted:
            raise ValueError(f"CANDIDATE_FISH_CLEARANCE: fish {name} failed after 28 attempts ({json.dumps(last_failure, sort_keys=True)})")

    active_geometry_records = [record for record in records if active.get(str(record["name"]), False)]
    camera_clearance_recheck = _camera_clearance(eye, records, matrices, active)
    if not camera_clearance_recheck["passed"]:
        raise ValueError(f"CANDIDATE_CAMERA_CLEARANCE: final camera clearance failed ({json.dumps(camera_clearance_recheck, sort_keys=True)})")

    _camera_pose(camera, eye, target, lens, camera_roll)
    scene.camera = camera
    try:
        scene.frame_set(1)
    except (AttributeError, RuntimeError):
        pass
    update = getattr(getattr(bpy, "context", None), "view_layer", None)
    if update is not None and callable(getattr(update, "update", None)):
        update.update()

    visible_names = [str(record["name"]) for record in records if active.get(str(record["name"]), False)]
    instances = [
        _record_output(record, bool(active.get(str(record["name"]), False)), matrices[str(record["name"])])
        for record in records
    ]
    for row in instances:
        row["layer"] = str(next(record["layer"] for record in records if record["name"] == row["name"]))
        pose = fish_pose_metadata.get(str(row["name"]))
        if pose is not None:
            row["pose"] = {
                key: _json_pose_value(value)
                for key, value in pose.items()
            }
    terrain_checks = {
        "passed": True,
        "method": "all fish evaluated vertices against seeded sandbed_height",
        "fish_count": len(fish_records),
        "minimum_fish_terrain_gap_m": float(min((_terrain_gap(record, matrices[record["name"]], floor_z, seed_phase) for record in fish_records), default=0.0)),
    }
    collision_checks = {
        "camera": camera_checks,
        "camera_clearance": camera_checks,
        "camera_final": camera_clearance_recheck,
        "fish_body_fin_proxy": {
            "passed": True,
            "method": "sphere_broadphase_oriented_bbox_15_axis_plus_full_vertex_terrain_gap",
            "margin_m": 0.045,
            "records": fish_checks,
        },
        "terrain": terrain_checks,
        "active_geometry_object_count": len(active_geometry_records),
        "background_active_names": [str(record["name"]) for record in background_active],
    }
    pitch_down_actual = float(camera_checks.get("pitch_down_actual_deg", math.degrees(math.atan2(float(eye.z - target.z), math.hypot(float(target.x - eye.x), float(target.y - eye.y))))))
    plan = {
        "seed_streams": {key: int(value) for key, value in seed_streams.items()},
        "state_index": int(state_index),
        "attempt_index": int(attempt_index),
        "camera": {
            "eye": _tuple3(eye),
            "target": _tuple3(target),
            "lens_mm": float(lens),
            "roll_deg": float(camera_roll),
            "pitch_down_actual_deg": pitch_down_actual,
            "height_above_seabed_m": float(eye.z - sandbed_height(float(eye.x), float(eye.y), floor_z=floor_z, seed_phase=seed_phase)),
            "seabed_height_m": float(sandbed_height(float(eye.x), float(eye.y), floor_z=floor_z, seed_phase=seed_phase)),
            "submersion_surface_m": float(surface_z - eye.z),
        },
        "instances": instances,
        "pool_registry": instances,
        "visible_object_names": visible_names,
        "active_object_names": visible_names,
        "visible_object_names_semantics": "active pool objects selected for label export; pixel visibility is measured from GT",
        "visible_fish_count": int(sum(1 for record in records if int(record["semantic_id"]) == _SEMANTIC_FISH and active.get(str(record["name"]), False))),
        "visible_fish_by_layer": fish_counts_by_layer,
        "collision_checks": collision_checks,
        "layout_family": layout_family,
        "layout_seed": int(layout_seed),
        "world_vertex_max_error_m": float(public_layout.get("world_vertex_max_error_m", 0.0)),
    }
    # Reassert JSON safety before handing the plan to labels/metadata writers.
    try:
        json.dumps(plan, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"geometry state plan is not JSON-safe: {exc}") from exc
    return plan


__all__ = ["DEFAULT_CAMERA_SAMPLING", "DEFAULT_OBJECT_SAMPLING", "prepare_layout", "apply_geometry_state"]
