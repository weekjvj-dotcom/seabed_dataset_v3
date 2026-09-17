"""Geometry-derived depth and integer labels for the seabed dataset.

The exporter deliberately traces a BVH built from the same evaluated meshes
that are present in the scene.  It does not use the renderer's depth pass,
which would make the units and camera-space convention dependent on colour
management or render settings.  Water and interface objects are filtered
before the BVH is built.

Blender is imported lazily enough for :mod:`seabed_dataset.src.labels.formats`
to remain usable in a normal Python interpreter.  Calling a geometry function
without Blender raises a clear ``RuntimeError``.
"""

from __future__ import annotations

import json
import math
import numbers
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

from .formats import write_exr, write_png

try:  # Keep formats.py and static tests importable outside Blender.
    import bpy  # type: ignore
    from mathutils import Vector  # type: ignore
    from mathutils.bvhtree import BVHTree  # type: ignore
except ImportError:  # pragma: no cover - exercised by normal CPython imports
    bpy = None  # type: ignore
    Vector = BVHTree = None  # type: ignore


SEMANTIC_NAMES: dict[int, str] = {
    0: "background",
    1: "sand",
    2: "rock",
    3: "coral",
    4: "grass",
    5: "fish",
}

def _require_blender() -> None:
    if bpy is None or Vector is None or BVHTree is None:
        raise RuntimeError("geometry labels require Blender's bpy/mathutils modules")


def _prop(obj: Any, key: str, default: Any = None) -> Any:
    """Read a Blender custom property or a light-weight test-double value."""

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


def _truthy_property(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off", "none", "null"}
    return bool(value)


def _integer_id(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer in {minimum}..{maximum}")
    if isinstance(value, numbers.Integral):
        integer = int(value)
    elif isinstance(value, numbers.Real):
        value_f = float(value)
        if not math.isfinite(value_f) or not value_f.is_integer():
            raise ValueError(f"{label} must be an integer in {minimum}..{maximum}")
        integer = int(value_f)
    else:
        raise ValueError(f"{label} must be an integer in {minimum}..{maximum}")
    if integer < minimum or integer > maximum:
        raise ValueError(f"{label}={integer} is outside {minimum}..{maximum}")
    return integer


def _object_name(obj: Any) -> str:
    name = getattr(obj, "name", None)
    return str(name) if name is not None else "<unnamed>"


def _pointer_key(obj: Any) -> tuple[str, int | str]:
    as_pointer = getattr(obj, "as_pointer", None)
    if callable(as_pointer):
        try:
            return ("pointer", int(as_pointer()))
        except (TypeError, ValueError):
            pass
    return ("identity", id(obj))


def _render_visibility_reason(scene: Any, obj: Any) -> str | None:
    """Return why an object is outside the scene's render-visible set."""

    if _truthy_property(getattr(obj, "hide_render", False)):
        return "hide_render=True"
    visible_camera = _prop(obj, "visible_camera", None)
    if visible_camera is None:
        visible_camera = getattr(obj, "visible_camera", True)
    if not _truthy_property(visible_camera):
        return "visible_camera=False"

    root = getattr(scene, "collection", None)
    if root is None:
        return "object has no scene collection path"
    object_key = _pointer_key(obj)
    visited: set[tuple[tuple[str, int | str], bool]] = set()

    def walk(collection: Any, ancestors_visible: bool) -> bool:
        collection_key = _pointer_key(collection)
        state = (collection_key, ancestors_visible)
        if state in visited:
            return False
        visited.add(state)
        visible = ancestors_visible and not _truthy_property(
            getattr(collection, "hide_render", False)
        )
        if visible:
            members = getattr(collection, "objects", ())
            if any(_pointer_key(member) == object_key for member in members):
                return True
        for child in getattr(collection, "children", ()):
            if walk(child, visible):
                return True
        return False

    if not walk(root, True):
        return "object has no visible collection path"
    return None


def _target_records(scene: Any, objects: Iterable[Any] | None) -> list[dict[str, Any]]:
    """Validate and normalize the explicit/default target object selection."""

    _require_blender()
    explicit = objects is not None
    candidates = list(objects) if explicit else list(getattr(scene, "objects", ()))
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int | str]] = set()
    instance_semantics: dict[int, int] = {}
    for obj in candidates:
        pointer = _pointer_key(obj)
        if pointer in seen:
            raise ValueError(f"target object {_object_name(obj)!r} was listed more than once")
        seen.add(pointer)
        if _truthy_property(_prop(obj, "is_water", False)):
            continue
        obj_type = getattr(obj, "type", None)
        if obj_type != "MESH":
            if explicit:
                raise TypeError(f"target object {_object_name(obj)!r} is not a MESH")
            continue
        visibility_reason = _render_visibility_reason(scene, obj)
        if visibility_reason is not None:
            if explicit:
                raise ValueError(
                    f"target object {_object_name(obj)!r} is not render-visible: {visibility_reason}"
                )
            continue
        raw_semantic = _prop(obj, "semantic_id", None)
        if raw_semantic is None:
            if explicit:
                raise ValueError(f"target object {_object_name(obj)!r} lacks semantic_id")
            continue
        # Default selection uses semantic_id > 0.  Explicit selection still
        # treats zero as invalid: background is not a target mesh.
        if isinstance(raw_semantic, numbers.Real) and not isinstance(raw_semantic, bool):
            raw_semantic_f = float(raw_semantic)
            if math.isfinite(raw_semantic_f) and raw_semantic_f == 0.0 and not explicit:
                continue
        semantic = _integer_id(raw_semantic, f"{_object_name(obj)}.semantic_id", 1, 5)
        raw_instance = _prop(obj, "instance_id", None)
        if raw_instance is None:
            raise ValueError(f"target object {_object_name(obj)!r} lacks instance_id")
        instance = _integer_id(raw_instance, f"{_object_name(obj)}.instance_id", 1, 65535)
        previous_semantic = instance_semantics.get(instance)
        if previous_semantic is not None and previous_semantic != semantic:
            raise ValueError(
                f"instance_id {instance} is used by semantic categories "
                f"{previous_semantic} and {semantic}"
            )
        instance_semantics[instance] = semantic
        morphology = _prop(obj, "asset_morphology", None)
        records.append(
            {
                "object": obj,
                "name": _object_name(obj),
                "semantic_id": semantic,
                "instance_id": instance,
                "asset_morphology": None if morphology is None else str(morphology),
            }
        )
    return records


def _build_world_bvh(scene: Any, records: Sequence[dict[str, Any]]) -> tuple[Any, list[dict[str, Any]] | None]:
    """Build one world-space BVH and return its triangle-to-label map."""

    _require_blender()
    if not records:
        return None, None
    context = bpy.context
    try:
        view_layer = scene.view_layers[0]
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError("the target scene must provide at least one view layer") from exc
    temp_override = getattr(context, "temp_override", None)
    if not callable(temp_override):
        raise RuntimeError("this Blender version lacks context.temp_override()")
    # A depsgraph belongs to a scene/view-layer context.  Do not use a method
    # on ``scene`` (regular bpy scenes do not provide one) or silently trace
    # whichever scene happens to be active in the UI.
    with temp_override(scene=scene, view_layer=view_layer):
        depsgraph = context.evaluated_depsgraph_get()
        update = getattr(depsgraph, "update", None)
        if callable(update):
            update()
    vertices: list[Any] = []
    triangles: list[tuple[int, int, int]] = []
    triangle_labels: list[dict[str, Any]] = []
    for record in records:
        obj = record["object"]
        obj_eval = obj.evaluated_get(depsgraph)
        mesh = None
        try:
            try:
                mesh = obj_eval.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
            except TypeError:  # Blender versions before the depsgraph argument
                mesh = obj_eval.to_mesh()
            if mesh is None:
                continue
            calc_loop_triangles = getattr(mesh, "calc_loop_triangles", None)
            if callable(calc_loop_triangles):
                calc_loop_triangles()
            obj_matrix = obj_eval.matrix_world
            base_index = len(vertices)
            for vertex in mesh.vertices:
                vertices.append(obj_matrix @ vertex.co)
            loop_triangles = getattr(mesh, "loop_triangles", ())
            for loop_triangle in loop_triangles:
                indices = tuple(int(index) for index in loop_triangle.vertices)
                if len(indices) != 3:
                    continue
                triangles.append(
                    (base_index + indices[0], base_index + indices[1], base_index + indices[2])
                )
                triangle_labels.append(record)
        finally:
            clear_mesh = getattr(obj_eval, "to_mesh_clear", None)
            if callable(clear_mesh):
                clear_mesh()
    if not triangles:
        return None, None
    try:
        tree = BVHTree.FromPolygons(vertices, triangles, all_triangles=True)
    except TypeError:  # defensive compatibility with older mathutils builds
        tree = BVHTree.FromPolygons(vertices, triangles)
    return tree, triangle_labels


def _render_dimensions(scene: Any) -> tuple[int, int]:
    render = getattr(scene, "render", None)
    if render is None:
        raise ValueError("scene.render is required for camera metadata")
    try:
        base_width = int(render.resolution_x)
        base_height = int(render.resolution_y)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("scene.render.resolution_x/y must be positive integers") from exc
    percentage = float(getattr(render, "resolution_percentage", 100.0))
    if not math.isfinite(percentage) or percentage <= 0.0:
        raise ValueError("scene.render.resolution_percentage must be positive and finite")
    # Blender truncates the percentage-scaled dimensions (for example,
    # 103 * 33% becomes 33, rather than Python round()'s 34).
    width = int(base_width * percentage / 100.0)
    height = int(base_height * percentage / 100.0)
    if width <= 0 or height <= 0:
        raise ValueError("effective render resolution is empty")
    return width, height


def _frame_intrinsics(scene: Any, camera: Any, width: int, height: int) -> tuple[float, float, float, float, float]:
    """Return fx, fy, cx, cy and local frame z in pixel units.

    ``Camera.view_frame`` is the authoritative Blender projection, including
    sensor fit, pixel aspect and lens shift.  Deriving K from its four corners
    avoids duplicating Blender's AUTO fit edge cases.
    """

    camera_data = getattr(camera, "data", None)
    if camera_data is None:
        raise ValueError("scene.camera.data is required")
    if getattr(camera_data, "type", "PERSP") != "PERSP":
        raise ValueError("geometry labels currently require a perspective camera")
    frame_method = getattr(camera_data, "view_frame", None)
    if callable(frame_method):
        try:
            frame = list(frame_method(scene=scene))
        except TypeError:
            frame = list(frame_method())
        if len(frame) < 4:
            raise ValueError("camera.view_frame() returned fewer than four corners")
        x_values = [float(point.x) for point in frame]
        y_values = [float(point.y) for point in frame]
        z_values = [float(point.z) for point in frame]
        xmin, xmax = min(x_values), max(x_values)
        ymin, ymax = min(y_values), max(y_values)
        z = sum(z_values) / len(z_values)
        frame_width, frame_height = xmax - xmin, ymax - ymin
        if not all(math.isfinite(value) for value in (xmin, xmax, ymin, ymax, z)):
            raise ValueError("camera.view_frame() returned non-finite coordinates")
        if frame_width <= 0.0 or frame_height <= 0.0 or abs(z) <= 1.0e-12:
            raise ValueError("camera.view_frame() returned a degenerate frame")
        # view_frame points lie on the camera's negative-Z image plane.
        focal_depth = -z
        fx = focal_depth * width / frame_width
        fy = focal_depth * height / frame_height
        # ``cx``/``cy`` are principal points in pixel-edge coordinates.  Pixel
        # centres are supplied later as (u + 0.5, v + 0.5), so adding another
        # half pixel here would shift every ray by half a pixel.
        cx = -xmin * width / frame_width
        cy = ymax * height / frame_height
        return fx, fy, cx, cy, z

    # The fallback is useful for small test doubles.  Real Blender cameras
    # provide view_frame(), so all sensor-fit details above remain authoritative
    # for production exports.
    lens = float(getattr(camera_data, "lens", 50.0))
    sensor_width = float(getattr(camera_data, "sensor_width", 36.0))
    sensor_height = float(getattr(camera_data, "sensor_height", 24.0))
    if not all(math.isfinite(value) and value > 0.0 for value in (lens, sensor_width, sensor_height)):
        raise ValueError("camera lens and sensor dimensions must be positive and finite")
    pixel_aspect_x = float(getattr(getattr(scene, "render", None), "pixel_aspect_x", 1.0))
    pixel_aspect_y = float(getattr(getattr(scene, "render", None), "pixel_aspect_y", 1.0))
    if not all(math.isfinite(value) and value > 0.0 for value in (pixel_aspect_x, pixel_aspect_y)):
        raise ValueError("render pixel aspect must be positive and finite")
    fit = str(getattr(camera_data, "sensor_fit", "AUTO")).upper()
    if fit == "AUTO":
        fit = "HORIZONTAL" if width * pixel_aspect_x >= height * pixel_aspect_y else "VERTICAL"
    if fit == "HORIZONTAL":
        fx = lens / sensor_width * width
        fy = fx * pixel_aspect_x / pixel_aspect_y
    elif fit == "VERTICAL":
        fy = lens / sensor_height * height
        fx = fy * pixel_aspect_y / pixel_aspect_x
    else:
        raise ValueError(f"unsupported camera sensor_fit {fit!r}")
    shift_x = float(getattr(camera_data, "shift_x", 0.0))
    shift_y = float(getattr(camera_data, "shift_y", 0.0))
    cx = width * (0.5 - shift_x)
    cy = height * (0.5 + shift_y)
    return fx, fy, cx, cy, -1.0


def _matrix_rows(matrix: Any) -> list[list[float]]:
    try:
        rows = [[float(matrix[row][column]) for column in range(4)] for row in range(4)]
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("camera matrix_world must be a 4x4 matrix") from exc
    if not all(math.isfinite(value) for row in rows for value in row):
        raise ValueError("camera matrix_world contains non-finite values")
    return rows


def camera_metadata(scene: Any) -> dict[str, Any]:
    """Return the calibrated perspective camera metadata used by the ray tracer."""

    _require_blender()
    camera = getattr(scene, "camera", None)
    if camera is None:
        raise ValueError("scene.camera is required for geometry labels")
    width, height = _render_dimensions(scene)
    camera_data = getattr(camera, "data", None)
    if camera_data is None:
        raise ValueError("scene.camera.data is required")
    lens_mm = float(getattr(camera_data, "lens", 50.0))
    sensor_width_mm = float(getattr(camera_data, "sensor_width", 36.0))
    sensor_height_mm = float(getattr(camera_data, "sensor_height", 24.0))
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (lens_mm, sensor_width_mm, sensor_height_mm)
    ):
        raise ValueError("camera lens and sensor dimensions must be positive and finite")
    pixel_aspect_x = float(getattr(scene.render, "pixel_aspect_x", 1.0))
    pixel_aspect_y = float(getattr(scene.render, "pixel_aspect_y", 1.0))
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (pixel_aspect_x, pixel_aspect_y)
    ):
        raise ValueError("render pixel aspect must be positive and finite")
    clip_start = float(getattr(camera_data, "clip_start", 0.1))
    clip_end = float(getattr(camera_data, "clip_end", 1000.0))
    if not (
        math.isfinite(clip_start)
        and math.isfinite(clip_end)
        and clip_start > 0.0
        and clip_end > clip_start
    ):
        raise ValueError("camera clip_start/clip_end must be finite with 0 < start < end")
    dof = getattr(camera_data, "dof", None)
    dof_enabled = bool(getattr(dof, "use_dof", False)) if dof is not None else False
    motion_blur = bool(getattr(getattr(scene, "render", None), "use_motion_blur", False))
    if dof_enabled or motion_blur:
        raise ValueError("DOF and motion blur must be disabled for deterministic geometry labels")
    fx, fy, cx, cy, frame_z = _frame_intrinsics(scene, camera, width, height)
    if not all(math.isfinite(value) and value > 0.0 for value in (fx, fy)):
        raise ValueError("camera intrinsics fx/fy must be positive and finite")
    matrix_world = getattr(camera, "matrix_world", None)
    if matrix_world is None:
        raise ValueError("scene.camera.matrix_world is required")
    camera_to_world = _matrix_rows(matrix_world)
    try:
        world_to_camera = _matrix_rows(matrix_world.inverted())
    except (AttributeError, ValueError, TypeError) as exc:
        raise ValueError("camera.matrix_world must be invertible") from exc
    render = scene.render
    pixel_aspect = [pixel_aspect_x, pixel_aspect_y]
    return {
        "projection": "perspective",
        "resolution": {"width": width, "height": height},
        "resolution_scene": {
            "width": int(render.resolution_x),
            "height": int(render.resolution_y),
            "percentage": float(getattr(render, "resolution_percentage", 100.0)),
        },
        "pixel_aspect": pixel_aspect,
        "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        "camera_to_world": camera_to_world,
        "world_to_camera": world_to_camera,
        "camera_name": str(getattr(camera, "name", "Camera")),
        "lens_mm": lens_mm,
        "sensor_width_mm": sensor_width_mm,
        "sensor_height_mm": sensor_height_mm,
        "sensor_fit": str(getattr(camera_data, "sensor_fit", "AUTO")),
        "shift": [
            float(getattr(camera_data, "shift_x", 0.0)),
            float(getattr(camera_data, "shift_y", 0.0)),
        ],
        "clip": [
            clip_start,
            clip_end,
        ],
        "dof_enabled": dof_enabled,
        "motion_blur_enabled": motion_blur,
        "camera_frame_z": frame_z,
        "units": "metres (1 Blender unit = 1 m)",
        "orientation": {
            "camera_local_x": "right",
            "camera_local_y": "up",
            "camera_local_negative_z": "forward",
            "image_origin": "top_left",
            "pixel_center": "(u + 0.5, v + 0.5)",
        },
    }


def _validate_pixel(scene: Any, u: int | float, v: int | float, metadata: dict[str, Any] | None = None) -> tuple[int, int]:
    if isinstance(u, bool) or isinstance(v, bool) or not isinstance(u, numbers.Real) or not isinstance(v, numbers.Real):
        raise ValueError("pixel coordinates u and v must be finite numbers")
    u_f, v_f = float(u), float(v)
    if not math.isfinite(u_f) or not math.isfinite(v_f) or not u_f.is_integer() or not v_f.is_integer():
        raise ValueError("pixel coordinates u and v must be integer pixel indices")
    u_i, v_i = int(u_f), int(v_f)
    if metadata is None:
        metadata = camera_metadata(scene)
    resolution = metadata["resolution"]
    if not 0 <= u_i < int(resolution["width"]) or not 0 <= v_i < int(resolution["height"]):
        raise ValueError(
            f"pixel ({u_i}, {v_i}) is outside {resolution['width']}x{resolution['height']} image"
        )
    return u_i, v_i


def ray_for_pixel(scene: Any, u: int | float, v: int | float) -> tuple[Any, Any]:
    """Return the world-space origin and unit direction through pixel ``(u,v)``."""

    _require_blender()
    metadata = camera_metadata(scene)
    u_i, v_i = _validate_pixel(scene, u, v, metadata)
    return _ray_for_pixel_metadata(scene, u_i, v_i, metadata)


def _ray_for_pixel_metadata(scene: Any, u: int, v: int, metadata: dict[str, Any]) -> tuple[Any, Any]:
    """Fast pixel ray helper for an export whose camera metadata is fixed."""

    fx = float(metadata["K"][0][0])
    fy = float(metadata["K"][1][1])
    cx = float(metadata["K"][0][2])
    cy = float(metadata["K"][1][2])
    camera_ray = Vector(
        (
            (u + 0.5 - cx) / fx,
            (cy - (v + 0.5)) / fy,
            -1.0,
        )
    ).normalized()
    camera = scene.camera
    matrix_world = camera.matrix_world
    origin = matrix_world.translation.copy()
    direction = (matrix_world.to_3x3() @ camera_ray).normalized()
    return origin, direction


def _ray_clip_segment(
    origin: Any,
    direction: Any,
    world_to_camera: Any,
    clip_start: float,
    clip_end: float,
) -> tuple[Any, float]:
    """Return a ray origin at the near plane and its finite far length.

    The camera clips in camera-space -Z, while ``direction`` is unit length in
    world space.  Consequently near/far distances along an oblique ray are
    ``clip / cos(theta)``.  The exported range is still measured from the
    original camera origin, after the BVH returns a hit.
    """

    try:
        local_direction = (world_to_camera.to_3x3() @ direction).normalized()
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("camera matrix cannot transform a world ray") from exc
    cos_theta = -float(local_direction.z)
    if not math.isfinite(cos_theta) or cos_theta <= 0.0:
        raise RuntimeError("camera ray does not point through the camera's negative-Z half-space")
    near_distance = float(clip_start) / cos_theta
    far_distance = float(clip_end) / cos_theta
    if not (
        math.isfinite(near_distance)
        and math.isfinite(far_distance)
        and far_distance > near_distance
    ):
        raise RuntimeError("camera clip planes produce an invalid world-ray segment")
    return origin + direction * near_distance, far_distance - near_distance


def _json_safe(value: Any) -> Any:
    """Convert Blender scalar/vector values if an unexpected one reaches metadata."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def export_labels(scene: Any, output_dir: str | Path, objects: Iterable[Any] | None = None) -> dict[str, Any]:
    """Export range/Z depth, semantic, instance and valid-mask labels.

    ``objects`` is an explicit iterable of target mesh objects.  When omitted,
    all mesh objects with ``semantic_id > 0`` are selected.  Objects carrying
    a truthy ``is_water`` custom property are always excluded.  The returned
    dictionary is also written to ``metadata.json`` beside the image files.
    """

    _require_blender()
    metadata = camera_metadata(scene)
    width = int(metadata["resolution"]["width"])
    height = int(metadata["resolution"]["height"])
    records = _target_records(scene, objects)
    tree, triangle_labels = _build_world_bvh(scene, records)
    camera = scene.camera
    world_to_camera = camera.matrix_world.inverted()
    clip_start = float(metadata["clip"][0])
    clip_end = float(metadata["clip"][1])
    depth_range = [[0.0 for _ in range(width)] for _ in range(height)]
    depth_camera_z = [[0.0 for _ in range(width)] for _ in range(height)]
    semantic = [[0 for _ in range(width)] for _ in range(height)]
    instance = [[0 for _ in range(width)] for _ in range(height)]
    valid = [[0 for _ in range(width)] for _ in range(height)]
    semantic_counts: dict[int, int] = {identifier: 0 for identifier in SEMANTIC_NAMES if identifier > 0}
    instance_counts: dict[int, int] = {}

    if tree is not None and triangle_labels:
        for v in range(height):
            for u in range(width):
                origin, direction = _ray_for_pixel_metadata(scene, u, v, metadata)
                cast_origin, cast_distance = _ray_clip_segment(
                    origin,
                    direction,
                    world_to_camera,
                    clip_start,
                    clip_end,
                )
                hit = tree.ray_cast(cast_origin, direction, cast_distance)
                location, _normal, triangle_index, _distance = hit
                if location is None or triangle_index is None or triangle_index < 0:
                    continue
                try:
                    record = triangle_labels[int(triangle_index)]
                except (IndexError, TypeError, ValueError) as exc:
                    raise RuntimeError("BVH returned an invalid triangle index") from exc
                hit_point = location
                range_m = float((hit_point - origin).length)
                camera_z = float(-(world_to_camera @ hit_point).z)
                if not math.isfinite(range_m) or not math.isfinite(camera_z):
                    raise RuntimeError("BVH intersection produced a non-finite depth")
                semantic_id = int(record["semantic_id"])
                instance_id = int(record["instance_id"])
                depth_range[v][u] = range_m
                depth_camera_z[v][u] = camera_z
                semantic[v][u] = semantic_id
                instance[v][u] = instance_id
                valid[v][u] = 255
                semantic_counts[semantic_id] = semantic_counts.get(semantic_id, 0) + 1
                instance_counts[instance_id] = instance_counts.get(instance_id, 0) + 1

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "depth_range_m": "depth_range_m.exr",
        "depth_camera_z_m": "depth_camera_z_m.exr",
        "semantic_id": "semantic_id.png",
        "instance_id": "instance_id.png",
        "valid_mask": "valid_mask.png",
    }
    instance_mapping: dict[str, dict[str, Any]] = {}
    for record in records:
        instance_key = str(record["instance_id"])
        entry = instance_mapping.setdefault(
            instance_key,
            {
                "semantic_id": int(record["semantic_id"]),
                "semantic_name": SEMANTIC_NAMES[int(record["semantic_id"])],
                "objects": [],
                "asset_morphologies": [],
                "visible_pixel_count": 0,
            },
        )
        if int(entry["semantic_id"]) != int(record["semantic_id"]):
            raise RuntimeError("instance mapping changed category after BVH construction")
        if record["name"] not in entry["objects"]:
            entry["objects"].append(record["name"])
        morphology = record.get("asset_morphology")
        if morphology is not None and morphology not in entry["asset_morphologies"]:
            entry["asset_morphologies"].append(morphology)
    for instance_key, count in instance_counts.items():
        if str(instance_key) in instance_mapping:
            instance_mapping[str(instance_key)]["visible_pixel_count"] = int(count)

    metadata.update(
        {
            "format": "seabed_geometry_labels_v1",
            "files": files,
            "image_origin": "top_left",
            "pixel_sampling": "pixel_center",
            "depth_definition": {
                "depth_range_m": "Euclidean distance from camera origin to nearest evaluated target triangle, metres",
                "depth_camera_z_m": "positive camera-space -Z coordinate of the same intersection, metres",
                "miss_value": 0.0,
            },
            "semantic_classes": {str(identifier): name for identifier, name in SEMANTIC_NAMES.items()},
            "semantic_pixel_counts": semantic_counts,
            "instance_pixel_counts": instance_counts,
            "instances": instance_mapping,
            "instance_map": instance_mapping,
            "target_object_count": len(records),
            "target_objects": [record["name"] for record in records],
            "water_objects_excluded": True,
            "edge_note": "Integer labels use nearest BVH intersection per pixel center and have no sRGB conversion or interpolation; RGB antialiasing boundaries may mix objects, so validate labels on non-edge pixels.",
        }
    )

    temporary_output = Path(tempfile.mkdtemp(prefix=".labels-", dir=str(output)))
    try:
        # Construct every image below in a private sibling directory first.
        # The later os.replace calls are per-file publishes, not a multi-file
        # filesystem transaction.
        write_exr(temporary_output / files["depth_range_m"], depth_range, width, height, channel="Y")
        write_exr(temporary_output / files["depth_camera_z_m"], depth_camera_z, width, height, channel="Y")
        write_png(temporary_output / files["semantic_id"], semantic, width, height, bit_depth=16)
        write_png(temporary_output / files["instance_id"], instance, width, height, bit_depth=16)
        write_png(temporary_output / files["valid_mask"], valid, width, height, bit_depth=8)
        metadata = _json_safe(metadata)
        metadata["files"]["metadata"] = "metadata.json"
        metadata_path = temporary_output / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        for filename in files.values():
            os.replace(temporary_output / filename, output / filename)
        # Metadata is published last so a successful metadata read implies the
        # five image files were fully written before this final per-file replace.
        os.replace(metadata_path, output / "metadata.json")
    finally:
        shutil.rmtree(temporary_output, ignore_errors=True)
    return metadata


__all__ = [
    "SEMANTIC_NAMES",
    "camera_metadata",
    "ray_for_pixel",
    "export_labels",
]
