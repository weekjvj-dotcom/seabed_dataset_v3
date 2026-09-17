"""Finite, native Cycles water volume and water-air interface.

This module only builds Blender data.  It does not render, read pixels, apply
an image-space attenuation formula, or register a Blender UI class.  The
volume is one closed rectangular mesh with two material slots:

* the upward-facing top polygon connects both a white, zero-roughness
  ``GlassBSDF`` surface and the native volume shader;
* the bottom and four side polygons connect the same volume shader only.

The two material slots use one node-group definition and receive identical
coefficient sockets.  Keeping the boundary in one mesh avoids the accidental
double attenuation caused by overlapping volume objects.

``bpy`` is imported defensively so that the schema and validation helpers can
be checked with the system Python interpreter.  Calling a build or metadata
operation still requires a real Blender Python environment.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping

try:  # Blender's Python exposes bpy; regular unit tests do not.
    import bpy  # type: ignore
except ImportError:  # pragma: no cover - exercised by the system-Python tests
    bpy = None  # type: ignore


WATER_SCHEMA_VERSION = "native_water.v1"
_REQUIRED_PRESET_KEYS = ("id", "absorption", "scatter", "g", "ior")
_GROUP_INPUTS = (
    "Absorption Coefficients",
    "Scatter Coefficients",
    "Anisotropy g",
    "Surface IOR",
)


def _number(value, name: str) -> float:
    """Return a finite float or raise a user-facing configuration error."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _rgb(value, name: str) -> tuple[float, float, float]:
    """Validate an RGB coefficient vector in inverse metres."""

    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain exactly three numbers")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain exactly three numbers") from exc
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three numbers")
    result = tuple(_number(component, f"{name}[{index}]") for index, component in enumerate(values))
    if any(component < 0.0 for component in result):
        raise ValueError(f"{name} coefficients must be non-negative")
    return result  # type: ignore[return-value]


def _json_compatible(value):
    """Convert Blender ID properties and mathutils arrays to JSON values.

    Blender may expose a nested custom-property dictionary as an
    ``IDPropertyGroup``.  It is mapping-like but is not accepted by
    ``json.dumps``.  Preserve its structure recursively instead of converting
    it to a lossy string.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("water metadata contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    # IDPropertyGroup is mapping-like on Blender versions where it does not
    # register as collections.abc.Mapping.
    if hasattr(value, "keys") and hasattr(value, "__getitem__"):
        try:
            return {
                str(key): _json_compatible(value[key])
                for key in value.keys()
            }
        except (AttributeError, KeyError, TypeError):
            pass
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    # bpy_prop_array, Vector, and similar numeric containers are iterable but
    # do not necessarily inherit list/tuple.
    try:
        if not isinstance(value, (bytes, bytearray)):
            return [_json_compatible(item) for item in value]
    except TypeError:
        pass
    raise TypeError(f"water metadata contains unsupported value {type(value).__name__}")


def _validate_preset(preset: Mapping) -> dict:
    """Validate and copy a water preset before touching Blender data.

    The returned dictionary contains only JSON-compatible values.  Extra keys
    are accepted for forward compatibility but are not consumed by the native
    water builder.
    """

    if not isinstance(preset, Mapping):
        raise ValueError("water preset must be a mapping")
    missing = [key for key in _REQUIRED_PRESET_KEYS if key not in preset]
    if missing:
        raise ValueError(f"water preset is missing required keys: {', '.join(missing)}")

    preset_id = preset["id"]
    if not isinstance(preset_id, str) or not preset_id.strip():
        raise ValueError("water preset id must be a non-empty string")
    if len(preset_id) > 96 or re.search(r"[\x00-\x1f\x7f/\\]", preset_id):
        raise ValueError("water preset id contains an invalid character")

    absorption = _rgb(preset["absorption"], "absorption")
    scatter = _rgb(preset["scatter"], "scatter")
    g = _number(preset["g"], "g")
    if not -1.0 < g < 1.0:
        raise ValueError("g must be strictly between -1 and 1 for Henyey-Greenstein")
    ior = _number(preset["ior"], "ior")
    # IOR=1.0 is a useful explicitly defined no-refraction control for
    # isolated interface experiments; ordinary water presets use 1.333 or
    # another measured/declared water-air value.
    if not 1.0 <= ior <= 4.0:
        raise ValueError("ior must be at least 1 and no greater than 4")

    return {
        "id": preset_id,
        "absorption": list(absorption),
        "scatter": list(scatter),
        "g": g,
        "ior": ior,
    }


def _validate_geometry(surface_z, extent, bottom_z) -> tuple[float, float, float]:
    """Validate finite water bounds before creating any Blender datablock."""

    top = _number(surface_z, "surface_z")
    horizontal_extent = _number(extent, "extent")
    bottom = _number(bottom_z, "bottom_z")
    if horizontal_extent <= 0.0:
        raise ValueError("extent must be greater than zero")
    if top <= bottom:
        raise ValueError("surface_z must be above bottom_z")
    return top, horizontal_extent, bottom


def _require_bpy():
    if bpy is None:
        raise RuntimeError("native water construction requires Blender's bpy module")
    return bpy


def _socket(node, name: str):
    socket = node.inputs.get(name) if hasattr(node.inputs, "get") else None
    if socket is None:
        raise RuntimeError(f"{node.bl_idname} is missing input socket {name!r}")
    return socket


def _output_socket(node, name: str):
    socket = node.outputs.get(name) if hasattr(node.outputs, "get") else None
    if socket is None:
        raise RuntimeError(f"{node.bl_idname} is missing output socket {name!r}")
    return socket


def _set_socket_value(node, name: str, value) -> None:
    socket = _socket(node, name)
    if isinstance(value, (tuple, list)):
        socket.default_value = tuple(float(item) for item in value)
    else:
        socket.default_value = float(value)


def _socket_value(socket):
    value = socket.default_value
    if isinstance(value, (str, bytes)):
        return value
    try:
        values = tuple(value)
    except TypeError:
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    return [float(item) for item in values]


def _safe_custom_set(datablock, key: str, value) -> None:
    """Set an ID property using JSON for nested values Blender cannot store."""

    try:
        datablock[key] = value
    except (TypeError, ValueError):
        datablock[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)


def _custom_json(datablock, key: str, default):
    try:
        value = datablock.get(key, default)
    except AttributeError:
        return default
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return default
        return _json_compatible(parsed)
    return _json_compatible(value)


def _make_interface_socket(group, *, name: str, in_out: str, socket_type: str, default, description: str):
    socket = group.interface.new_socket(name=name, in_out=in_out, socket_type=socket_type)
    try:
        socket.default_value = default
    except (AttributeError, TypeError, ValueError):
        pass
    try:
        socket.description = description
    except (AttributeError, TypeError):
        pass
    return socket


def _build_native_group(name: str, preset: Mapping, created: dict | None = None):
    """Create the shared group with direct native coefficient links."""

    blender = _require_bpy()
    group = blender.data.node_groups.new(name, "ShaderNodeTree")
    if created is not None:
        created["node_groups"].append(group)
    _make_interface_socket(
        group,
        name="Absorption Coefficients",
        in_out="INPUT",
        socket_type="NodeSocketVector",
        default=tuple(preset["absorption"]),
        description="RGB absorption coefficient in m^-1; connected directly to the native volume node.",
    )
    _make_interface_socket(
        group,
        name="Scatter Coefficients",
        in_out="INPUT",
        socket_type="NodeSocketVector",
        default=tuple(preset["scatter"]),
        description="RGB scattering coefficient in m^-1; connected directly to the native volume node.",
    )
    g_socket = _make_interface_socket(
        group,
        name="Anisotropy g",
        in_out="INPUT",
        socket_type="NodeSocketFloat",
        default=float(preset["g"]),
        description="Henyey-Greenstein anisotropy; -1 < g < 1.",
    )
    try:
        g_socket.min_value = -0.999999
        g_socket.max_value = 0.999999
    except (AttributeError, TypeError, ValueError):
        pass
    ior_socket = _make_interface_socket(
        group,
        name="Surface IOR",
        in_out="INPUT",
        socket_type="NodeSocketFloat",
        default=float(preset["ior"]),
        description="Water-air GlassBSDF IOR. This does not drive the volume phase IOR.",
    )
    try:
        ior_socket.min_value = 1.0
        ior_socket.max_value = 4.0
    except (AttributeError, TypeError, ValueError):
        pass
    _make_interface_socket(
        group,
        name="Surface",
        in_out="OUTPUT",
        socket_type="NodeSocketShader",
        default=None,
        description="Glass surface output for the top interface only.",
    )
    _make_interface_socket(
        group,
        name="Volume",
        in_out="OUTPUT",
        socket_type="NodeSocketShader",
        default=None,
        description="Finite water volume output for the whole closed boundary.",
    )

    nodes = group.nodes
    links = group.links
    group_input = nodes.new("NodeGroupInput")
    group_input.name = "Water controls input"
    group_input.label = "Native water controls"
    group_input.location = (-640, 80)

    coefficients = nodes.new("ShaderNodeVolumeCoefficients")
    coefficients.name = "Native Volume Coefficients"
    coefficients.label = "RGB m^-1 / Henyey-Greenstein"
    coefficients.location = (-40, -100)
    coefficients.phase = "HENYEY_GREENSTEIN"
    # Blender exposes several additional sockets in the node enumeration, but
    # some are phase-dependent internal sockets.  Leave every such socket at
    # the native default; only the three HG controls below are addressable and
    # part of this module's contract.  In particular, the surface IOR must not
    # be redirected to the volume node's phase-specific IOR socket.

    links.new(
        _output_socket(group_input, "Absorption Coefficients"),
        _socket(coefficients, "Absorption Coefficients"),
    )
    links.new(
        _output_socket(group_input, "Scatter Coefficients"),
        _socket(coefficients, "Scatter Coefficients"),
    )
    links.new(_output_socket(group_input, "Anisotropy g"), _socket(coefficients, "Anisotropy"))

    glass = nodes.new("ShaderNodeBsdfGlass")
    glass.name = "White Water-Air Glass"
    glass.label = "White Glass / roughness 0"
    glass.location = (-20, 230)
    _set_socket_value(glass, "Color", (1.0, 1.0, 1.0, 1.0))
    _set_socket_value(glass, "Roughness", 0.0)
    # Keep the Glass socket's own value meaningful even though the exposed
    # group control is linked to it.  This makes the effective surface IOR
    # inspectable from either side of the node group.
    _set_socket_value(glass, "IOR", preset["ior"])
    links.new(_output_socket(group_input, "Surface IOR"), _socket(glass, "IOR"))

    group_output = nodes.new("NodeGroupOutput")
    group_output.name = "Water shader outputs"
    group_output.location = (280, 80)
    links.new(_output_socket(glass, "BSDF"), _socket(group_output, "Surface"))
    links.new(_output_socket(coefficients, "Volume"), _socket(group_output, "Volume"))
    nodes.active = coefficients

    _safe_custom_set(group, "is_water_node_group", True)
    _safe_custom_set(group, "native_volume_shader", "ShaderNodeVolumeCoefficients")
    _safe_custom_set(group, "volume_phase", "HENYEY_GREENSTEIN")
    _safe_custom_set(
        group,
        "volume_ior_policy",
        "not applicable for HENYEY_GREENSTEIN; surface IOR belongs to GlassBSDF",
    )
    return group


def _make_material(name: str, group, preset: Mapping, *, connect_surface: bool, created: dict | None = None):
    """Create one material slot from the shared group.

    ``connect_surface=False`` leaves the group's Glass output unused, so that
    side and bottom polygons have a volume-only material.
    """

    blender = _require_bpy()
    material = blender.data.materials.new(name)
    if created is not None:
        created["materials"].append(material)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    controls = nodes.new("ShaderNodeGroup")
    controls.name = "Native Water Controls"
    controls.label = "Native coefficients / surface IOR"
    controls.width = 300
    controls.node_tree = group
    _set_socket_value(controls, "Absorption Coefficients", preset["absorption"])
    _set_socket_value(controls, "Scatter Coefficients", preset["scatter"])
    _set_socket_value(controls, "Anisotropy g", preset["g"])
    _set_socket_value(controls, "Surface IOR", preset["ior"])
    controls.location = (-360, 0)

    output = nodes.new("ShaderNodeOutputMaterial")
    output.name = "Water Material Output"
    output.location = (40, 0)
    links.new(_output_socket(controls, "Volume"), _socket(output, "Volume"))
    if connect_surface:
        links.new(_output_socket(controls, "Surface"), _socket(output, "Surface"))
    nodes.active = controls

    _safe_custom_set(material, "is_water_material", True)
    _safe_custom_set(material, "water_surface_connected", bool(connect_surface))
    _safe_custom_set(material, "water_group_name", group.name)
    return material


def _build_box_mesh(name: str, extent: float, bottom_z: float, surface_z: float, created: dict | None = None):
    """Build a single eight-vertex, outward-oriented closed box."""

    blender = _require_bpy()
    e = float(extent)
    b = float(bottom_z)
    t = float(surface_z)
    vertices = [
        (-e, -e, b),
        (e, -e, b),
        (e, e, b),
        (-e, e, b),
        (-e, -e, t),
        (e, -e, t),
        (e, e, t),
        (-e, e, t),
    ]
    # Bottom -Z, top +Z, then -Y/+X/+Y/-X sides.  The top face is polygon 1.
    faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    mesh = blender.data.meshes.new(name)
    if created is not None:
        created["meshes"].append(mesh)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    return mesh, vertices, faces


def _scene_water_objects(scene):
    try:
        return [obj for obj in scene.objects if bool(obj.get("is_water", False))]
    except AttributeError:
        return []


def _find_group_nodes(handle: Mapping) -> list:
    group = handle.get("node_group")
    materials = handle.get("materials", [])
    if group is None:
        raise ValueError("water handle has no node_group")
    result = []
    seen = set()
    for material in materials:
        tree = getattr(material, "node_tree", None)
        if tree is None:
            continue
        for node in tree.nodes:
            if node.type == "GROUP" and node.node_tree == group and id(node) not in seen:
                result.append(node)
                seen.add(id(node))
    if not result:
        raise ValueError("water handle contains no material control node")
    return result


def _find_group_node_for_material(material, group):
    tree = getattr(material, "node_tree", None)
    if tree is None:
        return None
    for node in tree.nodes:
        if node.type == "GROUP" and node.node_tree == group:
            return node
    return None


def _find_nested_node(group, bl_idname: str):
    for node in group.nodes:
        if node.bl_idname == bl_idname:
            return node
    return None


def _check_handle(handle: Mapping) -> Mapping:
    if not isinstance(handle, Mapping):
        raise ValueError("water handle must be a mapping returned by build_water")
    objects = handle.get("objects")
    if not isinstance(objects, (list, tuple)) or not objects:
        raise ValueError("water handle must contain at least one water object")
    for obj in objects:
        try:
            is_water = bool(obj.get("is_water", False))
        except AttributeError as exc:
            raise ValueError("water handle contains an invalid object") from exc
        if not is_water:
            raise ValueError("water handle contains an object without is_water=True")
    return handle


def _set_preset_on_nodes(nodes: list, preset: Mapping):
    required = set(_GROUP_INPUTS)
    for node in nodes:
        available = {socket.name for socket in node.inputs}
        missing = required - available
        if missing:
            raise RuntimeError(f"native water control node is missing inputs: {sorted(missing)}")


def _snapshot_preset_nodes(nodes: list) -> list[tuple[object, dict]]:
    return [(node, {name: _socket_value(_socket(node, name)) for name in _GROUP_INPUTS}) for node in nodes]


def _values_match(actual, expected, tolerance: float = 1e-7) -> bool:
    if isinstance(actual, list) or isinstance(expected, list):
        if not isinstance(actual, list) or not isinstance(expected, list) or len(actual) != len(expected):
            return False
        return all(_values_match(a, e, tolerance) for a, e in zip(actual, expected))
    try:
        return math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance)
    except (TypeError, ValueError):
        return actual == expected


def _restore_preset_nodes(snapshot) -> list[str]:
    """Restore and verify node values, returning explicit rollback errors."""

    errors = []
    for node, values in snapshot:
        for name, value in values.items():
            try:
                _set_socket_value(node, name, value)
                actual = _socket_value(_socket(node, name))
                if not _values_match(actual, value):
                    errors.append(
                        f"node {getattr(node, 'name', '<unnamed>')} socket {name!r} "
                        f"did not restore (actual={actual!r}, expected={value!r})"
                    )
            except Exception as exc:  # pragma: no cover - Blender-only failure path
                errors.append(
                    f"node {getattr(node, 'name', '<unnamed>')} socket {name!r}: {exc!r}"
                )
    return errors


def _restore_socket(socket, value, label: str, errors: list[str]) -> None:
    """Restore and verify one socket while collecting, never hiding, errors."""

    try:
        socket.default_value = tuple(value) if isinstance(value, list) else value
        actual = _socket_value(socket)
        if not _values_match(actual, value):
            errors.append(f"{label} did not restore (actual={actual!r}, expected={value!r})")
    except Exception as exc:  # pragma: no cover - Blender-only failure path
        errors.append(f"{label}: {exc!r}")


def _restore_custom_value(datablock, key: str, previous, label: str, errors: list[str]) -> None:
    try:
        if previous is None:
            try:
                present = key in datablock
            except TypeError:
                present = datablock.get(key, None) is not None
            if present:
                del datablock[key]
        else:
            _safe_custom_set(datablock, key, previous)
        actual = datablock.get(key, None)
        if actual != previous:
            errors.append(f"{label} did not restore (actual={actual!r}, expected={previous!r})")
    except Exception as exc:  # pragma: no cover - Blender-only failure path
        errors.append(f"{label}: {exc!r}")


def _set_node_preset(node, preset: Mapping) -> None:
    _set_socket_value(node, "Absorption Coefficients", preset["absorption"])
    _set_socket_value(node, "Scatter Coefficients", preset["scatter"])
    _set_socket_value(node, "Anisotropy g", preset["g"])
    _set_socket_value(node, "Surface IOR", preset["ior"])


def build_water(scene, surface_z: float, preset: dict, extent: float = 70.0, bottom_z: float = -3.0) -> dict:
    """Create one persistent finite water volume for ``scene``.

    The full horizontal extent is ``[-extent, extent]`` on both axes.  The
    volume occupies ``bottom_z <= z <= surface_z``.  Invalid inputs are fully
    validated before any Blender collection, node group, mesh, or material is
    created.
    """

    validated_preset = _validate_preset(preset)
    top, horizontal_extent, bottom = _validate_geometry(surface_z, extent, bottom_z)
    if scene is None or not hasattr(scene, "collection"):
        raise ValueError("scene must be a Blender scene")
    blender = _require_bpy()
    existing = _scene_water_objects(scene)
    if existing:
        names = ", ".join(getattr(obj, "name", "<unnamed>") for obj in existing)
        raise ValueError(
            "scene already contains water object(s); enable/disable or update the existing handle "
            f"instead of adding an overlapping volume: {names}"
        )

    identifier = validated_preset["id"]
    created = {
        "collections": [],
        "node_groups": [],
        "meshes": [],
        "objects": [],
        "materials": [],
    }
    try:
        collection = blender.data.collections.new(f"Water Volume {identifier}")
        created["collections"].append(collection)
        scene.collection.children.link(collection)
        collection.hide_render = False
        _safe_custom_set(collection, "is_water_collection", True)
        _safe_custom_set(collection, "water_schema", WATER_SCHEMA_VERSION)

        group = _build_native_group(
            f"Native Water Controls {identifier}", validated_preset, created=created
        )
        mesh, vertices, faces = _build_box_mesh(
            f"Closed Water Boundary {identifier}",
            horizontal_extent,
            bottom,
            top,
            created=created,
        )
        obj = blender.data.objects.new(f"Water Volume {identifier}", mesh)
        created["objects"].append(obj)
        collection.objects.link(obj)
        obj.hide_render = False
        obj.location = (0.0, 0.0, 0.0)

        top_material = _make_material(
            f"Water Top Glass and Volume {identifier}",
            group,
            validated_preset,
            connect_surface=True,
            created=created,
        )
        boundary_material = _make_material(
            f"Water Side Bottom Volume {identifier}",
            group,
            validated_preset,
            connect_surface=False,
            created=created,
        )
        mesh.materials.append(top_material)
        mesh.materials.append(boundary_material)
        for polygon in mesh.polygons:
            polygon.material_index = 0 if polygon.index == 1 else 1

        # Custom properties make the saved .blend self-describing without
        # adding semantic/instance labels that belong to ecology/labels.
        _safe_custom_set(obj, "is_water", True)
        _safe_custom_set(obj, "water_role", "one_closed_finite_volume")
        _safe_custom_set(obj, "water_preset_id", identifier)
        _safe_custom_set(obj, "surface_z", top)
        _safe_custom_set(obj, "bottom_z", bottom)
        _safe_custom_set(obj, "extent", horizontal_extent)
        _safe_custom_set(obj, "top_polygon_index", 1)
        _safe_custom_set(
            obj,
            "water_face_roles_json",
            {
                "bottom": [0],
                "top": [1],
                "side_y_min": [2],
                "side_x_max": [3],
                "side_y_max": [4],
                "side_x_min": [5],
            },
        )
        _safe_custom_set(
            obj,
            "material_roles_json",
            {"top": top_material.name, "side_bottom": boundary_material.name},
        )
        _safe_custom_set(obj, "ecology_labels_present", False)
        _safe_custom_set(obj, "native_shader_only", True)
        _safe_custom_set(obj, "volume_coefficients_units", "m^-1")
        _safe_custom_set(obj, "top_normal_policy", "outward +Z")

        _safe_custom_set(collection, "water_preset_id", identifier)
        _safe_custom_set(collection, "object_name", obj.name)

        handle = {
            "schema_version": WATER_SCHEMA_VERSION,
            "is_water": True,
            "scene": scene,
            "collection": collection,
            "objects": [obj],
            "mesh": mesh,
            "materials": [top_material, boundary_material],
            "top_material": top_material,
            "boundary_material": boundary_material,
            "node_group": group,
            "surface_z": top,
            "bottom_z": bottom,
            "extent": horizontal_extent,
            "preset": dict(validated_preset),
        }
        return handle
    except Exception as exc:
        cleanup_errors = _cleanup_created_datablocks(created)
        if cleanup_errors:
            note = "native water build cleanup encountered errors: " + "; ".join(cleanup_errors)
            try:
                exc.add_note(note)
            except AttributeError:  # pragma: no cover - old Blender Python fallback
                pass
        raise


def set_water_enabled(handle, enabled: bool):
    """Hide or reveal every water object represented by ``handle``."""

    checked = _check_handle(handle)
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a bool")
    objects = list(checked["objects"])
    collection = checked.get("collection")
    # Validate all targets before changing any state.
    for obj in objects:
        if not hasattr(obj, "hide_render"):
            raise ValueError("water handle contains an object without hide_render")
    if collection is not None and not hasattr(collection, "hide_render"):
        raise ValueError("water handle contains an invalid collection")

    for obj in objects:
        obj.hide_render = not enabled
    if collection is not None:
        collection.hide_render = not enabled
    return checked


def set_water_preset(handle, preset: dict):
    """Atomically update all native control sockets in a water handle.

    Validation and socket-shape checks happen before the first assignment.  A
    rollback snapshot also protects against a Blender socket assignment error,
    leaving a previous valid preset intact.
    """

    checked = _check_handle(handle)
    validated_preset = _validate_preset(preset)
    nodes = _find_group_nodes(checked)
    _set_preset_on_nodes(nodes, validated_preset)
    group = checked.get("node_group")
    glass = _find_nested_node(group, "ShaderNodeBsdfGlass")
    if glass is None:
        raise RuntimeError("water node group lacks the native GlassBSDF node")
    glass_ior_socket = _socket(glass, "IOR")
    snapshot = _snapshot_preset_nodes(nodes)
    old_glass_ior = _socket_value(glass_ior_socket)
    old_ids = []
    for obj in checked["objects"]:
        old_ids.append((obj, obj.get("water_preset_id", None)))
    collection = checked.get("collection")
    old_collection_id = collection.get("water_preset_id", None) if collection is not None else None
    try:
        for node in nodes:
            _set_node_preset(node, validated_preset)
        # Keep the Glass socket fallback synchronized with the exposed group
        # value.  The link remains in place, so the group socket is still the
        # effective control and the fallback is useful to an inspector.
        _set_socket_value(glass, "IOR", validated_preset["ior"])
        for obj in checked["objects"]:
            _safe_custom_set(obj, "water_preset_id", validated_preset["id"])
        if collection is not None:
            _safe_custom_set(collection, "water_preset_id", validated_preset["id"])
        # Keep the convenience field current for callers; metadata reads the
        # native sockets again and does not trust this cached dictionary.
        checked["preset"] = dict(validated_preset)
    except Exception as exc:
        rollback_errors = _restore_preset_nodes(snapshot)
        _restore_socket(glass_ior_socket, old_glass_ior, "Glass IOR", rollback_errors)
        for obj, previous in old_ids:
            _restore_custom_value(obj, "water_preset_id", previous, f"{obj.name}.water_preset_id", rollback_errors)
        if collection is not None:
            _restore_custom_value(
                collection,
                "water_preset_id",
                old_collection_id,
                "collection.water_preset_id",
                rollback_errors,
            )
        if rollback_errors:
            detail = "; ".join(rollback_errors)
            raise RuntimeError(
                "native water preset update failed and rollback was incomplete: " + detail
            ) from exc
        raise
    return checked


def _world_coordinate(obj, vertex):
    coordinate = vertex.co
    matrix = getattr(obj, "matrix_world", None)
    if matrix is not None:
        try:
            coordinate = matrix @ coordinate
        except TypeError:
            pass
    return [float(coordinate[index]) for index in range(3)]


def _world_normal(obj, polygon):
    normal = polygon.normal
    matrix = getattr(obj, "matrix_world", None)
    if matrix is not None:
        try:
            normal = matrix.to_3x3() @ normal
        except (AttributeError, TypeError):
            pass
    try:
        length = float(normal.length)
    except AttributeError:
        length = math.sqrt(sum(float(normal[index]) ** 2 for index in range(3)))
    if length <= 0.0:
        return [0.0, 0.0, 0.0]
    return [float(normal[index]) / length for index in range(3)]


def _geometry_metadata(obj):
    mesh = obj.data
    vertices = [_world_coordinate(obj, vertex) for vertex in mesh.vertices]
    bounds = []
    if vertices:
        bounds = [[min(point[index] for point in vertices), max(point[index] for point in vertices)] for index in range(3)]

    edge_counts = {}
    for polygon in mesh.polygons:
        indices = list(polygon.vertices)
        for first, second in zip(indices, indices[1:] + indices[:1]):
            edge = tuple(sorted((int(first), int(second))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    closed = bool(edge_counts) and all(count == 2 for count in edge_counts.values())
    roles = _custom_json(obj, "water_face_roles_json", {})
    role_normals = {}
    for role, polygon_indices in roles.items():
        role_normals[role] = [_world_normal(obj, mesh.polygons[int(index)]) for index in polygon_indices]

    slot_names = []
    for slot in mesh.materials:
        slot_names.append(getattr(slot, "name", ""))
    material_indices = {str(int(poly.index)): int(poly.material_index) for poly in mesh.polygons}
    has_semantic = False
    has_instance = False
    try:
        keys = set(obj.keys())
        has_semantic = "semantic_id" in keys
        has_instance = "instance_id" in keys
    except AttributeError:
        pass
    top_index = int(obj.get("top_polygon_index", 1))
    top_normal = role_normals.get("top", [[0.0, 0.0, 0.0]])[0]
    return {
        "name": getattr(obj, "name", ""),
        "type": getattr(obj, "type", "MESH"),
        "is_water": bool(obj.get("is_water", False)),
        "hide_render": bool(getattr(obj, "hide_render", False)),
        "vertex_count": len(mesh.vertices),
        "polygon_count": len(mesh.polygons),
        "edge_count": len(edge_counts),
        "closed_surface": closed,
        "bounds_m": bounds,
        "face_roles": roles,
        "role_normals": role_normals,
        "top_polygon_index": top_index,
        "top_outward_normal": top_normal,
        "material_slots": slot_names,
        "material_indices": material_indices,
        "has_semantic_id": has_semantic,
        "has_instance_id": has_instance,
    }


def _cleanup_created_datablocks(created: dict) -> list[str]:
    """Remove only datablocks created by one failed build transaction."""

    blender = _require_bpy()
    errors = []

    def remove_many(container, key: str, **kwargs):
        for datablock in reversed(created.get(key, [])):
            try:
                if datablock.name in container:
                    container.remove(datablock, **kwargs)
            except Exception as exc:  # pragma: no cover - Blender-only cleanup path
                errors.append(f"{key}:{getattr(datablock, 'name', '<unnamed>')}: {exc!r}")

    # Unlink objects before meshes/materials, then remove the collection last.
    remove_many(blender.data.objects, "objects", do_unlink=True)
    remove_many(blender.data.materials, "materials", do_unlink=True)
    remove_many(blender.data.meshes, "meshes", do_unlink=True)
    remove_many(blender.data.node_groups, "node_groups", do_unlink=True)
    remove_many(blender.data.collections, "collections", do_unlink=True)
    return errors


def _link_metadata(socket):
    links = list(getattr(socket, "links", []))
    if not links:
        return {"linked": False, "from_node": None, "from_socket": None}
    link = links[0]
    return {
        "linked": True,
        "from_node": getattr(getattr(link, "from_node", None), "name", None),
        "from_socket": getattr(getattr(link, "from_socket", None), "name", None),
    }


def _interface_metadata(group):
    result = []
    interface = getattr(group, "interface", None)
    items = getattr(interface, "items_tree", []) if interface is not None else []
    for item in items:
        if getattr(item, "item_type", None) != "SOCKET":
            continue
        entry = {
            "name": getattr(item, "name", ""),
            "identifier": getattr(item, "identifier", ""),
            "in_out": getattr(item, "in_out", ""),
            "socket_type": getattr(item, "socket_type", ""),
        }
        if hasattr(item, "default_value"):
            try:
                entry["default_value"] = _socket_value(item)
            except (AttributeError, TypeError, ValueError):
                pass
        result.append(entry)
    return result


def _material_metadata(material, group):
    node = _find_group_node_for_material(material, group)
    tree = getattr(material, "node_tree", None)
    output = None
    if tree is not None:
        output = next((item for item in tree.nodes if item.bl_idname == "ShaderNodeOutputMaterial"), None)
    if node is None:
        return {
            "name": getattr(material, "name", ""),
            "surface_connected": False,
            "volume_connected": False,
            "control_inputs": {},
        }
    control_inputs = {}
    for name in _GROUP_INPUTS:
        control_inputs[name] = _socket_value(_socket(node, name))
    return {
        "name": getattr(material, "name", ""),
        "surface_connected": bool(output and _socket(output, "Surface").is_linked),
        "volume_connected": bool(output and _socket(output, "Volume").is_linked),
        "control_node": getattr(node, "name", ""),
        "control_inputs": control_inputs,
    }


def _count_drivers(datablocks):
    count = 0
    seen = set()
    for datablock in datablocks:
        if datablock is None or id(datablock) in seen:
            continue
        seen.add(id(datablock))
        animation = getattr(datablock, "animation_data", None)
        if animation is None:
            continue
        count += len(getattr(animation, "drivers", []))
    return count


def water_metadata(handle) -> dict:
    """Read serializable geometry and native-node state from a water handle."""

    checked = _check_handle(handle)
    group = checked.get("node_group")
    if group is None:
        raise ValueError("water handle has no node_group")
    materials = list(checked.get("materials", []))
    nodes = _find_group_nodes(checked)
    volume = _find_nested_node(group, "ShaderNodeVolumeCoefficients")
    glass = _find_nested_node(group, "ShaderNodeBsdfGlass")
    if volume is None or glass is None:
        raise RuntimeError("water node group lacks the native volume or GlassBSDF node")

    control_input_sets = []
    for node in nodes:
        control_input_sets.append({name: _socket_value(_socket(node, name)) for name in _GROUP_INPUTS})
    primary_inputs = control_input_sets[0]
    actual_preset_id = ""
    try:
        actual_preset_id = str(checked["objects"][0].get("water_preset_id", ""))
    except (AttributeError, KeyError):
        pass

    # The coefficient and anisotropy sockets are intentionally linked from
    # the material's group node.  Their own defaults are therefore only the
    # node fallback; report both the effective group values and those raw
    # defaults so metadata remains truthful after set_water_preset().
    coefficient_inputs = {
        "absorption": list(primary_inputs["Absorption Coefficients"]),
        "scatter": list(primary_inputs["Scatter Coefficients"]),
        "anisotropy": float(primary_inputs["Anisotropy g"]),
        "phase_ior_applicable": False,
        "ior": None,
        "native_socket_defaults": {
            "absorption": _socket_value(_socket(volume, "Absorption Coefficients")),
            "scatter": _socket_value(_socket(volume, "Scatter Coefficients")),
            "anisotropy": float(_socket(volume, "Anisotropy").default_value),
        },
    }
    direct_links = {
        "absorption": _link_metadata(_socket(volume, "Absorption Coefficients")),
        "scatter": _link_metadata(_socket(volume, "Scatter Coefficients")),
        "anisotropy": _link_metadata(_socket(volume, "Anisotropy")),
        "phase_ior_applicable": False,
        "ior": None,
    }
    surface_ior_socket = _socket(glass, "IOR")
    surface = {
        "node_type": glass.bl_idname,
        "color": _socket_value(_socket(glass, "Color")),
        "roughness": float(_socket(glass, "Roughness").default_value),
        "ior": float(primary_inputs["Surface IOR"]),
        "glass_socket_default_ior": float(surface_ior_socket.default_value),
        "ior_link": _link_metadata(surface_ior_socket),
    }
    geometry = [_geometry_metadata(obj) for obj in checked["objects"]]
    top_material = checked.get("top_material")
    boundary_material = checked.get("boundary_material")
    material_records = []
    for material in materials:
        material_records.append(_material_metadata(material, group))

    object_enabled = all(not bool(getattr(obj, "hide_render", False)) for obj in checked["objects"])
    collection = checked.get("collection")
    collection_enabled = bool(collection is None or not getattr(collection, "hide_render", False))
    all_enabled = object_enabled and collection_enabled
    bounds = geometry[0]["bounds_m"] if geometry else []
    roles = geometry[0].get("face_roles", {}) if geometry else {}
    top_role = roles.get("top", [])
    side_roles = [role for role in roles if role.startswith("side_")]
    volume_vectors_match = all(
        record["Absorption Coefficients"] == primary_inputs["Absorption Coefficients"]
        and record["Scatter Coefficients"] == primary_inputs["Scatter Coefficients"]
        and record["Anisotropy g"] == primary_inputs["Anisotropy g"]
        for record in control_input_sets
    )

    # The metadata is deliberately composed from current datablocks and node
    # sockets.  ``handle['preset']`` is only a convenience cache for callers.
    result = {
        "schema_version": WATER_SCHEMA_VERSION,
        "is_water": True,
        "enabled": all_enabled,
        "preset": {
            "id": actual_preset_id,
            "absorption": list(primary_inputs["Absorption Coefficients"]),
            "scatter": list(primary_inputs["Scatter Coefficients"]),
            "g": float(primary_inputs["Anisotropy g"]),
            "ior": float(surface["ior"]),
        },
        "units": {
            "geometry": "m (1 Blender unit = 1 m by project contract)",
            "absorption": "m^-1",
            "scatter": "m^-1",
        },
        "boundary": {
            "bounds_m": bounds,
            "surface_z": bounds[2][1] if len(bounds) > 2 else None,
            "bottom_z": bounds[2][0] if len(bounds) > 2 else None,
            "extent_x": bounds[0] if len(bounds) > 0 else [],
            "extent_y": bounds[1] if len(bounds) > 1 else [],
            "closed": all(item.get("closed_surface", False) for item in geometry),
            "object_count": len(geometry),
            "volume_object_count": len(geometry),
            "top_polygon_count": len(top_role),
            "top_outward_normal": geometry[0].get("top_outward_normal", []) if geometry else [],
            "side_roles_volume_only": side_roles,
            "top_and_sides_single_mesh": len(geometry) == 1,
            "material_slots_share_volume_coefficients": volume_vectors_match,
            "horizontal_extent": [
                bounds[0] if len(bounds) > 0 else [],
                bounds[1] if len(bounds) > 1 else [],
            ],
            "vertical_extent": bounds[2] if len(bounds) > 2 else [],
        },
        "objects": geometry,
        "materials": material_records,
        "nodes": {
            "node_group": {
                "name": getattr(group, "name", ""),
                "interface": _interface_metadata(group),
                "control_node_count": len(nodes),
                "interface_defaults_scope": (
                    "Initial node-group socket defaults only; effective preset values are read from "
                    "the outer material control node inputs."
                ),
            },
            "volume": {
                "node_type": volume.bl_idname,
                "phase": getattr(volume, "phase", ""),
                "inputs": coefficient_inputs,
                "direct_input_links": direct_links,
                "phase_ior_applicable": False,
                "ior": None,
                "ior_policy": "not applicable for HENYEY_GREENSTEIN; preset IOR belongs only to GlassBSDF",
            },
            "surface": surface,
            "top_surface_connected": bool(top_material and _material_metadata(top_material, group)["surface_connected"]),
            "top_volume_connected": bool(top_material and _material_metadata(top_material, group)["volume_connected"]),
            "side_bottom_surface_connected": bool(
                boundary_material and _material_metadata(boundary_material, group)["surface_connected"]
            ),
            "side_bottom_volume_connected": bool(
                boundary_material and _material_metadata(boundary_material, group)["volume_connected"]
            ),
            "material_slot_volume_inputs_match": volume_vectors_match,
        },
        "integrity": {
            "all_objects_marked_is_water": all(item.get("is_water", False) for item in geometry),
            "ecology_labels_present": any(item.get("has_semantic_id", False) or item.get("has_instance_id", False) for item in geometry),
            "custom_driver_count": _count_drivers([group, *materials, *checked["objects"]]),
            "registered_ui": False,
            "single_non_overlapping_volume": len(geometry) == 1 and all(item.get("closed_surface", False) for item in geometry),
        },
        "physical_limitations": [
            "Homogeneous isotropic RGB absorption/scatter in a finite rectangular volume; coefficients are synthetic configuration values unless separately calibrated.",
            "The top is a static planar water-air GlassBSDF interface. Waves, foam, bubbles, salinity variation, and wavelength-dependent dispersion are outside this module.",
            "Side and bottom boundaries carry volume only, so they do not add fictitious glass walls. Rays that reach the finite boundary are truncated by this modeled domain.",
            "The public preset IOR is read from the white zero-roughness GlassBSDF surface. The HENYEY_GREENSTEIN volume phase does not use a volume IOR socket.",
            "The primary camera should remain inside the water volume for the intended underwater view; interface experiments require a separate explicitly configured scene.",
        ],
    }
    # Convert nested IDPropertyGroup/bpy arrays before callers serialize the
    # metadata into sample metadata.json.
    return _json_compatible(result)


__all__ = [
    "build_water",
    "set_water_enabled",
    "set_water_preset",
    "water_metadata",
]
