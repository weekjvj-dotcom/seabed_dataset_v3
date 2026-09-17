"""Deterministic, native Blender lighting rigs for the v2 seabed scenes.

The sampler in this module deliberately has no Blender dependency.  It makes
one immutable lighting plan for a geometry state and an attempt.  The Blender
builder consumes that plan and creates independent World and Light datablocks
for one of ``natural``, ``artificial`` or ``mixed``.  In particular, the
artificial lamps are real ``SPOT`` lights; this module does not add an emitter
mesh, a photograph, a Light Path transparency trick, or an RGB post-filter.

The native temperature property was added in different Blender releases under
slightly different capability sets.  We therefore inspect the running Blender
RNA at build time.  When both ``use_temperature`` and ``temperature`` exist,
they are used and read back.  If that native pair is unavailable, the fallback
is an inspectable native Blackbody -> Emission -> Light Output node graph.  A
fallback is never silently represented as a native temperature property in the
metadata.
"""

from __future__ import annotations

import json
import math
import operator
import random
from collections.abc import Mapping
from typing import Any

from .contracts import derive_seed


MODES = ("natural", "artificial", "mixed")
DEFAULT_ROOT_SEED = 72026
DEFAULT_SAMPLING = {
    "sky_strength": (0.08, 0.35),
    "sun_elevation_deg": (20.0, 75.0),
    "sun_rotation_deg": (0.0, 360.0),
    "spot_energy": (100.0, 600.0),
    "spot_full_angle_deg": (35.0, 70.0),
    "spot_blend": (0.3, 0.7),
    "temperature_k": (4500.0, 6500.0),
}
DEFAULT_LAMP_POSITION = {
    "LeftSpot": (-0.30, 0.10, -0.15),
    "RightSpot": (0.30, 0.10, -0.15),
}
TARGET_LOCAL = (0.0, 0.0, -4.0)


def _finite_number(value: Any, name: str) -> float:
    """Return a finite float while rejecting booleans and text values."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    """Validate a stable state/attempt index without coercing arbitrary text."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(result)


def _as_range(value: Any, name: str, default: tuple[float, float]) -> tuple[float, float]:
    """Read a scalar or a two-element inclusive sampling range.

    A scalar is accepted as a useful calibration override meaning a frozen
    value.  The public configuration normally uses ``[low, high]``.
    """

    if value is None:
        value = default
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number or a two-element range")
    if isinstance(value, (int, float)):
        result = _finite_number(value, name)
        return result, result
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a number or a two-element range")
    lo = _finite_number(value[0], f"{name}[0]")
    hi = _finite_number(value[1], f"{name}[1]")
    if lo > hi:
        raise ValueError(f"{name} lower bound exceeds upper bound")
    return lo, hi


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def _derive_seed(root_seed: int, *parts: Any) -> int:
    """Call the project's single SHA-derived seed helper."""

    result = derive_seed(root_seed, *parts)
    if isinstance(result, bool) or not isinstance(result, int):
        raise ValueError("contracts.derive_seed must return an integer")
    return int(result)


def _sampling_config(cfg: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    if not isinstance(cfg, Mapping):
        raise TypeError("cfg must be a mapping")
    supplied = cfg.get("lighting_sampling", {})
    if supplied is None:
        supplied = {}
    if not isinstance(supplied, Mapping):
        raise ValueError("lighting_sampling must be a mapping")
    result = {
        key: _as_range(supplied.get(key), key, default)
        for key, default in DEFAULT_SAMPLING.items()
    }

    lo, hi = result["sky_strength"]
    if lo < 0:
        raise ValueError("sky_strength must be non-negative")
    lo, hi = result["sun_elevation_deg"]
    if lo < 1 or hi > 89:
        raise ValueError("sun_elevation_deg must be between 1 and 89 degrees")
    lo, hi = result["sun_rotation_deg"]
    if lo < -360 or hi > 360:
        raise ValueError("sun_rotation_deg must be between -360 and 360 degrees")
    lo, hi = result["spot_energy"]
    if lo < 0 or hi > 5000:
        raise ValueError("spot_energy must be between 0 and 5000 BlenderW")
    lo, hi = result["spot_full_angle_deg"]
    if lo < 1 or hi > 179:
        raise ValueError("spot_full_angle_deg must be between 1 and 179 degrees")
    lo, hi = result["spot_blend"]
    if lo < 0 or hi > 1:
        raise ValueError("spot_blend must be between 0 and 1")
    lo, hi = result["temperature_k"]
    if lo < 1000 or hi > 20000:
        raise ValueError("temperature_k must be between 1000 K and 20000 K")
    return result


def _uniform(rng: random.Random, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return lo if lo == hi else rng.uniform(lo, hi)


def _lamp_energy_pair(rng: random.Random, bounds: tuple[float, float]) -> tuple[float, float, float]:
    """Sample a base energy and a small, bounded left/right asymmetry."""

    base = _uniform(rng, bounds)
    if bounds[0] == bounds[1]:
        return base, base, base
    left = _clamp(base * (1.0 + rng.uniform(-0.035, 0.035)), *bounds)
    right = _clamp(base * (1.0 + rng.uniform(-0.035, 0.035)), *bounds)
    return base, left, right


def sample_lighting(cfg: Mapping[str, Any], state_index: int, attempt_index: int = 0) -> dict[str, Any]:
    """Sample one deterministic natural-sky and dual-lamp lighting plan.

    The returned JSON plan contains both resource families.  A caller should
    sample once and pass this same plan to all three mode builds.  This makes
    Natural and Mixed share the exact sky and Artificial and Mixed share the
    exact lamp candidates.  The seed streams are derived from root, layout,
    state, and attempt; no earlier call can move a later state's stream.
    """

    state_index = _nonnegative_int(state_index, "state_index")
    attempt_index = _nonnegative_int(attempt_index, "attempt_index")
    if not isinstance(cfg, Mapping):
        raise TypeError("cfg must be a mapping")

    root_value = cfg.get("root_seed", DEFAULT_ROOT_SEED)
    root_seed = _nonnegative_int(root_value, "root_seed")
    layout_value = cfg.get("_layout_seed")
    if layout_value is None:
        layouts = cfg.get("layouts")
        if not isinstance(layouts, (list, tuple)) or not layouts:
            raise ValueError("cfg.layouts is required when _layout_seed is absent")
        first_layout = layouts[0]
        if not isinstance(first_layout, Mapping) or "seed" not in first_layout:
            raise ValueError("cfg.layouts[0].seed is required when _layout_seed is absent")
        layout_value = first_layout["seed"]
    layout_seed = _nonnegative_int(layout_value, "layout_seed")
    bounds = _sampling_config(cfg)

    lighting_seed = _derive_seed(root_seed, "lighting", layout_seed, state_index, attempt_index)
    sky_seed = _derive_seed(root_seed, "lighting", layout_seed, "sky", state_index, attempt_index)
    lamps_seed = _derive_seed(root_seed, "lighting", layout_seed, "lamps", state_index, attempt_index)
    direction_seed = _derive_seed(root_seed, "lighting", layout_seed, "lamp_direction", state_index, attempt_index)
    sky_rng = random.Random(sky_seed)
    lamps_rng = random.Random(lamps_seed)
    direction_rng = random.Random(direction_seed)

    sky_strength = _uniform(sky_rng, bounds["sky_strength"])
    sun_elevation = _uniform(sky_rng, bounds["sun_elevation_deg"])
    sun_rotation = _uniform(sky_rng, bounds["sun_rotation_deg"])
    base_energy, left_energy, right_energy = _lamp_energy_pair(lamps_rng, bounds["spot_energy"])
    cone_deg = _uniform(lamps_rng, bounds["spot_full_angle_deg"])
    blend = _uniform(lamps_rng, bounds["spot_blend"])
    temperature = _uniform(lamps_rng, bounds["temperature_k"])

    lamps = []
    for name, energy in (("LeftSpot", left_energy), ("RightSpot", right_energy)):
        pos = DEFAULT_LAMP_POSITION[name]
        # Direction jitter is intentionally small and in the camera's local
        # frame.  The lamp positions themselves remain at the approved anchors.
        target = (
            TARGET_LOCAL[0] + direction_rng.uniform(-0.10, 0.10),
            TARGET_LOCAL[1] + direction_rng.uniform(-0.08, 0.08),
            TARGET_LOCAL[2] + direction_rng.uniform(-0.12, 0.12),
        )
        lamps.append(
            {
                "name": name,
                "position_local": [float(v) for v in pos],
                "target_local": [float(v) for v in target],
                "energy_w": float(energy),
                "spot_full_angle_deg": float(cone_deg),
                "spot_size_rad": float(math.radians(cone_deg)),
                "spot_blend": float(blend),
                "temperature_k": float(temperature),
            }
        )

    seed_streams = {
        "lighting": int(lighting_seed),
        "sky": int(sky_seed),
        "lamps": int(lamps_seed),
        "lamp_direction": int(direction_seed),
    }
    plan = {
        "schema_version": 2,
        "root_seed": root_seed,
        "layout_seed": layout_seed,
        "state_index": state_index,
        "attempt_index": attempt_index,
        "seed": int(lighting_seed),
        "seed_streams": seed_streams,
        "sky": {
            "type": "MULTIPLE_SCATTERING_SKY",
            "sky_type": "MULTIPLE_SCATTERING",
            "sun_disc": False,
            "strength": float(sky_strength),
            "sun_elevation_deg": float(sun_elevation),
            "sun_rotation_deg": float(sun_rotation),
        },
        "lamps": lamps,
        # Flat aliases make plans easy to inspect and preserve compatibility
        # with small calibration scripts that read one field at a time.
        "sky_strength": float(sky_strength),
        "sun_elevation_deg": float(sun_elevation),
        "sun_rotation_deg": float(sun_rotation),
        "spot_energy": float(base_energy),
        "spot_full_angle_deg": float(cone_deg),
        "spot_blend": float(blend),
        "temperature_k": float(temperature),
        "sampling": {key: [float(lo), float(hi)] for key, (lo, hi) in bounds.items()},
    }
    # Fail here rather than letting a non-finite calibration value reach a
    # saved manifest.  This also documents that the result is JSON-safe.
    json.dumps(plan, ensure_ascii=False, allow_nan=False, sort_keys=True)
    return plan


def _bpy_module():
    try:
        import bpy  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised outside Blender
        raise RuntimeError("build_lighting requires Blender's bpy module") from exc
    return bpy


def _mathutils_types():
    try:
        from mathutils import Matrix, Vector  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised outside Blender
        raise RuntimeError("build_lighting requires Blender's mathutils module") from exc
    return Matrix, Vector


def _safe_idprop(obj: Any, key: str, value: Any) -> None:
    try:
        obj[key] = value
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
        # Metadata still reads the native RNA fields.  Custom properties are
        # convenience provenance and must not make a valid Blender rig fail.
        return


def _input_socket(node: Any, name: str) -> Any:
    inputs = getattr(node, "inputs", None)
    if inputs is None:
        raise RuntimeError(f"Node {getattr(node, 'name', '<unnamed>')} has no inputs")
    try:
        return inputs[name]
    except (KeyError, IndexError, TypeError):
        for socket in inputs:
            if getattr(socket, "name", None) == name or getattr(socket, "identifier", None) == name:
                return socket
    raise RuntimeError(f"Required node input {name!r} is unavailable")


def _set_required(obj: Any, attr: str, value: Any, description: str) -> None:
    if not hasattr(obj, attr):
        raise RuntimeError(f"Blender native API lacks required {description} field {attr!r}")
    try:
        setattr(obj, attr, value)
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        raise RuntimeError(f"Could not set native {description} field {attr!r}: {exc}") from exc


def _set_if_present(obj: Any, attr: str, value: Any) -> bool:
    if not hasattr(obj, attr):
        return False
    try:
        setattr(obj, attr, value)
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return False
    return True


def _native_update() -> None:
    """Ask Blender to evaluate changed RNA before reading effective values."""

    try:
        import bpy  # type: ignore

        view_layer = getattr(getattr(bpy, "context", None), "view_layer", None)
        if view_layer is not None and hasattr(view_layer, "update"):
            view_layer.update()
    except (ImportError, AttributeError, RuntimeError):
        return


def _blackbody_rgb_linear(temperature_k: float) -> list[float]:
    """Approximate linear RGB emitted by Blender's native Blackbody node.

    This is provenance for the fallback metadata only.  The render still uses
    the native Blackbody shader.  The approximation follows the common CCT to
    sRGB curve and converts it to linear Rec.709 values for a useful readback.
    """

    t = _clamp(float(temperature_k), 1000.0, 40000.0) / 100.0
    if t <= 66.0:
        red = 255.0
        green = 99.4708025861 * math.log(max(t, 1e-9)) - 161.1195681661
        blue = 0.0 if t <= 19.0 else 138.5177312231 * math.log(t - 10.0) - 305.0447927307
    else:
        red = 329.698727446 * ((t - 60.0) ** -0.1332047592)
        green = 288.1221695283 * ((t - 60.0) ** -0.0755148492)
        blue = 255.0

    srgb = [_clamp(v / 255.0, 0.0, 1.0) for v in (red, green, blue)]
    return [
        float(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4)
        for v in srgb
    ]


def _rgb3(values: Any, name: str) -> list[float]:
    """Validate and return exactly three finite RGB channels."""

    try:
        channels = [float(v) for v in tuple(values)]
        if len(channels) < 3:
            raise ValueError(f"{name} has fewer than three channels")
        channels = channels[:3]
        if not all(math.isfinite(v) for v in channels):
            raise ValueError(f"{name} contains a non-finite channel")
        return channels
    except (TypeError, IndexError, ValueError, RuntimeError) as exc:
        raise RuntimeError(f"Could not read valid RGB channels from {name}") from exc


def _read_color(light_data: Any, attr: str = "color") -> list[float]:
    """Read a Blender color sequence and fail explicitly when it is unreadable."""

    color = getattr(light_data, attr, None)
    return _rgb3(color, f"Light.{attr}")


def _configure_temperature(light_data: Any, temperature_k: float) -> tuple[str, list[float], dict[str, Any]]:
    """Configure a native temperature path and return source/readback data."""

    temperature_k = _finite_number(temperature_k, "temperature_k")
    fallback_rgb = _blackbody_rgb_linear(temperature_k)
    native_fields = {
        "use_temperature": hasattr(light_data, "use_temperature"),
        "temperature": hasattr(light_data, "temperature"),
    }
    if all(native_fields.values()):
        try:
            setattr(light_data, "temperature", temperature_k)
            setattr(light_data, "use_temperature", True)
            _native_update()
            actual_enabled = bool(getattr(light_data, "use_temperature"))
            actual_temperature = float(getattr(light_data, "temperature"))
            if not actual_enabled or not math.isfinite(actual_temperature):
                raise RuntimeError("Blender did not retain the native temperature value")
            base_color = _read_color(light_data, "color")
            # Blender 5.2.1 exposes this computed native output separately;
            # ``color`` remains the editable base RGB and stays white in the
            # runtime probe for both 4500 K and 6500 K.
            if not hasattr(light_data, "temperature_color"):
                raise RuntimeError("Blender native temperature_color readback is unavailable")
            effective = _read_color(light_data, "temperature_color")
            return (
                "native_light_temperature",
                effective,
                {
                    "api": native_fields,
                    "use_temperature": actual_enabled,
                    "temperature_k": actual_temperature,
                    "base_color": base_color,
                    "temperature_color": effective,
                    "effective_rgb_source": "Light.temperature_color",
                },
            )
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            # A present native API that cannot be configured or read back must
            # fail explicitly.  Falling through would leave use_temperature
            # active while metadata described an unrelated estimate.
            raise RuntimeError(f"Native Light temperature configuration/readback failed: {exc}") from exc

    # A native light node graph is an actual render path and remains inspectable
    # after the .blend is opened without this Python module.
    if not hasattr(light_data, "use_nodes") or not hasattr(light_data, "node_tree"):
        raise RuntimeError("This Blender build exposes neither a usable native light temperature property nor a native light node graph")
    try:
        light_data.use_nodes = True
        tree = light_data.node_tree
        if tree is None:
            raise RuntimeError("light node_tree is None after enabling use_nodes")
        nodes = tree.nodes
        links = tree.links
        nodes.clear()
        blackbody = nodes.new("ShaderNodeBlackbody")
        blackbody.name = "Temperature Blackbody"
        _input_socket(blackbody, "Temperature").default_value = temperature_k
        emission = nodes.new("ShaderNodeEmission")
        emission.name = "Temperature Emission"
        strength = next((s for s in emission.inputs if getattr(s, "name", None) == "Strength"), None)
        if strength is not None:
            strength.default_value = 1.0
        output = nodes.new("ShaderNodeOutputLight")
        output.name = "Light Output"
        links.new(blackbody.outputs[0], _input_socket(emission, "Color"))
        links.new(emission.outputs[0], _input_socket(output, "Surface"))
        try:
            base_color = _read_color(light_data, "color")
            base_color_source = "Light.color"
        except RuntimeError:
            base_color = None
            base_color_source = "unavailable"
        effective = _rgb3(fallback_rgb, "Blackbody estimate")
        return (
            "native_blackbody_shader",
            effective,
            {
                "api": native_fields,
                "use_temperature": False,
                "temperature_k": temperature_k,
                "base_color": base_color,
                "base_color_source": base_color_source,
                "temperature_color": effective,
                "effective_rgb_source": "native_blackbody_shader_output_estimate",
            },
        )
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise RuntimeError(f"Could not construct native Blackbody light shader: {exc}") from exc


def _plan_sky(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    sky = plan.get("sky")
    if isinstance(sky, Mapping):
        return sky
    required = ("sky_strength", "sun_elevation_deg", "sun_rotation_deg")
    if not all(key in plan for key in required):
        raise ValueError("lighting_plan lacks sky parameters")
    return {
        "type": "MULTIPLE_SCATTERING_SKY",
        "sky_type": "MULTIPLE_SCATTERING",
        "sun_disc": False,
        "strength": plan["sky_strength"],
        "sun_elevation_deg": plan["sun_elevation_deg"],
        "sun_rotation_deg": plan["sun_rotation_deg"],
    }


def _plan_lamps(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    lamps = plan.get("lamps")
    if lamps is None:
        lamps = [plan.get("left_lamp"), plan.get("right_lamp")]
    if not isinstance(lamps, (list, tuple)) or len(lamps) != 2 or not all(isinstance(v, Mapping) for v in lamps):
        raise ValueError("lighting_plan.lamps must contain exactly two lamp mappings")
    return list(lamps)


def _lamp_value(lamp: Mapping[str, Any], key: str, aliases: tuple[str, ...] = ()) -> Any:
    for candidate in (key, *aliases):
        if candidate in lamp:
            return lamp[candidate]
    raise ValueError(f"Lamp plan lacks {key}")


def _create_world(bpy: Any, mode: str, sky_plan: Mapping[str, Any]) -> Any:
    world = bpy.data.worlds.new(f"Lighting_{mode.capitalize()}_World")
    world.use_nodes = True
    tree = world.node_tree
    if tree is None:
        raise RuntimeError("World node_tree is unavailable")
    nodes = tree.nodes
    links = tree.links
    nodes.clear()
    sky = nodes.new("ShaderNodeTexSky")
    # Keep the conventional Blender node names so manual inspection and the
    # v1 state reader can find the native nodes without special cases.
    sky.name = "Sky Texture"
    _set_required(sky, "sky_type", "MULTIPLE_SCATTERING", "sky")
    _set_required(sky, "sun_disc", False, "sky")
    _set_required(
        sky,
        "sun_elevation",
        math.radians(_finite_number(sky_plan.get("sun_elevation_deg"), "sun_elevation_deg")),
        "sky",
    )
    _set_required(
        sky,
        "sun_rotation",
        math.radians(_finite_number(sky_plan.get("sun_rotation_deg"), "sun_rotation_deg")),
        "sky",
    )
    background = nodes.new("ShaderNodeBackground")
    background.name = "Background"
    strength = _input_socket(background, "Strength")
    requested_strength = _finite_number(sky_plan.get("strength"), "sky strength")
    actual_strength = 0.0 if mode == "artificial" else requested_strength
    strength.default_value = actual_strength
    output = nodes.new("ShaderNodeOutputWorld")
    output.name = "World Output"
    links.new(sky.outputs[0], _input_socket(background, "Color"))
    links.new(background.outputs[0], _input_socket(output, "Surface"))

    _safe_idprop(world, "lighting_mode", mode)
    _safe_idprop(world, "sky_model", "MULTIPLE_SCATTERING")
    _safe_idprop(world, "sun_disc", False)
    _safe_idprop(world, "sky_strength_input", requested_strength)
    _safe_idprop(world, "background_strength_actual", actual_strength)
    _safe_idprop(world, "sun_elevation_deg_input", float(sky_plan["sun_elevation_deg"]))
    _safe_idprop(world, "sun_rotation_deg_input", float(sky_plan["sun_rotation_deg"]))
    return world


def _link_collection(scene: Any, collection: Any) -> None:
    """Link a mode-owned collection to one scene without duplicating it."""

    root = getattr(scene, "collection", None)
    children = getattr(root, "children", None)
    if root is None or children is None:
        raise RuntimeError("scene has no root collection for native lights")
    try:
        children.link(collection)
    except RuntimeError as exc:
        users = getattr(collection, "users_scene", ())
        if not any(item is scene for item in users):
            raise RuntimeError(f"Could not link light collection to scene: {exc}") from exc


def _link_object(collection: Any, obj: Any) -> None:
    if collection is None or not hasattr(collection, "objects"):
        raise RuntimeError("native light collection is unavailable")
    try:
        collection.objects.link(obj)
    except RuntimeError as exc:
        # An already linked object is harmless; other link failures should be
        # surfaced because they would make the saved rig incomplete.
        users = getattr(obj, "users_collection", ())
        if not any(item is collection for item in users):
            raise RuntimeError(f"Could not link light object to scene: {exc}") from exc


def build_lighting(scene: Any, camera: Any, lighting_plan: Mapping[str, Any], mode: str) -> dict[str, Any]:
    """Create one independent native light rig and assign its World to scene."""

    bpy = _bpy_module()
    Matrix, Vector = _mathutils_types()
    if not isinstance(lighting_plan, Mapping):
        raise TypeError("lighting_plan must be a mapping")
    mode = str(mode).lower()
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    if scene is None or camera is None:
        raise ValueError("scene and camera are required")
    layout_value = lighting_plan.get("layout_seed")
    layout_seed = _nonnegative_int(layout_value, "layout_seed") if layout_value is not None else None
    sky_plan = _plan_sky(lighting_plan)
    lamp_plans = _plan_lamps(lighting_plan)
    world = _create_world(bpy, mode, sky_plan)
    collection = bpy.data.collections.new(f"Lighting_{mode.capitalize()}_Lights")
    _link_collection(scene, collection)
    _set_if_present(collection, "hide_render", False)
    _set_if_present(collection, "hide_viewport", False)
    _safe_idprop(collection, "lighting_mode", mode)
    _safe_idprop(world, "layout_seed", layout_seed)
    _safe_idprop(collection, "layout_seed", layout_seed)
    _safe_idprop(collection, "fixture_geometry", False)
    objects = []

    for index, lamp_plan in enumerate(lamp_plans):
        fallback_name = "LeftSpot" if index == 0 else "RightSpot"
        label = str(lamp_plan.get("name", fallback_name))
        if label not in ("LeftSpot", "RightSpot"):
            label = fallback_name
        data = bpy.data.lights.new(f"Lighting_{mode.capitalize()}_{label}", type="SPOT")
        obj = bpy.data.objects.new(f"Lighting_{mode.capitalize()}_{label}", data)
        _link_object(collection, obj)
        obj.parent = camera
        # Identity inverse means matrix_local is the camera-local transform and
        # moving/rotating the camera updates matrix_world naturally.
        obj.matrix_parent_inverse = Matrix.Identity(4)

        position = _lamp_value(lamp_plan, "position_local")
        target = _lamp_value(lamp_plan, "target_local")
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            raise ValueError(f"{label} position_local must contain three values")
        if not isinstance(target, (list, tuple)) or len(target) != 3:
            raise ValueError(f"{label} target_local must contain three values")
        position_v = Vector([_finite_number(v, f"{label} position") for v in position])
        target_v = Vector([_finite_number(v, f"{label} target") for v in target])
        direction = target_v - position_v
        if direction.length <= 1e-9:
            raise ValueError(f"{label} target is coincident with the lamp")
        rotation = direction.to_track_quat("-Z", "Y")
        obj.location = position_v
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = rotation

        requested_energy = _finite_number(
            _lamp_value(lamp_plan, "energy_w", ("energy",)),
            f"{label} energy",
        )
        actual_energy = 0.0 if mode == "natural" else requested_energy
        data.energy = actual_energy
        cone_deg = _finite_number(
            _lamp_value(lamp_plan, "spot_full_angle_deg", ("full_angle_deg",)),
            f"{label} spot full angle",
        )
        if not 0 < cone_deg < 180:
            raise ValueError(f"{label} spot full angle must be between 0 and 180 degrees")
        data.spot_size = math.radians(cone_deg)
        blend = _finite_number(_lamp_value(lamp_plan, "spot_blend"), f"{label} spot blend")
        if not 0 <= blend <= 1:
            raise ValueError(f"{label} spot blend must be between 0 and 1")
        data.spot_blend = blend
        _set_if_present(data, "use_shadow", True)
        _set_if_present(data, "shadow_soft_size", 0.025)
        _set_if_present(data, "shape", "DISK")
        _set_if_present(data, "use_soft_falloff", True)
        _set_if_present(data, "normalize", True)
        _set_if_present(data, "exposure", 0.0)

        temperature = _finite_number(_lamp_value(lamp_plan, "temperature_k"), f"{label} temperature")
        source, effective_rgb, temperature_readback = _configure_temperature(data, temperature)
        _safe_idprop(data, "lighting_mode", mode)
        _safe_idprop(data, "layout_seed", layout_seed)
        _safe_idprop(data, "fixture_role", label)
        _safe_idprop(data, "requested_energy_w", requested_energy)
        _safe_idprop(data, "temperature_input_k", temperature)
        _safe_idprop(data, "temperature_source", source)
        _safe_idprop(data, "effective_rgb", effective_rgb)
        _safe_idprop(data, "temperature_readback", temperature_readback)
        _safe_idprop(data, "base_color_readback", temperature_readback.get("base_color"))
        _safe_idprop(data, "temperature_color_readback", temperature_readback.get("temperature_color"))
        _safe_idprop(data, "effective_rgb_source", temperature_readback.get("effective_rgb_source"))
        _safe_idprop(data, "spot_full_angle_deg_input", cone_deg)
        _safe_idprop(data, "spot_size_rad_actual", float(data.spot_size))
        _safe_idprop(data, "spot_blend_actual", float(data.spot_blend))
        _safe_idprop(obj, "lighting_mode", mode)
        _safe_idprop(obj, "layout_seed", layout_seed)
        _safe_idprop(obj, "fixture_role", label)
        _safe_idprop(obj, "position_local_input", [float(v) for v in position_v])
        _safe_idprop(obj, "target_local_input", [float(v) for v in target_v])
        _safe_idprop(obj, "requested_energy_w", requested_energy)
        _safe_idprop(obj, "actual_energy_w", float(data.energy))
        _safe_idprop(obj, "temperature_input_k", temperature)
        _safe_idprop(obj, "temperature_source", source)
        _safe_idprop(obj, "effective_rgb", effective_rgb)
        _safe_idprop(obj, "base_color_readback", temperature_readback.get("base_color"))
        _safe_idprop(obj, "temperature_color_readback", temperature_readback.get("temperature_color"))
        _safe_idprop(obj, "effective_rgb_source", temperature_readback.get("effective_rgb_source"))
        _safe_idprop(obj, "spot_full_angle_deg_input", cone_deg)
        _safe_idprop(obj, "spot_blend_actual", float(data.spot_blend))
        objects.append(obj)

    scene.world = world
    _safe_idprop(scene, "lighting_mode", mode)
    # Preserve the requested relation for callers that inspect a handle before
    # reading metadata; the actual matrices remain in the Blender objects.
    return {"objects": objects, "world": world, "mode": mode, "collection": collection}


def _json_safe(value: Any, seen: set[int] | None = None) -> Any:
    """Recursively convert Blender RNA/mathutils values to JSON-safe data."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if seen is None:
        seen = set()
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in seen:
            return "<recursive>"
        seen.add(marker)
        result = {str(key): _json_safe(item, seen) for key, item in value.items()}
        seen.remove(marker)
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        marker = id(value)
        if marker in seen:
            return "<recursive>"
        seen.add(marker)
        result = [_json_safe(item, seen) for item in value]
        seen.remove(marker)
        return result
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    # Blender's Color wrapper in 5.2.1 is sequence-convertible even though it
    # does not expose ``__iter__``.  Prefer its numeric tuple to an RNA repr.
    if value.__class__.__name__ in {"Color", "ColorRGBA"}:
        try:
            return [_json_safe(item, seen) for item in tuple(value)]
        except (IndexError, KeyError, TypeError, ValueError, RuntimeError):
            pass
    # Blender's Color RNA value supports ``len``/indexing but, on the probed
    # 5.2.1 build, does not advertise ``__iter__``.  Read that sequence before
    # falling back to its opaque string representation.
    if hasattr(value, "__len__") and hasattr(value, "__getitem__"):
        marker = id(value)
        if marker in seen:
            return "<recursive>"
        try:
            length = len(value)
            if 0 <= length <= 64:
                seen.add(marker)
                result = [_json_safe(value[index], seen) for index in range(length)]
                seen.remove(marker)
                return result
        except (IndexError, KeyError, TypeError, ValueError, RuntimeError):
            pass
    # Matrix, Vector and Color all expose an iterable numeric representation.
    if hasattr(value, "__iter__"):
        marker = id(value)
        if marker in seen:
            return "<recursive>"
        seen.add(marker)
        try:
            result = [_json_safe(item, seen) for item in value]
        except TypeError:
            result = str(value)
        seen.remove(marker)
        return result
    if isinstance(value, (complex,)):
        return {"real": float(value.real), "imag": float(value.imag)}
    return str(value)


def _id_properties(obj: Any) -> dict[str, Any]:
    try:
        keys = list(obj.keys())
        return {str(key): _json_safe(obj[key]) for key in keys}
    except (AttributeError, KeyError, TypeError, RuntimeError):
        return {}


def _matrix_value(value: Any) -> Any:
    try:
        return [[float(v) for v in row] for row in value]
    except (TypeError, ValueError):
        return _json_safe(value)


def _node_tree_metadata(tree: Any, stack: set[int] | None = None) -> Any:
    if tree is None:
        return None
    stack = set() if stack is None else set(stack)
    marker = id(tree)
    if marker in stack:
        return {"identifier": str(getattr(tree, "name", "<unnamed>")), "recursive": True}
    stack.add(marker)
    nodes_meta = []
    nodes = sorted(list(getattr(tree, "nodes", ())), key=lambda node: str(getattr(node, "name", "")))
    for node in nodes:
        sockets = {}
        for socket in getattr(node, "inputs", ()):
            if hasattr(socket, "default_value"):
                try:
                    sockets[str(getattr(socket, "identifier", getattr(socket, "name", "")))] = _json_safe(
                        socket.default_value
                    )
                except (AttributeError, TypeError, ValueError, RuntimeError):
                    continue
        properties = {}
        for attr in (
            "sky_type",
            "sun_disc",
            "sun_elevation",
            "sun_rotation",
            "temperature",
            "mode",
            "distribution",
            "space",
        ):
            if hasattr(node, attr):
                try:
                    properties[attr] = _json_safe(getattr(node, attr))
                except (AttributeError, TypeError, ValueError, RuntimeError):
                    continue
        nested = getattr(node, "node_tree", None)
        row = {
            "name": str(getattr(node, "name", "")),
            "label": str(getattr(node, "label", "")),
            "type": str(getattr(node, "bl_idname", getattr(node, "type", ""))),
            "mute": bool(getattr(node, "mute", False)),
            "hide": bool(getattr(node, "hide", False)),
            "inputs": sockets,
            "properties": properties,
        }
        if nested is not None:
            row["node_tree"] = _node_tree_metadata(nested, stack)
        nodes_meta.append(row)
    links_meta = []
    for link in getattr(tree, "links", ()):
        links_meta.append(
            {
                "from_node": str(getattr(getattr(link, "from_node", None), "name", "")),
                "from_socket": str(
                    getattr(
                        getattr(link, "from_socket", None),
                        "identifier",
                        getattr(getattr(link, "from_socket", None), "name", ""),
                    )
                ),
                "to_node": str(getattr(getattr(link, "to_node", None), "name", "")),
                "to_socket": str(
                    getattr(
                        getattr(link, "to_socket", None),
                        "identifier",
                        getattr(getattr(link, "to_socket", None), "name", ""),
                    )
                ),
            }
        )
    stack.remove(marker)
    return {
        "identifier": str(getattr(tree, "name", "<unnamed>")),
        "type": str(getattr(tree, "bl_idname", "NodeTree")),
        "nodes": nodes_meta,
        "links": sorted(links_meta, key=lambda item: (item["from_node"], item["to_node"], item["to_socket"])),
    }


def _actual_fields(obj: Any, names: tuple[str, ...]) -> dict[str, Any]:
    values = {}
    for name in names:
        if hasattr(obj, name):
            try:
                values[name] = _json_safe(getattr(obj, name))
            except (AttributeError, TypeError, ValueError, RuntimeError):
                continue
    return values


def _collection_semantics(collection: Any) -> dict[str, Any] | None:
    """Return stable collection identity without stringifying RNA structs."""

    if collection is None:
        return None
    return {
        "name": str(getattr(collection, "name", "")),
        "objects": sorted(str(getattr(item, "name", "")) for item in getattr(collection, "objects", ())),
        "children": sorted(str(getattr(item, "name", "")) for item in getattr(collection, "children", ())),
    }


def _light_linking_semantics(obj: Any) -> dict[str, Any] | None:
    """Read ObjectLightLinking's meaningful fields, never its RNA repr."""

    linking = getattr(obj, "light_linking", None)
    if linking is None:
        return None
    receiver = _collection_semantics(getattr(linking, "receiver_collection", None))
    blocker = _collection_semantics(getattr(linking, "blocker_collection", None))
    lightgroup = None
    for attr in ("lightgroup", "light_group"):
        if hasattr(obj, attr):
            try:
                value = getattr(obj, attr)
                if value not in (None, ""):
                    lightgroup = str(value)
                    break
                if lightgroup is None:
                    lightgroup = None if value in (None, "") else str(value)
            except (AttributeError, TypeError, ValueError, RuntimeError):
                continue
    return {
        "receiver_collection": receiver,
        "blocker_collection": blocker,
        "lightgroup": lightgroup,
    }


def _driver_metadata(obj: Any) -> dict[str, Any]:
    animation = getattr(obj, "animation_data", None)
    if animation is None:
        return {"has_animation_data": False, "driver_paths": []}
    drivers = []
    for entry in getattr(animation, "drivers", ()):
        drivers.append(str(getattr(getattr(entry, "data_path", None), "__str__", lambda: "")()))
    return {"has_animation_data": True, "driver_paths": drivers}


def _light_metadata(obj: Any) -> dict[str, Any]:
    data = getattr(obj, "data", None)
    custom_obj = _id_properties(obj)
    custom_data = _id_properties(data)
    matrix_world = _matrix_value(getattr(obj, "matrix_world", None))
    matrix_local = _matrix_value(getattr(obj, "matrix_local", getattr(obj, "matrix_world", None)))
    matrix_parent_inverse = _matrix_value(getattr(obj, "matrix_parent_inverse", None))
    parent = getattr(obj, "parent", None)
    actual = _actual_fields(
        data,
        (
            "type",
            "energy",
            "color",
            "temperature",
            "temperature_color",
            "use_temperature",
            "spot_size",
            "spot_blend",
            "shape",
            "size",
            "size_y",
            "shadow_soft_size",
            "use_shadow",
            "use_custom_distance",
            "cutoff_distance",
            "volume_factor",
            "diffuse_factor",
            "specular_factor",
            "transmission_factor",
            "spread",
        ),
    )
    try:
        cone_rad = float(getattr(data, "spot_size"))
        cone_deg = math.degrees(cone_rad)
    except (AttributeError, TypeError, ValueError):
        cone_rad = None
        cone_deg = None
    temperature_input = custom_data.get("temperature_input_k", custom_obj.get("temperature_input_k"))
    effective_rgb = custom_data.get("effective_rgb", custom_obj.get("effective_rgb"))
    temperature_source = custom_data.get("temperature_source", custom_obj.get("temperature_source"))
    base_color_readback = custom_data.get("base_color_readback", custom_obj.get("base_color_readback"))
    temperature_color_readback = custom_data.get(
        "temperature_color_readback", custom_obj.get("temperature_color_readback")
    )
    effective_rgb_source = custom_data.get("effective_rgb_source", custom_obj.get("effective_rgb_source"))
    if effective_rgb is not None:
        effective_rgb = _rgb3(effective_rgb, "effective_rgb")
    if base_color_readback is not None:
        base_color_readback = _rgb3(base_color_readback, "base_color_readback")
    if temperature_color_readback is not None:
        temperature_color_readback = _rgb3(temperature_color_readback, "temperature_color_readback")
    actual_base_color = actual.get("color")
    if actual_base_color is not None:
        actual_base_color = _rgb3(actual_base_color, "Light.color readback")
    actual_temperature_color = actual.get("temperature_color")
    if actual_temperature_color is not None:
        actual_temperature_color = _rgb3(actual_temperature_color, "Light.temperature_color readback")
    try:
        local_position = [float(matrix_local[0][3]), float(matrix_local[1][3]), float(matrix_local[2][3])]
    except (IndexError, TypeError):
        local_position = None
    visibility = _actual_fields(
        obj,
        (
            "hide_render",
            "hide_viewport",
            "visible_camera",
            "visible_diffuse",
            "visible_glossy",
            "visible_transmission",
            "visible_volume_scatter",
            "visible_shadow",
            "is_holdout",
            "is_shadow_catcher",
        ),
    )
    data_tree = _node_tree_metadata(getattr(data, "node_tree", None)) if getattr(data, "use_nodes", False) else None
    light_linking = _light_linking_semantics(obj)
    return {
        "name": str(getattr(obj, "name", "")),
        "data_name": str(getattr(data, "name", "")),
        "object_type": str(getattr(obj, "type", "")),
        # Frequently inspected scalar aliases are copied from the readback
        # below; the nested ``actual`` mapping remains the complete set.
        "type": actual.get("type"),
        "energy": actual.get("energy"),
        "color": actual.get("color"),
        "base_color": base_color_readback if base_color_readback is not None else actual_base_color,
        "use_temperature": actual.get("use_temperature"),
        "temperature_k": actual.get("temperature"),
        "temperature_color": temperature_color_readback if temperature_color_readback is not None else actual_temperature_color,
        "effective_rgb": effective_rgb,
        "effective_rgb_source": effective_rgb_source,
        "spot_size_rad": cone_rad,
        "spot_full_angle_deg": cone_deg,
        "spot_blend": actual.get("spot_blend"),
        "use_shadow": actual.get("use_shadow"),
        "actual": actual,
        "spot": {
            "size_rad": cone_rad,
            "full_angle_deg": cone_deg,
            "blend": actual.get("spot_blend"),
        },
        "temperature": {
            "input_k": temperature_input,
            "source": temperature_source,
            "effective_rgb": effective_rgb,
            "effective_rgb_source": effective_rgb_source,
            "base_color": base_color_readback if base_color_readback is not None else actual_base_color,
            "temperature_color": temperature_color_readback if temperature_color_readback is not None else actual_temperature_color,
            "actual_native_fields": {
                key: actual[key] for key in ("use_temperature", "temperature") if key in actual
            },
        },
        "transform": {
            "parent": str(getattr(parent, "name", "")) if parent is not None else None,
            "matrix_local": matrix_local,
            "matrix_parent_inverse": matrix_parent_inverse,
            "matrix_world": matrix_world,
            "position_local": local_position,
            "target_local": custom_obj.get("target_local_input"),
        },
        "visibility": visibility,
        "shadow": {
            key: actual[key] for key in ("use_shadow", "shadow_soft_size") if key in actual
        },
        "light_linking": light_linking,
        "drivers": {"object": _driver_metadata(obj), "data": _driver_metadata(data)},
        "custom_properties": {"object": custom_obj, "data": custom_data},
        "node_graph": data_tree,
        "collections": [str(getattr(item, "name", "")) for item in getattr(obj, "users_collection", ())],
    }


def _world_metadata(world: Any) -> dict[str, Any]:
    tree = getattr(world, "node_tree", None)
    custom_properties = _id_properties(world)
    sky_rows = []
    background_rows = []
    for node in getattr(tree, "nodes", ()) if tree is not None else ():
        if str(getattr(node, "bl_idname", "")) == "ShaderNodeTexSky":
            sky_rows.append(
                _actual_fields(node, ("sky_type", "sun_disc", "sun_elevation", "sun_rotation"))
            )
        elif str(getattr(node, "bl_idname", "")) == "ShaderNodeBackground":
            strength = None
            try:
                strength = _json_safe(_input_socket(node, "Strength").default_value)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                pass
            background_rows.append(
                {"strength": strength, "color": _json_safe(getattr(_input_socket(node, "Color"), "default_value", None))}
            )
    return {
        "name": str(getattr(world, "name", "")),
        "use_nodes": bool(getattr(world, "use_nodes", False)),
        "layout_seed": custom_properties.get("layout_seed"),
        "custom_properties": custom_properties,
        "sky": sky_rows,
        "background": background_rows,
        "sky_type": sky_rows[0].get("sky_type") if sky_rows else None,
        "sun_disc": sky_rows[0].get("sun_disc") if sky_rows else None,
        "sun_elevation_rad": sky_rows[0].get("sun_elevation") if sky_rows else None,
        "sun_rotation_rad": sky_rows[0].get("sun_rotation") if sky_rows else None,
        "background_strength": background_rows[0].get("strength") if background_rows else None,
        "node_graph": _node_tree_metadata(tree),
        "drivers": _driver_metadata(world),
    }


def _collection_metadata(collection: Any) -> dict[str, Any] | None:
    if collection is None:
        return None
    return {
        "name": str(getattr(collection, "name", "")),
        "hide_render": bool(getattr(collection, "hide_render", False)),
        "hide_viewport": bool(getattr(collection, "hide_viewport", False)),
        "custom_properties": _id_properties(collection),
        "objects": [str(getattr(obj, "name", "")) for obj in getattr(collection, "objects", ())],
        "children": [str(getattr(child, "name", "")) for child in getattr(collection, "children", ())],
        "drivers": _driver_metadata(collection),
    }


def lighting_metadata(handle: Mapping[str, Any]) -> dict[str, Any]:
    """Read actual native rig values into a JSON-safe metadata dictionary."""

    if not isinstance(handle, Mapping):
        raise TypeError("handle must be a mapping")
    mode = str(handle.get("mode", "")).lower()
    if mode not in MODES:
        raise ValueError(f"handle.mode must be one of {', '.join(MODES)}")
    objects = handle.get("objects")
    if not isinstance(objects, (list, tuple)) or len(objects) != 2:
        raise ValueError("handle.objects must contain exactly two native lights")
    world = handle.get("world")
    if world is None:
        raise ValueError("handle.world is required")
    _native_update()
    collection = handle.get("collection")
    world_metadata = _world_metadata(world)
    objects_metadata = [_light_metadata(obj) for obj in objects]
    temperature_sources = [item["temperature"].get("source") for item in objects_metadata]
    effective_sources = [item["temperature"].get("effective_rgb_source") for item in objects_metadata]
    native_temperature_readback = all(
        source == "native_light_temperature" and effective == "Light.temperature_color"
        for source, effective in zip(temperature_sources, effective_sources)
    )
    if native_temperature_readback:
        temperature_readback_status = "native_light_temperature"
    elif all(source == "native_blackbody_shader" for source in temperature_sources):
        temperature_readback_status = "native_blackbody_shader_estimate"
    else:
        temperature_readback_status = "mixed_or_unavailable"
    result = {
        "schema_version": 2,
        "mode": mode,
        "layout_seed": world_metadata.get("layout_seed"),
        "world": world_metadata,
        "collection": _collection_metadata(collection),
        "objects": objects_metadata,
        "temperature_sources": temperature_sources,
        "resource_identity": {
            "world_name": str(getattr(world, "name", "")),
            "object_names": [str(getattr(obj, "name", "")) for obj in objects],
            "data_names": [str(getattr(getattr(obj, "data", None), "name", "")) for obj in objects],
        },
        "native_constraints": {
            "fixture_geometry": False,
            "light_path_transparency": False,
            "image_emission": False,
            "temperature_is_read_back": native_temperature_readback,
            "temperature_readback_status": temperature_readback_status,
            "power_unit": "BlenderW demonstration parameter; not calibrated device power",
        },
    }
    safe = _json_safe(result)
    json.dumps(safe, ensure_ascii=False, allow_nan=False, sort_keys=True)
    return safe


__all__ = ["MODES", "sample_lighting", "build_lighting", "lighting_metadata"]
