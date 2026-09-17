"""Batch-scale v3 runner: one randomized scene, one light plan, four states.

This file is intentionally an additive batch driver.  The existing v3 source
and first-round outputs remain untouched.  It runs inside one Blender process
and imports the already validated v3 generator modules; the only v3 behavior
selected here is to build one lighting group per scene instead of all three
mode groups.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path
from random import Random

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

import bpy  # type: ignore
import numpy as np  # type: ignore
from mathutils import Vector  # type: ignore

from src.contracts import (
    REFERENCE_FILES,
    SAMPLE_FILES,
    atomic_json,
    canonical_hash,
    derive_seed,
    file_hash,
    load_config,
    seal_record,
)
from src.gates import geometry_gate
from src.labels import export_labels
from src.labels.formats import read_exr, read_png
from src.lighting import lighting_metadata
from src.quality import geometry_metrics, rgb_decision, rgb_metrics
from src.render import configure, read_render_rgb, select_device
from src.scene import build_geometry, build_lighting_groups, geometry_constraints, validate_light_positions
from src.assets import build_ecosystem
from src.randomization import apply_geometry_state, prepare_layout
from src.lighting import build_lighting, sample_lighting
from src.assets.external import REPLACEMENTS
from src.state import assert_clear_scene, geometry_digest, state_digest, water_template
from src.storage import event, promote_directory, readable_bundle
from src.water import build_water, set_water_enabled, set_water_preset, water_metadata


class CandidateRejected(Exception):
    def __init__(self, reason: str, evidence: dict | None = None):
        super().__init__(reason)
        self.evidence = evidence


_SCAN_CACHE: dict[str, bpy.types.Object] = {}
_IN_PLACE = False


def _world_bounds(object_: bpy.types.Object) -> tuple[Vector, Vector]:
    points = [object_.matrix_world @ Vector(corner) for corner in object_.bound_box]
    return (
        Vector((min(point.x for point in points), min(point.y for point in points), min(point.z for point in points))),
        Vector((max(point.x for point in points), max(point.y for point in points), max(point.z for point in points))),
    )


def _apply_cached_scanned_replacements(objects, instances, morphology_counts):
    """Reuse scan meshes packed in a verified v3 blend; never import a GLB."""

    by_name = {object_.name: object_ for object_ in objects}
    rows = []
    for spec in REPLACEMENTS:
        target = by_name.get(spec.target_name)
        source = _SCAN_CACHE.get(spec.target_name)
        if target is None or source is None:
            raise RuntimeError("Missing cached v3 scan asset: " + spec.target_name)
        previous_morphology = str(target.get("asset_morphology", "unknown"))
        old_mesh = target.data
        target.data = source.data.copy()
        target.rotation_mode = "XYZ"
        # The cached v3 blend already stores the scan mesh after its original
        # import-scale and layout-pivot conversion.  Applying the GLB
        # import-scale a second time would enlarge it by roughly 10x.  Keep
        # the cached local mesh at unit transform and let prepare_layout apply
        # the batch scene's pivot/pose contract once.
        target.rotation_euler = (0.0, 0.0, 0.0)
        target.scale = (1.0, 1.0, 1.0)
        target.data.name = f"{target.name}_CachedV3ScanMesh"
        target.data["asset_source"] = "cc0_imported"
        target.data["asset_uid"] = spec.asset_uid
        target.data["source_relative_path"] = spec.source_relative_path
        target.data["source_sha256"] = spec.source_sha256
        target["asset_source"] = "cc0_imported"
        target["asset_uid"] = spec.asset_uid
        target["asset_morphology"] = spec.morphology
        target["external_source_relative_path"] = spec.source_relative_path
        target["external_source_sha256"] = spec.source_sha256
        target["external_source_url"] = spec.source_url
        target["external_license"] = "CC0"
        target["external_import_scale"] = float(spec.uniform_scale)
        target["external_import_yaw_rad"] = float(spec.yaw_radians)
        target["external_images_packed_in_glb"] = True
        if spec.fish:
            target["asset_forward"] = [float(np.cos(spec.yaw_radians)), float(np.sin(spec.yaw_radians)), 0.0]
            target["asset_up"] = [0.0, 0.0, 1.0]
        for material in target.data.materials:
            if material is None:
                continue
            material["asset_source"] = "cc0_imported"
            material["asset_uid"] = spec.asset_uid
            material["external_license"] = "CC0"
            material["source_relative_path"] = spec.source_relative_path
        if old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)
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
            raise RuntimeError("Missing cached scan instance record: " + target.name)
        rows.append({"asset_uid": spec.asset_uid, "target_name": target.name, "instance_id": instance_id, "semantic_id": int(target["semantic_id"]), "morphology": spec.morphology, "source_relative_path": spec.source_relative_path, "source_sha256": spec.source_sha256, "source_url": spec.source_url, "license": "CC0", "uniform_scale": float(spec.uniform_scale), "yaw_radians": float(spec.yaw_radians), "images_packed": True, "cache_source": "v3_native_blend"})
    return rows


def configure_template_asset_cache() -> None:
    """Bind the v3 ecology builder to scan meshes already present in a blend."""

    global _SCAN_CACHE
    _SCAN_CACHE = {}
    for spec in REPLACEMENTS:
        source = bpy.data.objects.get(spec.target_name)
        if source is None or source.type != "MESH":
            raise RuntimeError("Template blend lacks scan object " + spec.target_name)
        if source.data is None or "CC0ScanMesh" not in source.data.name:
            raise RuntimeError("Template object is not a verified CC0 scan mesh: " + spec.target_name)
        _SCAN_CACHE[spec.target_name] = source
        # Keep the immutable template objects out of the global name namespace
        # used by the v3 builder.  Otherwise Blender suffixes each new target
        # as ``.001`` and the replacement contract cannot find it.
        source.name = "__V3_TEMPLATE_" + spec.target_name
        for parent in list(source.users_collection):
            parent.objects.unlink(source)
    import src.assets.ecology as ecology
    ecology.apply_scanned_replacements = _apply_cached_scanned_replacements


def build_geometry_cached(cfg: dict, layout: dict, state_index: int, attempt_index: int) -> dict:
    """Build one v3 scene without wm.read_factory_settings or GLB import."""

    if not _SCAN_CACHE:
        raise RuntimeError("Template asset cache is not configured")
    before = {
        "objects": {obj.as_pointer() for obj in bpy.data.objects},
        "scenes": {scene.as_pointer() for scene in bpy.data.scenes},
        "collections": {collection.as_pointer() for collection in bpy.data.collections},
        "meshes": {mesh.as_pointer() for mesh in bpy.data.meshes},
        "cameras": {camera.as_pointer() for camera in bpy.data.cameras},
        "lights": {light.as_pointer() for light in bpy.data.lights},
        "worlds": {world.as_pointer() for world in bpy.data.worlds},
        "materials": {material.as_pointer() for material in bpy.data.materials},
        "node_groups": {group.as_pointer() for group in bpy.data.node_groups},
    }
    base = bpy.data.scenes.new(f"L{layout['seed']}_G{state_index:03d}_Geometry")
    configure(base, cfg)
    ecosystem = build_ecosystem(base, layout["seed"], 0.0)
    camera = bpy.data.objects.new("Camera_Main", bpy.data.cameras.new("Camera_Main_Pinhole"))
    base.collection.objects.link(camera)
    base.camera = camera
    camera.data.type = "PERSP"
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.sensor_width = 36
    camera.data.clip_start = 0.01
    camera.data.clip_end = 150
    camera.data.dof.use_dof = False
    world = bpy.data.worlds.new("Geometry_Check_Black")
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background") or world.node_tree.nodes.new("ShaderNodeBackground")
    background.inputs["Strength"].default_value = 0
    base.world = world
    prepare_layout(base, ecosystem)
    local = dict(cfg, _layout_seed=layout["seed"])
    plan = apply_geometry_state(base, ecosystem, camera, local, state_index, attempt_index)
    base.view_layers[0].update()
    active = [obj for obj in ecosystem["objects"] if obj.name in set(plan["visible_object_names"])]
    if not active:
        raise ValueError("No active target objects")
    geometry_constraints(base, active, min(cfg["depths_m"]))
    device = select_device(base)
    lighting_plan = sample_lighting(local, state_index, attempt_index)
    handle = {"base": base, "ecology": ecosystem, "camera": camera, "geometry_plan": plan, "active_objects": active, "objects": ecosystem["objects"], "device": device, "lighting_plan": lighting_plan, "groups": [], "_batch_before": before}
    return handle


def build_geometry_in_place(cfg: dict, layout: dict, state_index: int, attempt_index: int) -> dict:
    """Build v3 geometry in the one scene loaded from the scan-cache blend."""

    if not _SCAN_CACHE:
        raise RuntimeError("Template asset cache is not configured")
    base = bpy.context.scene
    before = {
        "objects": {obj.as_pointer() for obj in bpy.data.objects},
        "scenes": {scene.as_pointer() for scene in bpy.data.scenes},
        "collections": {collection.as_pointer() for collection in bpy.data.collections},
        "meshes": {mesh.as_pointer() for mesh in bpy.data.meshes},
        "cameras": {camera.as_pointer() for camera in bpy.data.cameras},
        "lights": {light.as_pointer() for light in bpy.data.lights},
        "worlds": {world.as_pointer() for world in bpy.data.worlds},
        "materials": {material.as_pointer() for material in bpy.data.materials},
        "node_groups": {group.as_pointer() for group in bpy.data.node_groups},
    }
    base.name = f"L{layout['seed']}_G{state_index:03d}_Geometry"
    configure(base, cfg)
    ecosystem = build_ecosystem(base, layout["seed"], 0.0)
    camera = bpy.data.objects.new("Camera_Main", bpy.data.cameras.new("Camera_Main_Pinhole"))
    base.collection.objects.link(camera)
    base.camera = camera
    camera.data.type = "PERSP"
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.sensor_width = 36
    camera.data.clip_start = 0.01
    camera.data.clip_end = 150
    camera.data.dof.use_dof = False
    world = bpy.data.worlds.new("Geometry_Check_Black")
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background") or world.node_tree.nodes.new("ShaderNodeBackground")
    background.inputs["Strength"].default_value = 0
    base.world = world
    prepare_layout(base, ecosystem)
    local = dict(cfg, _layout_seed=layout["seed"])
    plan = apply_geometry_state(base, ecosystem, camera, local, state_index, attempt_index)
    base.view_layers[0].update()
    active = [obj for obj in ecosystem["objects"] if obj.name in set(plan["visible_object_names"])]
    if not active:
        raise ValueError("No active target objects")
    geometry_constraints(base, active, min(cfg["depths_m"]))
    device = select_device(base)
    lighting_plan = sample_lighting(local, state_index, attempt_index)
    return {"base": base, "ecology": ecosystem, "camera": camera, "geometry_plan": plan, "active_objects": active, "objects": ecosystem["objects"], "device": device, "lighting_plan": lighting_plan, "groups": [], "_batch_before": before, "_in_place": True}


def build_lighting_in_place(handle: dict, cfg: dict, state: dict, water_presets: list[dict]) -> dict:
    """Create one native rig in the existing scene and return its water plan."""

    scene = handle["base"]
    mode = str(state["lighting_mode"])
    rig = build_lighting(scene, handle["camera"], handle["lighting_plan"], mode)
    rig["clearance"] = validate_light_positions(handle, rig, cfg)
    scene.world = rig["world"]
    scene["lighting_mode"] = mode
    handle["lighting"] = rig
    handle["water_presets"] = water_presets
    handle["groups"] = [{"mode": mode, "clear": scene, "lighting": rig, "variants": []}]
    return handle


def cleanup_cached_handle(handle: dict | None) -> None:
    """Remove only datablocks created after the template blend was loaded."""

    if not handle or "_batch_before" not in handle:
        return
    before = handle["_batch_before"]
    template_scene = next((scene for scene in bpy.data.scenes if scene.as_pointer() in before["scenes"]), None)
    if template_scene is not None:
        window = bpy.context.window
        if window is None:
            windows = list(getattr(getattr(bpy.context, "window_manager", None), "windows", ()))
            window = windows[0] if windows else None
        if window is not None:
            window.scene = template_scene
    for obj in list(bpy.data.objects):
        if obj.as_pointer() not in before["objects"]:
            bpy.data.objects.remove(obj, do_unlink=True)
    for scene in list(bpy.data.scenes):
        if scene.as_pointer() not in before["scenes"]:
            bpy.data.scenes.remove(scene)
    for collection in list(bpy.data.collections):
        if collection.as_pointer() not in before["collections"] and collection.users == 0:
            bpy.data.collections.remove(collection)
    for datablocks, key in ((bpy.data.meshes, "meshes"), (bpy.data.cameras, "cameras"), (bpy.data.lights, "lights"), (bpy.data.worlds, "worlds")):
        for datablock in list(datablocks):
            if datablock.as_pointer() not in before[key] and datablock.users == 0:
                datablocks.remove(datablock)
    for datablocks, key in ((bpy.data.materials, "materials"), (bpy.data.node_groups, "node_groups")):
        for datablock in list(datablocks):
            if datablock.as_pointer() not in before[key] and datablock.users == 0:
                datablocks.remove(datablock)


def _window_and_view_layer(scene):
    """Return a valid GUI window after v3 factory-reset has rebuilt Blender data."""

    window = bpy.context.window
    if window is None:
        windows = list(getattr(getattr(bpy.context, "window_manager", None), "windows", ()))
        if windows:
            window = windows[0]
    if window is None:
        raise RuntimeError("v3 batch needs one Blender window for an in-process render")
    view_layer = getattr(window, "view_layer", None) or scene.view_layers[0]
    return window, view_layer


def render_rgb(scene, stem):
    """Render with v3's encoding contract without relying on context.window."""

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    window, view_layer = _window_and_view_layer(scene)
    window.scene = scene
    view_layer.update()
    start = time.perf_counter()
    with bpy.context.temp_override(window=window, scene=scene, view_layer=view_layer):
        bpy.ops.render.render(scene=scene.name, write_still=False)
    result = bpy.data.images["Render Result"]
    settings = scene.render.image_settings
    settings.color_mode = "RGB"
    settings.file_format = "OPEN_EXR"
    settings.color_depth = "32"
    settings.exr_codec = "ZIP"
    result.save_render(str(stem) + ".exr", scene=scene)
    settings.file_format = "PNG"
    settings.color_depth = "16"
    settings.compression = 20
    result.save_render(str(stem) + ".png", scene=scene)
    rgb = read_render_rgb(Path(str(stem) + ".exr"))
    if not np.isfinite(rgb).all():
        raise ValueError("Nonfinite rendered RGB")
    return {"seconds": time.perf_counter() - start, "resolution": [rgb.shape[1], rgb.shape[0]], "linear_rgb_mean": rgb.mean(axis=(0, 1)).tolist(), "min": float(rgb.min()), "max": float(rgb.max()), "engine": "CYCLES", "device": scene.cycles.device, "samples": scene.cycles.samples, "seed": scene.cycles.seed, "denoising": False, "exr": {"pixel_type": "FLOAT32", "compression": "ZIP", "space": "Linear Rec.709", "display_transform_applied": False}, "png": {"bits": 16, "display": "sRGB", "view_transform": "Standard", "look": "None", "exposure": 0, "gamma": 1}, "rgb_pixels_generated_by": "Cycles path tracing"}


def batch_source_hash() -> str:
    """Hash the unchanged v3 generator plus this additive batch driver."""

    paths = sorted((ROOT / "src").rglob("*.py")) + [ROOT / "run.py", Path(__file__).resolve()]
    catalog = ROOT / "assets" / "asset_catalog.json"
    asset_root = ROOT / "assets" / "source"
    return canonical_hash(
        {
            "v3_source": {str(path.relative_to(ROOT)): file_hash(path) for path in paths},
            "asset_catalog": file_hash(catalog),
            "asset_files": {str(path.relative_to(ROOT)): file_hash(path) for path in sorted(asset_root.rglob("*")) if path.is_file()},
        }
    )


def runtime_signature(cfg: dict) -> dict:
    configure(bpy.context.scene, cfg)
    device = select_device(bpy.context.scene)
    ocio = Path(bpy.utils.resource_path("LOCAL")) / "datafiles/colormanagement/config.ocio"
    if not ocio.is_file():
        raise RuntimeError("Native OCIO config was not found")
    return {
        "blender": bpy.app.version_string,
        "build_hash": bpy.app.build_hash.decode(),
        "python": platform.python_version(),
        "os": platform.mac_ver()[0],
        "device": device,
        "working_space": "Linear Rec.709",
        "display": "Standard/sRGB",
        "ocio_config_sha256": file_hash(ocio),
        "autoexec_disabled_cli": "--disable-autoexec" in sys.argv,
        "user_preference_scripts_auto_execute": bpy.context.preferences.filepaths.use_scripts_auto_execute,
    }


def validate_batch_config(cfg: dict) -> None:
    if int(cfg.get("scene_count", 0)) != 500:
        raise ValueError("v3_500_scenes requires scene_count=500")
    schedule = cfg.get("batch_layout_schedule") or {}
    if schedule.get("scheme") != "derive_seed" or schedule.get("namespace") != "scene_layout":
        raise ValueError("Invalid v3 batch layout seed schedule")
    lighting = cfg.get("single_lighting_schedule") or {}
    if lighting.get("type") != "balanced_cyclic" or type(lighting.get("offset")) is not int:
        raise ValueError("Invalid single-light schedule")
    if not 1 <= int(cfg.get("batch_max_attempts", cfg["max_attempts"])) <= 24:
        raise ValueError("Invalid batch_max_attempts")
    if tuple(cfg["lighting_modes"]) != ("natural", "artificial", "mixed"):
        raise ValueError("v3 batch lighting modes must be natural/artificial/mixed")
    counts = {mode: 0 for mode in cfg["lighting_modes"]}
    for number in range(1, 501):
        counts[cfg["lighting_modes"][(number - 1 + lighting["offset"]) % 3]] += 1
    if sorted(counts.values()) != [166, 167, 167]:
        raise ValueError("Single-light schedule is not balanced 166/167/167")


def batch_plan(cfg: dict) -> dict:
    validate_batch_config(cfg)
    states = []
    references = []
    pairs = []
    modes = cfg["lighting_modes"]
    mode_offset = int(cfg["single_lighting_schedule"]["offset"])
    split = cfg.get("batch_split_schedule", {"train": 400, "validation": 50, "test": 50})
    if sum(split.values()) != 500:
        raise ValueError("batch_split_schedule must partition 500 scenes")
    seeds = set()
    for number in range(1, 501):
        scene_id = f"scene_{number:04d}"
        layout_seed = derive_seed(cfg["root_seed"], "scene_layout", number)
        if layout_seed in seeds:
            raise ValueError("Duplicate v3 batch layout seed")
        seeds.add(layout_seed)
        split_id = "train" if number <= split["train"] else ("validation" if number <= split["train"] + split["validation"] else "test")
        mode = modes[(number - 1 + mode_offset) % len(modes)]
        item = {
            "state_key": scene_id,
            "scene_id": scene_id,
            "scene_number": number,
            "layout_seed": layout_seed,
            "state_index": 0,
            "layout_family": {101: "curved_dual_ridge", 202: "left_concentrated_island", 303: "diagonal_scatter"}.get(layout_seed, ("curved_dual_ridge", "left_concentrated_island", "diagonal_scatter")[abs(layout_seed) % 3]),
            "split": split_id,
            "lighting_mode": mode,
        }
        states.append(item)
        references.append({**item, "reference_key": f"{scene_id}_{mode}"})
        for preset in cfg["presets"]:
            pairs.append({**item, "sample_id": f"{scene_id}_{mode}_d5_{preset['id']}", "seabed_depth_m": 5, "water_id": preset["id"]})
    if len(states) != 500 or len(references) != 500 or len(pairs) != 1500:
        raise ValueError("v3 batch plan cardinality is not 500/500/1500")
    return {
        "schema_version": 2,
        "batch_schema": "v3_single_lighting_four_state",
        "geometry_states": states,
        "references": references,
        "pairs": pairs,
        "lighting_mode_counts": {mode: sum(state["lighting_mode"] == mode for state in states) for mode in modes},
    }


def ensure_plan(root: Path, plan: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "plan.json"
    if path.exists():
        if canonical_hash(json.loads(path.read_text())) != canonical_hash(plan):
            raise ValueError("Existing v3 batch output has a different plan; choose a new output root")
    elif any(root.iterdir()):
        raise ValueError("Nonempty v3 batch output root without plan.json")
    else:
        atomic_json(path, plan)


def verify_calibration_gate(cfg: dict, source: str, config_hash: str, runtime: dict) -> dict:
    path = ROOT / cfg["calibration_gate"]
    if not path.is_file():
        raise RuntimeError("v3 batch calibration gate is missing: " + str(path))
    gate = json.loads(path.read_text())
    if gate.get("schema_version") != 1 or gate.get("status") != "pass" or gate.get("scene_contract") != "single_lighting_four_state":
        raise RuntimeError("v3 batch calibration gate is not passed")
    if gate.get("source_sha256") != source or gate.get("config_sha256") != config_hash or gate.get("runtime_sha256") != canonical_hash(runtime):
        raise RuntimeError("v3 batch calibration gate signature mismatch")
    if gate.get("lighting_mode_counts") != {"natural": 167, "artificial": 167, "mixed": 166}:
        raise RuntimeError("v3 batch calibration gate mode counts mismatch")
    for evidence in gate.get("evidence", []):
        evidence_path = ROOT / evidence["path"]
        if not evidence_path.is_file() or file_hash(evidence_path) != evidence["sha256"]:
            raise RuntimeError("v3 batch calibration evidence changed: " + str(evidence_path))
    return {"path": str(path.relative_to(ROOT)), "sha256": file_hash(path)}


def water_plan(cfg: dict, state: dict, attempt: int) -> dict:
    seed = derive_seed(cfg["root_seed"], "water", state["layout_seed"], state["state_index"], attempt)
    rng = Random(seed)
    lo, hi = cfg.get("density_multiplier", [1, 1])
    multiplier = rng.uniform(lo, hi)
    variants = []
    for base in cfg["presets"]:
        preset = dict(base)
        preset["absorption"] = [multiplier * value for value in base["absorption"]]
        preset["scatter"] = [multiplier * value for value in base["scatter"]]
        variants.append(preset)
    return {"seed": seed, "density_multiplier": multiplier, "base_presets": cfg["presets"], "effective_presets": variants}


def label_arrays(folder: Path):
    return (
        np.asarray(read_exr(folder / "depth_range_m.exr")["pixels"], dtype=np.float32),
        np.asarray(read_png(folder / "semantic_id.png")["pixels"], dtype=np.uint16),
        np.asarray(read_png(folder / "instance_id.png")["pixels"], dtype=np.uint16),
        np.asarray(read_png(folder / "valid_mask.png")["pixels"], dtype=np.uint8),
    )


def assert_unchanged(scene, objects, digest: str) -> None:
    current, _ = state_digest(scene, objects)
    if current != digest:
        raise RuntimeError("FATAL_V3_BATCH_PAIR_STATE_MISMATCH: " + scene.name)


def rebuild_manifest(root: Path, plan: dict) -> list[dict]:
    rows: list[dict] = []
    for state in plan["geometry_states"]:
        checkpoint = root / "states" / state["state_key"] / "accepted.json"
        if checkpoint.is_file():
            value = json.loads(checkpoint.read_text())
            if value.get("status") == "accepted":
                rows.extend(value.get("manifest_rows", []))
    rows.sort(key=lambda row: row["sample_id"])
    atomic_json(root / "batch_manifest.json", {"schema_version": 1, "rows": rows, "pairs": len(rows), "references": len({row["reference_id"] for row in rows})})
    (root / "manifest.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    return rows


def native_save(handle: dict, path: Path, cfg: dict, metadata: dict) -> None:
    """Save one representative native v3 scene when explicitly requested."""

    path.parent.mkdir(parents=True, exist_ok=True)
    window, view_layer = _window_and_view_layer(handle["groups"][0]["clear"])
    if handle.get("base") and handle["base"].name in bpy.data.scenes:
        base = handle["base"]
        window.scene = handle["groups"][0]["clear"]
        bpy.data.scenes.remove(base)
        handle["base"] = None
    window.scene = handle["groups"][0]["variants"][1]["scene"]
    for scene in bpy.data.scenes:
        scene.render.filepath = "//manual_" + scene.name + ".png"
    doc = bpy.data.texts.get("START_HERE_Lighting_and_Data") or bpy.data.texts.new("START_HERE_Lighting_and_Data")
    doc.clear()
    doc.write("Native Cycles v3 batch scene: one deterministic lighting mode with Clear/mild/medium/strong paired states. Clear excludes seawater volume and air-water surface. No project Python or autoexec is required for manual F12.\n\n" + json.dumps({"config": cfg, "metadata": metadata}, ensure_ascii=False, indent=2))
    for image in bpy.data.images:
        relative = image.get("source_relative_path") if hasattr(image, "get") else None
        if not relative:
            continue
        source = (ROOT / str(relative)).resolve()
        if not source.is_file() or not source.is_relative_to(ROOT.resolve()):
            raise RuntimeError("Native save refused missing/outside-v3 image: " + str(relative))
        relative_to_blend = Path(__import__("os").path.relpath(source, start=path.parent.resolve())).as_posix()
        image.filepath = "//" + relative_to_blend
        image.filepath_raw = "//" + relative_to_blend
    result = bpy.ops.wm.save_as_mainfile(filepath=str(path), compress=True, relative_remap=False)
    if result != {"FINISHED"} or not path.is_file() or path.stat().st_size < 1024:
        raise RuntimeError("Native v3 batch blend save failed")


def checkpoint_complete(root: Path, state: dict, plan: dict, identity: dict) -> bool:
    path = root / "states" / state["state_key"] / "accepted.json"
    if not path.is_file():
        return False
    try:
        checkpoint = json.loads(path.read_text())
        if checkpoint.get("status") != "accepted" or any(checkpoint.get(key) != value for key, value in {**state, **identity}.items()):
            return False
        if len(checkpoint.get("manifest_rows", [])) != 3:
            return False
        for row in checkpoint["manifest_rows"]:
            sample = root / row["sample_path"]
            reference = root / row["reference_path"]
            if not readable_bundle(reference, "reference") or not readable_bundle(sample, "sample"):
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError):
        return False


def run_scene_in_place(cfg: dict, plan: dict, state: dict, root: Path, identity: dict) -> dict:
    """Render one scene's four states without creating additional Blender scenes."""

    key = state["state_key"]
    started = time.time()
    for attempt in range(int(cfg.get("batch_max_attempts", cfg["max_attempts"]))):
        folder = root / "attempts" / key / f"a{attempt:02d}"
        if folder.exists():
            previous = root / "rejected" / key / (f"a{attempt:02d}_previous_" + str(time.time_ns()))
            previous.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(folder), str(previous))
        folder.mkdir(parents=True, exist_ok=True)
        event(root, "candidate_started", state_key=key, state_index=0, attempt_index=attempt, status="running")
        handle = None
        try:
            layout = {"seed": state["layout_seed"], "split": state["split"]}
            handle = build_geometry_in_place(cfg, layout, 0, attempt)
            pool = handle["objects"]
            active = handle["active_objects"]
            geo_plan = handle["geometry_plan"]
            registry = {str(item["instance_id"]): item for item in geo_plan["instances"]}
            geo_sha, geo_snapshot = geometry_digest(handle["base"], pool)
            geometry_id = "geo_" + geo_sha[:24]
            geometa = export_labels(handle["base"], folder / "geometry", active)
            depth, sem, inst, mask = label_arrays(folder / "geometry")
            geometry_quality = geometry_gate(geometry_metrics(depth, sem, inst, registry, cfg), inst, registry, cfg)
            atomic_json(folder / "geometry_plan.json", geo_plan)
            atomic_json(folder / "geometry_quality.json", geometry_quality)
            if not geometry_quality["passed"]:
                raise CandidateRejected("geometry:" + ",".join(geometry_quality["reasons"]), geometry_quality)

            water = water_plan(cfg, state, attempt)
            build_lighting_in_place(handle, cfg, state, water["effective_presets"])
            scene = handle["base"]
            rig_meta = lighting_metadata(handle["lighting"])
            rig_meta["clearance"] = handle["lighting"]["clearance"]
            assert_clear_scene(scene, pool)
            digest, scene_snapshot = state_digest(scene, pool)
            assert_unchanged(scene, pool, digest)
            reference_id = "ref_" + digest[:24]
            common = {
                **identity,
                **state,
                "reference_id": reference_id,
                "geometry_id": geometry_id,
                "state_sha256": digest,
                "lighting": rig_meta,
                "geometry_quality": geometry_quality,
                "render_contract": {"engine": "CYCLES", "resolution": cfg["render"]["resolution"], "samples": cfg["render"]["samples"], "seed": cfg["render"]["seed"], "denoising": False},
            }
            ref_stage = folder / "references" / reference_id
            ref_target = root / "references" / reference_id
            ref_stage.mkdir(parents=True, exist_ok=True)
            ref_stats = render_rgb(scene, ref_stage / "clear_rgb")
            assert_unchanged(scene, pool, digest)
            for name in ("depth_range_m.exr", "depth_camera_z_m.exr", "semantic_id.png", "instance_id.png", "valid_mask.png"):
                shutil.copy2(folder / "geometry" / name, ref_stage / name)
            clear_quality = rgb_decision(rgb_metrics(read_render_rgb(ref_stage / "clear_rgb.exr"), mask, sem), cfg, state["lighting_mode"])
            atomic_json(ref_stage / "quality.json", clear_quality)
            if not clear_quality["passed"]:
                raise CandidateRejected("clear:" + ",".join(clear_quality["reasons"]), clear_quality)
            ref_meta = {
                **common,
                "state": scene_snapshot,
                "geometry_state": geo_snapshot,
                "geometry_plan": geo_plan,
                "full_instance_registry": registry,
                "labels": geometa,
                "rgb_quality": clear_quality,
                "render": ref_stats,
                "clear_definition": "No seawater volume or air-water surface",
                "water_objects_in_scene": [obj.name for obj in scene.objects if obj.get("is_water")],
                "asset_summary": handle["ecology"]["summary"],
            }
            atomic_json(ref_stage / "metadata.json", ref_meta)
            seal_record(ref_stage, REFERENCE_FILES, common)
            if not readable_bundle(ref_stage, "reference", common):
                raise RuntimeError("FATAL_V3_BATCH_REFERENCE_EXPORT_INVALID")

            rows = []
            to_promote = [(ref_stage, ref_target)]
            water_handle = None
            for index, preset in enumerate(water["effective_presets"]):
                if water_handle is None:
                    water_handle = build_water(scene, 5, preset, extent=70, bottom_z=-3)
                    handle["water"] = water_handle
                else:
                    set_water_preset(water_handle, preset)
                set_water_enabled(water_handle, True)
                sample_id = f"{key}_{state['lighting_mode']}_d5_{preset['id']}"
                sample_stage = folder / "samples" / sample_id
                sample_target = root / "samples" / sample_id
                sample_stage.mkdir(parents=True, exist_ok=True)
                water_meta = water_metadata(water_handle)
                template = water_template(water_handle)
                expected = {
                    **common,
                    "sample_id": sample_id,
                    "reference_path": str(ref_target.relative_to(root)),
                    "seabed_depth_m": 5,
                    "water_id": preset["id"],
                    "camera_submersion_m": 5 - handle["camera"].matrix_world.translation.z,
                    "water": water_meta,
                    "water_template_sha256": canonical_hash(template),
                }
                assert_unchanged(scene, pool, digest)
                stats = render_rgb(scene, sample_stage / "degraded_rgb")
                assert_unchanged(scene, pool, digest)
                sample_quality = rgb_decision(rgb_metrics(read_render_rgb(sample_stage / "degraded_rgb.exr"), mask, sem), cfg, state["lighting_mode"])
                atomic_json(sample_stage / "quality.json", sample_quality)
                if not sample_quality["passed"]:
                    raise CandidateRejected(sample_id + ":" + ",".join(sample_quality["reasons"]), sample_quality)
                sample_meta = {**expected, "water_plan": water, "water_template": template, "rgb_quality": sample_quality, "render": stats, "state_after_sha256": digest, "visible_fish_count": geometry_quality["metrics"]["visible_fish_count"]}
                atomic_json(sample_stage / "metadata.json", sample_meta)
                seal = seal_record(sample_stage, SAMPLE_FILES, expected)
                if not readable_bundle(sample_stage, "sample", expected):
                    raise RuntimeError("FATAL_V3_BATCH_SAMPLE_EXPORT_INVALID")
                to_promote.append((sample_stage, sample_target))
                rows.append({"sample_id": sample_id, "state_key": key, "scene_id": key, "scene_number": state["scene_number"], "state_index": 0, "geometry_id": geometry_id, "reference_id": reference_id, "reference_path": str(ref_target.relative_to(root)), "sample_path": str(sample_target.relative_to(root)), "layout_seed": state["layout_seed"], "layout_family": state["layout_family"], "split": state["split"], "lighting_mode": state["lighting_mode"], "water_id": preset["id"], "seabed_depth_m": 5, "accepted_attempt": attempt})
                set_water_enabled(water_handle, False)
                print("V3_BATCH_PAIR_COMPLETE " + sample_id, flush=True)
            for stage, target in to_promote:
                promote_directory(stage, target, root / "rejected" / "repaired_bundles")
            checkpoint = {**identity, **state, "status": "accepted", "accepted_attempt": attempt, "geometry_id": geometry_id, "geometry_plan": geo_plan, "lighting_plan": handle["lighting_plan"], "water_plan": water, "geometry_quality": geometry_quality, "reference_ids": [reference_id], "manifest_rows": rows, "blend": None, "blend_sha256": None, "seconds": time.time() - started}
            atomic_json(root / "states" / key / "accepted.json", checkpoint)
            event(root, "state_accepted", state_key=key, state_index=0, attempt_index=attempt, lighting_mode=state["lighting_mode"], status="accepted")
            cleanup_cached_handle(handle)
            return checkpoint
        except CandidateRejected as exc:
            event(root, "candidate_rejected", state_key=key, state_index=0, attempt_index=attempt, status="rejected", invalid_reason=str(exc), candidate_evidence=exc.evidence)
            cleanup_cached_handle(handle)
        except ValueError as exc:
            if not str(exc).startswith(("CANDIDATE_", "camera placement", "fish placement", "No active target")):
                cleanup_cached_handle(handle)
                raise
            event(root, "candidate_rejected", state_key=key, state_index=0, attempt_index=attempt, status="rejected", invalid_reason=str(exc))
            cleanup_cached_handle(handle)
        except BaseException:
            cleanup_cached_handle(handle)
            raise
    event(root, "state_failed", state_key=key, state_index=0, status="failed", invalid_reason="max_attempts_exhausted")
    raise RuntimeError("v3 batch scene exhausted max_attempts: " + key)


def run_scene(cfg: dict, plan: dict, state: dict, root: Path, identity: dict, save_native_blend: bool = False) -> dict:
    key = state["state_key"]
    started = time.time()
    for attempt in range(int(cfg["max_attempts"])):
        folder = root / "attempts" / key / f"a{attempt:02d}"
        if folder.exists():
            previous = root / "rejected" / key / (f"a{attempt:02d}_previous_" + str(time.time_ns()))
            previous.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(folder), str(previous))
        folder.mkdir(parents=True, exist_ok=True)
        event(root, "candidate_started", state_key=key, state_index=0, attempt_index=attempt, status="running")
        handle = None
        try:
            layout = {"seed": state["layout_seed"], "split": state["split"]}
            handle = build_geometry(cfg, layout, 0, attempt)
            pool = handle["objects"]
            active = handle["active_objects"]
            geo_plan = handle["geometry_plan"]
            registry = {str(item["instance_id"]): item for item in geo_plan["instances"]}
            geo_sha, geo_snapshot = geometry_digest(handle["base"], pool)
            geometry_id = "geo_" + geo_sha[:24]
            geometa = export_labels(handle["base"], folder / "geometry", active)
            depth, sem, inst, mask = label_arrays(folder / "geometry")
            geometry_quality = geometry_gate(geometry_metrics(depth, sem, inst, registry, cfg), inst, registry, cfg)
            atomic_json(folder / "geometry_plan.json", geo_plan)
            atomic_json(folder / "geometry_quality.json", geometry_quality)
            if not geometry_quality["passed"]:
                raise CandidateRejected("geometry:" + ",".join(geometry_quality["reasons"]), geometry_quality)
            water = water_plan(cfg, state, attempt)
            single_cfg = dict(cfg)
            single_cfg["lighting_modes"] = [state["lighting_mode"]]
            groups = build_lighting_groups(handle, single_cfg, key, water["effective_presets"])
            if len(groups) != 1 or groups[0]["mode"] != state["lighting_mode"]:
                raise RuntimeError("FATAL_V3_BATCH_EXPECTED_ONE_LIGHTING_GROUP")
            group = groups[0]
            clear = group["clear"]
            assert_clear_scene(clear, pool)
            digest, scene_snapshot = state_digest(clear, pool)
            for variant in group["variants"]:
                assert_unchanged(variant["scene"], pool, digest)
            rig_meta = lighting_metadata(group["lighting"])
            rig_meta["clearance"] = validate_light_positions(handle, group["lighting"], cfg)
            reference_id = "ref_" + digest[:24]
            common = {
                **identity,
                **state,
                "reference_id": reference_id,
                "geometry_id": geometry_id,
                "state_sha256": digest,
                "lighting": rig_meta,
                "geometry_quality": geometry_quality,
                "render_contract": {"engine": "CYCLES", "resolution": cfg["render"]["resolution"], "samples": cfg["render"]["samples"], "seed": cfg["render"]["seed"], "denoising": False},
            }
            ref_stage = folder / "references" / reference_id
            ref_target = root / "references" / reference_id
            ref_stage.mkdir(parents=True, exist_ok=True)
            ref_stats = render_rgb(clear, ref_stage / "clear_rgb")
            assert_unchanged(clear, pool, digest)
            for name in ("depth_range_m.exr", "depth_camera_z_m.exr", "semantic_id.png", "instance_id.png", "valid_mask.png"):
                shutil.copy2(folder / "geometry" / name, ref_stage / name)
            clear_quality = rgb_decision(rgb_metrics(read_render_rgb(ref_stage / "clear_rgb.exr"), mask, sem), cfg, state["lighting_mode"])
            atomic_json(ref_stage / "quality.json", clear_quality)
            if not clear_quality["passed"]:
                raise CandidateRejected("clear:" + ",".join(clear_quality["reasons"]), clear_quality)
            ref_meta = {
                **common,
                "state": scene_snapshot,
                "geometry_state": geo_snapshot,
                "geometry_plan": geo_plan,
                "full_instance_registry": registry,
                "labels": geometa,
                "rgb_quality": clear_quality,
                "render": ref_stats,
                "clear_definition": "No seawater volume or air-water surface",
                "water_objects_in_scene": [obj.name for obj in clear.objects if obj.get("is_water")],
                "asset_summary": handle["ecology"]["summary"],
            }
            atomic_json(ref_stage / "metadata.json", ref_meta)
            seal_record(ref_stage, REFERENCE_FILES, common)
            if not readable_bundle(ref_stage, "reference", common):
                raise RuntimeError("FATAL_V3_BATCH_REFERENCE_EXPORT_INVALID")
            rows = []
            samples_to_promote = [(ref_stage, ref_target)]
            for variant in group["variants"]:
                preset = variant["preset"]
                wet = variant["scene"]
                sample_id = f"{key}_{state['lighting_mode']}_d5_{preset['id']}"
                sample_stage = folder / "samples" / sample_id
                sample_target = root / "samples" / sample_id
                sample_stage.mkdir(parents=True, exist_ok=True)
                expected = {
                    **common,
                    "sample_id": sample_id,
                    "reference_path": str(ref_target.relative_to(root)),
                    "seabed_depth_m": 5,
                    "water_id": preset["id"],
                    "camera_submersion_m": 5 - handle["camera"].matrix_world.translation.z,
                    "water": water_metadata(variant["water"]),
                    "water_template_sha256": canonical_hash(water_template(variant["water"])),
                }
                assert_unchanged(wet, pool, digest)
                stats = render_rgb(wet, sample_stage / "degraded_rgb")
                assert_unchanged(wet, pool, digest)
                sample_quality = rgb_decision(rgb_metrics(read_render_rgb(sample_stage / "degraded_rgb.exr"), mask, sem), cfg, state["lighting_mode"])
                atomic_json(sample_stage / "quality.json", sample_quality)
                if not sample_quality["passed"]:
                    raise CandidateRejected(sample_id + ":" + ",".join(sample_quality["reasons"]), sample_quality)
                sample_meta = {**expected, "water_plan": water, "water_template": water_template(variant["water"]), "rgb_quality": sample_quality, "render": stats, "state_after_sha256": digest, "visible_fish_count": geometry_quality["metrics"]["visible_fish_count"]}
                atomic_json(sample_stage / "metadata.json", sample_meta)
                seal = seal_record(sample_stage, SAMPLE_FILES, expected)
                if not readable_bundle(sample_stage, "sample", expected):
                    raise RuntimeError("FATAL_V3_BATCH_SAMPLE_EXPORT_INVALID")
                samples_to_promote.append((sample_stage, sample_target))
                rows.append({"sample_id": sample_id, "state_key": key, "scene_id": key, "scene_number": state["scene_number"], "state_index": 0, "geometry_id": geometry_id, "reference_id": reference_id, "reference_path": str(ref_target.relative_to(root)), "sample_path": str(sample_target.relative_to(root)), "layout_seed": state["layout_seed"], "layout_family": state["layout_family"], "split": state["split"], "lighting_mode": state["lighting_mode"], "water_id": preset["id"], "seabed_depth_m": 5, "accepted_attempt": attempt})
                print("V3_BATCH_PAIR_COMPLETE " + sample_id, flush=True)
            if save_native_blend:
                native_save(handle, root / "scenes" / f"{key}.blend", cfg, {"state": state, "geometry_id": geometry_id, "reference_id": reference_id, "lighting_plan": handle["lighting_plan"], "water_plan": water})
            for stage, target in samples_to_promote:
                promote_directory(stage, target, root / "rejected" / "repaired_bundles")
            checkpoint = {
                **identity,
                **state,
                "status": "accepted",
                "accepted_attempt": attempt,
                "geometry_id": geometry_id,
                "geometry_plan": geo_plan,
                "lighting_plan": handle["lighting_plan"],
                "water_plan": water,
                "geometry_quality": geometry_quality,
                "reference_ids": [reference_id],
                "manifest_rows": rows,
                "blend": str((root / "scenes" / f"{key}.blend").relative_to(root)) if save_native_blend else None,
                "blend_sha256": file_hash(root / "scenes" / f"{key}.blend") if save_native_blend else None,
                "seconds": time.time() - started,
            }
            atomic_json(root / "states" / key / "accepted.json", checkpoint)
            event(root, "state_accepted", state_key=key, state_index=0, attempt_index=attempt, lighting_mode=state["lighting_mode"], status="accepted")
            cleanup_cached_handle(handle)
            return checkpoint
        except CandidateRejected as exc:
            event(root, "candidate_rejected", state_key=key, state_index=0, attempt_index=attempt, status="rejected", invalid_reason=str(exc), candidate_evidence=exc.evidence)
        except ValueError as exc:
            if not str(exc).startswith(("CANDIDATE_", "camera placement", "fish placement", "No active target")):
                cleanup_cached_handle(handle)
                raise
            event(root, "candidate_rejected", state_key=key, state_index=0, attempt_index=attempt, status="rejected", invalid_reason=str(exc))
            cleanup_cached_handle(handle)
        except BaseException:
            cleanup_cached_handle(handle)
            raise
    event(root, "state_failed", state_key=key, state_index=0, status="failed", invalid_reason="max_attempts_exhausted")
    raise RuntimeError("v3 batch scene exhausted max_attempts: " + key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=500)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--save-native", action="store_true")
    parser.add_argument("--quit", action="store_true", help="quit Blender after the batch range completes")
    parser.add_argument("--preview-tag", default="current", help="unique label for an isolated preview root")
    parser.add_argument("--template-cache", action="store_true", help="reuse scan meshes from the loaded v3 blend and skip GLB import")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    cfg = load_config(args.config)
    validate_batch_config(cfg)
    if not 1 <= args.start <= args.end <= 500:
        raise ValueError("scene range must be within 1..500")
    plan = batch_plan(cfg)
    cfg_hash = canonical_hash(cfg)
    source = batch_source_hash()
    runtime = runtime_signature(cfg)
    if args.template_cache:
        configure_template_asset_cache()
        globals()["build_geometry"] = build_geometry_in_place
        globals()["_IN_PLACE"] = True
    root = ROOT / cfg["output_root"]
    if args.preview:
        safe_tag = "".join(character for character in str(args.preview_tag) if character.isalnum() or character in "-_") or "current"
        root = ROOT / "previews" / ("v3_500_" + cfg_hash[:12] + "_" + source[:8] + "_" + safe_tag)
    root.mkdir(parents=True, exist_ok=True)
    gate = None if args.preview else verify_calibration_gate(cfg, source, cfg_hash, runtime)
    identity = {"schema_version": 2, "batch_dataset_id": cfg.get("batch_dataset_id", "seabed-v3-500-scenes"), "source_sha256": source, "config_sha256": cfg_hash, "runtime_signature": runtime, "runtime_sha256": canonical_hash(runtime), "calibration_gate": gate}
    full_plan = {**plan, "config": cfg, "config_hash": cfg_hash, "source_hash": source, "runtime": runtime, "calibration_gate": identity["calibration_gate"], "batch_source_note": "v3 source plus additive tools/run_500_scenes.py"}
    ensure_plan(root, full_plan)
    accepted_count = 0
    for state in plan["geometry_states"][args.start - 1 : args.end]:
        if checkpoint_complete(root, state, full_plan, identity):
            accepted_count += 1
            print("V3_BATCH_SCENE_REUSED " + state["scene_id"], flush=True)
            continue
        if _IN_PLACE:
            run_scene_in_place(cfg, full_plan, state, root, identity)
        else:
            run_scene(cfg, full_plan, state, root, identity, save_native_blend=args.save_native)
        accepted_count += 1
        rebuild_manifest(root, full_plan)
    rebuild_manifest(root, full_plan)
    atomic_json(root / "batch_run.json", {"status": "complete" if accepted_count == args.end - args.start + 1 else "partial", "start": args.start, "end": args.end, "accepted_in_range": accepted_count, "root": str(root), "source_sha256": source, "config_sha256": cfg_hash, "runtime_sha256": canonical_hash(runtime), "lighting_mode_counts": plan["lighting_mode_counts"]})
    print("V3_BATCH_COMPLETE " + json.dumps({"start": args.start, "end": args.end, "accepted_in_range": accepted_count, "root": str(root)}, ensure_ascii=False), flush=True)
    if args.quit:
        bpy.ops.wm.quit_blender()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        (ROOT / "reports" / "last_batch_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
