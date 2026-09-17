#!/usr/bin/env python3
"""Independent strong-water seed-noise check for the formal v2 dataset.

Run this file inside one background Blender process, for example::

    Blender --background --factory-startup --disable-autoexec \
      --python seabed_dataset_v2/tests/validate_dataset_noise_v2.py -- \
      /path/to/seabed_dataset_v2/outputs/first_round_v1

The accepted native files and formal RGB/GT files are read only.  A second
render is written below ``reports/dataset_noise_v2`` and the native blend is
never saved after it is opened.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "seabed_dataset/.deps/py313"))

from src.contracts import atomic_json, canonical_hash, file_hash
from src.gates import noise_gate
from src.labels.formats import read_png
from src.quality import noise_metrics
from src.render import read_render_rgb, render_rgb, select_device


MODES = ("natural", "artificial", "mixed")
WATER_ID = "strong"
SECOND_SEED_OFFSET = 9011
EXPECTED_GROUPS = 9
EXPECTED_SAMPLES = 512
REPORT_ROOT = ROOT / "reports" / "dataset_noise_v2"


class RuntimeMismatch(RuntimeError):
    def __init__(self, message: str, details: dict[str, Any]):
        super().__init__(message)
        self.details = details


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _progress(path: Path, report: dict[str, Any]) -> None:
    atomic_json(path, report)


def _failure(state_key: str, mode: str, reason: str, **extra: Any) -> dict[str, Any]:
    result = {
        "state_key": state_key,
        "lighting_mode": mode,
        "water_id": WATER_ID,
        "status": "failed",
        "passed": False,
        "error": reason,
    }
    result.update(extra)
    return result


def _expected_states(plan: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    states = []
    for item in plan.get("geometry_states", []):
        if isinstance(item, dict) and isinstance(item.get("state_key"), str):
            states.append(item["state_key"])
    if not states:
        states = sorted({str(row.get("state_key")) for row in rows if row.get("state_key")})
    return states[:3]


def _scene_name(state_key: str, mode: str, row: dict[str, Any]) -> str:
    depth = row.get("seabed_depth_m", 5)
    try:
        depth_text = f"{float(depth):g}"
    except (TypeError, ValueError):
        depth_text = str(depth)
    return f"{state_key}_{mode}_D{depth_text}_{WATER_ID}"


def _native_scene_summary(scene: Any) -> dict[str, Any]:
    lights = []
    for obj in scene.objects:
        if obj.type != "LIGHT":
            continue
        data = obj.data
        lights.append(
            {
                "name": obj.name,
                "type": data.type,
                "energy": float(data.energy),
                "parent": obj.parent.name if obj.parent else None,
            }
        )
    return {
        "name": scene.name,
        "camera": scene.camera.name if scene.camera else None,
        "world": scene.world.name if scene.world else None,
        "lights": lights,
        "water_objects": [obj.name for obj in scene.objects if obj.get("is_water")],
        "engine": scene.render.engine,
        "resolution": [int(scene.render.resolution_x), int(scene.render.resolution_y)],
        "samples": int(scene.cycles.samples),
        "seed": int(scene.cycles.seed),
        "device": scene.cycles.device,
        "denoising": bool(scene.cycles.use_denoising),
    }


def _render_setting_snapshot(scene: Any) -> dict[str, Any]:
    image_settings = scene.render.image_settings
    return {
        "file_format": image_settings.file_format,
        "color_mode": image_settings.color_mode,
        "color_depth": image_settings.color_depth,
        "compression": image_settings.compression,
        "exr_codec": image_settings.exr_codec,
        "seed": int(scene.cycles.seed),
        "samples": int(scene.cycles.samples),
        "device": scene.cycles.device,
    }


def _restore_render_settings(scene: Any, snapshot: dict[str, Any]) -> None:
    image_settings = scene.render.image_settings
    for key in ("file_format", "color_mode", "color_depth", "compression", "exr_codec"):
        try:
            setattr(image_settings, key, snapshot[key])
        except (AttributeError, TypeError, ValueError, RuntimeError):
            pass
    scene.cycles.seed = snapshot["seed"]
    scene.cycles.samples = snapshot["samples"]
    scene.cycles.device = snapshot["device"]


def _runtime_check(scene: Any, selected_device: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Read runtime facts directly, without configuring or mutating the scene."""

    raw_build_hash = getattr(getattr(bpy, "app", None), "build_hash", "")
    if isinstance(raw_build_hash, bytes):
        build_hash = raw_build_hash.decode(errors="replace")
    else:
        build_hash = str(raw_build_hash)
    actual: dict[str, Any] = {
        "blender": str(getattr(getattr(bpy, "app", None), "version_string", "")),
        "build_hash": build_hash,
        "python": platform.python_version(),
        "os": platform.mac_ver()[0],
        "device": selected_device,
        "working_space": "Linear Rec.709",
        "display": f"{scene.view_settings.view_transform}/{scene.display_settings.display_device}",
        "autoexec_disabled_cli": "--disable-autoexec" in sys.argv,
        "user_preference_scripts_auto_execute": bool(
            bpy.context.preferences.filepaths.use_scripts_auto_execute
        ),
    }
    try:
        ocio = Path(bpy.utils.resource_path("LOCAL")) / "datafiles/colormanagement/config.ocio"
        actual["ocio_config_sha256"] = file_hash(ocio) if ocio.is_file() else None
    except (AttributeError, OSError, TypeError, ValueError):
        actual["ocio_config_sha256"] = None
    expected = plan.get("runtime") if isinstance(plan.get("runtime"), dict) else {}
    comparisons = {}
    for key in actual:
        if key not in expected:
            continue
        comparisons[key] = {
            "expected": expected[key],
            "actual": actual[key],
            "match": canonical_hash(expected[key]) == canonical_hash(actual[key]),
        }
    mismatches = [key for key, item in comparisons.items() if not item["match"]]
    return {
        "expected": expected,
        "actual": actual,
        "comparisons": comparisons,
        "compared_fields": sorted(comparisons),
        "missing_expected_fields": sorted(set(actual) - set(expected)),
        "mismatches": mismatches,
        "all_match": not mismatches,
    }


def _load_group_inputs(root: Path, row: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    sample_dir = root / row["sample_path"]
    reference_dir = root / row["reference_path"]
    sample_meta_path = sample_dir / "metadata.json"
    reference_meta_path = reference_dir / "metadata.json"
    if not sample_meta_path.is_file() or not reference_meta_path.is_file():
        raise RuntimeError("sample/reference metadata is missing")
    sample_meta = _json(sample_meta_path)
    reference_meta = _json(reference_meta_path)
    if sample_meta.get("sample_id") != row.get("sample_id"):
        raise RuntimeError("sample metadata identity does not match manifest")
    if sample_meta.get("reference_id") != row.get("reference_id"):
        raise RuntimeError("sample metadata reference does not match manifest")
    if sample_meta.get("lighting_mode") != row.get("lighting_mode"):
        raise RuntimeError("sample metadata lighting mode does not match manifest")
    metadata_water_id = sample_meta.get("water_id")
    if metadata_water_id is not None and metadata_water_id != WATER_ID:
        raise RuntimeError("sample metadata is not the strong-water variant")
    water_metadata = sample_meta.get("water")
    if isinstance(water_metadata, dict):
        preset_metadata = water_metadata.get("preset")
        if isinstance(preset_metadata, dict) and preset_metadata.get("id") not in (None, WATER_ID):
            raise RuntimeError("sample native water metadata is not the strong preset")
    if sample_meta.get("reference_path") not in (None, row.get("reference_path")):
        raise RuntimeError("sample metadata reference_path does not match manifest")
    if "lighting" in sample_meta and canonical_hash(sample_meta.get("lighting")) != canonical_hash(reference_meta.get("lighting")):
        raise RuntimeError("sample and Clear lighting metadata differ")
    if reference_meta.get("reference_id") != row.get("reference_id"):
        raise RuntimeError("reference metadata identity does not match manifest")
    if reference_meta.get("lighting_mode") != row.get("lighting_mode"):
        raise RuntimeError("reference metadata lighting mode does not match manifest")
    expected_source = plan.get("source_hash")
    expected_config = plan.get("config_hash")
    expected_runtime = canonical_hash(plan.get("runtime", {})) if plan.get("runtime") is not None else None
    signatures = {
        "source_sha256": sample_meta.get("source_sha256"),
        "config_sha256": sample_meta.get("config_sha256"),
        "runtime_sha256": sample_meta.get("runtime_sha256"),
        "expected_source_sha256": expected_source,
        "expected_config_sha256": expected_config,
        "expected_runtime_sha256": expected_runtime,
    }
    if expected_source is not None and sample_meta.get("source_sha256") != expected_source:
        raise RuntimeError("sample source signature does not match plan")
    if expected_config is not None and sample_meta.get("config_sha256") != expected_config:
        raise RuntimeError("sample config signature does not match plan")
    if expected_runtime is not None and sample_meta.get("runtime_sha256") != expected_runtime:
        raise RuntimeError("sample runtime signature does not match plan")
    for key, expected in (("source_sha256", expected_source), ("config_sha256", expected_config), ("runtime_sha256", expected_runtime)):
        if expected is not None and reference_meta.get(key) != expected:
            raise RuntimeError(f"reference {key} does not match plan")
    render = sample_meta.get("render")
    if not isinstance(render, dict):
        raise RuntimeError("sample render metadata is missing")
    original_seed = render.get("seed")
    original_samples = render.get("samples")
    if isinstance(original_seed, bool) or not isinstance(original_seed, int):
        raise RuntimeError("sample render.seed is not an integer")
    if isinstance(original_samples, bool) or not isinstance(original_samples, int):
        raise RuntimeError("sample render.samples is not an integer")
    if original_samples != EXPECTED_SAMPLES:
        raise RuntimeError(f"formal strong sample is not {EXPECTED_SAMPLES} samples")
    reference_render = reference_meta.get("render")
    if isinstance(reference_render, dict):
        if reference_render.get("seed") not in (None, original_seed) or reference_render.get("samples") not in (None, original_samples):
            raise RuntimeError("Clear and strong render seeds/samples differ")
    clear_exr = reference_dir / "clear_rgb.exr"
    strong_exr = sample_dir / "degraded_rgb.exr"
    mask_png = reference_dir / "valid_mask.png"
    for path in (clear_exr, strong_exr, mask_png):
        if not path.is_file():
            raise RuntimeError(f"required source file is missing: {path.name}")
    return {
        "sample_dir": sample_dir,
        "reference_dir": reference_dir,
        "sample_meta": sample_meta,
        "reference_meta": reference_meta,
        "clear_exr": clear_exr,
        "strong_exr": strong_exr,
        "mask_png": mask_png,
        "original_seed": int(original_seed),
        "original_samples": int(original_samples),
        "second_seed": int(original_seed) + SECOND_SEED_OFFSET,
        "signatures": signatures,
    }


def _run_group(root: Path, report_dir: Path, row: dict[str, Any], checkpoint: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    state_key = str(row["state_key"])
    mode = str(row["lighting_mode"])
    inputs = _load_group_inputs(root, row, plan)
    blend_value = checkpoint.get("blend")
    if not isinstance(blend_value, str) or not blend_value:
        raise RuntimeError("accepted checkpoint has no blend path")
    blend = (root / blend_value).resolve() if not Path(blend_value).is_absolute() else Path(blend_value).resolve()
    if not blend.is_file():
        raise RuntimeError("accepted native blend is missing")
    actual_blend_sha = file_hash(blend)
    expected_blend_sha = checkpoint.get("blend_sha256")
    if expected_blend_sha != actual_blend_sha:
        raise RuntimeError("accepted native blend SHA-256 does not match checkpoint")
    expected_blend_bytes = checkpoint.get("blend_bytes")
    if expected_blend_bytes is not None and int(expected_blend_bytes) != blend.stat().st_size:
        raise RuntimeError("accepted native blend byte count does not match checkpoint")
    open_result = bpy.ops.wm.open_mainfile(filepath=str(blend), load_ui=False, use_scripts=False)
    if open_result != {"FINISHED"}:
        raise RuntimeError(f"open_mainfile failed: {open_result}")
    scene_name = _scene_name(state_key, mode, row)
    scene = bpy.data.scenes.get(scene_name)
    if scene is None:
        candidates = [
            candidate
            for candidate in bpy.data.scenes
            if candidate.name.startswith(f"{state_key}_{mode}_D") and candidate.name.endswith(f"_{WATER_ID}")
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"native strong scene is unavailable: {scene_name}")
        scene = candidates[0]
    native = _native_scene_summary(scene)
    if native["camera"] is None or native["world"] is None:
        raise RuntimeError("native scene has no camera or World")
    if len(native["lights"]) != 2 or any(item["type"] != "SPOT" for item in native["lights"]):
        raise RuntimeError("native strong scene does not contain two Spot lights")
    if not native["water_objects"]:
        raise RuntimeError("native strong scene has no water volume object")
    expected_resolution = inputs["sample_meta"].get("render", {}).get("resolution")
    if expected_resolution and native["resolution"] != [int(expected_resolution[0]), int(expected_resolution[1])]:
        raise RuntimeError("native resolution does not match sample metadata")
    if native["engine"] != "CYCLES" or native["samples"] != inputs["original_samples"] or native["seed"] != inputs["original_seed"]:
        raise RuntimeError("native scene render settings do not match original sample metadata")
    clear = read_render_rgb(inputs["clear_exr"])
    strong = read_render_rgb(inputs["strong_exr"])
    mask_payload = read_png(inputs["mask_png"])
    mask = np.asarray(mask_payload["pixels"], dtype=np.uint8)
    if set(np.unique(mask)).difference({0, 255}):
        raise RuntimeError("valid_mask is not encoded as 0/255")
    valid_mask = mask == 255
    if clear.shape != strong.shape or clear.shape[:2] != valid_mask.shape:
        raise RuntimeError("clear/strong/mask dimensions do not match")
    group_dir = report_dir / state_key / mode
    group_dir.mkdir(parents=True, exist_ok=True)
    original_settings = _render_setting_snapshot(scene)
    selected_device = select_device(scene)
    runtime = _runtime_check(scene, selected_device, plan)
    if not runtime["all_match"]:
        raise RuntimeMismatch("native runtime signature does not match plan", runtime)
    if int(scene.cycles.samples) != inputs["original_samples"] or int(scene.cycles.seed) != inputs["original_seed"]:
        raise RuntimeError("select_device changed native seed/sample before the second render")
    second_stem = group_dir / "strong_second_seed"
    render_stats: dict[str, Any] | None = None
    try:
        scene.cycles.seed = inputs["second_seed"]
        scene.cycles.samples = inputs["original_samples"]
        render_stats = render_rgb(scene, second_stem)
    finally:
        _restore_render_settings(scene, original_settings)
    settings_restored = _render_setting_snapshot(scene) == original_settings
    if not settings_restored:
        raise RuntimeError("render_settings_restored=False; group rejected")
    blend_sha_after = file_hash(blend)
    if blend_sha_after != actual_blend_sha:
        raise RuntimeError("accepted native blend SHA-256 changed after read-only render")
    second = read_render_rgb(Path(str(second_stem) + ".exr"))
    metrics = noise_metrics(clear, strong, second, valid_mask)
    cfg = plan.get("config")
    if not isinstance(cfg, dict):
        raise RuntimeError("plan has no configuration for noise gate")
    decision = noise_gate(metrics, cfg)
    output_files = {}
    for suffix in (".exr", ".png"):
        path = Path(str(second_stem) + suffix)
        if not path.is_file():
            raise RuntimeError(f"second render output is missing: {path.name}")
        output_files[path.name] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    return {
        "state_key": state_key,
        "lighting_mode": mode,
        "water_id": WATER_ID,
        "status": "passed" if decision["passed"] else "failed_gate",
        "passed": bool(decision["passed"]),
        "reference_id": row.get("reference_id"),
        "sample_id": row.get("sample_id"),
        "accepted_attempt": row.get("accepted_attempt"),
        "blend": _relative(root, blend),
        "blend_sha256_before": actual_blend_sha,
        "blend_sha256_after": blend_sha_after,
        "blend_unchanged": True,
        "source_sha256": inputs["signatures"]["source_sha256"],
        "config_sha256": inputs["signatures"]["config_sha256"],
        "runtime_sha256": inputs["signatures"]["runtime_sha256"],
        "original_seed": inputs["original_seed"],
        "second_seed": inputs["second_seed"],
        "samples": inputs["original_samples"],
        "selected_device": selected_device,
        "runtime": runtime,
        "native_scene": native,
        "render": render_stats,
        "noise": metrics,
        "decision": decision,
        "source_files": {
            "clear_rgb_exr": _relative(root, inputs["clear_exr"]),
            "strong_rgb_exr": _relative(root, inputs["strong_exr"]),
            "valid_mask": _relative(root, inputs["mask_png"]),
        },
        "second_render_files": output_files,
        "render_settings_restored": settings_restored,
    }


def main() -> int:
    if "--" not in sys.argv or len(sys.argv) <= sys.argv.index("--") + 1:
        raise SystemExit("usage: validate_dataset_noise_v2.py -- <formal_output_root>")
    root = Path(sys.argv[sys.argv.index("--") + 1]).expanduser().resolve()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_ROOT / "result.json"
    report: dict[str, Any] = {
        "schema_version": 2,
        "status": "running",
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": str(root),
        "expected_groups": EXPECTED_GROUPS,
        "water_id": WATER_ID,
        "second_seed_offset": SECOND_SEED_OFFSET,
        "groups": [],
    }
    _progress(report_path, report)
    manifest_path = root / "manifest.jsonl"
    plan_path = root / "plan.json"
    if not manifest_path.is_file() or not plan_path.is_file():
        report.update(
            {
                "status": "not_ready",
                "reason": "formal manifest/plan is not present; no formal 27-pair dataset is available",
                "completed_groups": 0,
                "passed_groups": 0,
                "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        _progress(report_path, report)
        print(json.dumps({"status": report["status"], "groups": 0}, ensure_ascii=False))
        return 2
    try:
        plan = _json(plan_path)
        rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError, TypeError) as exc:
        report.update({"status": "not_ready", "reason": f"formal index read failed: {exc}"})
        _progress(report_path, report)
        return 2
    if not isinstance(plan, dict) or not isinstance(rows, list):
        report.update({"status": "not_ready", "reason": "formal plan/manifest has invalid JSON structure"})
        _progress(report_path, report)
        return 2
    states = _expected_states(plan, rows)
    report.update(
        {
            "source_sha256": plan.get("source_hash"),
            "config_sha256": plan.get("config_hash"),
            "runtime_sha256": canonical_hash(plan.get("runtime", {})),
            "manifest_rows": len(rows),
            "geometry_states": states,
            "planned_groups": len(states) * len(MODES),
        }
    )
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = (str(row.get("state_key")), str(row.get("lighting_mode")), str(row.get("water_id")))
        by_key.setdefault(key, []).append(row)
    if len(rows) != 27 or len(states) != 3:
        report.update(
            {
                "status": "not_ready",
                "reason": "formal manifest is not the complete 27-pair/3-state first-round grid",
                "completed_groups": 0,
                "passed_groups": 0,
                "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        _progress(report_path, report)
        print(json.dumps({"status": report["status"], "groups": 0}, ensure_ascii=False))
        return 2
    for state_key in states:
        checkpoint_path = root / "states" / state_key / "accepted.json"
        checkpoint: dict[str, Any] | None = None
        checkpoint_error = None
        if checkpoint_path.is_file():
            try:
                checkpoint = _json(checkpoint_path)
                if not isinstance(checkpoint, dict) or checkpoint.get("status") != "accepted":
                    checkpoint_error = "accepted checkpoint is not accepted"
            except (OSError, ValueError, TypeError) as exc:
                checkpoint_error = f"accepted checkpoint read failed: {exc}"
        else:
            checkpoint_error = "accepted checkpoint is missing"
        for mode in MODES:
            matches = by_key.get((state_key, mode, WATER_ID), [])
            if len(matches) != 1:
                group = _failure(state_key, mode, f"expected one formal strong row, found {len(matches)}")
            elif checkpoint_error:
                group = _failure(state_key, mode, checkpoint_error)
            else:
                try:
                    group = _run_group(root, REPORT_ROOT, matches[0], checkpoint, plan)
                except RuntimeMismatch as exc:
                    group = _failure(state_key, mode, str(exc), runtime=exc.details)
                except Exception as exc:
                    group = _failure(state_key, mode, f"{type(exc).__name__}: {exc}")
            report["groups"].append(group)
            report["completed_groups"] = len(report["groups"])
            report["passed_groups"] = sum(bool(item.get("passed")) for item in report["groups"])
            print(
                f"DATASET_NOISE_GROUP {state_key} {mode} {group['status']} "
                f"{report['completed_groups']}/{EXPECTED_GROUPS}",
                flush=True,
            )
            _progress(report_path, report)
    all_passed = len(report["groups"]) == EXPECTED_GROUPS and all(item.get("passed") is True for item in report["groups"])
    report.update(
        {
            "status": "pass" if all_passed else "fail",
            "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "all_groups_passed": all_passed,
            "method": "native accepted blend reopened read-only; same 512 samples, second seed original+9011; src.quality.noise_metrics + src.gates.noise_gate",
        }
    )
    _progress(report_path, report)
    print(json.dumps({"status": report["status"], "groups": len(report["groups"]), "passed": report["passed_groups"]}, ensure_ascii=False))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
