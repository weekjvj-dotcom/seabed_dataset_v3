#!/usr/bin/env python3
"""Assemble the evidence-backed v2 acceptance report.

This file intentionally uses only the Python standard library.  It does not
open Blender, render, import project runtime modules, or change any dataset
artifact.  The default invocation reads the completed first-round output and
the reports written by the independent validation jobs, then writes
``reports/acceptance.json`` and ``reports/ACCEPTANCE.md``.  Missing or failed
evidence produces an ``incomplete`` report and a non-zero exit status.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_RELATIVE = Path("configs/first_round.json")
REPORTS_RELATIVE = Path("reports")
EXPECTED_MODES = ("natural", "artificial", "mixed")
EXPECTED_WATERS = ("mild", "medium", "strong")
EXPECTED_GEOMETRIES = 3
EXPECTED_REFERENCES = 9
EXPECTED_PAIRS = 27
EXPECTED_NATIVE_FILES = 3
EXPECTED_CALIBRATION_SAMPLES = 512
EXPECTED_CALIBRATION_LOW_SAMPLES = 128
EXPECTED_INSTANCE_PIXELS = 16
EXPECTED_RGB_NRMSE = 0.001
EXPECTED_BASELINE_SOURCE_SHA256 = (
    "0e5978ad4d0ecff9ee993ff30efcc5202a90d5cebd0b55e68c15ecf5438c7e09"
)
EXPECTED_BASELINE_BLEND_SHA256 = (
    "8e578590764fa1836fb3c69b94504637af2a68868fee1c9f77b13b8a68452a74"
)
PASS_STATUSES = {
    "pass",
    "passed",
    "complete",
    "completed",
    "ready",
    "validated",
    "pass_with_non_bitwise_rgb_rerender",
}


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hash(root: Path) -> str:
    source_root = root / "src"
    paths = sorted(source_root.rglob("*.py")) + [root / "run.py"]
    entries = {
        str(path.relative_to(root).as_posix()): _file_hash(path)
        for path in paths
        if path.is_file()
    }
    return _canonical_hash(entries)


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _path(root: Path, value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else root / candidate


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return os.path.relpath(path.resolve(), root.resolve()).replace(os.sep, "/")


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class Audit:
    """Collect failures without allowing one missing file to hide others."""

    def __init__(self, project_root: Path, reports_root: Path) -> None:
        self.project_root = project_root
        self.reports_root = reports_root
        self.missing: list[str] = []
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.evidence: dict[str, dict[str, Any]] = {}
        self.checks: dict[str, dict[str, Any]] = {}

    def _label(self, path: Path) -> str:
        return _relative(self.project_root, path)

    def load(self, path: Path, name: str, *, required: bool = True) -> Any | None:
        label = self._label(path)
        record = {"path": label, "exists": path.is_file()}
        self.evidence[name] = record
        if not path.is_file():
            if required:
                self.missing.append(label)
            else:
                self.warnings.append(f"optional evidence missing: {label}")
            return None
        try:
            value = _json(path)
        except (OSError, ValueError, TypeError) as exc:
            self.failures.append(f"{label}: invalid JSON ({exc})")
            record["json_error"] = str(exc)
            return None
        record["json"] = True
        return value

    def text(self, path: Path, name: str, *, required: bool = True) -> str | None:
        label = self._label(path)
        record = {"path": label, "exists": path.is_file()}
        self.evidence[name] = record
        if not path.is_file():
            if required:
                self.missing.append(label)
            else:
                self.warnings.append(f"optional evidence missing: {label}")
            return None
        try:
            value = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            self.failures.append(f"{label}: cannot read ({exc})")
            return None
        record["read"] = True
        return value

    def check(self, name: str, passed: bool, details: Any = None, error: str | None = None) -> bool:
        item: dict[str, Any] = {"passed": bool(passed)}
        if details is not None:
            item["details"] = details
        if error:
            item["error"] = error
        self.checks[name] = item
        if not passed and error:
            self.failures.append(f"{name}: {error}")
        elif not passed and details is not None:
            self.failures.append(f"{name}: {details}")
        elif not passed:
            self.failures.append(name)
        return bool(passed)

    def require(self, condition: bool, name: str, details: Any = None) -> bool:
        return self.check(name, condition, details=details)


def _status_pass(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in PASS_STATUSES


def _same(value: Any, expected: Any) -> bool:
    return _canonical_hash(value) == _canonical_hash(expected)


def _sealed_files(audit: Audit, directory: Path, expected: Iterable[str], label: str) -> bool:
    success_path = directory / "success.json"
    if not success_path.is_file():
        audit.failures.append(f"{_relative(audit.project_root, success_path)}: missing success.json")
        return False
    try:
        success = _json(success_path)
    except (OSError, ValueError, TypeError) as exc:
        audit.failures.append(f"{_relative(audit.project_root, success_path)}: invalid success JSON ({exc})")
        return False
    expected_set = set(expected)
    files = success.get("files") if isinstance(success, dict) else None
    ok = isinstance(success, dict) and success.get("schema_version") == 2 and isinstance(files, dict)
    if ok and set(files) != expected_set:
        audit.failures.append(f"{label}: success file set mismatch")
        ok = False
    if ok:
        for name, item in files.items():
            path = directory / name
            if not isinstance(item, dict) or not path.is_file():
                ok = False
                audit.failures.append(f"{label}: sealed file missing or malformed: {name}")
                continue
            try:
                actual_bytes = path.stat().st_size
                actual_hash = _file_hash(path)
            except OSError as exc:
                ok = False
                audit.failures.append(f"{label}: cannot hash {name}: {exc}")
                continue
            if item.get("bytes") != actual_bytes or item.get("sha256") != actual_hash:
                ok = False
                audit.failures.append(f"{label}: hash/byte mismatch: {name}")
    if ok and not isinstance(success.get("metadata"), dict):
        ok = False
        audit.failures.append(f"{label}: success metadata is missing or malformed")
    if ok and isinstance(files, dict) and isinstance(success.get("metadata"), dict):
        metadata_path = directory / "metadata.json"
        try:
            metadata = _json(metadata_path)
        except (OSError, ValueError, TypeError) as exc:
            ok = False
            audit.failures.append(f"{label}: metadata JSON cannot be read ({exc})")
        else:
            for key, expected_value in success["metadata"].items():
                if not isinstance(metadata, dict) or key not in metadata or not _same(metadata[key], expected_value):
                    ok = False
                    audit.failures.append(f"{label}: sealed metadata mismatch: {key}")
    return bool(ok)


def _manifest(root: Path, plan: dict[str, Any], audit: Audit) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    path = root / "manifest.jsonl"
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        audit.missing.append(_relative(audit.project_root, path))
        return rows, {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        audit.failures.append(f"{_relative(audit.project_root, path)}: cannot read ({exc})")
        return rows, {}
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            audit.failures.append(f"manifest line {line_number}: blank line")
            continue
        try:
            value = json.loads(line)
        except ValueError as exc:
            audit.failures.append(f"manifest line {line_number}: invalid JSON ({exc})")
            continue
        if not isinstance(value, dict):
            audit.failures.append(f"manifest line {line_number}: row is not an object")
            continue
        rows.append(value)

    jobs = {
        str(item.get("sample_id")): item
        for item in plan.get("pairs", [])
        if isinstance(item, dict) and item.get("sample_id") is not None
    }
    ids = [str(row.get("sample_id")) for row in rows]
    duplicates = sorted(item for item, count in Counter(ids).items() if count > 1)
    audit.require(len(rows) == EXPECTED_PAIRS, "formal_pair_count", {"actual": len(rows), "expected": EXPECTED_PAIRS})
    audit.require(len(set(ids)) == len(ids), "formal_sample_ids_unique", duplicates or True)
    audit.require(set(ids) == set(jobs), "formal_manifest_matches_plan", {"missing": sorted(set(jobs) - set(ids)), "extra": sorted(set(ids) - set(jobs))})
    compare_fields = (
        "state_key",
        "state_index",
        "layout_seed",
        "layout_family",
        "split",
        "lighting_mode",
        "water_id",
        "seabed_depth_m",
    )
    for row in rows:
        sample_id = str(row.get("sample_id"))
        expected = jobs.get(sample_id)
        if expected is None:
            continue
        for field in compare_fields:
            if row.get(field) != expected.get(field):
                audit.failures.append(f"manifest {sample_id}: {field} does not match plan")
    return rows, jobs


def _identity(plan: dict[str, Any]) -> dict[str, Any]:
    runtime = plan.get("runtime", {})
    return {
        "schema_version": 2,
        "source_sha256": plan.get("source_hash"),
        "config_sha256": plan.get("config_hash"),
        "runtime_sha256": _canonical_hash(runtime),
        "calibration_gate": plan.get("calibration_gate"),
        "baseline_source_sha256": plan.get("baseline_source_sha256"),
        "baseline_blend_sha256": plan.get("baseline_blend_sha256"),
    }


def _check_identity(audit: Audit, value: Any, expected: dict[str, Any], label: str) -> bool:
    if not isinstance(value, dict):
        audit.failures.append(f"{label}: metadata is not an object")
        return False
    ok = True
    for key, expected_value in expected.items():
        if key not in value or not _same(value.get(key), expected_value):
            audit.failures.append(f"{label}: identity field {key} mismatch")
            ok = False
    return ok


def _render_contract(plan: dict[str, Any]) -> dict[str, Any]:
    config = plan.get("config") if isinstance(plan.get("config"), dict) else {}
    render = config.get("render") if isinstance(config.get("render"), dict) else {}
    return {
        "engine": "CYCLES",
        "resolution": render.get("resolution"),
        "samples": render.get("samples"),
        "seed": render.get("seed"),
        "denoising": False,
    }


def _quality_summary(quality: Any) -> dict[str, Any]:
    if not isinstance(quality, dict):
        return {"passed": False}
    metrics = quality.get("metrics") if isinstance(quality.get("metrics"), dict) else {}
    bands = metrics.get("bands") if isinstance(metrics.get("bands"), dict) else {}
    band_summary: dict[str, Any] = {}
    for band in ("near", "middle", "far", "background"):
        item = bands.get(band) if isinstance(bands.get(band), dict) else {}
        band_summary[band] = {
            "fraction_of_valid": item.get("fraction_of_valid"),
            "ecology_pixel_count": item.get("ecology_pixel_count", item.get("ecology_pixels")),
        }
    meaningful = quality.get("meaningful_visibility") if isinstance(quality.get("meaningful_visibility"), dict) else {}
    morphologies = meaningful.get("morphologies") if isinstance(meaningful.get("morphologies"), dict) else {}
    return {
        "passed": quality.get("passed") is True,
        "valid_fraction": metrics.get("valid_fraction"),
        "bands": band_summary,
        "meaningful_visibility": {
            "minimum_pixels": meaningful.get("minimum_pixels", EXPECTED_INSTANCE_PIXELS),
            "fish_count": meaningful.get("fish_count", metrics.get("visible_fish_count")),
            "fish_morphologies": morphologies.get("fish", metrics.get("fish_morphologies", [])),
            "coral_morphologies": morphologies.get("coral", metrics.get("coral_morphologies", [])),
        },
    }


def _passed_mapping(value: Any) -> bool:
    return isinstance(value, dict) and value.get("passed") is True


def _accepted_states(
    root: Path,
    plan: dict[str, Any],
    rows: list[dict[str, Any]],
    audit: Audit,
) -> list[dict[str, Any]]:
    expected_identity = _identity(plan)
    state_items = [item for item in plan.get("geometry_states", []) if isinstance(item, dict)]
    summaries: list[dict[str, Any]] = []
    for state in state_items:
        key = str(state.get("state_key"))
        checkpoint_path = root / "states" / key / "accepted.json"
        checkpoint = audit.load(checkpoint_path, f"accepted:{key}")
        if not isinstance(checkpoint, dict):
            continue
        label = f"accepted:{key}"
        audit.require(checkpoint.get("schema_version") == 2, label + ":schema_version", checkpoint.get("schema_version"))
        audit.require(checkpoint.get("status") == "accepted", label + ":status", checkpoint.get("status"))
        _check_identity(audit, checkpoint, expected_identity, label)
        for field in ("state_key", "layout_seed", "state_index", "layout_family", "split"):
            audit.require(checkpoint.get(field) == state.get(field), label + f":{field}", {"actual": checkpoint.get(field), "expected": state.get(field)})
        attempt = checkpoint.get("accepted_attempt")
        config = plan.get("config") if isinstance(plan.get("config"), dict) else {}
        audit.require(
            type(attempt) is int and 0 <= attempt < int(config.get("max_attempts", 0)),
            label + ":accepted_attempt",
            attempt,
        )
        state_rows = [row for row in rows if row.get("state_key") == key]
        cp_rows = checkpoint.get("manifest_rows")
        audit.require(isinstance(cp_rows, list), label + ":manifest_rows_type")
        if isinstance(cp_rows, list):
            audit.require(
                {str(item.get("sample_id")) for item in cp_rows if isinstance(item, dict)}
                == {str(item.get("sample_id")) for item in state_rows},
                label + ":manifest_job_set",
            )
            for row in cp_rows:
                if isinstance(row, dict) and row.get("accepted_attempt") != attempt:
                    audit.failures.append(f"{label}: manifest row attempt mismatch")
        refs = sorted({str(row.get("reference_id")) for row in state_rows})
        checkpoint_refs = checkpoint.get("reference_ids")
        audit.require(isinstance(checkpoint_refs, list) and set(checkpoint_refs) == set(refs), label + ":reference_set", {"actual": checkpoint_refs, "expected": refs})
        geometry_plan = checkpoint.get("geometry_plan") if isinstance(checkpoint.get("geometry_plan"), dict) else {}
        audit.require(geometry_plan.get("state_index") == state.get("state_index"), label + ":geometry_plan_state")
        audit.require(geometry_plan.get("attempt_index") == attempt, label + ":geometry_plan_attempt")
        audit.require(geometry_plan.get("layout_seed") == state.get("layout_seed"), label + ":geometry_plan_layout")
        quality = checkpoint.get("geometry_quality")
        audit.require(isinstance(quality, dict) and quality.get("passed") is True, label + ":geometry_quality")
        blend_value = checkpoint.get("blend")
        blend = _path(root, blend_value) if isinstance(blend_value, str) else None
        blend_ok = blend is not None and blend.is_file()
        if blend_ok:
            actual_hash = _file_hash(blend)
            actual_bytes = blend.stat().st_size
            blend_ok = actual_hash == checkpoint.get("blend_sha256") and actual_bytes == checkpoint.get("blend_bytes")
        audit.require(blend_ok, label + ":native_blend_hash", {"blend": _relative(audit.project_root, blend) if blend else blend_value})
        camera = geometry_plan.get("camera") if isinstance(geometry_plan.get("camera"), dict) else {}
        summaries.append(
            {
                "state_key": key,
                "layout_seed": state.get("layout_seed"),
                "layout_family": state.get("layout_family"),
                "accepted_attempt": attempt,
                "geometry_id": checkpoint.get("geometry_id"),
                "reference_ids": refs,
                "camera": {
                    key: camera.get(key)
                    for key in (
                        "eye",
                        "target",
                        "roll_deg",
                        "pitch_down_actual_deg",
                        "pitch_down_deg",
                        "height_above_seabed_m",
                        "submersion_surface_m",
                    )
                    if key in camera
                },
                "quality": _quality_summary(quality),
                "native": {
                    "path": _relative(audit.project_root, blend) if blend else blend_value,
                    "sha256": checkpoint.get("blend_sha256"),
                    "bytes": checkpoint.get("blend_bytes"),
                },
            }
        )
    audit.require(len(summaries) == EXPECTED_GEOMETRIES, "accepted_geometry_count", {"actual": len(summaries), "expected": EXPECTED_GEOMETRIES})
    audit.require(len({item.get("geometry_id") for item in summaries}) == EXPECTED_GEOMETRIES, "geometry_ids_distinct")
    audit.require(len({item.get("native", {}).get("path") for item in summaries}) == EXPECTED_NATIVE_FILES, "native_files_distinct")
    return summaries


def _bundles(
    root: Path,
    plan: dict[str, Any],
    rows: list[dict[str, Any]],
    audit: Audit,
) -> dict[str, Any]:
    identity = _identity(plan)
    contract = _render_contract(plan)
    by_reference: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_reference[str(row.get("reference_id"))].append(row)
    reference_summaries: list[dict[str, Any]] = []
    sample_summaries: list[dict[str, Any]] = []
    for reference_id, group in sorted(by_reference.items()):
        first = group[0]
        reference_dir = root / str(first.get("reference_path", ""))
        metadata_path = reference_dir / "metadata.json"
        metadata = audit.load(metadata_path, f"reference_metadata:{reference_id}")
        if not isinstance(metadata, dict):
            continue
        label = f"reference:{reference_id}"
        audit.require(_sealed_files(audit, reference_dir, ("clear_rgb.exr", "clear_rgb.png", "depth_range_m.exr", "depth_camera_z_m.exr", "semantic_id.png", "instance_id.png", "valid_mask.png", "metadata.json"), label + ":seal"), label + ":seal")
        audit.require(metadata.get("schema_version") == 2, label + ":schema_version")
        _check_identity(audit, metadata, identity, label)
        for field in ("state_key", "state_index", "layout_seed", "layout_family", "split", "geometry_id", "reference_id", "lighting_mode"):
            audit.require(metadata.get(field) == first.get(field), label + f":{field}")
        audit.require(metadata.get("clear_definition") == "No seawater volume or air-water surface", label + ":clear_definition")
        audit.require(metadata.get("water_objects_in_scene") == [], label + ":clear_water_objects")
        audit.require(_passed_mapping(metadata.get("geometry_quality")), label + ":geometry_quality")
        audit.require(_passed_mapping(metadata.get("rgb_quality")), label + ":rgb_quality")
        audit.require(_same(metadata.get("render_contract"), contract), label + ":render_contract")
        lighting = metadata.get("lighting") if isinstance(metadata.get("lighting"), dict) else {}
        audit.require(lighting.get("mode") == first.get("lighting_mode"), label + ":lighting_mode")
        audit.require(isinstance(lighting.get("objects"), list) and len(lighting["objects"]) == 2, label + ":lighting_objects")
        audit.require(isinstance(metadata.get("state"), dict) and isinstance(metadata.get("geometry_state"), dict), label + ":state_snapshots")
        reference_summaries.append(
            {
                "reference_id": reference_id,
                "state_key": first.get("state_key"),
                "lighting_mode": first.get("lighting_mode"),
                "geometry_id": first.get("geometry_id"),
                "state_sha256": metadata.get("state_sha256"),
                "path": _relative(audit.project_root, reference_dir),
                "clear_definition": metadata.get("clear_definition"),
                "water_objects_in_scene": metadata.get("water_objects_in_scene"),
            }
        )
        for row in sorted(group, key=lambda item: str(item.get("water_id"))):
            sample_id = str(row.get("sample_id"))
            sample_dir = root / str(row.get("sample_path", ""))
            sample_metadata = audit.load(sample_dir / "metadata.json", f"sample_metadata:{sample_id}")
            if not isinstance(sample_metadata, dict):
                continue
            sample_label = f"sample:{sample_id}"
            audit.require(_sealed_files(audit, sample_dir, ("degraded_rgb.exr", "degraded_rgb.png", "metadata.json"), sample_label + ":seal"), sample_label + ":seal")
            audit.require(sample_metadata.get("schema_version") == 2, sample_label + ":schema_version")
            _check_identity(audit, sample_metadata, identity, sample_label)
            for field in ("sample_id", "state_key", "state_index", "layout_seed", "layout_family", "split", "geometry_id", "reference_id", "lighting_mode", "seabed_depth_m"):
                audit.require(sample_metadata.get(field) == row.get(field), sample_label + f":{field}")
            audit.require(sample_metadata.get("state_sha256") == metadata.get("state_sha256"), sample_label + ":state_sha256")
            audit.require(sample_metadata.get("state_after_sha256") == sample_metadata.get("state_sha256"), sample_label + ":state_after_sha256")
            audit.require(_passed_mapping(sample_metadata.get("geometry_quality")), sample_label + ":geometry_quality")
            audit.require(_passed_mapping(sample_metadata.get("rgb_quality")), sample_label + ":rgb_quality")
            audit.require(_same(sample_metadata.get("render_contract"), contract), sample_label + ":render_contract")
            water = sample_metadata.get("water") if isinstance(sample_metadata.get("water"), dict) else {}
            boundary = water.get("boundary") if isinstance(water.get("boundary"), dict) else {}
            metadata_water_id = sample_metadata.get("water_id")
            if metadata_water_id is None and isinstance(water.get("preset"), dict):
                metadata_water_id = water["preset"].get("id")
            audit.require(metadata_water_id == row.get("water_id"), sample_label + ":water_id")
            audit.require(water.get("enabled") is True and boundary.get("closed") is True, sample_label + ":water_boundary")
            sample_summaries.append(
                {
                    "sample_id": sample_id,
                    "path": _relative(audit.project_root, sample_dir),
                    "reference_id": reference_id,
                    "lighting_mode": row.get("lighting_mode"),
                    "water_id": row.get("water_id"),
                    "state_key": row.get("state_key"),
                    "state_sha256": sample_metadata.get("state_sha256"),
                }
            )
    audit.require(len(reference_summaries) == EXPECTED_REFERENCES, "reference_count", {"actual": len(reference_summaries), "expected": EXPECTED_REFERENCES})
    audit.require(len(sample_summaries) == EXPECTED_PAIRS, "sample_count", {"actual": len(sample_summaries), "expected": EXPECTED_PAIRS})
    refs_by_state_mode = {(item["state_key"], item["lighting_mode"]) for item in reference_summaries}
    audit.require(len(refs_by_state_mode) == EXPECTED_REFERENCES, "reference_state_mode_pairs")
    for reference_id, group in by_reference.items():
        audit.require(len(group) == len(EXPECTED_WATERS), f"reference:{reference_id}:water_count", len(group))
        audit.require({str(row.get("water_id")) for row in group} == set(EXPECTED_WATERS), f"reference:{reference_id}:water_set")
    return {"references": reference_summaries, "samples": sample_summaries}


def _check_calibration(
    audit: Audit,
    reports_root: Path,
    plan: dict[str, Any],
    gate_path: Path,
) -> dict[str, Any]:
    calibration_dir = gate_path.parent
    result_path = calibration_dir / "result.json"
    calibration_label = calibration_dir.name
    gate = audit.load(gate_path, f"{calibration_label}_gate")
    result = audit.load(result_path, f"{calibration_label}_result")
    summary: dict[str, Any] = {"gate": {}, "result": {}}
    expected_runtime = _canonical_hash(plan.get("runtime", {}))
    if isinstance(gate, dict):
        audit.require(gate.get("status") == "pass", "calibration_gate_status", gate.get("status"))
        audit.require(gate.get("source_sha256") == plan.get("source_hash"), "calibration_gate_source")
        audit.require(gate.get("config_sha256") == plan.get("config_hash"), "calibration_gate_config")
        audit.require(gate.get("runtime_sha256") == expected_runtime, "calibration_gate_runtime")
        audit.require(gate.get("samples") == EXPECTED_CALIBRATION_SAMPLES, "calibration_gate_samples", gate.get("samples"))
        gate_modes = gate.get("lighting_modes")
        audit.require(isinstance(gate_modes, list) and len(gate_modes) == len(EXPECTED_MODES) and set(gate_modes) == set(EXPECTED_MODES), "calibration_gate_modes")
        noise = gate.get("noise") if isinstance(gate.get("noise"), dict) else {}
        convergence = gate.get("noise_convergence") if isinstance(gate.get("noise_convergence"), dict) else {}
        light_noise: dict[str, Any] = {}
        for mode in EXPECTED_MODES:
            item = noise.get(mode) if isinstance(noise.get(mode), dict) else {}
            metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
            conv = convergence.get(mode) if isinstance(convergence.get(mode), dict) else {}
            ratio = metrics.get("effect_to_noise_ratio")
            structure_ratio = metrics.get("structure_noise_ratio")
            audit.require(item.get("passed") is True, f"calibration_noise:{mode}", item.get("reasons", []))
            audit.require(conv.get("passed") is True, f"calibration_convergence:{mode}")
            audit.require(conv.get("low_samples") == EXPECTED_CALIBRATION_LOW_SAMPLES and conv.get("high_samples") == EXPECTED_CALIBRATION_SAMPLES, f"calibration_convergence_samples:{mode}")
            high_over_low = conv.get("noise_ratio_high_over_low")
            audit.require(_finite_number(high_over_low) and float(high_over_low) < 1.0, f"calibration_convergence_ratio:{mode}", high_over_low)
            light_noise[mode] = {
                "effect_to_noise_ratio": ratio,
                "structure_noise_ratio": structure_ratio,
                "noise_rms": metrics.get("noise_rms"),
                "low_samples": conv.get("low_samples"),
                "high_samples": conv.get("high_samples"),
                "noise_ratio_high_over_low": high_over_low,
            }
        summary["noise"] = light_noise
        evidence = gate.get("evidence")
        evidence_ok = isinstance(evidence, list) and bool(evidence)
        verified_evidence: list[dict[str, Any]] = []
        if not evidence_ok:
            audit.failures.append("calibration gate: evidence list is missing or empty")
        else:
            for item in evidence:
                if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
                    evidence_ok = False
                    audit.failures.append("calibration gate: malformed evidence item")
                    continue
                actual = _path(audit.project_root, item["path"])
                if not actual.is_file() or _file_hash(actual) != item["sha256"]:
                    evidence_ok = False
                    audit.failures.append(f"calibration gate evidence mismatch: {item['path']}")
                else:
                    verified_evidence.append({"path": _relative(audit.project_root, actual), "sha256": item["sha256"]})
        audit.require(evidence_ok, "calibration_gate_evidence")
        expected_gate = {"path": _relative(audit.project_root, gate_path), "sha256": _file_hash(gate_path) if gate_path.is_file() else None}
        audit.require(_same(plan.get("calibration_gate"), expected_gate), "calibration_gate_plan_link", {"actual": plan.get("calibration_gate"), "expected": expected_gate})
        summary["gate"] = {
            "status": gate.get("status"),
            "scope": gate.get("scope"),
            "samples": gate.get("samples"),
            "modes": list(gate.get("lighting_modes", [])),
            "evidence_count": len(gate.get("evidence", [])) if isinstance(gate.get("evidence"), list) else 0,
            "evidence": verified_evidence,
        }
    if isinstance(result, dict):
        audit.require(result.get("status") == "pass", "calibration_result_status", result.get("status"))
        audit.require(result.get("source_sha256") == plan.get("source_hash"), "calibration_result_source")
        audit.require(_same(result.get("config"), plan.get("config")), "calibration_result_config")
        audit.require(_canonical_hash(result.get("runtime", {})) == expected_runtime, "calibration_result_runtime")
        renders = result.get("renders") if isinstance(result.get("renders"), list) else []
        expected_render_keys = {(mode, condition) for mode in EXPECTED_MODES for condition in ("clear", "mild", "medium", "strong")}
        actual_render_keys = {(item.get("mode"), item.get("condition")) for item in renders if isinstance(item, dict)}
        audit.require(len(renders) == 12 and actual_render_keys == expected_render_keys, "calibration_render_grid")
        audit.require(all(_passed_mapping(item.get("quality")) for item in renders if isinstance(item, dict)), "calibration_render_quality")
        audit.require(_passed_mapping(result.get("geometry_quality")), "calibration_geometry_quality")
        for key in ("noise", "noise_convergence"):
            values = result.get(key) if isinstance(result.get(key), dict) else {}
            audit.require(set(values) == set(EXPECTED_MODES), f"calibration_result_{key}_modes")
            audit.require(all(isinstance(item, dict) and item.get("passed") is True for item in values.values()), f"calibration_result_{key}_passed")
        summary["result"] = {"status": result.get("status"), "renders": len(renders)}
    initial_path = reports_root / "calibration_128_initial" / "result.json"
    initial = audit.load(initial_path, "calibration_128_initial", required=False)
    if isinstance(initial, dict):
        natural = initial.get("noise", {}).get("natural", {}) if isinstance(initial.get("noise"), dict) else {}
        metrics = natural.get("metrics", {}) if isinstance(natural, dict) else {}
        summary["initial_128"] = {
            "status": initial.get("status"),
            "natural_structure_noise_ratio": metrics.get("structure_noise_ratio"),
            "recorded_failure": initial.get("status") == "fail" and _finite_number(metrics.get("structure_noise_ratio")) and float(metrics["structure_noise_ratio"]) < 5.0,
        }
    return summary


def _check_generation_initial(audit: Audit, reports_root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    path = reports_root / "generation_initial.json"
    value = audit.load(path, "generation_initial")
    summary: dict[str, Any] = {}
    if not isinstance(value, dict):
        return summary
    audit.require(_status_pass(value.get("status")), "generation_initial_status", value.get("status"))
    audit.require(value.get("source_sha256") == plan.get("source_hash"), "generation_initial_source")
    audit.require(value.get("config_sha256") == plan.get("config_hash"), "generation_initial_config")
    audit.require(value.get("pairs") == EXPECTED_PAIRS and value.get("planned_pairs") == EXPECTED_PAIRS, "generation_initial_pair_grid", {"pairs": value.get("pairs"), "planned_pairs": value.get("planned_pairs")})
    audit.require(value.get("references") == EXPECTED_REFERENCES and value.get("states") == EXPECTED_GEOMETRIES, "generation_initial_state_grid", {"references": value.get("references"), "states": value.get("states")})
    for key in ("pairs", "pair_count", "generated_pairs"):
        if key in value:
            audit.require(value[key] == EXPECTED_PAIRS, "generation_initial_pair_count", value[key])
            break
    duration_keys = ("generation_seconds", "initial_generation_seconds", "duration_seconds", "elapsed_seconds", "seconds", "wall_seconds")
    duration = next((value.get(key) for key in duration_keys if _finite_number(value.get(key))), None)
    if duration is None:
        for key, item in value.items():
            if "second" in str(key).lower() and _finite_number(item):
                duration = item
                break
    audit.require(_finite_number(duration) and float(duration) >= 0.0, "generation_initial_actual_seconds", duration)
    summary["status"] = value.get("status")
    summary["seconds"] = float(duration) if _finite_number(duration) else None
    summary["path"] = _relative(audit.project_root, path)
    return summary


def _check_exports(audit: Audit, reports_root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    value = audit.load(reports_root / "exports_v2.json", "exports_v2")
    summary = value if isinstance(value, dict) else {}
    if isinstance(value, dict):
        audit.require(value.get("status") == "pass", "exports_status", value.get("status"))
        audit.require(value.get("pairs") == EXPECTED_PAIRS, "exports_pairs", value.get("pairs"))
        audit.require(value.get("references") == EXPECTED_REFERENCES, "exports_references", value.get("references"))
        audit.require(value.get("geometry_states") == EXPECTED_GEOMETRIES, "exports_geometry_states", value.get("geometry_states"))
        audit.require(value.get("allow_partial") is False, "exports_not_partial", value.get("allow_partial"))
        audit.require(value.get("source_sha256") == plan.get("source_hash"), "exports_source")
    return {"status": summary.get("status"), "pairs": summary.get("pairs"), "references": summary.get("references"), "geometry_states": summary.get("geometry_states")}


def _check_dataset_noise(audit: Audit, reports_root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    value = audit.load(reports_root / "dataset_noise_v2" / "result.json", "dataset_noise_v2")
    summary = value if isinstance(value, dict) else {}
    if isinstance(value, dict):
        groups = value.get("groups") if isinstance(value.get("groups"), list) else []
        audit.require(value.get("status") == "pass", "dataset_noise_status", value.get("status"))
        if "source_sha256" in value:
            audit.require(value.get("source_sha256") == plan.get("source_hash"), "dataset_noise_source")
        if "config_sha256" in value:
            audit.require(value.get("config_sha256") == plan.get("config_hash"), "dataset_noise_config")
        if "runtime_sha256" in value:
            audit.require(value.get("runtime_sha256") == _canonical_hash(plan.get("runtime", {})), "dataset_noise_runtime")
        audit.require(len(groups) == EXPECTED_GEOMETRIES * len(EXPECTED_MODES), "dataset_noise_group_count", len(groups))
        audit.require(all(isinstance(item, dict) and item.get("passed") is True for item in groups), "dataset_noise_groups_passed")
        audit.require(all(isinstance(item, dict) and item.get("samples") == EXPECTED_CALIBRATION_SAMPLES for item in groups), "dataset_noise_samples")
        group_keys = {(item.get("state_key"), item.get("lighting_mode"), item.get("water_id")) for item in groups if isinstance(item, dict)}
        audit.require(len(group_keys) == len(groups) and len({item.get("state_key") for item in groups if isinstance(item, dict)}) == EXPECTED_GEOMETRIES, "dataset_noise_group_keys")
        audit.require({item.get("lighting_mode") for item in groups if isinstance(item, dict)} == set(EXPECTED_MODES), "dataset_noise_modes")
        audit.require({item.get("water_id") for item in groups if isinstance(item, dict)} == {"strong"}, "dataset_noise_water")
    return {"status": summary.get("status"), "groups": len(summary.get("groups", [])) if isinstance(summary.get("groups"), list) else 0}


def _check_resume(audit: Audit, reports_root: Path, formal_root: Path) -> dict[str, Any]:
    resume_dir = reports_root / f"resume_{formal_root.name}"
    resume_label = resume_dir.name
    value = audit.load(resume_dir / "result.json", resume_label)
    summary = value if isinstance(value, dict) else {}
    if isinstance(value, dict):
        audit.require(value.get("status") in {"pass", "pass_with_non_bitwise_rgb_rerender"}, "resume_status", value.get("status"))
        audit.require(value.get("pairs") == EXPECTED_PAIRS, "resume_pairs", value.get("pairs"))
        audit.require(value.get("manifest_unique") is True, "resume_manifest_unique")
        audit.require(value.get("cached_files_unchanged_byte_exactly") is True, "resume_cached_bytes")
        audit.require(value.get("gt_restored_byte_exactly") is True, "resume_gt_bytes")
        limit = value.get("rgb_rerender_nrmse_limit")
        audit.require(_finite_number(limit) and float(limit) <= EXPECTED_RGB_NRMSE, "resume_rgb_nrmse_limit", limit)
        steps = value.get("steps") if isinstance(value.get("steps"), list) else []
        audit.require(bool(steps) and all(isinstance(item, dict) and item.get("exit_code") == item.get("expected_exit") for item in steps), "resume_step_results")
    return {"status": summary.get("status"), "pairs": summary.get("pairs"), "rgb_rerender_nrmse_limit": summary.get("rgb_rerender_nrmse_limit")}


def _check_native(audit: Audit, reports_root: Path) -> dict[str, Any]:
    value = audit.load(reports_root / "native_cli_v2" / "result.json", "native_cli_v2")
    summary = value if isinstance(value, dict) else {}
    if isinstance(value, dict):
        checks = value.get("checks") if isinstance(value.get("checks"), list) else []
        audit.require(value.get("status") == "pass", "native_cli_status", value.get("status"))
        audit.require(value.get("distinct_native_files") == EXPECTED_NATIVE_FILES, "native_cli_file_count", value.get("distinct_native_files"))
        native_modes = value.get("lighting_modes")
        audit.require(isinstance(native_modes, list) and set(native_modes) == set(EXPECTED_MODES), "native_cli_modes")
        audit.require(len({item.get("state_key") for item in checks if isinstance(item, dict)}) == EXPECTED_GEOMETRIES, "native_cli_states")
        audit.require(all(item.get("exit_code") == 0 and item.get("project_python_invoked") is False and item.get("autoexec_disabled") is True for item in checks if isinstance(item, dict)), "native_cli_no_project_python")
        format_ok = True
        for item in checks:
            if not isinstance(item, dict):
                format_ok = False
                continue
            png = item.get("png") if isinstance(item.get("png"), dict) else {}
            limit = item.get("display8_rmse_limit")
            if (png.get("width"), png.get("height"), png.get("bits"), png.get("color_type")) != (256, 256, 16, 2):
                format_ok = False
            if not _finite_number(item.get("display8_rmse")) or not _finite_number(limit) or float(item["display8_rmse"]) > float(limit):
                format_ok = False
        audit.require(format_ok, "native_cli_png_format_and_compare")
    return {"status": summary.get("status"), "checks": len(summary.get("checks", [])) if isinstance(summary.get("checks"), list) else 0, "distinct_native_files": summary.get("distinct_native_files"), "lighting_modes": summary.get("lighting_modes")}


def _check_viewer(audit: Audit, reports_root: Path) -> dict[str, Any]:
    value = audit.load(reports_root / "viewer_validation.json", "viewer_validation")
    summary = value if isinstance(value, dict) else {}
    formal = value.get("formal") if isinstance(value, dict) and isinstance(value.get("formal"), dict) else {}
    viewer = value.get("viewer") if isinstance(value, dict) and isinstance(value.get("viewer"), dict) else {}
    if isinstance(value, dict):
        audit.require(_status_pass(formal.get("status")), "viewer_formal_status", formal.get("status"))
        audit.require(not formal.get("errors") and formal.get("manifest_exists") is True and formal.get("valid_pairs") == EXPECTED_PAIRS and formal.get("valid_references") == EXPECTED_REFERENCES, "viewer_formal_complete", formal)
        audit.require(viewer.get("card_count") == EXPECTED_PAIRS, "viewer_card_count", viewer.get("card_count"))
        audit.require(viewer.get("formal_manifest_is_authoritative") is True, "viewer_manifest_authoritative")
        audit.require(viewer.get("formal_pairs_never_filled_by_placeholders") is True, "viewer_no_placeholders")
        local = viewer.get("local_resource_validation") if isinstance(viewer.get("local_resource_validation"), dict) else {}
        audit.require(local.get("passed") is True or local.get("exists") is True and not local.get("missing_links"), "viewer_local_resources", local)
        html_value = local.get("path") or viewer.get("html")
        html_path = _path(audit.project_root, html_value) if isinstance(html_value, str) else None
        audit.require(html_path is not None and html_path.is_file(), "viewer_html_exists", _relative(audit.project_root, html_path) if html_path else html_value)
        contact = viewer.get("contact_sheet") if isinstance(viewer.get("contact_sheet"), dict) else {}
        if "placeholder_rows" in contact:
            audit.require(contact.get("placeholder_rows") == 0, "viewer_contact_no_placeholders", contact.get("placeholder_rows"))
    return {"formal_status": formal.get("status"), "valid_pairs": formal.get("valid_pairs"), "valid_references": formal.get("valid_references"), "card_count": viewer.get("card_count")}


def _select_combined_native_log(reports_root: Path) -> Path:
    """Choose the final native log without binding acceptance to an old name."""

    preferred = reports_root / "combined_native_metadata_fix.log"
    if preferred.is_file():
        return preferred
    candidates = sorted(reports_root.glob("combined_native*.log"), key=lambda path: path.stat().st_mtime if path.exists() else 0.0, reverse=True)
    return candidates[0] if candidates else preferred


def _check_logs_and_faults(audit: Audit, reports_root: Path) -> dict[str, Any]:
    combined_path = _select_combined_native_log(reports_root)
    combined = audit.text(combined_path, "combined_native_log")
    pure = audit.text(reports_root / "pure_tests_v2.log", "pure_tests_v2_log")
    audit.require(bool(combined and re.search(r"Ran\s+40 tests", combined) and re.search(r"(?:^|\n)OK(?:\n|$)", combined)), "combined_native_40_tests")
    audit.require(bool(pure and re.search(r"Ran\s+20 tests", pure) and re.search(r"(?:^|\n)OK(?:\n|$)", pure)), "pure_20_tests")
    faults: dict[str, Any] = {}
    for name in ("pipeline_faults", "supervisor_faults"):
        value = audit.load(reports_root / name / "result.json", name)
        faults[name] = {"status": value.get("status") if isinstance(value, dict) else None}
        if isinstance(value, dict):
            scenarios = value.get("scenarios") if isinstance(value.get("scenarios"), list) else []
            audit.require(value.get("status") == "pass" and bool(scenarios) and all(isinstance(item, dict) and item.get("passed") is True for item in scenarios), name + "_pass")
    return {"combined_native_log": _relative(audit.project_root, combined_path), "faults": faults}


def _check_baseline(audit: Audit, reports_root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    path = reports_root / "baseline_provenance.json"
    value = audit.load(path, "baseline_provenance")
    summary: dict[str, Any] = {}
    if isinstance(value, dict):
        audit.require(value.get("baseline_source_sha256") == EXPECTED_BASELINE_SOURCE_SHA256, "baseline_v1_source_frozen")
        audit.require(value.get("baseline_blend_sha256") == EXPECTED_BASELINE_BLEND_SHA256, "baseline_v1_blend_record_frozen")
        audit.require(plan.get("baseline_source_sha256") == value.get("baseline_source_sha256"), "plan_baseline_source")
        audit.require(plan.get("baseline_blend_sha256") == value.get("baseline_blend_sha256"), "plan_baseline_blend")
        blend_value = value.get("baseline_blend")
        blend = _path(audit.project_root, blend_value) if isinstance(blend_value, str) else None
        blend_ok = blend is not None and blend.is_file() and _file_hash(blend) == value.get("baseline_blend_sha256")
        audit.require(blend_ok, "baseline_original_blend_hash", _relative(audit.project_root, blend) if blend else blend_value)
        summary = {"source_sha256": value.get("baseline_source_sha256"), "blend": _relative(audit.project_root, blend) if blend else blend_value, "blend_sha256": value.get("baseline_blend_sha256")}
    return summary


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _markdown(report: dict[str, Any]) -> str:
    status = report.get("status")
    passed = status == "pass_with_documented_limits"
    lines = [
        "# seabed_dataset_v2 验收报告",
        "",
        f"状态：`{status}`。" + ("所有发布门槛均有明确证据。" if passed else "证据尚不完整或存在失败项，未宣称正式完成。"),
        "",
        "本报告由标准库脚本读取实际输出与独立验证证据生成。它不生成 RGB/GT，不启动 Blender，也不把故障注入、占位图或校准预览计入正式数据。",
        "",
        "## 数量与配对",
        "",
        f"计划数量为 {report['counts']['planned_pairs']} pairs / {report['counts']['planned_references']} references / {report['counts']['planned_geometries']} geometries；实际 manifest 为 {report['counts']['actual_pairs']} / {report['counts']['actual_references']} / {report['counts']['actual_geometries']}。正式发布要求三者分别为 27 / 9 / 3。",
        "",
        "每个 reference 必须对应同一 geometry state 和 lighting mode 下的 mild、medium、strong；sample 必须保持 reference_id、state_sha256、相机、鱼姿态和灯光摘要一致。",
        "",
        "## Geometry states",
        "",
        "| state | layout | accepted attempt | camera pose/pitch | near / middle / far / background valid fraction | ecology pixels | meaningful fish / morphology |",
        "|---|---|---:|---|---|---|---|",
    ]
    for item in report.get("state_summaries", []):
        quality = item.get("quality", {})
        bands = quality.get("bands", {})
        fractions = "/".join(str(bands.get(name, {}).get("fraction_of_valid")) for name in ("near", "middle", "far", "background"))
        ecology = "/".join(str(bands.get(name, {}).get("ecology_pixel_count")) for name in ("near", "middle", "far", "background"))
        meaningful = quality.get("meaningful_visibility", {})
        morphology = ", ".join(map(str, meaningful.get("fish_morphologies", []))) + " / " + ", ".join(map(str, meaningful.get("coral_morphologies", [])))
        camera = item.get("camera", {})
        pose = f"eye={camera.get('eye')}; target={camera.get('target')}; pitch={camera.get('pitch_down_actual_deg', camera.get('pitch_down_deg'))}°"
        lines.append(f"| {item.get('state_key')} | {item.get('layout_family')} | {item.get('accepted_attempt')} | {pose} | {fractions} | {ecology} | {meaningful.get('fish_count')} / {morphology} |")
    lines += [
        "",
        "## Lighting and calibration",
        "",
        "| lighting | effect/noise | structure/noise | high/low noise |",
        "|---|---:|---:|---:|",
    ]
    for mode, item in report.get("calibration", {}).get("noise", {}).items():
        lines.append(f"| {mode} | {item.get('effect_to_noise_ratio')} | {item.get('structure_noise_ratio')} | {item.get('noise_ratio_high_over_low')} |")
    initial = report.get("calibration", {}).get("initial_128")
    if initial:
        lines += ["", f"128-sample 历史记录：Natural structure/noise={initial.get('natural_structure_noise_ratio')}，该次失败被保留；最终 512-sample calibration gate 状态为 `{report.get('calibration', {}).get('gate', {}).get('status', 'pass')}`。"]
    generation = report.get("generation_initial", {})
    lines += [
        "",
        "## RGB、GT、格式、恢复与原生文件",
        "",
        f"初始化生成耗时采用 `generation_initial.json` 的实际 seconds：`{generation.get('seconds')}`。该数值不被扩写为整台 8 GB 机器的正式渲染承诺。",
        "",
        f"独立 exports：`{report.get('exports', {}).get('status')}`；dataset noise：`{report.get('dataset_noise', {}).get('status')}`；resume：`{report.get('resume', {}).get('status')}`；native CLI：`{report.get('native_cli', {}).get('status')}`；viewer：实际 cards={report.get('viewer', {}).get('card_count')}。",
        "",
        "reference/sample 封包验证 schema、文件 SHA/字节数、RGB/GT 连接、Cycles/PNG16/EXR32、valid mask、深度范围和配对摘要。缓存命中要求字节不变；Metal RGB 重渲染使用 NRMSE ≤ 0.001，GT 和缓存文件保持严格字节一致。",
        "",
        "## 可靠证据",
        "",
    ]
    for name, item in report.get("evidence", {}).items():
        lines.append(f"1. `{name}`：`{item.get('path')}`")
    if report.get("missing_evidence"):
        lines += ["", "缺失证据："]
        lines.extend(f"1. `{path}`" for path in report["missing_evidence"])
    if report.get("failures"):
        lines += ["", "失败或不一致："]
        lines.extend(f"1. {failure}" for failure in report["failures"])
    history = report.get("history", {})
    if history:
        lines += ["", "## 历史记录（不参与当前验收）", ""]
        lines.append(f"当前 formal output：`{history.get('current_output')}`；当前 calibration：`{history.get('current_calibration')}`。")
        old_output = history.get("superseded_output")
        if old_output:
            lines.append(f"旧 output 保留为历史：`{old_output.get('path')}`，存在={old_output.get('exists')}，不计入当前 pairs/references。")
        old_calibration = history.get("superseded_calibration")
        if old_calibration:
            lines.append(f"旧 calibration 保留为历史：`{old_calibration.get('path')}`，存在={old_calibration.get('exists')}，不计入当前 gate。")
        cache_failure = history.get("cache_identity_failure")
        if cache_failure:
            lines.append(f"旧缓存失败记录：`{cache_failure.get('path')}`，原因={cache_failure.get('cause')}；该失败仅作修复历史。")
    lines += [
        "",
        "## 已披露限制",
        "",
        "1. 资产是程序化、风格化的珊瑚礁、鱼、岩礁、海草和起伏沙床，不是实测海域重建。",
        "2. 主体纹理使用对象局部米制坐标；对象 scale 扰动会改变世界空间纹理尺寸。",
        "3. 鱼、相机和无实体灯源使用保守的 OBB/包围体与点源净空检查，不等同精确实体或光束碰撞检测。",
        "4. RGB 水体吸收/散射系数是合成预设；原生玻璃界面的绝对 eta² 辐亮度没有标定。",
        "5. Natural/Artificial/Mixed 与三档水体的正式配对只接受完整 27 对；校准、故障注入和占位预览不计入正式数量。",
        "6. 三个 geometry states 同属一个布局族的工程验证集，不能证明训练泛化或域外性能。",
        "7. Stage C 的 216 对和任何网络训练均未执行。",
        "",
    ]
    return "\n".join(lines)


def _historical_summary(
    project_root: Path,
    reports_root: Path,
    current_formal_root: Path,
    current_calibration_dir: Path,
    audit: Audit,
) -> dict[str, Any]:
    """Keep superseded cache/calibration failures visible without gating release."""

    failure_path = reports_root / "cache_identity_failure.json"
    failure = audit.load(failure_path, "historical_cache_identity_failure", required=False)
    summary: dict[str, Any] = {"not_used_for_acceptance": True}
    if isinstance(failure, dict):
        old_output = failure.get("affected_output")
        old_root = _path(project_root, old_output) if isinstance(old_output, str) else None
        summary["cache_identity_failure"] = {
            "path": _relative(project_root, failure_path),
            "status": failure.get("status"),
            "affected_output": _relative(project_root, old_root) if old_root else old_output,
            "cause": failure.get("cause"),
            "old_data_preserved": failure.get("old_data_preserved"),
        }
        summary["superseded_output"] = {
            "path": _relative(project_root, old_root) if old_root else old_output,
            "exists": bool(old_root and old_root.exists()),
        }
    calibration_name = current_calibration_dir.name
    if calibration_name.endswith("_v2"):
        historical_calibration = reports_root / calibration_name[:-3]
        historical_result = historical_calibration / "result.json"
        historical_value = audit.load(historical_result, "historical_calibration_result", required=False)
        summary["superseded_calibration"] = {
            "path": _relative(project_root, historical_calibration),
            "exists": historical_calibration.exists(),
            "status": historical_value.get("status") if isinstance(historical_value, dict) else None,
            "not_used_for_acceptance": True,
        }
    summary["current_output"] = _relative(project_root, current_formal_root)
    summary["current_calibration"] = _relative(project_root, current_calibration_dir)
    return summary


def assemble(project_root: Path, formal_root: Path | None = None, reports_root: Path | None = None, config_path: Path | None = None) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    reports_root = Path(reports_root or (project_root / REPORTS_RELATIVE)).resolve()
    audit = Audit(project_root, reports_root)
    config_file = Path(config_path or (project_root / CONFIG_RELATIVE)).resolve()
    current_config = audit.load(config_file, "current_config")
    current_config = current_config if isinstance(current_config, dict) else {}
    configured_output = current_config.get("output_root")
    configured_formal_root = _path(project_root, configured_output) if isinstance(configured_output, str) else None
    if formal_root is None:
        formal_root = configured_formal_root or (project_root / "outputs" / "missing_current_output")
    formal_root = Path(formal_root).resolve()
    if configured_formal_root is not None:
        audit.require(formal_root == configured_formal_root.resolve(), "formal_root_matches_current_config", {"actual": _relative(project_root, formal_root), "config": _relative(project_root, configured_formal_root)})
    plan_path = formal_root / "plan.json"
    plan = audit.load(plan_path, "formal_plan")
    plan = plan if isinstance(plan, dict) else {}
    if current_config:
        audit.require(_same(plan.get("config"), current_config), "plan_config_matches_current_config")
    current_source = _source_hash(project_root)
    plan_source = plan.get("source_hash")
    audit.require(plan.get("schema_version") == 2, "plan_schema_version", plan.get("schema_version"))
    audit.require(current_source == plan_source, "source_sha256_matches_plan", {"actual": current_source, "plan": plan_source})
    planned_geometries = len(plan.get("geometry_states", [])) if isinstance(plan.get("geometry_states"), list) else 0
    planned_references = len(plan.get("references", [])) if isinstance(plan.get("references"), list) else 0
    planned_pairs = len(plan.get("pairs", [])) if isinstance(plan.get("pairs"), list) else 0
    audit.require(planned_geometries == EXPECTED_GEOMETRIES, "planned_geometry_count", planned_geometries)
    audit.require(planned_references == EXPECTED_REFERENCES, "planned_reference_count", planned_references)
    audit.require(planned_pairs == EXPECTED_PAIRS, "planned_pair_count", planned_pairs)
    rows, _ = _manifest(formal_root, plan, audit)
    state_summaries = _accepted_states(formal_root, plan, rows, audit)
    bundles = _bundles(formal_root, plan, rows, audit)
    gate_value = current_config.get("calibration_gate")
    gate_path = _path(project_root, gate_value) if isinstance(gate_value, str) else project_root / "reports" / "missing_current_calibration_gate.json"
    calibration = _check_calibration(audit, reports_root, plan, gate_path)
    generation_initial = _check_generation_initial(audit, reports_root, plan)
    exports = _check_exports(audit, reports_root, plan)
    dataset_noise = _check_dataset_noise(audit, reports_root, plan)
    resume = _check_resume(audit, reports_root, formal_root)
    native_cli = _check_native(audit, reports_root)
    viewer = _check_viewer(audit, reports_root)
    faults = _check_logs_and_faults(audit, reports_root)
    baseline = _check_baseline(audit, reports_root, plan)
    history = _historical_summary(project_root, reports_root, formal_root, gate_path.parent, audit)
    actual_references = len(bundles.get("references", []))
    actual_pairs = len(bundles.get("samples", []))
    actual_geometries = len({item.get("state_key") for item in bundles.get("samples", [])})
    counts = {
        "planned_pairs": planned_pairs,
        "planned_references": planned_references,
        "planned_geometries": planned_geometries,
        "actual_pairs": actual_pairs,
        "actual_references": actual_references,
        "actual_geometries": actual_geometries,
        "native_files": len({item.get("native", {}).get("path") for item in state_summaries}),
    }
    audit.require(actual_pairs == EXPECTED_PAIRS, "actual_pair_count", actual_pairs)
    audit.require(actual_references == EXPECTED_REFERENCES, "actual_reference_count", actual_references)
    audit.require(actual_geometries == EXPECTED_GEOMETRIES, "actual_geometry_count", actual_geometries)
    ready = not audit.missing and not audit.failures and all(item.get("passed") is True for item in audit.checks.values())
    report = {
        "schema_version": 2,
        "status": "pass_with_documented_limits" if ready else "incomplete",
        "ready_for_release": ready,
        "scope": {
            "stage_a_calibration": True,
            "stage_b_first_round_pairs": EXPECTED_PAIRS,
            "stage_c_216_pairs_executed": False,
            "network_training_executed": False,
            "fault_injection_or_placeholder_data_counted_as_formal": False,
        },
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "project_root": str(project_root),
        "formal_root": _relative(project_root, formal_root),
        "source_sha256": {"plan": plan_source, "actual": current_source, "matches": current_source == plan_source},
        "counts": counts,
        "state_summaries": state_summaries,
        "references": bundles.get("references", []),
        "samples": bundles.get("samples", []),
        "generation_initial": generation_initial,
        "calibration": calibration,
        "exports": exports,
        "dataset_noise": dataset_noise,
        "resume": resume,
        "native_cli": native_cli,
        "viewer": viewer,
        "fault_injection_validation": faults,
        "baseline": baseline,
        "history": history,
        "evidence": audit.evidence,
        "checks": audit.checks,
        "missing_evidence": sorted(set(audit.missing)),
        "failures": audit.failures,
        "warnings": audit.warnings,
        "limitations": [
            "programmatic stylized assets, not measured seabed reconstruction",
            "object-local metre texture coordinates change world texture size under object scale perturbation",
            "conservative OBB/point-source clearance proxies",
            "synthetic RGB water coefficients",
            "absolute eta^2 cross-interface radiance is not calibrated",
            "Metal RGB numerical rerender uses NRMSE <= 0.001 while cache and GT files remain byte exact",
            "three geometry states share one layout family and do not establish generalization",
            "Stage C 216 pairs and network training were not executed",
        ],
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / CONFIG_RELATIVE), help="current v2 configuration")
    parser.add_argument("--root", default=None, help="optional formal output root override; defaults to config output_root")
    parser.add_argument("--reports-root", default=str(PROJECT_ROOT / REPORTS_RELATIVE), help="evidence reports root")
    parser.add_argument("--json-out", default=str(PROJECT_ROOT / REPORTS_RELATIVE / "acceptance.json"), help="machine-readable output")
    parser.add_argument("--md-out", default=str(PROJECT_ROOT / REPORTS_RELATIVE / "ACCEPTANCE.md"), help="Chinese report output")
    args = parser.parse_args(argv)
    project_root = PROJECT_ROOT
    formal_root = _path(project_root, args.root) if args.root is not None else None
    reports_root = _path(project_root, args.reports_root)
    report = assemble(project_root, formal_root, reports_root, _path(project_root, args.config))
    json_text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_write(_path(project_root, args.json_out), json_text)
    _atomic_write(_path(project_root, args.md_out), _markdown(report))
    print(json.dumps({"status": report["status"], "pairs": report["counts"]["actual_pairs"], "references": report["counts"]["actual_references"], "geometries": report["counts"]["actual_geometries"], "missing": len(report["missing_evidence"]), "failures": len(report["failures"])}, ensure_ascii=False))
    return 0 if report["ready_for_release"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
