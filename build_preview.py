#!/usr/bin/env python3
"""Build an entirely local preview for the seabed v2 dataset.

The formal manifest is the only source of formal pairs.  When that manifest is
not available yet, this script can use the checked-in 128-sample calibration
images as a clearly labelled development preview.  Missing geometry states are
rendered as explicit placeholders in the nine-row contact sheet; they are
never counted as generated pairs.

No Blender process is needed.  RGB/GT source files are opened read-only and
copied only to the generated ``preview_assets`` directory.  Range thumbnails
are derived into that directory and do not modify the source EXR.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - installation-specific
    raise SystemExit("build_preview.py requires numpy and Pillow") from exc


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "first_round.json"


def _configured_relative_root(key: str, fallback: str) -> Path:
    """Read the active v2 path from config without touching any dataset files."""

    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return Path(value)
    except (OSError, TypeError, ValueError):
        pass
    return Path(fallback)


FORMAL_RELATIVE = _configured_relative_root("output_root", "outputs/first_round_v2")
CALIBRATION_RELATIVE = Path("reports/calibration_final_v2")
DEFAULT_HTML = PROJECT_ROOT / "preview.html"
DEFAULT_CONTACT = PROJECT_ROOT / "reports/contact_sheet.png"
DEFAULT_REPORT = PROJECT_ROOT / "reports/viewer_validation.json"
DEFAULT_ASSETS = PROJECT_ROOT / "preview_assets"
MODES = ("natural", "artificial", "mixed")
WATERS = ("mild", "medium", "strong")
EXPECTED_GEOMETRY_STATES = 3
EXPECTED_FORMAL_PAIRS = len(MODES) * len(WATERS) * EXPECTED_GEOMETRY_STATES
EXPECTED_FORMAL_REFERENCES = len(MODES) * EXPECTED_GEOMETRY_STATES

try:
    # The reader is dependency-free and follows this project's top-left,
    # float32, metre-labelled EXR convention.
    import sys

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.labels.formats import read_exr
except ImportError:  # pragma: no cover - source checkout issue
    read_exr = None


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_child(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty relative path")
    candidate = Path(value)
    resolved_root = root.resolve()
    resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"{label} escapes the formal output root")
    return resolved


def _mode(value: Any, label: str = "lighting_mode") -> str:
    if not isinstance(value, str) or value.lower() not in MODES:
        raise ValueError(f"{label} must be one of {', '.join(MODES)}")
    return value.lower()


def _water(value: Any, label: str = "water_id") -> str:
    if not isinstance(value, str) or value.lower() not in WATERS:
        raise ValueError(f"{label} must be one of {', '.join(WATERS)}")
    return value.lower()


def _state_key(value: Any) -> str:
    if isinstance(value, str) and value:
        return value
    raise ValueError("state_key must be a non-empty string")


def _readable_png(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return tuple(int(v) for v in image.size)
    except (OSError, ValueError, TypeError):
        return None


def _metric(mapping: Any, *keys: str) -> Any:
    current = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _record_metrics(reference_metadata: dict[str, Any], sample_metadata: dict[str, Any]) -> dict[str, Any]:
    geometry = reference_metadata.get("geometry_quality") or {}
    reference_quality = reference_metadata.get("rgb_quality") or {}
    sample_quality = sample_metadata.get("rgb_quality") or {}
    geometry_metrics = geometry.get("metrics") if isinstance(geometry, dict) else {}
    reference_metrics = reference_quality.get("metrics") if isinstance(reference_quality, dict) else {}
    sample_metrics = sample_quality.get("metrics") if isinstance(sample_quality, dict) else {}
    return {
        "valid_fraction": _metric(geometry_metrics, "valid_fraction"),
        "near_fraction_of_valid": _metric(geometry_metrics, "bands", "near", "fraction_of_valid"),
        "middle_fraction_of_valid": _metric(geometry_metrics, "bands", "middle", "fraction_of_valid"),
        "far_fraction_of_valid": _metric(geometry_metrics, "bands", "far", "fraction_of_valid"),
        "clear_mean_linear": _metric(reference_metrics, "mean_linear"),
        "sample_mean_linear": _metric(sample_metrics, "mean_linear"),
        "sample_display_dynamic_range": _metric(sample_metrics, "display_dynamic_range"),
        "sample_clipped_fraction": _metric(sample_metrics, "clipped_fraction"),
        "reference_quality_passed": reference_quality.get("passed") if isinstance(reference_quality, dict) else None,
        "sample_quality_passed": sample_quality.get("passed") if isinstance(sample_quality, dict) else None,
    }


def _record_source_paths(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "clear": str(record["clear_path"]),
        "degraded": str(record["degraded_path"]),
        "range": str(record["range_path"]) if record.get("range_path") else None,
        "reference_metadata": str(record["reference_metadata_path"]),
        "sample_metadata": str(record["sample_metadata_path"]) if record.get("sample_metadata_path") else None,
        "blend": str(record["blend_path"]) if record.get("blend_path") else None,
    }


def _validate_formal_row(root: Path, row: dict[str, Any], index: int) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    required = ("sample_id", "reference_id", "reference_path", "sample_path", "state_key", "lighting_mode", "water_id")
    missing = [key for key in required if key not in row]
    if missing:
        return None, [f"manifest row {index} missing {', '.join(missing)}"]
    try:
        sample_id = str(row["sample_id"])
        reference_id = str(row["reference_id"])
        state_key = _state_key(row["state_key"])
        mode = _mode(row["lighting_mode"])
        water = _water(row["water_id"])
        reference_dir = _safe_child(root, row["reference_path"], f"manifest row {index} reference_path")
        sample_dir = _safe_child(root, row["sample_path"], f"manifest row {index} sample_path")
    except (TypeError, ValueError) as exc:
        return None, [f"manifest row {index}: {exc}"]
    reference_metadata_path = reference_dir / "metadata.json"
    sample_metadata_path = sample_dir / "metadata.json"
    paths = {
        "clear_path": reference_dir / "clear_rgb.png",
        "degraded_path": sample_dir / "degraded_rgb.png",
        "range_path": reference_dir / "depth_range_m.exr",
    }
    for label, path in (("reference metadata", reference_metadata_path), ("sample metadata", sample_metadata_path), *paths.items()):
        if not path.is_file():
            errors.append(f"{sample_id}: missing {label} {path}")
    if errors:
        return None, errors
    try:
        reference_metadata = _json_load(reference_metadata_path)
        sample_metadata = _json_load(sample_metadata_path)
    except (OSError, ValueError, TypeError) as exc:
        return None, [f"{sample_id}: metadata read failed: {exc}"]
    if not isinstance(reference_metadata, dict) or not isinstance(sample_metadata, dict):
        return None, [f"{sample_id}: metadata must be JSON objects"]
    # The manifest is authoritative for the row identity.  Current pipeline
    # metadata stores the preset under ``water.preset.id``; older/alternate
    # bundles may expose a direct ``water_id``.  Prefer the direct field, then
    # use the native water metadata, and reject a row with neither identity.
    metadata_water_id = sample_metadata.get("water_id")
    if metadata_water_id is None:
        water_metadata = sample_metadata.get("water")
        preset_metadata = water_metadata.get("preset") if isinstance(water_metadata, dict) else None
        metadata_water_id = preset_metadata.get("id") if isinstance(preset_metadata, dict) else None
    checks = (
        (reference_metadata.get("reference_id"), reference_id, "reference_id"),
        (reference_metadata.get("lighting_mode"), mode, "reference lighting_mode"),
        (sample_metadata.get("sample_id"), sample_id, "sample_id"),
        (sample_metadata.get("reference_id"), reference_id, "sample reference_id"),
        (sample_metadata.get("lighting_mode"), mode, "sample lighting_mode"),
        (sample_metadata.get("reference_path"), row.get("reference_path"), "sample reference_path"),
    )
    for actual, expected, label in checks:
        if actual != expected:
            errors.append(f"{sample_id}: {label} {actual!r} != {expected!r}")
    if metadata_water_id is None:
        errors.append(f"{sample_id}: sample metadata has neither water_id nor water.preset.id")
    elif metadata_water_id != water:
        errors.append(f"{sample_id}: sample water identity {metadata_water_id!r} != {water!r}")
    lighting = reference_metadata.get("lighting")
    if isinstance(lighting, dict) and lighting.get("mode") not in (None, mode):
        errors.append(f"{sample_id}: embedded lighting mode does not match manifest")
    clear_size = _readable_png(paths["clear_path"])
    degraded_size = _readable_png(paths["degraded_path"])
    if clear_size is None:
        errors.append(f"{sample_id}: clear PNG cannot be decoded")
    if degraded_size is None:
        errors.append(f"{sample_id}: degraded PNG cannot be decoded")
    if clear_size and degraded_size and clear_size != degraded_size:
        errors.append(f"{sample_id}: clear/degraded dimensions differ")
    if errors:
        return None, errors
    blend_path = root / "scenes" / f"{state_key}.blend"
    if not blend_path.is_file():
        candidate = reference_metadata.get("native_blend")
        if isinstance(candidate, str):
            try:
                blend_path = _safe_child(root, candidate, f"{sample_id} native_blend")
            except ValueError:
                blend_path = None
        else:
            blend_path = None
    record = {
        "record_id": sample_id,
        "sample_id": sample_id,
        "reference_id": reference_id,
        "state_key": state_key,
        "state_index": row.get("state_index"),
        "layout_seed": row.get("layout_seed"),
        "layout_family": row.get("layout_family"),
        "lighting_mode": mode,
        "water_id": water,
        "accepted_attempt": row.get("accepted_attempt"),
        "source_kind": "formal",
        "source_status": "formal_manifest",
        "clear_path": paths["clear_path"],
        "degraded_path": paths["degraded_path"],
        "range_path": paths["range_path"],
        "reference_metadata_path": reference_metadata_path,
        "sample_metadata_path": sample_metadata_path,
        "blend_path": blend_path,
        "reference_metadata": reference_metadata,
        "sample_metadata": sample_metadata,
        "metrics": _record_metrics(reference_metadata, sample_metadata),
    }
    return record, []


def scan_formal(formal_root: Path) -> dict[str, Any]:
    """Read and cross-check formal manifest rows without changing any source."""

    manifest = formal_root / "manifest.jsonl"
    result: dict[str, Any] = {
        "root": str(formal_root),
        "manifest": str(manifest),
        "manifest_exists": manifest.is_file(),
        "status": "not_ready" if not manifest.is_file() else "partial",
        "rows_seen": 0,
        "valid_pairs": 0,
        "valid_references": 0,
        "errors": [],
        "records": [],
    }
    if not manifest.is_file():
        result["errors"].append("Formal manifest is not present; the 27-pair release has not been generated")
        return result
    records: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        result["errors"].append(f"Cannot read formal manifest: {exc}")
        return result
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        result["rows_seen"] += 1
        try:
            row = json.loads(line)
        except (ValueError, TypeError) as exc:
            result["errors"].append(f"manifest row {index} is invalid JSON: {exc}")
            continue
        if not isinstance(row, dict):
            result["errors"].append(f"manifest row {index} is not an object")
            continue
        try:
            sample_id = str(row.get("sample_id", ""))
        except Exception:
            sample_id = ""
        if sample_id in seen_samples:
            result["errors"].append(f"duplicate sample_id {sample_id!r}")
            continue
        seen_samples.add(sample_id)
        record, errors = _validate_formal_row(formal_root, row, index)
        result["errors"].extend(errors)
        if record:
            records.append(record)
    result["records"] = records
    result["valid_pairs"] = len(records)
    result["valid_references"] = len({record["reference_id"] for record in records})
    reference_locations: dict[str, set[str]] = defaultdict(set)
    reference_modes: dict[str, set[str]] = defaultdict(set)
    for record in records:
        reference_locations[record["reference_id"]].add(str(record["reference_metadata_path"]))
        reference_modes[record["reference_id"]].add(record["lighting_mode"])
    inconsistent_references = sorted(
        reference_id
        for reference_id, locations in reference_locations.items()
        if len(locations) != 1 or len(reference_modes[reference_id]) != 1
    )
    if inconsistent_references:
        result["errors"].append(f"reference IDs point to multiple paths or modes: {inconsistent_references}")
    result["inconsistent_references"] = inconsistent_references
    state_keys = sorted({record["state_key"] for record in records})
    actual_grid = {(record["state_key"], record["lighting_mode"], record["water_id"]) for record in records}
    expected_grid = {
        (state, mode, water)
        for state in state_keys
        for mode in MODES
        for water in WATERS
    } if len(state_keys) == EXPECTED_GEOMETRY_STATES else set()
    result["state_keys"] = state_keys
    result["state_count"] = len(state_keys)
    result["missing_grid"] = sorted(expected_grid - actual_grid)
    result["unexpected_grid"] = sorted(actual_grid - expected_grid) if expected_grid else sorted(actual_grid)
    if result["rows_seen"] == 0:
        result["status"] = "not_ready"
    elif (
        not result["errors"]
        and result["valid_pairs"] == EXPECTED_FORMAL_PAIRS
        and result["valid_references"] == EXPECTED_FORMAL_REFERENCES
        and result["state_count"] == EXPECTED_GEOMETRY_STATES
        and not result["missing_grid"]
        and not result["unexpected_grid"]
    ):
        result["status"] = "complete"
    else:
        result["status"] = "partial"
    return result


def _calibration_render_map(result: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    mapped = {}
    for row in result.get("renders", []):
        if not isinstance(row, dict):
            continue
        try:
            mode = _mode(row.get("mode"))
            condition = str(row.get("condition", ""))
        except ValueError:
            continue
        mapped[(mode, condition)] = row
    return mapped


def scan_calibration(calibration_root: Path) -> dict[str, Any]:
    """Index the optional 128-sample calibration gallery."""

    result_path = calibration_root / "result.json"
    output: dict[str, Any] = {
        "root": str(calibration_root),
        "result": str(result_path),
        "result_exists": result_path.is_file(),
        "status": "unavailable" if not result_path.is_file() else "available_development_only",
        "source_status": None,
        "valid_pairs": 0,
        "valid_references": 0,
        "errors": [],
        "records": [],
    }
    if not result_path.is_file():
        output["errors"].append("Calibration result is not present")
        return output
    try:
        calibration = _json_load(result_path)
    except (OSError, ValueError, TypeError) as exc:
        output["status"] = "invalid"
        output["errors"].append(f"Calibration result cannot be read: {exc}")
        return output
    if not isinstance(calibration, dict):
        output["status"] = "invalid"
        output["errors"].append("Calibration result must be a JSON object")
        return output
    output["source_status"] = calibration.get("status")
    config = calibration.get("config") if isinstance(calibration.get("config"), dict) else {}
    layouts = config.get("layouts") if isinstance(config.get("layouts"), list) else []
    layout_seed = layouts[0].get("seed", 101) if layouts and isinstance(layouts[0], dict) else 101
    geometry_plan = calibration.get("geometry_plan") if isinstance(calibration.get("geometry_plan"), dict) else {}
    state_index = geometry_plan.get("state_index", 0)
    state_key = f"l{layout_seed}_g{int(state_index):03d}"
    render_map = _calibration_render_map(calibration)
    geometry_dir = calibration_root / "geometry"
    range_path = geometry_dir / "depth_range_m.exr"
    geometry_metadata_path = geometry_dir / "metadata.json"
    for mode in MODES:
        clear_row = render_map.get((mode, "clear"))
        clear_path = calibration_root / mode / "clear_rgb.png"
        if not clear_path.is_file() or _readable_png(clear_path) is None:
            output["errors"].append(f"Calibration {mode}: clear image is unavailable")
            continue
        for water in WATERS:
            degraded_path = calibration_root / mode / f"{water}.png"
            row = render_map.get((mode, water))
            if not degraded_path.is_file() or _readable_png(degraded_path) is None:
                output["errors"].append(f"Calibration {mode}/{water}: degraded image is unavailable")
                continue
            clear_quality = clear_row.get("quality", {}) if isinstance(clear_row, dict) else {}
            sample_quality = row.get("quality", {}) if isinstance(row, dict) else {}
            reference_metadata = {
                "reference_id": f"calibration_{state_key}_{mode}",
                "state_key": state_key,
                "state_index": state_index,
                "layout_seed": layout_seed,
                "lighting_mode": mode,
                "lighting": clear_row.get("lighting") if isinstance(clear_row, dict) else None,
                "geometry_quality": calibration.get("geometry_quality"),
                "rgb_quality": clear_quality,
                "source_kind": "calibration_128",
            }
            sample_metadata = {
                "sample_id": f"calibration_{state_key}_{mode}_{water}",
                "reference_id": reference_metadata["reference_id"],
                "lighting_mode": mode,
                "water_id": water,
                "rgb_quality": sample_quality,
                "source_kind": "calibration_128",
            }
            record = {
                "record_id": sample_metadata["sample_id"],
                "sample_id": sample_metadata["sample_id"],
                "reference_id": reference_metadata["reference_id"],
                "state_key": state_key,
                "state_index": state_index,
                "layout_seed": layout_seed,
                "layout_family": None,
                "lighting_mode": mode,
                "water_id": water,
                "accepted_attempt": geometry_plan.get("attempt_index", 0),
                "source_kind": "calibration_128",
                "source_status": calibration.get("status"),
                "clear_path": clear_path,
                "degraded_path": degraded_path,
                "range_path": range_path if range_path.is_file() else None,
                "reference_metadata_path": geometry_metadata_path if geometry_metadata_path.is_file() else result_path,
                "sample_metadata_path": None,
                "blend_path": calibration_root / "calibration.blend" if (calibration_root / "calibration.blend").is_file() else None,
                "reference_metadata": reference_metadata,
                "sample_metadata": sample_metadata,
                "metrics": {
                    "valid_fraction": _metric(clear_quality, "metrics", "valid_fraction"),
                    "near_fraction_of_valid": _metric(calibration.get("geometry_quality"), "metrics", "bands", "near", "fraction_of_valid"),
                    "middle_fraction_of_valid": _metric(calibration.get("geometry_quality"), "metrics", "bands", "middle", "fraction_of_valid"),
                    "far_fraction_of_valid": _metric(calibration.get("geometry_quality"), "metrics", "bands", "far", "fraction_of_valid"),
                    "clear_mean_linear": _metric(clear_row, "stats", "linear_rgb_mean") if isinstance(clear_row, dict) else None,
                    "sample_mean_linear": _metric(row, "stats", "linear_rgb_mean") if isinstance(row, dict) else None,
                    "sample_display_dynamic_range": _metric(sample_quality, "metrics", "display_dynamic_range"),
                    "sample_clipped_fraction": _metric(sample_quality, "metrics", "clipped_fraction"),
                    "reference_quality_passed": clear_quality.get("passed") if isinstance(clear_quality, dict) else None,
                    "sample_quality_passed": sample_quality.get("passed") if isinstance(sample_quality, dict) else None,
                },
            }
            output["records"].append(record)
    output["valid_pairs"] = len(output["records"])
    output["valid_references"] = len({record["reference_id"] for record in output["records"]})
    return output


def _slug(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return text.strip("_") or "item"


def _copy_asset(source: Path, asset_dir: Path, name: str) -> Path | None:
    if source is None or not source.is_file():
        return None
    target = asset_dir / f"{_slug(name)}{source.suffix.lower()}"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(source, target)
    except OSError:
        return None
    return target


def _range_thumbnail(source: Path | None, target: Path, tile_size: int) -> Path | None:
    """Write a display-only range thumbnail; preserve the source EXR."""

    if source is None or not source.is_file() or read_exr is None:
        return None
    try:
        payload = read_exr(source)
        values = np.asarray(payload["pixels"], dtype=np.float32)
        if values.ndim != 2 or values.size == 0:
            return None
        finite = np.isfinite(values) & (values > 0)
        if not finite.any():
            return None
        low = float(np.percentile(values[finite], 2))
        high = float(np.percentile(values[finite], 98))
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            high = low + 1.0
        normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
        display = np.where(finite, (1.0 - normalized) * 255.0, 0.0).astype(np.uint8)
        image = Image.fromarray(display, mode="L").convert("RGB")
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, format="PNG")
        return target
    except (OSError, ValueError, TypeError, KeyError, RuntimeError):
        return None


def _font() -> Any:
    try:
        return ImageFont.load_default()
    except OSError:  # pragma: no cover - Pillow always ships a default font
        return None


def _placeholder(tile_size: int, text: str) -> Image.Image:
    image = Image.new("RGB", (tile_size, tile_size), (35, 40, 47))
    draw = ImageDraw.Draw(image)
    font = _font()
    lines = text.split("\n")
    heights = []
    widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    total_h = sum(heights) + max(0, len(lines) - 1) * 3
    y = (tile_size - total_h) // 2
    for line, width, height in zip(lines, widths, heights):
        draw.text(((tile_size - width) // 2, y), line, fill=(220, 225, 230), font=font)
        y += height + 3
    return image


def _load_tile(path: Path | None, tile_size: int, label: str) -> Image.Image:
    if path is None or not path.is_file():
        return _placeholder(tile_size, label)
    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
            resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
            image.thumbnail((tile_size, tile_size), resampling)
            canvas = Image.new("RGB", (tile_size, tile_size), (10, 14, 18))
            canvas.paste(image, ((tile_size - image.width) // 2, (tile_size - image.height) // 2))
            return canvas
    except (OSError, ValueError):
        return _placeholder(tile_size, label)


def _display_record(record: dict[str, Any], output_html: Path, asset_dir: Path, tile_size: int, cache: dict[str, Path]) -> dict[str, Any]:
    key = record["record_id"]
    display: dict[str, Any] = dict(record)
    for field, source_key in (("clear_asset", "clear_path"), ("degraded_asset", "degraded_path")):
        source = Path(record[source_key]) if record.get(source_key) else None
        cache_key = f"{field}:{source}"
        if cache_key not in cache:
            copied = _copy_asset(source, asset_dir, f"{_slug(key)}_{field}") if source else None
            cache[cache_key] = copied
        display[field] = cache[cache_key]
    range_source = Path(record["range_path"]) if record.get("range_path") else None
    range_cache_key = f"range:{range_source}"
    if range_cache_key not in cache:
        target = asset_dir / f"{_slug(key)}_range.png"
        cache[range_cache_key] = _range_thumbnail(range_source, target, tile_size)
    display["range_asset"] = cache[range_cache_key]
    return display


def _relative_link(base: Path, target: Path | None) -> str | None:
    if target is None:
        return None
    try:
        # Resolve both sides so macOS /var -> /private/var aliases do not turn
        # an existing local asset into a false missing-link report.
        return os.path.relpath(str(target.resolve()), str(base.parent.resolve())).replace(os.sep, "/")
    except (TypeError, ValueError):
        return None


def _format_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, list):
        return ", ".join(_format_value(item) for item in value)
    return str(value)


def _record_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": record["record_id"],
        "state_key": record["state_key"],
        "lighting_mode": record["lighting_mode"],
        "water_id": record["water_id"],
        "source_kind": record["source_kind"],
        "reference_id": record["reference_id"],
        "metrics": record.get("metrics", {}),
    }


def _lighting_summary(record: dict[str, Any]) -> str:
    lighting = record.get("reference_metadata", {}).get("lighting")
    if not isinstance(lighting, dict):
        return "lighting metadata unavailable"
    world = lighting.get("world") if isinstance(lighting.get("world"), dict) else {}
    background = world.get("background_strength")
    objects = lighting.get("objects") if isinstance(lighting.get("objects"), list) else []
    energies = [item.get("energy") for item in objects if isinstance(item, dict) and item.get("energy") is not None]
    temperatures = [item.get("temperature_k") for item in objects if isinstance(item, dict) and item.get("temperature_k") is not None]
    sources = sorted({
        str(item.get("effective_rgb_source"))
        for item in objects
        if isinstance(item, dict) and item.get("effective_rgb_source")
    })
    return (
        f"world strength={_format_value(background)} · "
        f"spot energy={_format_value(energies)} · "
        f"temperature K={_format_value(temperatures)} · "
        f"RGB source={_format_value(sources)}"
    )


def _html_record_card(record: dict[str, Any], output_html: Path) -> str:
    metrics = record.get("metrics", {})
    source_paths = _record_source_paths(record)
    links = []
    for label, key in (("reference metadata", "reference_metadata"), ("sample metadata", "sample_metadata"), ("range EXR", "range"), ("native blend", "blend")):
        target = Path(source_paths[key]) if source_paths.get(key) else None
        relative = _relative_link(output_html, target)
        if relative and target and target.is_file():
            links.append(f'<a href="{html.escape(relative, quote=True)}">{html.escape(label)}</a>')
    link_html = " · ".join(links) if links else "无本地元数据/Blend链接"
    clear = _relative_link(output_html, Path(record["clear_asset"]))
    degraded = _relative_link(output_html, Path(record["degraded_asset"]))
    range_asset = _relative_link(output_html, Path(record["range_asset"])) if record.get("range_asset") else None
    source_label = "FORMAL" if record["source_kind"] == "formal" else "CALIBRATION_128 · 开发预览"
    metrics_text = (
        f"valid={_format_value(metrics.get('valid_fraction'))} · "
        f"near={_format_value(metrics.get('near_fraction_of_valid'))} · "
        f"middle={_format_value(metrics.get('middle_fraction_of_valid'))} · "
        f"far={_format_value(metrics.get('far_fraction_of_valid'))} · "
        f"sample mean={_format_value(metrics.get('sample_mean_linear'))}"
    )
    images = []
    if clear:
        images.append(f'<img src="{html.escape(clear, quote=True)}" alt="Clear" loading="lazy">')
    if degraded:
        images.append(f'<img src="{html.escape(degraded, quote=True)}" alt="{html.escape(record["water_id"])}" loading="lazy">')
    if range_asset:
        images.append(f'<img src="{html.escape(range_asset, quote=True)}" alt="Range" loading="lazy">')
    return (
        f'<article class="pair-card" data-state="{html.escape(str(record["state_key"]), quote=True)}" '
        f'data-mode="{html.escape(record["lighting_mode"], quote=True)}" '
        f'data-water="{html.escape(record["water_id"], quote=True)}">'
        f'<div class="card-head"><span>{html.escape(str(record["state_key"]))} · {html.escape(record["lighting_mode"])} · {html.escape(record["water_id"])}</span>'
        f'<span class="source {"formal" if record["source_kind"] == "formal" else "calibration"}">{html.escape(source_label)}</span></div>'
        f'<div class="images">{"".join(images)}</div>'
        f'<p class="meta">reference=<code>{html.escape(str(record["reference_id"]))}</code> · sample=<code>{html.escape(str(record["sample_id"]))}</code></p>'
        f'<p class="metrics">{html.escape(metrics_text)}<br>{html.escape(_lighting_summary(record))}</p>'
        f'<p class="links">{link_html}</p></article>'
    )


def _expected_grid(formal: dict[str, Any], calibration: dict[str, Any]) -> list[tuple[str, str]]:
    all_records = formal.get("records", []) + calibration.get("records", [])
    states = sorted({record.get("state_key") for record in all_records if record.get("state_key")})
    if states:
        # The release is a single layout x three geometry states.  Preserve
        # the expected first three state keys when they are visible in config
        # or calibration; append observed formal keys only for partial runs.
        layout_seed = None
        for record in all_records:
            if record.get("layout_seed") is not None:
                layout_seed = record["layout_seed"]
                break
        if layout_seed is not None:
            expected_states = [f"l{layout_seed}_g{index:03d}" for index in range(EXPECTED_GEOMETRY_STATES)]
            states = expected_states + [state for state in states if state not in expected_states]
    else:
        states = [f"l101_g{index:03d}" for index in range(EXPECTED_GEOMETRY_STATES)]
    return [(state, mode) for state in states[:EXPECTED_GEOMETRY_STATES] for mode in MODES]


def _contact_sheet(records: Iterable[dict[str, Any]], output_path: Path, output_html: Path, tile_size: int) -> dict[str, Any]:
    by_grid: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = (str(record["state_key"]), str(record["lighting_mode"]))
        existing = by_grid.get(key)
        # Formal assets have precedence over optional calibration assets in a
        # partial development directory.
        if existing is None or (existing["source_kind"] != "formal" and record["source_kind"] == "formal"):
            by_grid[key] = record
    layout = _expected_grid({"records": [r for r in records if r["source_kind"] == "formal"]}, {"records": [r for r in records if r["source_kind"] != "formal"]})
    left_label_width = 190
    header_height = 40
    row_height = tile_size + 30
    columns = ("Clear", "mild", "medium", "strong", "Range")
    image = Image.new("RGB", (left_label_width + tile_size * len(columns), header_height + row_height * len(layout)), (16, 21, 27))
    draw = ImageDraw.Draw(image)
    font = _font()
    for column_index, column in enumerate(columns):
        x = left_label_width + column_index * tile_size
        draw.rectangle((x, 0, x + tile_size - 1, header_height - 1), fill=(30, 41, 52))
        draw.text((x + 10, 13), column, fill=(235, 241, 246), font=font)
    actual_rows = 0
    placeholders = 0
    row_entries = []
    for row_index, (state, mode) in enumerate(layout):
        record = by_grid.get((state, mode))
        y = header_height + row_index * row_height
        draw.rectangle((0, y, left_label_width - 1, y + tile_size - 1), fill=(25, 32, 40))
        source_text = "FORMAL" if record and record["source_kind"] == "formal" else "CALIBRATION_128" if record else "NOT GENERATED"
        draw.multiline_text((10, y + 12), f"{state}\n{mode}\n{source_text}", fill=(224, 230, 235), font=font, spacing=2)
        tiles = []
        if record:
            actual_rows += 1
            tiles = [
                _load_tile(record.get("clear_asset"), tile_size, "MISSING CLEAR"),
                _load_tile(record.get("mild_asset"), tile_size, "MISSING MILD"),
                _load_tile(record.get("medium_asset"), tile_size, "MISSING MEDIUM"),
                _load_tile(record.get("strong_asset"), tile_size, "MISSING STRONG"),
                _load_tile(record.get("range_asset"), tile_size, "RANGE UNAVAILABLE"),
            ]
            # A row record is one water pair.  The contact sheet needs all
            # three water columns from the same mode/reference.
            row_records = [r for r in records if r["state_key"] == state and r["lighting_mode"] == mode and r["source_kind"] == record["source_kind"]]
            water_map = {r["water_id"]: r for r in row_records}
            tiles[1] = _load_tile(water_map.get("mild", {}).get("degraded_asset") if water_map.get("mild") else None, tile_size, "MISSING MILD")
            tiles[2] = _load_tile(water_map.get("medium", {}).get("degraded_asset") if water_map.get("medium") else None, tile_size, "MISSING MEDIUM")
            tiles[3] = _load_tile(water_map.get("strong", {}).get("degraded_asset") if water_map.get("strong") else None, tile_size, "MISSING STRONG")
            row_entries.append({"state_key": state, "lighting_mode": mode, "source_kind": record["source_kind"], "water_records": sorted(water_map)})
        else:
            placeholders += 1
            tiles = [_placeholder(tile_size, "NOT GENERATED") for _ in columns]
            row_entries.append({"state_key": state, "lighting_mode": mode, "source_kind": None, "water_records": []})
        for column_index, tile in enumerate(tiles):
            image.paste(tile, (left_label_width + column_index * tile_size, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    return {"path": str(output_path), "row_count": len(layout), "actual_rows": actual_rows, "placeholder_rows": placeholders, "rows": row_entries}


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for key, value in attrs:
            if key in ("src", "href") and value:
                self.links.append((tag, value))


def _validate_html(path: Path, expected_cards: int) -> dict[str, Any]:
    result = {"path": str(path), "exists": path.is_file(), "local_links": True, "missing_links": [], "external_links": [], "card_count": 0, "card_count_matches": False, "js": {}}
    if not path.is_file():
        result["missing_links"].append("preview.html is missing")
        return result
    text = path.read_text(encoding="utf-8")
    parser = _LinkParser()
    try:
        parser.feed(text)
    except Exception as exc:  # HTMLParser is intentionally tolerant
        result["missing_links"].append(f"HTML parse failed: {exc}")
    for tag, link in parser.links:
        if link.startswith(("#", "javascript:", "data:")):
            continue
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", link):
            result["external_links"].append(link)
            continue
        target_text = link.split("#", 1)[0].split("?", 1)[0]
        target = (path.parent / target_text).resolve()
        if not target.is_file():
            result["missing_links"].append(link)
    result["local_links"] = not result["missing_links"] and not result["external_links"]
    result["card_count"] = text.count('class="pair-card"')
    result["card_count_matches"] = result["card_count"] == expected_cards
    script_matches = re.findall(r"<script(?P<attrs>[^>]*)>(?P<body>.*?)</script>", text, flags=re.IGNORECASE | re.DOTALL)
    executable_scripts = [
        body
        for attrs, body in script_matches
        if body.strip() and not re.search(r"type\s*=\s*['\"]application/json['\"]", attrs, flags=re.IGNORECASE)
    ]
    node_path = shutil.which("node")
    js_result = {"node_available": bool(node_path), "checked": bool(executable_scripts), "passed": None, "error": None}
    if executable_scripts and node_path:
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".js", delete=False) as temp:
                temp.write("\n".join(executable_scripts))
                temp_path = Path(temp.name)
            completed = subprocess.run([node_path, "--check", str(temp_path)], capture_output=True, text=True, timeout=10)
            js_result["passed"] = completed.returncode == 0
            if completed.returncode:
                js_result["error"] = (completed.stderr or completed.stdout).strip()[:1000]
        except (OSError, subprocess.SubprocessError) as exc:
            js_result["error"] = str(exc)
        finally:
            try:
                temp_path.unlink()
            except (UnboundLocalError, OSError):
                pass
    elif executable_scripts:
        js_result["error"] = "node is unavailable; JavaScript syntax was not externally checked"
    result["js"] = js_result
    return result


def _html_document(records: list[dict[str, Any]], formal: dict[str, Any], calibration: dict[str, Any], output_path: Path, contact_sheet: Path, report_path: Path) -> str:
    cards = "\n".join(_html_record_card(record, output_path) for record in records)
    state_options = sorted({str(record["state_key"]) for record in records})
    option_html = '<option value="all">全部 geometry</option>' + "".join(f'<option value="{html.escape(value, quote=True)}">{html.escape(value)}</option>' for value in state_options)
    mode_options = '<option value="all">全部 lighting</option>' + "".join(f'<option value="{mode}">{mode}</option>' for mode in MODES)
    water_options = '<option value="all">全部 water</option>' + "".join(f'<option value="{water}">{water}</option>' for water in WATERS)
    formal_status = formal["status"]
    calibration_count = calibration.get("valid_pairs", 0)
    formal_note = (
        f"正式清单状态：<strong>{html.escape(formal_status)}</strong>，可验证 pairs={formal.get('valid_pairs', 0)}/{EXPECTED_FORMAL_PAIRS}，references={formal.get('valid_references', 0)}/{EXPECTED_FORMAL_REFERENCES}。"
    )
    calibration_note = f"当前开发预览使用 calibration_128 条件数：{calibration_count}；这些图像不会进入正式 manifest。"
    contact_link = _relative_link(output_path, contact_sheet)
    report_link = _relative_link(output_path, report_path)
    data = [_record_summary(record) for record in records]
    data_json = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    return f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Seabed Dataset v2 · Offline Preview</title>
<style>
:root {{ color-scheme: dark; --bg:#10151b; --panel:#1b2530; --muted:#a8b5c2; --accent:#63c5d8; --warn:#ffc857; }}
* {{ box-sizing:border-box; }} body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:#edf2f7; }}
main {{ max-width:1500px; margin:0 auto; padding:28px 22px 50px; }} h1 {{ margin:0 0 8px; font-size:28px; }} h2 {{ margin:28px 0 12px; font-size:20px; }}
.banner {{ background:#3b2d11; border:1px solid #876b2c; color:#ffe7a0; padding:14px 16px; border-radius:10px; line-height:1.55; }}
.facts {{ color:var(--muted); line-height:1.55; }} .facts strong {{ color:#f6f8fa; }} a {{ color:var(--accent); }} code {{ color:#d7f7ff; }}
.controls {{ display:flex; flex-wrap:wrap; gap:10px; padding:14px; background:var(--panel); border-radius:10px; }} label {{ color:var(--muted); }} select {{ margin-left:6px; background:#101820; color:#f4f8fb; border:1px solid #536474; border-radius:5px; padding:6px; }}
.pair-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(430px,1fr)); gap:14px; }} .pair-card {{ background:var(--panel); border:1px solid #2d3a46; border-radius:10px; overflow:hidden; }}
.card-head {{ display:flex; justify-content:space-between; gap:8px; padding:11px 13px; font-weight:600; }} .source {{ padding:2px 7px; border-radius:5px; font-size:11px; }} .source.formal {{ background:#164d42; color:#b7ffe4; }} .source.calibration {{ background:#644d18; color:#ffe6a0; }}
.images {{ display:grid; grid-template-columns:repeat(3,1fr); gap:2px; background:#0b0f13; }} .images img {{ display:block; width:100%; aspect-ratio:1; object-fit:cover; }} .meta,.metrics,.links {{ margin:9px 13px; color:var(--muted); font-size:12px; line-height:1.45; }} .metrics {{ color:#d9e4ed; }}
.contact {{ width:100%; max-width:1280px; border:1px solid #2d3a46; }} .small {{ font-size:12px; color:var(--muted); }}
@media (max-width:700px) {{ main {{ padding:18px 10px 30px; }} .pair-grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body><main>
<h1>Seabed Dataset v2 · Offline Preview</h1>
<div class="banner"><strong>开发预览状态</strong><br>{formal_note}<br>{html.escape(calibration_note)}<br>缺失状态/条件显示为未生成占位，不代表已完成渲染。</div>
<p class="facts">contact sheet：{f'<a href="{html.escape(contact_link, quote=True)}">打开 9 行检查表</a>' if contact_link else '尚未生成'} · validation：{f'<a href="{html.escape(report_link, quote=True)}">viewer_validation.json</a>' if report_link else '尚未生成'}</p>
<div class="controls"><label>Geometry<select id="state-filter">{option_html}</select></label><label>Lighting<select id="mode-filter">{mode_options}</select></label><label>Water<select id="water-filter">{water_options}</select></label><span id="visible-count" class="small"></span></div>
<h2>配对浏览</h2><p class="small">每张卡片的 Clear 与 degraded 通过同一 reference_id 配对；Natural/Artificial/Mixed 不共享 Clear。Range 为原始 depth_range_m.exr 的显示缩略图，原始 EXR 和 metadata 仍保留在本地输出目录。</p>
<section id="cards" class="pair-grid">{cards}</section>
<h2>资源与边界</h2><p class="facts">这是离线 HTML，未加载 CDN 或网络资源。校准图只用于查看原生光照和水体趋势，不能写入正式 manifest。正式生成后重新运行 build_preview.py 即可刷新正式卡片与 9 行表格。</p>
<script type="application/json" id="viewer-data">{data_json}</script>
<script>
(function() {{
  const cards = Array.from(document.querySelectorAll('.pair-card'));
  const state = document.getElementById('state-filter');
  const mode = document.getElementById('mode-filter');
  const water = document.getElementById('water-filter');
  const count = document.getElementById('visible-count');
  function update() {{
    let visible = 0;
    cards.forEach(card => {{
      const show = (state.value === 'all' || card.dataset.state === state.value) && (mode.value === 'all' || card.dataset.mode === mode.value) && (water.value === 'all' || card.dataset.water === water.value);
      card.hidden = !show; if (show) visible += 1;
    }});
    count.textContent = visible + ' cards';
  }}
  [state, mode, water].forEach(select => select.addEventListener('change', update)); update();
}})();
</script>
</main></body></html>
'''


def build_preview(
    *,
    formal_root: Path,
    calibration_root: Path,
    output_html: Path,
    contact_sheet: Path,
    report_path: Path,
    asset_dir: Path,
    tile_size: int = 220,
) -> dict[str, Any]:
    if tile_size < 64 or tile_size > 1024:
        raise ValueError("tile_size must be between 64 and 1024")
    formal = scan_formal(formal_root)
    calibration = scan_calibration(calibration_root)
    formal_records = formal.get("records", [])
    calibration_records = calibration.get("records", [])
    if formal.get("status") == "complete":
        records = formal_records
    else:
        # For a partial formal run, use calibration only for a grid key that
        # has no valid formal rows.  A formal row always wins, preventing a
        # natural calibration Clear from being paired with another source.
        formal_keys = {
            (record["state_key"], record["lighting_mode"], record["water_id"])
            for record in formal_records
        }
        records = formal_records + [
            record
            for record in calibration_records
            if (record["state_key"], record["lighting_mode"], record["water_id"]) not in formal_keys
        ]
    asset_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[str, Path] = {}
    display_records = [_display_record(record, output_html, asset_dir, tile_size, cache) for record in records]
    # Make all water siblings available to the contact-sheet builder as display
    # assets while retaining one card per pair in the HTML.
    by_group: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in display_records:
        by_group[(record["state_key"], record["lighting_mode"], record["water_id"])] = record
    for record in display_records:
        for water in WATERS:
            sibling = by_group.get((record["state_key"], record["lighting_mode"], water))
            record[f"{water}_asset"] = sibling.get("degraded_asset") if sibling else None
    contact = _contact_sheet(display_records, contact_sheet, output_html, tile_size)
    html_text = _html_document(display_records, formal, calibration, output_html, contact_sheet, report_path)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_text, encoding="utf-8")
    # The HTML links to the validation report itself.  Write the report once,
    # then re-run the local-link check so the delivered report reflects the
    # final on-disk state rather than a transient missing self-link.
    html_validation = _validate_html(output_html, len(display_records))
    report = {
        "schema_version": 1,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "project_root": str(PROJECT_ROOT),
        "formal": {key: value for key, value in formal.items() if key != "records"},
        "calibration": {key: value for key, value in calibration.items() if key != "records"},
        "viewer": {
            "html": str(output_html),
            "contact_sheet": contact,
            "asset_dir": str(asset_dir),
            "card_count": len(display_records),
            "local_resource_validation": html_validation,
            "contact_sheet_validation": contact,
            "formal_pairs_never_filled_by_placeholders": True,
            "formal_manifest_is_authoritative": True,
        },
        "records": [_record_summary(record) for record in display_records],
    }
    _write_json(report_path, report)
    report["viewer"]["local_resource_validation"] = _validate_html(output_html, len(display_records))
    _write_json(report_path, report)
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, default=PROJECT_ROOT / FORMAL_RELATIVE)
    parser.add_argument("--calibration-root", type=Path, default=PROJECT_ROOT / CALIBRATION_RELATIVE)
    parser.add_argument("--output-html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--contact-sheet", type=Path, default=DEFAULT_CONTACT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--tile-size", type=int, default=220)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_preview(
        formal_root=args.formal_root.resolve(),
        calibration_root=args.calibration_root.resolve(),
        output_html=args.output_html.resolve(),
        contact_sheet=args.contact_sheet.resolve(),
        report_path=args.report.resolve(),
        asset_dir=args.asset_dir.resolve(),
        tile_size=args.tile_size,
    )
    summary = {
        "formal_status": report["formal"]["status"],
        "formal_valid_pairs": report["formal"]["valid_pairs"],
        "calibration_valid_pairs": report["calibration"]["valid_pairs"],
        "html": report["viewer"]["html"],
        "contact_sheet": report["viewer"]["contact_sheet"]["path"],
        "viewer_validation": args.report.resolve().as_posix(),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
