#!/usr/bin/env python3
"""Build a deterministic local archive for the additive v3 500-scene batch.

This is an archive/packaging utility for the *existing* v3 batch runner.  It
does not render, import Blender, change the v3 generator, or reinterpret the
v3 algorithm.  The input is the runner's flat output root::

    root/
    references/scene_<number>_clear/
        clear_rgb.exr, clear_rgb.png, depth_range_m.exr,
        depth_camera_z_m.exr, semantic_id.png, instance_id.png,
        valid_mask.png, metadata.json, success.json
      samples/scene_0001_<mode>_d5_<water>/
        degraded_rgb.exr, degraded_rgb.png, metadata.json, success.json
      states/scene_0001/accepted.json
      manifest.jsonl
      batch_manifest.json

The generated staging directory is self-contained and has this layout::

    staging/
      scenes/scene_0001/{metadata.json,scene_manifest.json,clear/,mild/,medium/,strong/}
      archive_manifest_v3_batch.json
      volume_01_manifest.json ... volume_05_manifest.json
      volume_01.zip ... volume_05.zip
      SHA256SUMS.txt

The ZIPs contain only scene-relative POSIX paths.  Scene files, archive
manifests, and ZIP members are ordered deterministically; ZIP timestamps and
compression metadata are fixed.  All writes go to ``--staging`` (and an
optional explicit ``--report``), never to the flat input root.

Only the Python standard library is used.  A small, structurally complete
fixture can use ``--expected-scenes 1 --scenes-per-volume 1``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


sys.dont_write_bytecode = True

SCHEMA_VERSION = 1
BATCH_SCHEMA = "v3_single_lighting_four_state"
ARCHIVE_FORMAT = "seabed_dataset_v3_batch_scene_archive"
VERSION_NOTE = (
    "Additive v3 batch-scale packaging; reuses the completed v3 generator and "
    "does not introduce a new rendering algorithm."
)
FORMAL_SCENE_COUNT = 500
SCENES_PER_VOLUME = 100
STATES = ("clear", "mild", "medium", "strong")
DEGRADED_STATES = STATES[1:]
LIGHTING_MODES = ("natural", "artificial", "mixed")
WATER_IDS = ("mild", "medium", "strong")
SCENE_RE = re.compile(r"^scene_(?P<number>\d{4})$")
REFERENCE_RE = re.compile(r"^(?:ref_[0-9a-fA-F]+|scene_\d{4}_clear)$")
SAMPLE_RE = re.compile(
    r"^(?P<scene>scene_\d{4})_(?P<mode>natural|artificial|mixed)_d5_"
    r"(?P<water>mild|medium|strong)$"
)
CHUNK_SIZE = 1024 * 1024
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
IGNORED_NAMES = frozenset({".DS_Store"})

REFERENCE_FILES = (
    "clear_rgb.exr",
    "clear_rgb.png",
    "depth_range_m.exr",
    "depth_camera_z_m.exr",
    "semantic_id.png",
    "instance_id.png",
    "valid_mask.png",
    "metadata.json",
    "success.json",
)
SAMPLE_FILES = ("degraded_rgb.exr", "degraded_rgb.png", "metadata.json", "success.json")
GROUND_TRUTH_FILES = (
    "depth_range_m.exr",
    "depth_camera_z_m.exr",
    "semantic_id.png",
    "instance_id.png",
    "valid_mask.png",
)


class LayoutError(ValueError):
    """Raised when the v3 flat output cannot be packaged safely."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_text(value: Any) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_safe_relative_path(value: str) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    if value.startswith("/") or re.match(r"^[A-Za-z]:", value):
        return False
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or ".." in parsed.parts or "." in parsed.parts:
        return False
    return value == parsed.as_posix()


def expected_scene_ids(count: int) -> list[str]:
    if count < 1:
        raise ValueError("expected scene count must be positive")
    return [f"scene_{index:04d}" for index in range(1, count + 1)]


def _scene_number(scene_id: str) -> int | None:
    match = SCENE_RE.fullmatch(scene_id)
    return int(match.group("number")) if match else None


def _sample_identity(sample_id: str) -> tuple[str, str, str] | None:
    match = SAMPLE_RE.fullmatch(sample_id)
    if not match:
        return None
    return match.group("scene"), match.group("mode"), match.group("water")


def is_ground_truth_name(name: str) -> bool:
    """Identify the v3 depth/label names that must remain with clear."""

    lower = str(name).lower()
    if lower in GROUND_TRUTH_FILES:
        return True
    stem = Path(lower).stem
    tokens = {token for token in re.split(r"[^a-z0-9]+", stem) if token}
    return bool(tokens & {"depth", "semantic", "instance", "mask", "label", "ground", "truth", "gt"})


def _write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _write_json(path: Path, value: Any) -> None:
    _write_bytes(path, canonical_json_text(value).encode("utf-8"))


def _copy_file(source: Path, target: Path, *, hardlink: bool = False) -> None:
    """Atomically copy bytes without preserving host-dependent mtimes."""

    source = Path(source)
    target = Path(target)
    if not source.is_file() or source.is_symlink():
        raise LayoutError(f"source bundle file is missing or not regular: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if hardlink:
        try:
            os.link(source, target)
            return
        except OSError as exc:
            if exc.errno != getattr(os, "EXDEV", 18):
                raise
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            with source.open("rb") as source_handle:
                shutil.copyfileobj(source_handle, handle, length=CHUNK_SIZE)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LayoutError(f"invalid JSON {path}: {exc}") from exc


def _resolve_under(root: Path, relative: str) -> Path:
    if not is_safe_relative_path(relative):
        raise LayoutError(f"unsafe source relative path: {relative!r}")
    target = Path(root).joinpath(*PurePosixPath(relative).parts).resolve()
    root_resolved = Path(root).resolve()
    if not target.is_relative_to(root_resolved):
        raise LayoutError(f"source path escapes root: {relative!r}")
    return target


def _record(path: Path, relative_to: Path) -> dict[str, Any]:
    relative = path.resolve().relative_to(Path(relative_to).resolve()).as_posix()
    if not is_safe_relative_path(relative):
        raise LayoutError(f"unsafe generated relative path: {relative}")
    stat = path.stat()
    return {"path": relative, "bytes": int(stat.st_size), "sha256": sha256_file(path)}


def _record_map(records: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(item["path"]): item for item in records}


def _verify_success_record(
    bundle: Path,
    required: Sequence[str],
    success: Any,
    errors: list[str],
) -> None:
    if not isinstance(success, Mapping):
        errors.append(f"{bundle}: success.json must contain an object")
        return
    declared = success.get("files")
    if not isinstance(declared, Mapping):
        return
    for name in required:
        if name == "success.json":
            continue
        item = declared.get(name)
        if not isinstance(item, Mapping):
            errors.append(f"{bundle}: success.json is missing file seal for {name}")
            continue
        path = bundle / name
        if not path.is_file():
            continue
        declared_bytes = item.get("bytes", item.get("size"))
        if isinstance(declared_bytes, int) and not isinstance(declared_bytes, bool) and declared_bytes != path.stat().st_size:
            errors.append(f"{bundle}: success byte count mismatch for {name}")
        declared_hash = item.get("sha256")
        if isinstance(declared_hash, str) and declared_hash != sha256_file(path):
            errors.append(f"{bundle}: success SHA-256 mismatch for {name}")


def _validate_bundle(
    path: Path,
    kind: str,
    required: Sequence[str],
    expected: Mapping[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    path = Path(path).resolve()
    label = f"{kind} bundle {path}"
    files: dict[str, dict[str, Any]] = {}
    if not path.is_dir() or path.is_symlink():
        errors.append(f"{label}: directory is missing")
        return {"path": path, "metadata": {}, "success": {}, "files": files}
    for name in required:
        file_path = path / name
        if not file_path.is_file() or file_path.is_symlink():
            errors.append(f"{label}: missing required file {name}")
            continue
        files[name] = {
            "bytes": int(file_path.stat().st_size),
            "sha256": sha256_file(file_path),
        }
    metadata: Any = {}
    metadata_path = path / "metadata.json"
    if metadata_path.is_file():
        try:
            metadata = _read_json(metadata_path)
        except LayoutError as exc:
            errors.append(str(exc))
            metadata = {}
    if not isinstance(metadata, Mapping):
        errors.append(f"{label}: metadata.json must contain an object")
        metadata = {}
    success: Any = {}
    success_path = path / "success.json"
    if success_path.is_file():
        try:
            success = _read_json(success_path)
        except LayoutError as exc:
            errors.append(str(exc))
            success = {}
    if not isinstance(success, Mapping):
        errors.append(f"{label}: success.json must contain an object")
        success = {}
    _verify_success_record(path, required, success, errors)
    if kind == "sample":
        for child in path.iterdir():
            if child.is_file() and child.name not in required and is_ground_truth_name(child.name):
                errors.append(f"{label}: GT/label file is present in degraded bundle: {child.name}")
    for key, value in expected.items():
        if value is None:
            continue
        if key not in metadata:
            errors.append(f"{label}: metadata is missing identity field {key}")
        elif metadata.get(key) != value:
            errors.append(
                f"{label}: metadata identity mismatch for {key}: "
                f"{metadata.get(key)!r} != {value!r}"
            )
        success_metadata = success.get("metadata") if isinstance(success, Mapping) else None
        if isinstance(success_metadata, Mapping) and key in success_metadata and success_metadata[key] != value:
            errors.append(
                f"{label}: success metadata identity mismatch for {key}: "
                f"{success_metadata[key]!r} != {value!r}"
            )
    return {"path": path, "metadata": dict(metadata), "success": dict(success), "files": files}


def _canonical_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in sorted(rows, key=lambda item: str(item.get("sample_id", "")))]


def _load_batch_rows(root: Path, errors: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    batch_path = root / "batch_manifest.json"
    jsonl_path = root / "manifest.jsonl"
    batch: Any = {}
    rows_batch: list[Any] = []
    rows_jsonl: list[Any] = []
    if not batch_path.is_file():
        errors.append(f"missing batch manifest: {batch_path}")
    else:
        try:
            batch = _read_json(batch_path)
        except LayoutError as exc:
            errors.append(str(exc))
            batch = {}
    if isinstance(batch, Mapping) and isinstance(batch.get("rows"), list):
        rows_batch = batch["rows"]
    elif batch_path.is_file():
        errors.append("batch_manifest.json does not contain a rows list")
    if not jsonl_path.is_file():
        errors.append(f"missing JSONL manifest: {jsonl_path}")
    else:
        try:
            for line_number, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    errors.append(f"manifest.jsonl line {line_number} is not an object")
                else:
                    rows_jsonl.append(value)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"invalid manifest.jsonl: {exc}")
    valid_batch = [dict(row) for row in rows_batch if isinstance(row, Mapping)]
    valid_jsonl = [dict(row) for row in rows_jsonl if isinstance(row, Mapping)]
    if _canonical_rows(valid_batch) != _canonical_rows(valid_jsonl):
        errors.append("batch_manifest.json rows differ from manifest.jsonl rows")
    rows = _canonical_rows(valid_batch or valid_jsonl)
    if isinstance(batch, Mapping):
        if isinstance(batch.get("pairs"), int) and batch["pairs"] != len(rows):
            errors.append(f"batch_manifest.json pairs={batch['pairs']} but rows={len(rows)}")
    return rows, dict(batch) if isinstance(batch, Mapping) else {}


def _source_scene_dirs(root: Path, expected_ids: Sequence[str], errors: list[str]) -> None:
    states_root = root / "states"
    if not states_root.is_dir():
        errors.append(f"missing states directory: {states_root}")
        return
    actual = sorted(path.name for path in states_root.iterdir() if path.is_dir() and not path.is_symlink())
    expected = sorted(expected_ids)
    if actual != expected:
        errors.append(f"states scene IDs differ: expected {expected[:3]}..., observed {actual[:3]}...")


def collect_batch(root: Path, expected_count: int = FORMAL_SCENE_COUNT, *, strict: bool = True) -> dict[str, Any]:
    """Read and validate flat v3 batch inputs without writing anything."""

    root = Path(root).expanduser().resolve()
    errors: list[str] = []
    if not root.is_dir():
        raise LayoutError(f"flat batch root is not a directory: {root}")
    rows, batch_manifest = _load_batch_rows(root, errors)
    rows_by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        scene_id = row.get("scene_id", row.get("state_key"))
        if not isinstance(scene_id, str):
            errors.append(f"manifest row has no scene_id: {row!r}")
            continue
        rows_by_scene.setdefault(scene_id, []).append(row)
    expected_ids = expected_scene_ids(expected_count)
    actual_ids = sorted(rows_by_scene, key=lambda value: (_scene_number(value) or 10**9, value))
    if actual_ids != expected_ids:
        errors.append(
            f"batch scene IDs differ: expected {expected_ids[:3]}...{expected_ids[-1]}, "
            f"observed {actual_ids[:3]}..."
        )
    _source_scene_dirs(root, expected_ids, errors)

    scene_plans: dict[str, dict[str, Any]] = {}
    referenced_reference_paths: set[str] = set()
    referenced_sample_paths: set[str] = set()
    referenced_reference_ids: set[str] = set()
    for scene_id in expected_ids:
        scene_rows = rows_by_scene.get(scene_id, [])
        if len(scene_rows) != 3:
            errors.append(f"{scene_id}: expected three degraded rows, observed {len(scene_rows)}")
        modes = {row.get("lighting_mode") for row in scene_rows}
        modes.discard(None)
        if len(modes) != 1 or not modes.issubset(set(LIGHTING_MODES)):
            errors.append(f"{scene_id}: expected one lighting mode, observed {sorted(modes)}")
        water_ids = {row.get("water_id") for row in scene_rows}
        if water_ids != set(WATER_IDS):
            errors.append(f"{scene_id}: expected water IDs {WATER_IDS}, observed {sorted(water_ids)}")

        row_by_water: dict[str, dict[str, Any]] = {}
        reference_ids: set[str] = set()
        reference_path_values: set[str] = set()
        for row in scene_rows:
            water_id = row.get("water_id")
            sample_id = row.get("sample_id")
            if isinstance(water_id, str) and isinstance(sample_id, str):
                row_by_water[water_id] = row
                parsed = _sample_identity(sample_id)
                if parsed is None or parsed[0] != scene_id or parsed[2] != water_id or parsed[1] != row.get("lighting_mode"):
                    errors.append(f"{scene_id}: invalid sample_id identity: {sample_id}")
            reference_id = row.get("reference_id")
            if isinstance(reference_id, str):
                reference_ids.add(reference_id)
            reference_path = row.get("reference_path")
            if isinstance(reference_path, str):
                reference_path_values.add(reference_path)
            sample_path = row.get("sample_path")
            if isinstance(sample_path, str):
                try:
                    _resolve_under(root, sample_path)
                    referenced_sample_paths.add(PurePosixPath(sample_path).as_posix())
                except LayoutError as exc:
                    errors.append(f"{scene_id}: {exc}")
            else:
                errors.append(f"{scene_id}: row is missing sample_path")
        if len(reference_ids) != 1:
            errors.append(f"{scene_id}: expected one reference_id, observed {sorted(reference_ids)}")
        if len(reference_path_values) != 1:
            errors.append(f"{scene_id}: expected one reference_path, observed {sorted(reference_path_values)}")
        reference_id = next(iter(reference_ids), None)
        reference_path = next(iter(reference_path_values), None)
        reference_info = {"path": None, "metadata": {}, "success": {}, "files": {}}
        if not isinstance(reference_id, str):
            reference_id = ""
        if not isinstance(reference_path, str):
            reference_path = f"references/{reference_id}" if reference_id else ""
        if reference_path:
            try:
                reference_path = PurePosixPath(reference_path).as_posix()
                reference_dir = _resolve_under(root, reference_path)
                referenced_reference_paths.add(reference_path)
                if reference_id:
                    referenced_reference_ids.add(reference_id)
                reference_info = _validate_bundle(
                    reference_dir,
                    "reference",
                    REFERENCE_FILES,
                    {
                        "scene_id": scene_id,
                        "state_key": scene_id,
                        "reference_id": reference_id,
                        "lighting_mode": next(iter(modes), None),
                    },
                    errors,
                )
                allowed_reference_names = {reference_id, f"{scene_id}_clear"}
                if reference_dir.name not in allowed_reference_names:
                    errors.append(f"{scene_id}: reference directory {reference_dir.name!r} is not one of {sorted(allowed_reference_names)!r}")
            except LayoutError as exc:
                errors.append(f"{scene_id}: {exc}")

        samples: dict[str, dict[str, Any]] = {}
        for water_id in WATER_IDS:
            row = row_by_water.get(water_id, {})
            sample_id = row.get("sample_id") if isinstance(row, Mapping) else None
            sample_path = row.get("sample_path") if isinstance(row, Mapping) else None
            info = {"path": None, "metadata": {}, "success": {}, "files": {}}
            if isinstance(sample_id, str) and isinstance(sample_path, str):
                try:
                    sample_path = PurePosixPath(sample_path).as_posix()
                    sample_dir = _resolve_under(root, sample_path)
                    info = _validate_bundle(
                        sample_dir,
                        "sample",
                        SAMPLE_FILES,
                        {
                            "sample_id": sample_id,
                            "scene_id": scene_id,
                            "state_key": scene_id,
                            "reference_id": reference_id,
                            "lighting_mode": next(iter(modes), None),
                            "water_id": water_id,
                        },
                        errors,
                    )
                    if sample_dir.name != sample_id:
                        errors.append(f"{scene_id}: sample directory {sample_dir.name!r} != {sample_id!r}")
                except LayoutError as exc:
                    errors.append(f"{scene_id}: {exc}")
            else:
                errors.append(f"{scene_id}: missing sample row for {water_id}")
            samples[water_id] = {"row": dict(row), "info": info}

        checkpoint_path = root / "states" / scene_id / "accepted.json"
        checkpoint: Any = {}
        if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
            errors.append(f"{scene_id}: missing accepted checkpoint {checkpoint_path}")
        else:
            try:
                checkpoint = _read_json(checkpoint_path)
            except LayoutError as exc:
                errors.append(str(exc))
                checkpoint = {}
        if not isinstance(checkpoint, Mapping):
            errors.append(f"{scene_id}: accepted checkpoint must be an object")
            checkpoint = {}
        if checkpoint.get("status") != "accepted":
            errors.append(f"{scene_id}: accepted checkpoint status is not accepted")
        checkpoint_rows = checkpoint.get("manifest_rows")
        if isinstance(checkpoint_rows, list):
            valid_checkpoint_rows = [dict(row) for row in checkpoint_rows if isinstance(row, Mapping)]
            if _canonical_rows(valid_checkpoint_rows) != _canonical_rows(scene_rows):
                errors.append(f"{scene_id}: accepted checkpoint rows differ from batch manifest")
        else:
            errors.append(f"{scene_id}: accepted checkpoint has no manifest_rows list")
        for key, value in {"scene_id": scene_id, "state_key": scene_id}.items():
            if key in checkpoint and checkpoint[key] != value:
                errors.append(f"{scene_id}: checkpoint {key} identity mismatch")

        scene_plans[scene_id] = {
            "scene_id": scene_id,
            "scene_number": _scene_number(scene_id),
            "rows": _canonical_rows(scene_rows),
            "reference_id": reference_id,
            "reference_path": reference_path,
            "reference": reference_info,
            "lighting_mode": next(iter(modes), None),
            "samples": samples,
            "checkpoint_path": f"states/{scene_id}/accepted.json",
            "checkpoint": dict(checkpoint),
        }

    reference_root = root / "references"
    sample_root = root / "samples"
    actual_reference_paths = set()
    if reference_root.is_dir():
        actual_reference_paths = {
            f"references/{path.name}"
            for path in reference_root.iterdir()
            if path.is_dir() and not path.is_symlink() and REFERENCE_RE.fullmatch(path.name)
        }
    actual_sample_paths = set()
    if sample_root.is_dir():
        actual_sample_paths = {
            f"samples/{path.name}"
            for path in sample_root.iterdir()
            if path.is_dir() and not path.is_symlink() and SAMPLE_RE.fullmatch(path.name)
        }
    if actual_reference_paths != referenced_reference_paths:
        errors.append("reference bundle inventory differs from manifest rows")
    if actual_sample_paths != referenced_sample_paths:
        errors.append("sample bundle inventory differs from manifest rows")
    if len(referenced_reference_ids) != expected_count:
        errors.append(f"expected {expected_count} unique references, observed {len(referenced_reference_ids)}")
    if len(referenced_sample_paths) != expected_count * 3:
        errors.append(f"expected {expected_count * 3} unique samples, observed {len(referenced_sample_paths)}")

    result = {
        "root": root,
        "rows": rows,
        "batch_manifest": batch_manifest,
        "scene_ids": expected_ids,
        "scenes": scene_plans,
        "reference_count": len(referenced_reference_ids),
        "sample_count": len(referenced_sample_paths),
        "errors": sorted(set(errors)),
    }
    if strict and result["errors"]:
        raise LayoutError("; ".join(result["errors"][:12]))
    return result


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=name, date_time=ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (0o100644 & 0xFFFF) << 16
    info.internal_attr = 0
    info.extra = b""
    info.comment = b""
    return info


def _walk_records(scene_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(scene_root.rglob("*"), key=lambda item: item.as_posix()):
        if path.name in IGNORED_NAMES:
            continue
        if path.is_symlink():
            raise LayoutError(f"symlink is not allowed in staged scene: {path}")
        if path.is_file():
            records.append(_record(path, scene_root))
    records.sort(key=lambda item: str(item["path"]))
    return records


def _write_deterministic_zip(path: Path, records: Sequence[Mapping[str, Any]], scene_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            allowZip64=True,
        ) as archive:
            for record in sorted(records, key=lambda item: str(item["path"])):
                name = str(record["path"])
                if not is_safe_relative_path(name):
                    raise LayoutError(f"unsafe archive path: {name}")
                source = scene_root.joinpath(*PurePosixPath(name).parts)
                if not source.is_file() or source.is_symlink():
                    raise LayoutError(f"staged file disappeared: {source}")
                with archive.open(_zip_info(name), mode="w") as destination, source.open("rb") as source_handle:
                    shutil.copyfileobj(source_handle, destination, length=CHUNK_SIZE)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _scene_manifest(scene: Mapping[str, Any]) -> dict[str, Any]:
    reference_id = str(scene["reference_id"])
    mode = scene.get("lighting_mode")
    sample_map = {
        water_id: {
            "sample_id": item["row"].get("sample_id"),
            "path": f"{water_id}/metadata.json",
        }
        for water_id, item in scene["samples"].items()
    }
    shared = [f"clear/{name}" for name in GROUND_TRUTH_FILES]
    return {
        "schema_version": SCHEMA_VERSION,
        "batch_schema": BATCH_SCHEMA,
        "version_note": VERSION_NOTE,
        "scene_id": scene["scene_id"],
        "scene_number": scene["scene_number"],
        "reference_id": reference_id,
        "lighting_mode": mode,
        "states": {
            "clear": {
                "kind": "reference",
                "relative_path": "clear",
                "files": list(REFERENCE_FILES),
                "reference_id": reference_id,
                "ground_truth": shared,
            },
            **{
                state: {
                    "kind": "degraded",
                    "relative_path": state,
                    "files": list(SAMPLE_FILES),
                    "sample_id": sample_map[state]["sample_id"],
                    "ground_truth": [],
                }
                for state in DEGRADED_STATES
            },
        },
        "shared_ground_truth": {
            "authoritative_state": "clear",
            "paths": shared,
            "not_copied_to": list(DEGRADED_STATES),
        },
    }


def _scene_metadata(scene: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "batch_schema": BATCH_SCHEMA,
        "version_note": VERSION_NOTE,
        "scene_id": scene["scene_id"],
        "scene_number": scene["scene_number"],
        "reference_id": scene["reference_id"],
        "lighting_mode": scene.get("lighting_mode"),
        "state_order": list(STATES),
        "sample_ids": {
            water_id: item["row"].get("sample_id") for water_id, item in scene["samples"].items()
        },
        "source_paths": {
            "reference": scene["reference_path"],
            "samples": {
                water_id: item["row"].get("sample_path")
                for water_id, item in scene["samples"].items()
            },
            "accepted": scene["checkpoint_path"],
        },
        "ground_truth": {
            "shared_from": "clear",
            "files": list(GROUND_TRUTH_FILES),
        },
    }


def _copy_scene_to_staging(scene: Mapping[str, Any], staging: Path, *, hardlink_inputs: bool = False) -> None:
    scene_target = staging / "scenes" / str(scene["scene_id"])
    reference = scene["reference"]
    for name in REFERENCE_FILES:
        _copy_file(Path(reference["path"]) / name, scene_target / "clear" / name, hardlink=hardlink_inputs)
    for state in DEGRADED_STATES:
        info = scene["samples"][state]["info"]
        for name in SAMPLE_FILES:
            _copy_file(Path(info["path"]) / name, scene_target / state / name, hardlink=hardlink_inputs)
    _write_json(scene_target / "metadata.json", _scene_metadata(scene))
    _write_json(scene_target / "scene_manifest.json", _scene_manifest(scene))


def _staged_scene_summary(scene_id: str, scene_root: Path, source_scene: Mapping[str, Any]) -> dict[str, Any]:
    local_records = _walk_records(scene_root)
    records = [
        {**record, "path": f"{scene_id}/{record['path']}"}
        for record in local_records
    ]
    by_state: dict[str, list[str]] = {state: [] for state in STATES}
    root_files: list[str] = []
    for record in local_records:
        path = str(record["path"])
        parts = PurePosixPath(path).parts
        if len(parts) >= 2 and parts[0] in STATES:
            by_state[parts[0]].append(path)
        else:
            root_files.append(f"{scene_id}/{path}")
    return {
        "scene_id": scene_id,
        "scene_number": source_scene["scene_number"],
        "reference_id": source_scene["reference_id"],
        "lighting_mode": source_scene.get("lighting_mode"),
        "state_order": list(STATES),
        "states": {
            state: {
                "relative_path": state,
                "files": sorted(by_state[state]),
                "sample_id": None if state == "clear" else source_scene["samples"][state]["row"].get("sample_id"),
            }
            for state in STATES
        },
        "root_files": sorted(root_files),
        "shared_ground_truth": [f"clear/{name}" for name in GROUND_TRUTH_FILES],
        "file_count": len(records),
        "bytes": sum(int(record["bytes"]) for record in records),
        "files": records,
    }


def _volume_base(scenes: Sequence[Mapping[str, Any]], scenes_per_volume: int) -> list[dict[str, Any]]:
    volumes: list[dict[str, Any]] = []
    for offset in range(0, len(scenes), scenes_per_volume):
        selected = list(scenes[offset : offset + scenes_per_volume])
        index = offset // scenes_per_volume + 1
        records = [record for scene in selected for record in scene["files"]]
        records.sort(key=lambda item: str(item["path"]))
        volumes.append(
            {
                "volume_id": f"volume_{index:02d}",
                "volume_index": index,
                "scene_ids": [str(scene["scene_id"]) for scene in selected],
                "scene_count": len(selected),
                "scene_start": str(selected[0]["scene_id"]),
                "scene_end": str(selected[-1]["scene_id"]),
                "files": records,
                "file_count": len(records),
                "uncompressed_bytes": sum(int(record["bytes"]) for record in records),
            }
        )
    return volumes


def build_archive(
    root: Path,
    staging: Path,
    *,
    expected_count: int = FORMAL_SCENE_COUNT,
    scenes_per_volume: int = SCENES_PER_VOLUME,
    hardlink_inputs: bool = False,
) -> dict[str, Any]:
    """Assemble flat v3 bundles into deterministic scene volumes."""

    root = Path(root).expanduser().resolve()
    staging = Path(staging).expanduser().resolve()
    if scenes_per_volume < 1:
        raise LayoutError("--scenes-per-volume must be positive")
    if staging == root or staging.is_relative_to(root):
        raise LayoutError("--staging must be outside the flat input root")
    source = collect_batch(root, expected_count, strict=True)
    staging.mkdir(parents=True, exist_ok=True)

    for scene_id in source["scene_ids"]:
        _copy_scene_to_staging(source["scenes"][scene_id], staging, hardlink_inputs=hardlink_inputs)

    staged_scenes: list[dict[str, Any]] = []
    scene_root = staging / "scenes"
    for scene_id in source["scene_ids"]:
        staged_scenes.append(
            _staged_scene_summary(scene_id, scene_root / scene_id, source["scenes"][scene_id])
        )
    staged_scenes.sort(key=lambda scene: str(scene["scene_id"]))
    source_manifest_records = {
        name: _record(root / name, root)
        for name in ("batch_manifest.json", "manifest.jsonl")
        if (root / name).is_file()
    }
    source_rows_digest = sha256_bytes(canonical_json_bytes(source["rows"]))
    volumes = _volume_base(staged_scenes, scenes_per_volume)

    for volume in volumes:
        zip_name = f"{volume['volume_id']}.zip"
        zip_path = staging / zip_name
        _write_deterministic_zip(zip_path, volume["files"], scene_root)
        volume["zip_name"] = zip_name
        volume["zip_bytes"] = int(zip_path.stat().st_size)
        volume["zip_sha256"] = sha256_file(zip_path)

    for volume in volumes:
        manifest_name = f"{volume['volume_id']}_manifest.json"
        volume["manifest_name"] = manifest_name
        volume_manifest = {
            "schema_version": SCHEMA_VERSION,
            "archive_format": ARCHIVE_FORMAT,
            "batch_schema": BATCH_SCHEMA,
            "version_note": VERSION_NOTE,
            "volume_id": volume["volume_id"],
            "volume_index": volume["volume_index"],
            "scenes_per_volume": scenes_per_volume,
            "scene_ids": volume["scene_ids"],
            "scene_count": volume["scene_count"],
            "scene_start": volume["scene_start"],
            "scene_end": volume["scene_end"],
            "source_manifest_rows_sha256": source_rows_digest,
            "files": volume["files"],
            "file_count": volume["file_count"],
            "uncompressed_bytes": volume["uncompressed_bytes"],
            "zip": {
                "name": volume["zip_name"],
                "bytes": volume["zip_bytes"],
                "sha256": volume["zip_sha256"],
            },
        }
        _write_json(staging / manifest_name, volume_manifest)
        volume["manifest_sha256"] = sha256_file(staging / manifest_name)

    archive_manifest = {
        "schema_version": SCHEMA_VERSION,
        "archive_format": ARCHIVE_FORMAT,
        "batch_schema": BATCH_SCHEMA,
        "version_note": VERSION_NOTE,
        "source_layout": "v3_batch_flat",
        "scene_name_pattern": "scene_{number:04d}",
        "state_order": list(STATES),
        "lighting_modes": list(LIGHTING_MODES),
        "ground_truth_policy": {
            "authoritative_state": "clear",
            "shared_within_scene": True,
            "files": list(GROUND_TRUTH_FILES),
            "forbidden_states": list(DEGRADED_STATES),
        },
        "scene_count": len(staged_scenes),
        "scene_ids": [str(scene["scene_id"]) for scene in staged_scenes],
        "scenes_per_volume": scenes_per_volume,
        "volume_count": len(volumes),
        "source": {
            "manifest_rows_sha256": source_rows_digest,
            "input_manifests": source_manifest_records,
            "reference_count": source["reference_count"],
            "sample_count": source["sample_count"],
            "checkpoint_count": len(source["scene_ids"]),
        },
        "files": [record for scene in staged_scenes for record in scene["files"]],
        "scenes": staged_scenes,
        "volumes": [
            {
                "volume_id": volume["volume_id"],
                "volume_index": volume["volume_index"],
                "manifest_name": volume["manifest_name"],
                "manifest_sha256": volume["manifest_sha256"],
                "zip_name": volume["zip_name"],
                "zip_bytes": volume["zip_bytes"],
                "zip_sha256": volume["zip_sha256"],
                "scene_ids": volume["scene_ids"],
                "scene_count": volume["scene_count"],
                "file_count": volume["file_count"],
                "uncompressed_bytes": volume["uncompressed_bytes"],
            }
            for volume in volumes
        ],
        "determinism": {
            "file_order": "POSIX lexical path",
            "zip_timestamp": "1980-01-01T00:00:00",
            "zip_compression": "deflate level 9",
            "zip_comments": False,
        },
    }
    archive_name = "archive_manifest_v3_batch.json"
    _write_json(staging / archive_name, archive_manifest)
    archive_hash = sha256_file(staging / archive_name)

    checksum_names = [
        archive_name,
        *[str(volume["manifest_name"]) for volume in volumes],
        *[str(volume["zip_name"]) for volume in volumes],
    ]
    checksums: dict[str, str] = {}
    lines: list[str] = []
    for name in sorted(checksum_names):
        value = sha256_file(staging / name)
        checksums[name] = value
        lines.append(f"{value}  {name}")
    _write_bytes(staging / "SHA256SUMS.txt", ("\n".join(lines) + "\n").encode("utf-8"))

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "archive_format": ARCHIVE_FORMAT,
        "batch_schema": BATCH_SCHEMA,
        "version_note": VERSION_NOTE,
        "source_root": str(root),
        "staging": str(staging),
        "scene_count": len(staged_scenes),
        "scenes_per_volume": scenes_per_volume,
        "volume_count": len(volumes),
        "hardlink_inputs": hardlink_inputs,
        "source_manifest_rows_sha256": source_rows_digest,
        "archive_manifest": archive_name,
        "archive_manifest_sha256": archive_hash,
        "checksum_file": "SHA256SUMS.txt",
        "checksums": checksums,
        "generated_files": [archive_name, *[str(volume["manifest_name"]) for volume in volumes], *[str(volume["zip_name"]) for volume in volumes], "SHA256SUMS.txt"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Package existing flat v3 batch bundles into deterministic scene "
            "archives; this tool does not render or change the v3 algorithm."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", required=True, type=Path, help="flat v3 batch root with references/samples/states/manifests")
    parser.add_argument("--staging", required=True, type=Path, help="separate output directory for scenes, ZIPs, manifests, and SHA256SUMS.txt")
    parser.add_argument("--report", type=str, help="optional build report path; use '-' for complete JSON on stdout")
    parser.add_argument("--expected-scenes", type=int, default=FORMAL_SCENE_COUNT, help="exact contiguous scene count; set fixture size for a small test")
    parser.add_argument("--scenes-per-volume", type=int, default=SCENES_PER_VOLUME, help="scene count per volume; formal batch uses 100")
    parser.add_argument("--hardlink-inputs", action="store_true", help="hard-link source bundle files into staging when possible to avoid duplicating raw bytes")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = build_archive(
            args.root,
            args.staging,
            expected_count=args.expected_scenes,
            scenes_per_volume=args.scenes_per_volume,
            hardlink_inputs=args.hardlink_inputs,
        )
        if args.report == "-":
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        elif args.report:
            report_path = Path(args.report).expanduser().resolve()
            root = Path(args.root).expanduser().resolve()
            staging = Path(args.staging).expanduser().resolve()
            if report_path == root or report_path.is_relative_to(root):
                raise LayoutError("--report must not write inside the flat input root")
            _write_json(report_path, report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "scene_count": report["scene_count"],
                    "volume_count": report["volume_count"],
                    "archive_manifest_sha256": report["archive_manifest_sha256"],
                    "staging": report["staging"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except (LayoutError, OSError, ValueError) as exc:
        print(f"build_archive_manifest_v3_batch: ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
