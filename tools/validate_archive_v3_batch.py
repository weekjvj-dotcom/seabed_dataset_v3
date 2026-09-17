#!/usr/bin/env python3
"""Read-only validator for the additive v3 500-scene batch archive.

The input ``--root`` is the flat output of ``tools/run_500_scenes.py``.  The
validator checks its ``references/``, ``samples/``, ``states/``,
``manifest.jsonl`` and ``batch_manifest.json`` before checking the assembled
``--staging/scenes/scene_####`` tree and its five 100-scene ZIP volumes.

This is deliberately a v3 batch packaging check, not a v4 algorithm or a
rendering check.  It imports only the sibling standard-library builder module;
it never imports Blender or project runtime modules.  Input trees are never
modified.  A report is written only when ``--report`` is given, and must be
outside both input directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


sys.dont_write_bytecode = True

try:
    from build_archive_manifest_v3_batch import (  # type: ignore
        ARCHIVE_FORMAT,
        BATCH_SCHEMA,
        DEGRADED_STATES,
        FORMAL_SCENE_COUNT,
        GROUND_TRUTH_FILES,
        REFERENCE_FILES,
        SAMPLE_FILES,
        SCENES_PER_VOLUME,
        STATES,
        LayoutError,
        canonical_json_bytes,
        collect_batch,
        expected_scene_ids,
        is_safe_relative_path,
        sha256_file,
    )
except ImportError as exc:  # pragma: no cover - only relevant to unusual importers
    raise SystemExit(f"validator cannot import sibling builder: {exc}") from exc


MANIFEST_NAMES = ("archive_manifest_v3_batch.json", "archive_manifest.json")
CHECKSUM_NAMES = ("SHA256SUMS.txt", "SHA256SUMS", "sha256sums.txt")
SCENE_RE = re.compile(r"^scene_\d{4}$")


class Validation:
    """Accumulate independent checks for one stable JSON report."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.checks: dict[str, dict[str, Any]] = {}
        self.samples: list[dict[str, Any]] = []

    def pass_check(self, name: str, details: Mapping[str, Any] | None = None) -> None:
        self.checks[name] = {"status": "pass", **dict(details or {})}

    def fail_check(self, name: str, message: str, details: Mapping[str, Any] | None = None) -> None:
        self.checks[name] = {"status": "fail", "message": message, **dict(details or {})}
        self.errors.append(f"{name}: {message}")


def _read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _record_bytes(record: Mapping[str, Any]) -> int | None:
    value = record.get("bytes", record.get("size"))
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _safe_staging_file(staging: Path, name: Any) -> Path | None:
    if not isinstance(name, str) or not is_safe_relative_path(name):
        return None
    parsed = PurePosixPath(name)
    if len(parsed.parts) != 1:
        return None
    return staging / parsed.parts[0]


def _walk_records(scene_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(scene_root.rglob("*"), key=lambda item: item.as_posix()):
        if path.name == ".DS_Store":
            continue
        if path.is_symlink():
            raise LayoutError(f"symlink is not allowed in staging scene: {path}")
        if path.is_file():
            relative = path.relative_to(scene_root).as_posix()
            if not is_safe_relative_path(relative):
                raise LayoutError(f"unsafe staging scene path: {relative}")
            records.append(
                {
                    "path": relative,
                    "bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
    return records


def _global_scene_records(scene_id: str, scene_dir: Path) -> list[dict[str, Any]]:
    return [
        {**record, "path": f"{scene_id}/{record['path']}"}
        for record in _walk_records(scene_dir)
    ]


def _rows_map(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("sample_id")): row for row in rows if isinstance(row.get("sample_id"), str)}


def _compare_file(
    source: Path,
    target: Path,
    label: str,
    validation: Validation,
) -> None:
    if not source.is_file() or source.is_symlink():
        validation.errors.append(f"{label}: source file is missing")
        return
    if not target.is_file() or target.is_symlink():
        validation.errors.append(f"{label}: staged file is missing")
        return
    if source.stat().st_size != target.stat().st_size:
        validation.errors.append(f"{label}: byte count mismatch")
    if sha256_file(source) != sha256_file(target):
        validation.errors.append(f"{label}: SHA-256 mismatch")


def _metadata_identity(
    metadata: Mapping[str, Any],
    expected: Mapping[str, Any],
    label: str,
    validation: Validation,
) -> None:
    for key, value in expected.items():
        if value is None:
            continue
        if key not in metadata:
            validation.errors.append(f"{label}: missing metadata identity field {key}")
        elif metadata.get(key) != value:
            validation.errors.append(
                f"{label}: metadata identity mismatch for {key}: "
                f"{metadata.get(key)!r} != {value!r}"
            )


def _check_staged_scene(
    scene_id: str,
    scene_dir: Path,
    source_scene: Mapping[str, Any],
    root: Path,
    validation: Validation,
) -> list[dict[str, Any]]:
    """Check one assembled scene and compare every copied source byte."""

    records: list[dict[str, Any]] = []
    if not scene_dir.is_dir() or scene_dir.is_symlink():
        validation.errors.append(f"{scene_id}: staged scene directory is missing")
        return records
    try:
        records = _global_scene_records(scene_id, scene_dir)
    except (OSError, LayoutError) as exc:
        validation.errors.append(f"{scene_id}: cannot inventory staged files: {exc}")

    metadata_path = scene_dir / "metadata.json"
    manifest_path = scene_dir / "scene_manifest.json"
    metadata: Any = {}
    scene_manifest: Any = {}
    if not metadata_path.is_file() or metadata_path.is_symlink():
        validation.errors.append(f"{scene_id}: missing scene-level metadata.json")
    else:
        try:
            metadata = _read_json(metadata_path)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            validation.errors.append(f"{scene_id}: invalid scene metadata.json: {exc}")
    if not manifest_path.is_file() or manifest_path.is_symlink():
        validation.errors.append(f"{scene_id}: missing scene_manifest.json")
    else:
        try:
            scene_manifest = _read_json(manifest_path)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            validation.errors.append(f"{scene_id}: invalid scene_manifest.json: {exc}")

    expected_samples = {
        state: source_scene.get("samples", {}).get(state, {})
        for state in DEGRADED_STATES
    }
    expected_mode = source_scene.get("lighting_mode")
    _metadata_identity(
        metadata if isinstance(metadata, Mapping) else {},
        {
            "scene_id": scene_id,
            "reference_id": source_scene.get("reference_id"),
            "lighting_mode": expected_mode,
        },
        f"{scene_id}/metadata.json",
        validation,
    )
    if isinstance(metadata, Mapping) and metadata.get("state_order") != list(STATES):
        validation.errors.append(f"{scene_id}/metadata.json: state_order is not {list(STATES)!r}")

    if not isinstance(scene_manifest, Mapping):
        scene_manifest = {}
    _metadata_identity(
        scene_manifest,
        {
            "scene_id": scene_id,
            "reference_id": source_scene.get("reference_id"),
            "lighting_mode": expected_mode,
        },
        f"{scene_id}/scene_manifest.json",
        validation,
    )
    states_value = scene_manifest.get("states")
    if not isinstance(states_value, Mapping) or set(states_value) != set(STATES):
        validation.errors.append(f"{scene_id}/scene_manifest.json: states do not contain exactly {list(STATES)!r}")

    shared_expected = [f"clear/{name}" for name in GROUND_TRUTH_FILES]
    shared = scene_manifest.get("shared_ground_truth")
    if not isinstance(shared, Mapping) or shared.get("authoritative_state") != "clear" or shared.get("paths") != shared_expected:
        validation.errors.append(f"{scene_id}/scene_manifest.json: clear shared_ground_truth reference mismatch")
    if isinstance(shared, Mapping) and shared.get("not_copied_to") != list(DEGRADED_STATES):
        validation.errors.append(f"{scene_id}/scene_manifest.json: degraded GT exclusion list mismatch")

    state_required: dict[str, Sequence[str]] = {"clear": REFERENCE_FILES}
    state_required.update({state: SAMPLE_FILES for state in DEGRADED_STATES})
    for state in STATES:
        state_dir = scene_dir / state
        if not state_dir.is_dir() or state_dir.is_symlink():
            validation.errors.append(f"{scene_id}: missing staged state directory {state}")
            continue
        actual_names = sorted(
            path.name
            for path in state_dir.iterdir()
            if path.name != ".DS_Store"
        )
        expected_names = sorted(state_required[state])
        if actual_names != expected_names:
            validation.errors.append(
                f"{scene_id}/{state}: files differ; expected {expected_names}, observed {actual_names}"
            )
        if state == "clear":
            source_info = source_scene.get("reference", {})
            source_path = source_info.get("path") if isinstance(source_info, Mapping) else None
        else:
            source_info = expected_samples[state].get("info", {}) if isinstance(expected_samples[state], Mapping) else {}
            source_path = source_info.get("path") if isinstance(source_info, Mapping) else None
        if isinstance(source_path, Path):
            for name in state_required[state]:
                _compare_file(
                    source_path / name,
                    state_dir / name,
                    f"{scene_id}/{state}/{name}",
                    validation,
                )
        elif isinstance(source_path, str):
            for name in state_required[state]:
                _compare_file(
                    Path(source_path) / name,
                    state_dir / name,
                    f"{scene_id}/{state}/{name}",
                    validation,
                )

        state_value = states_value.get(state) if isinstance(states_value, Mapping) else None
        if not isinstance(state_value, Mapping):
            validation.errors.append(f"{scene_id}/scene_manifest.json: missing state record {state}")
            continue
        if state_value.get("files") != list(state_required[state]):
            validation.errors.append(f"{scene_id}/scene_manifest.json: file list mismatch for {state}")
        if state == "clear":
            if state_value.get("ground_truth") != shared_expected:
                validation.errors.append(f"{scene_id}/scene_manifest.json: clear GT list mismatch")
        elif state_value.get("ground_truth") != []:
            validation.errors.append(f"{scene_id}/scene_manifest.json: non-clear GT list is not empty for {state}")
        if state != "clear" and state_value.get("sample_id") != expected_samples[state].get("row", {}).get("sample_id"):
            validation.errors.append(f"{scene_id}/scene_manifest.json: sample identity mismatch for {state}")

    staged_root_files = sorted(
        path.name
        for path in scene_dir.iterdir()
        if path.is_file() and path.name != ".DS_Store"
    )
    if staged_root_files != ["metadata.json", "scene_manifest.json"]:
        validation.errors.append(
            f"{scene_id}: unexpected scene-root files; observed {staged_root_files}"
        )
    return records


def _sample_ids(scene_ids: Sequence[str], count: int, seed: int) -> list[str]:
    if count <= 0:
        return []
    ranked = sorted(
        scene_ids,
        key=lambda scene_id: (
            hashlib.sha256(f"v3-batch-sample:{seed}:{scene_id}".encode("utf-8")).hexdigest(),
            scene_id,
        ),
    )
    return sorted(ranked[: min(count, len(ranked))])


def _sample_extract(
    staging: Path,
    volume: Mapping[str, Any],
    expected_rows: Mapping[str, Mapping[str, Any]],
    sample_count: int,
    sample_seed: int,
    validation: Validation,
) -> None:
    zip_name = volume.get("zip_name")
    zip_path = _safe_staging_file(staging, zip_name)
    scene_ids = volume.get("scene_ids")
    if zip_path is None or not zip_path.is_file() or not isinstance(scene_ids, list):
        return
    selected = _sample_ids([item for item in scene_ids if isinstance(item, str)], sample_count, sample_seed)
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            file_names = sorted(
                info.filename for info in archive.infolist() if not info.is_dir()
            )
            for scene_id in selected:
                prefix = f"{scene_id}/"
                scene_names = [name for name in file_names if name.startswith(prefix)]
                result: dict[str, Any] = {
                    "volume_id": volume.get("volume_id"),
                    "scene_id": scene_id,
                    "status": "pass",
                }
                if not scene_names:
                    result["status"] = "fail"
                    result["message"] = "no ZIP entries"
                    validation.errors.append(f"sample_extract: {volume.get('volume_id')}/{scene_id} has no entries")
                    validation.samples.append(result)
                    continue
                with tempfile.TemporaryDirectory(prefix="v3-batch-archive-sample-") as directory:
                    temporary = Path(directory)
                    for name in scene_names:
                        if not is_safe_relative_path(name):
                            result["status"] = "fail"
                            validation.errors.append(f"sample_extract: unsafe ZIP path {name!r}")
                            continue
                        target = temporary.joinpath(*PurePosixPath(name).parts)
                        if not target.resolve().is_relative_to(temporary.resolve()):
                            result["status"] = "fail"
                            validation.errors.append(f"sample_extract: path escape {name!r}")
                            continue
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(name, "r") as source, target.open("wb") as destination:
                            shutil.copyfileobj(source, destination, length=1024 * 1024)
                        expected = expected_rows.get(name)
                        if expected is not None:
                            expected_bytes = _record_bytes(expected)
                            if expected_bytes is not None and target.stat().st_size != expected_bytes:
                                result["status"] = "fail"
                                validation.errors.append(f"sample_extract: size mismatch {name}")
                            expected_hash = expected.get("sha256")
                            if isinstance(expected_hash, str) and sha256_file(target) != expected_hash:
                                result["status"] = "fail"
                                validation.errors.append(f"sample_extract: SHA-256 mismatch {name}")
                    extracted_scene = temporary / scene_id
                    for state in STATES:
                        state_dir = extracted_scene / state
                        if not state_dir.is_dir() or not any(path.is_file() for path in state_dir.rglob("*")):
                            result["status"] = "fail"
                            validation.errors.append(f"sample_extract: missing/empty {scene_id}/{state}")
                    for name in ("metadata.json", "scene_manifest.json"):
                        path = extracted_scene / name
                        if not path.is_file():
                            result["status"] = "fail"
                            validation.errors.append(f"sample_extract: missing {scene_id}/{name}")
                        else:
                            try:
                                _read_json(path)
                            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                                result["status"] = "fail"
                                validation.errors.append(f"sample_extract: invalid {scene_id}/{name}: {exc}")
                validation.samples.append(result)
    except (OSError, zipfile.BadZipFile) as exc:
        validation.errors.append(f"sample_extract: cannot inspect {zip_path}: {exc}")


def _normalise_manifest_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping):
        return []
    raw = value.get("files", value.get("entries", value.get("records")))
    result: list[dict[str, Any]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, Mapping):
                result.append(dict(item))
            elif isinstance(item, str):
                result.append({"path": item})
    elif isinstance(raw, Mapping):
        for path, item in raw.items():
            row = dict(item) if isinstance(item, Mapping) else {}
            row.setdefault("path", str(path))
            result.append(row)
    return result


def _row_map(
    rows: Iterable[Mapping[str, Any]],
    validation: Validation,
    label: str,
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        path = row.get("path")
        if not isinstance(path, str) or not is_safe_relative_path(path):
            validation.errors.append(f"{label}: unsafe/missing relative path {path!r}")
            continue
        if path in result:
            validation.errors.append(f"{label}: duplicate path {path}")
            continue
        result[path] = row
    return result


def _validate_staging(
    root: Path,
    staging: Path,
    source: Mapping[str, Any],
    expected_count: int,
    validation: Validation,
) -> tuple[list[str], dict[str, Mapping[str, Any]]]:
    scenes_root = staging / "scenes"
    if not scenes_root.is_dir():
        validation.fail_check("staging_scenes", f"missing scenes directory: {scenes_root}")
        return [], {}
    actual_ids = sorted(
        path.name
        for path in scenes_root.iterdir()
        if path.is_dir() and not path.is_symlink() and SCENE_RE.fullmatch(path.name)
    )
    expected_ids = expected_scene_ids(expected_count)
    if actual_ids != expected_ids:
        validation.fail_check(
            "staging_scene_count",
            f"expected {expected_count} scene directories, observed {len(actual_ids)}",
            {"expected": expected_count, "observed": len(actual_ids)},
        )
    else:
        validation.pass_check("staging_scene_count", {"count": len(actual_ids)})
    staged_records: dict[str, Mapping[str, Any]] = {}
    source_scenes = source.get("scenes", {})
    for scene_id in expected_ids:
        source_scene = source_scenes.get(scene_id, {}) if isinstance(source_scenes, Mapping) else {}
        records = _check_staged_scene(
            scene_id,
            scenes_root / scene_id,
            source_scene,
            root,
            validation,
        )
        for record in records:
            path = str(record["path"])
            if path in staged_records:
                validation.errors.append(f"staging_files: duplicate path {path}")
            else:
                staged_records[path] = record
    if not any(item.startswith("staging_") for item in validation.errors):
        validation.pass_check("staging_scene_layout", {"scene_count": len(actual_ids), "file_count": len(staged_records)})
    return actual_ids, staged_records


def _validate_archive_files(
    staging: Path,
    manifest: Mapping[str, Any],
    staged_records: Mapping[str, Mapping[str, Any]],
    validation: Validation,
) -> None:
    manifest_rows = _row_map(_normalise_manifest_rows(manifest), validation, "archive_manifest_files")
    missing = sorted(set(staged_records) - set(manifest_rows))
    extra = sorted(set(manifest_rows) - set(staged_records))
    if missing:
        validation.errors.append(f"archive_manifest_files: missing {missing[:8]}")
    if extra:
        validation.errors.append(f"archive_manifest_files: unexpected {extra[:8]}")
    for path in sorted(set(staged_records) & set(manifest_rows)):
        expected = staged_records[path]
        declared = manifest_rows[path]
        if _record_bytes(expected) != _record_bytes(declared):
            validation.errors.append(f"archive_manifest_files: byte count mismatch {path}")
        if declared.get("sha256") != expected.get("sha256"):
            validation.errors.append(f"archive_manifest_files: SHA-256 mismatch {path}")
    if not any(item.startswith("archive_manifest_files:") for item in validation.errors):
        validation.pass_check("archive_manifest_files", {"file_count": len(manifest_rows)})


def _validate_volumes(
    staging: Path,
    manifest: Mapping[str, Any],
    staged_records: Mapping[str, Mapping[str, Any]],
    source_scene_ids: Sequence[str],
    expected_count: int,
    scenes_per_volume: int,
    sample_count: int,
    sample_seed: int,
    validation: Validation,
) -> dict[str, Any]:
    raw_volumes = manifest.get("volumes")
    volumes = raw_volumes if isinstance(raw_volumes, list) else []
    expected_volume_count = (expected_count + scenes_per_volume - 1) // scenes_per_volume
    if len(volumes) != expected_volume_count:
        validation.fail_check(
            "volume_count",
            f"expected {expected_volume_count}, observed {len(volumes)}",
            {"expected": expected_volume_count, "observed": len(volumes)},
        )
    else:
        validation.pass_check("volume_count", {"count": len(volumes)})
    results: dict[str, Any] = {}
    seen_scenes: list[str] = []
    for index, value in enumerate(volumes, start=1):
        expected_id = f"volume_{index:02d}"
        if not isinstance(value, Mapping):
            validation.errors.append(f"{expected_id}: volume entry is not an object")
            continue
        volume_id = value.get("volume_id", expected_id)
        if volume_id != expected_id:
            validation.errors.append(f"volume_order: expected {expected_id}, observed {volume_id!r}")
        scene_ids = value.get("scene_ids")
        scene_ids = scene_ids if isinstance(scene_ids, list) else []
        expected_scene_slice = list(source_scene_ids[(index - 1) * scenes_per_volume : index * scenes_per_volume])
        if scene_ids != expected_scene_slice:
            validation.errors.append(f"{expected_id}: scene slice mismatch")
        seen_scenes.extend(item for item in scene_ids if isinstance(item, str))
        manifest_name = value.get("manifest_name", f"{expected_id}_manifest.json")
        zip_name = value.get("zip_name", f"{expected_id}.zip")
        results[expected_id] = {
            "value": dict(value),
            "scene_ids": scene_ids,
            "manifest_name": manifest_name,
            "zip_name": zip_name,
        }
        volume_manifest_path = _safe_staging_file(staging, manifest_name)
        if volume_manifest_path is None or not volume_manifest_path.is_file():
            validation.errors.append(f"{expected_id}: missing volume manifest {manifest_name!r}")
            continue
        try:
            volume_manifest = _read_json(volume_manifest_path)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            validation.errors.append(f"{expected_id}: invalid volume manifest JSON: {exc}")
            continue
        if not isinstance(volume_manifest, Mapping):
            validation.errors.append(f"{expected_id}: volume manifest must be an object")
            continue
        results[expected_id]["manifest"] = volume_manifest
        declared_manifest_hash = value.get("manifest_sha256")
        actual_manifest_hash = sha256_file(volume_manifest_path)
        if declared_manifest_hash != actual_manifest_hash:
            validation.errors.append(f"{expected_id}: volume manifest SHA-256 mismatch")
        if volume_manifest.get("archive_format") not in (None, ARCHIVE_FORMAT):
            validation.errors.append(f"{expected_id}: unexpected archive_format")
        if volume_manifest.get("scene_ids") != expected_scene_slice:
            validation.errors.append(f"{expected_id}: volume manifest scene IDs mismatch")
        volume_rows = _row_map(
            _normalise_manifest_rows(volume_manifest),
            validation,
            f"{expected_id}_manifest_files",
        )
        expected_rows = {
            path: staged_records[path]
            for path in staged_records
            if path.split("/", 1)[0] in set(expected_scene_slice)
        }
        results[expected_id]["rows"] = volume_rows
        if set(volume_rows) != set(expected_rows):
            validation.errors.append(f"{expected_id}: volume manifest file set mismatch")
        for path in sorted(set(volume_rows) & set(expected_rows)):
            if volume_rows[path].get("sha256") != expected_rows[path].get("sha256"):
                validation.errors.append(f"{expected_id}: volume manifest SHA-256 mismatch for {path}")

        zip_path = _safe_staging_file(staging, zip_name)
        if zip_path is None or not zip_path.is_file():
            validation.errors.append(f"{expected_id}: missing ZIP {zip_name!r}")
            continue
        actual_zip_hash = sha256_file(zip_path)
        if value.get("zip_sha256") != actual_zip_hash:
            validation.errors.append(f"{expected_id}: ZIP SHA-256 mismatch")
        if value.get("zip_bytes") != zip_path.stat().st_size:
            validation.errors.append(f"{expected_id}: ZIP byte count mismatch")
        if not isinstance(volume_manifest.get("zip"), Mapping):
            validation.errors.append(f"{expected_id}: volume manifest has no zip record")
        else:
            zip_record = volume_manifest["zip"]
            if zip_record.get("name") != zip_name or zip_record.get("sha256") != actual_zip_hash or zip_record.get("bytes") != zip_path.stat().st_size:
                validation.errors.append(f"{expected_id}: volume manifest zip record mismatch")
        try:
            with zipfile.ZipFile(zip_path, "r") as archive:
                names: list[str] = []
                duplicate: set[str] = set()
                for info in archive.infolist():
                    raw_name = info.filename
                    name = raw_name[:-1] if info.is_dir() and raw_name.endswith("/") else raw_name
                    if not is_safe_relative_path(name):
                        validation.errors.append(f"{expected_id}: unsafe ZIP path {raw_name!r}")
                        continue
                    if info.is_dir():
                        continue
                    if name in names:
                        duplicate.add(name)
                    names.append(name)
                    if name not in expected_rows:
                        validation.errors.append(f"{expected_id}: unexpected ZIP path {name}")
                    elif _record_bytes(expected_rows[name]) is not None and info.file_size != _record_bytes(expected_rows[name]):
                        validation.errors.append(f"{expected_id}: ZIP size mismatch {name}")
                if duplicate:
                    validation.errors.append(f"{expected_id}: duplicate ZIP paths {sorted(duplicate)[:6]}")
                if set(names) != set(expected_rows):
                    validation.errors.append(f"{expected_id}: ZIP file set mismatch")
                bad_member = archive.testzip()
                if bad_member is not None:
                    validation.errors.append(f"{expected_id}: ZIP CRC failure at {bad_member}")
        except (OSError, zipfile.BadZipFile) as exc:
            validation.errors.append(f"{expected_id}: cannot inspect ZIP: {exc}")
            continue
        _sample_extract(
            staging,
            {**dict(value), "volume_id": expected_id, "scene_ids": scene_ids, "zip_name": zip_name},
            expected_rows,
            sample_count,
            sample_seed,
            validation,
        )
    if seen_scenes != list(source_scene_ids):
        validation.errors.append("volume_scene_partition: volumes do not partition source scenes exactly once")
    else:
        validation.pass_check("volume_scene_partition", {"scene_count": len(seen_scenes)})
    return results


def _validate_checksums(
    staging: Path,
    manifest_path: Path | None,
    volumes: Mapping[str, Any],
    validation: Validation,
) -> None:
    checksum_path = None
    for name in CHECKSUM_NAMES:
        candidate = staging / name
        if candidate.is_file() and not candidate.is_symlink():
            checksum_path = candidate
            break
    if checksum_path is None:
        validation.fail_check("sha256sums", "SHA256SUMS.txt is missing")
        return
    rows: dict[str, str] = {}
    pattern = re.compile(r"^([0-9A-Fa-f]{64})\s+(?:\*?)(.+?)\s*$")
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        validation.fail_check("sha256sums", f"cannot read checksums: {exc}")
        return
    for line_number, line in enumerate(lines, start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = pattern.match(line)
        if not match:
            validation.errors.append(f"sha256sums: malformed line {line_number}")
            continue
        digest, name = match.groups()
        if not is_safe_relative_path(name) or "/" in name:
            validation.errors.append(f"sha256sums: unsafe file name {name!r}")
            continue
        if name in rows:
            validation.errors.append(f"sha256sums: duplicate file name {name}")
        rows[name] = digest.lower()
    expected_names: list[str] = []
    if manifest_path is not None:
        expected_names.append(manifest_path.name)
    for value in volumes.values():
        if isinstance(value, Mapping):
            for key in ("manifest_name", "zip_name"):
                if isinstance(value.get(key), str):
                    expected_names.append(value[key])
    expected_names = sorted(set(expected_names))
    for name in sorted(set(expected_names) - set(rows)):
        validation.errors.append(f"sha256sums: missing checksum for {name}")
    for name in sorted(set(rows) - set(expected_names)):
        validation.errors.append(f"sha256sums: unexpected checksum entry {name}")
    for name in sorted(set(expected_names) & set(rows)):
        path = staging / name
        if not path.is_file() or sha256_file(path) != rows[name]:
            validation.errors.append(f"sha256sums: digest mismatch for {name}")
    if not any(item.startswith("sha256sums:") for item in validation.errors):
        validation.pass_check("sha256sums", {"file_count": len(expected_names)})


def _report_path_safe(path: Path, root: Path, staging: Path) -> bool:
    path = path.expanduser().resolve()
    return not (
        path == root
        or path == staging
        or path.is_relative_to(root)
        or path.is_relative_to(staging)
    )


def validate(
    root: Path,
    staging: Path,
    *,
    expected_count: int = FORMAL_SCENE_COUNT,
    scenes_per_volume: int = SCENES_PER_VOLUME,
    sample_count: int = 1,
    sample_seed: int = 4071,
) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    staging = Path(staging).expanduser().resolve()
    validation = Validation()
    if expected_count < 1:
        validation.fail_check("arguments", "expected scene count must be positive")
        expected_count = 1
    if scenes_per_volume < 1:
        validation.fail_check("arguments", "scenes per volume must be positive")
        scenes_per_volume = 1
    if sample_count < 0:
        validation.fail_check("arguments", "sample count must be non-negative")
        sample_count = 0
    if not root.is_dir():
        validation.fail_check("flat_root", f"not a directory: {root}")
    else:
        validation.pass_check("flat_root")
    source: dict[str, Any] = {"scene_ids": [], "scenes": {}, "rows": [], "errors": []}
    if root.is_dir():
        try:
            source = collect_batch(root, expected_count, strict=False)
        except (OSError, LayoutError, ValueError) as exc:
            validation.fail_check("flat_batch", str(exc))
        for error in source.get("errors", []):
            validation.errors.append(f"flat_batch: {error}")
        if not source.get("errors"):
            validation.pass_check(
                "flat_batch",
                {
                    "scene_count": len(source.get("scene_ids", [])),
                    "reference_count": source.get("reference_count"),
                    "sample_count": source.get("sample_count"),
                },
            )

    source_scene_ids = source.get("scene_ids", []) if isinstance(source.get("scene_ids"), list) else []
    source_scene_ids = [item for item in source_scene_ids if isinstance(item, str)]
    _, staged_records = _validate_staging(root, staging, source, expected_count, validation)

    manifest_path = None
    manifest: Mapping[str, Any] | None = None
    if staging.is_dir():
        for name in MANIFEST_NAMES:
            candidate = staging / name
            if candidate.is_file() and not candidate.is_symlink():
                manifest_path = candidate
                break
    if manifest_path is None:
        validation.fail_check("archive_manifest", "archive_manifest_v3_batch.json is missing")
    else:
        try:
            loaded = _read_json(manifest_path)
            manifest = loaded if isinstance(loaded, Mapping) else None
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            validation.fail_check("archive_manifest", f"invalid JSON: {exc}")
        if manifest is None and not validation.checks.get("archive_manifest", {}).get("status") == "fail":
            validation.fail_check("archive_manifest", "top-level value must be an object")

    volume_results: dict[str, Any] = {}
    if manifest is not None:
        if manifest.get("archive_format") != ARCHIVE_FORMAT:
            validation.fail_check("archive_format", f"expected {ARCHIVE_FORMAT!r}, observed {manifest.get('archive_format')!r}")
        else:
            validation.pass_check("archive_format")
        if manifest.get("batch_schema") != BATCH_SCHEMA:
            validation.fail_check("batch_schema", f"expected {BATCH_SCHEMA!r}, observed {manifest.get('batch_schema')!r}")
        else:
            validation.pass_check("batch_schema")
        if manifest.get("scene_count") != expected_count:
            validation.fail_check("archive_scene_count", f"expected {expected_count}, observed {manifest.get('scene_count')!r}")
        else:
            validation.pass_check("archive_scene_count", {"count": expected_count})
        if manifest.get("scene_ids") != source_scene_ids:
            validation.fail_check("archive_scene_ids", "archive scene IDs differ from flat source")
        else:
            validation.pass_check("archive_scene_ids", {"count": len(source_scene_ids)})
        if manifest.get("state_order") != list(STATES):
            validation.fail_check("archive_state_order", f"expected {list(STATES)!r}, observed {manifest.get('state_order')!r}")
        else:
            validation.pass_check("archive_state_order")
        declared_rows_digest = manifest.get("source", {}).get("manifest_rows_sha256") if isinstance(manifest.get("source"), Mapping) else None
        source_rows = source.get("rows", []) if isinstance(source.get("rows"), list) else []
        computed_rows_digest = hashlib.sha256(canonical_json_bytes(source_rows)).hexdigest()
        if declared_rows_digest != computed_rows_digest:
            validation.fail_check("source_manifest_rows_sha256", f"declared {declared_rows_digest!r} != computed {computed_rows_digest}")
        else:
            validation.pass_check("source_manifest_rows_sha256", {"sha256": computed_rows_digest})
        _validate_archive_files(staging, manifest, staged_records, validation)
        volume_results = _validate_volumes(
            staging,
            manifest,
            staged_records,
            source_scene_ids,
            expected_count,
            scenes_per_volume,
            sample_count,
            sample_seed,
            validation,
        )
    _validate_checksums(staging, manifest_path, volume_results, validation)

    errors = sorted(set(validation.errors))
    report = {
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "archive_format": ARCHIVE_FORMAT,
        "batch_schema": BATCH_SCHEMA,
        "version_note": "v3 batch packaging only; no new rendering algorithm",
        "root": str(root),
        "staging": str(staging),
        "expected_scene_count": expected_count,
        "observed_flat_scene_count": len(source_scene_ids),
        "observed_staged_file_count": len(staged_records),
        "scenes_per_volume": scenes_per_volume,
        "expected_volume_count": (expected_count + scenes_per_volume - 1) // scenes_per_volume,
        "observed_volume_count": len(volume_results),
        "sample_count_per_volume": sample_count,
        "sample_seed": sample_seed,
        "checks": validation.checks,
        "samples": sorted(validation.samples, key=lambda item: (str(item.get("volume_id")), str(item.get("scene_id")))),
        "errors": errors,
        "warnings": sorted(set(validation.warnings)),
    }
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only validation of flat v3 batch bundles, assembled four-state "
            "scene archives, five ZIP volumes, SHA-256 and sampled extraction."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", required=True, type=Path, help="flat v3_500_scenes output root")
    parser.add_argument("--staging", required=True, type=Path, help="assembled archive staging directory")
    parser.add_argument("--report", type=str, help="optional report path outside root/staging; '-' prints complete JSON")
    parser.add_argument("--expected-scenes", type=int, default=FORMAL_SCENE_COUNT, help="required scene count; set fixture size for a small test")
    parser.add_argument("--scenes-per-volume", type=int, default=SCENES_PER_VOLUME, help="required scene count per volume")
    parser.add_argument("--sample-count", "--sample-scenes", dest="sample_count", type=int, default=1, help="deterministic scenes sampled per volume")
    parser.add_argument("--sample-seed", type=int, default=4071, help="stable sample-selection seed")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.expanduser().resolve()
    staging = args.staging.expanduser().resolve()
    report = validate(
        root,
        staging,
        expected_count=args.expected_scenes,
        scenes_per_volume=args.scenes_per_volume,
        sample_count=args.sample_count,
        sample_seed=args.sample_seed,
    )
    if args.report == "-":
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["status"] == "pass" else 1
    if args.report:
        report_path = Path(args.report).expanduser().resolve()
        if not _report_path_safe(report_path, root, staging):
            print("validate_archive_v3_batch: ERROR: --report must be outside --root and --staging", file=sys.stderr)
            return 2
        try:
            from build_archive_manifest_v3_batch import _write_json  # type: ignore

            _write_json(report_path, report)
        except OSError as exc:
            print(f"validate_archive_v3_batch: ERROR: cannot write report: {exc}", file=sys.stderr)
            return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "observed_flat_scene_count": report["observed_flat_scene_count"],
                "observed_staged_file_count": report["observed_staged_file_count"],
                "observed_volume_count": report["observed_volume_count"],
                "errors": len(report["errors"]),
                "error_examples": report["errors"][:3],
                "samples": len(report["samples"]),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
