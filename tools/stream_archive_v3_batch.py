#!/usr/bin/env python3
"""Build one v3 batch archive volume at a time without copying raw bundles."""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path

from build_archive_manifest_v3_batch import (
    ARCHIVE_FORMAT,
    BATCH_SCHEMA,
    DEGRADED_STATES,
    GROUND_TRUTH_FILES,
    LIGHTING_MODES,
    REFERENCE_FILES,
    SAMPLE_FILES,
    SCENES_PER_VOLUME,
    STATES,
    _copy_scene_to_staging,
    _record,
    _scene_manifest,
    _scene_metadata,
    _staged_scene_summary,
    _volume_base,
    _write_deterministic_zip,
    _write_json,
    canonical_json_bytes,
    collect_batch,
    sha256_bytes,
    sha256_file,
)


VERSION_NOTE = (
    "Additive v3 batch-scale packaging; reuses the completed v3 generator and "
    "does not introduce a new rendering algorithm."
)
WORK_STATE = "archive_work_state.json"


def _success_records(bundle: Path, required: tuple[str, ...]) -> tuple[dict, dict, dict[str, dict]]:
    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    success = json.loads((bundle / "success.json").read_text(encoding="utf-8"))
    declared = success.get("files") if isinstance(success, dict) else None
    if not isinstance(declared, dict):
        raise RuntimeError(f"success.json has no files map: {bundle}")
    records = {}
    for name in required:
        path = bundle / name
        item = declared.get(name)
        if not path.is_file():
            raise RuntimeError(f"validated bundle is incomplete: {bundle}/{name}")
        if name == "success.json":
            records[name] = {"path": name, "bytes": int(path.stat().st_size), "sha256": sha256_file(path)}
        elif not isinstance(item, dict):
            raise RuntimeError(f"validated bundle has no file seal: {bundle}/{name}")
        else:
            records[name] = {"path": name, "bytes": int(item["bytes"]), "sha256": str(item["sha256"])}
    return metadata, success, records


def _fast_source_from_validation(root: Path, expected_count: int, validation_report: Path) -> dict:
    report = json.loads(validation_report.read_text(encoding="utf-8"))
    if report.get("status") != "pass" or report.get("scenes") != expected_count or report.get("pairs") != expected_count * 3:
        raise RuntimeError("validated report is not a passing report for the requested scene count")
    rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_ids = [f"scene_{index:04d}" for index in range(1, expected_count + 1)]
    by_scene: dict[str, list[dict]] = {}
    for row in rows:
        by_scene.setdefault(str(row["scene_id"]), []).append(row)
    if sorted(by_scene) != expected_ids or any(len(by_scene[scene_id]) != 3 for scene_id in expected_ids):
        raise RuntimeError("validated flat manifest does not contain the expected scene rows")
    scenes = {}
    for scene_id in expected_ids:
        scene_rows = by_scene[scene_id]
        reference_paths = {str(row["reference_path"]) for row in scene_rows}
        reference_ids = {str(row["reference_id"]) for row in scene_rows}
        if len(reference_paths) != 1 or len(reference_ids) != 1:
            raise RuntimeError(f"{scene_id}: reference identity is not shared")
        reference_path = root / next(iter(reference_paths))
        ref_meta, ref_success, ref_records = _success_records(reference_path, REFERENCE_FILES)
        samples = {}
        for water_id in ("mild", "medium", "strong"):
            row = next(row for row in scene_rows if row["water_id"] == water_id)
            sample_path = root / str(row["sample_path"])
            sample_meta, sample_success, sample_records = _success_records(sample_path, SAMPLE_FILES)
            samples[water_id] = {
                "row": dict(row),
                "info": {"path": sample_path, "metadata": sample_meta, "success": sample_success, "files": sample_records},
            }
        checkpoint_path = root / "states" / scene_id / "accepted.json"
        scenes[scene_id] = {
            "scene_id": scene_id,
            "scene_number": int(scene_id.split("_")[1]),
            "reference_id": next(iter(reference_ids)),
            "reference_path": next(iter(reference_paths)),
            "reference": {"path": reference_path, "metadata": ref_meta, "success": ref_success, "files": ref_records},
            "lighting_mode": scene_rows[0]["lighting_mode"],
            "samples": samples,
            "checkpoint_path": f"states/{scene_id}/accepted.json",
            "checkpoint": json.loads(checkpoint_path.read_text(encoding="utf-8")),
        }
    source_manifest_records = {
        name: _record(root / name, root)
        for name in ("batch_manifest.json", "manifest.jsonl")
        if (root / name).is_file()
    }
    return {
        "scene_ids": expected_ids,
        "scenes": scenes,
        "rows": rows,
        "reference_count": expected_count,
        "sample_count": expected_count * 3,
        "source_manifest_records": source_manifest_records,
    }


def _fast_staged_scene_summary(scene_id: str, scene_root: Path, source_scene: dict) -> dict:
    records = []
    by_state = {state: [] for state in STATES}
    for name, item in source_scene["reference"]["files"].items():
        record = {"path": f"{scene_id}/clear/{name}", "bytes": item["bytes"], "sha256": item["sha256"]}
        records.append(record)
        by_state["clear"].append(f"clear/{name}")
    for state in DEGRADED_STATES:
        for name, item in source_scene["samples"][state]["info"]["files"].items():
            record = {"path": f"{scene_id}/{state}/{name}", "bytes": item["bytes"], "sha256": item["sha256"]}
            records.append(record)
            by_state[state].append(f"{state}/{name}")
    for name in ("metadata.json", "scene_manifest.json"):
        record = _record(scene_root / name, scene_root)
        record["path"] = f"{scene_id}/{name}"
        records.append(record)
    records.sort(key=lambda item: str(item["path"]))
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
        "root_files": sorted([f"{scene_id}/metadata.json", f"{scene_id}/scene_manifest.json"]),
        "shared_ground_truth": [f"clear/{name}" for name in GROUND_TRUTH_FILES],
        "file_count": len(records),
        "bytes": sum(int(record["bytes"]) for record in records),
        "files": records,
    }


def _load_or_prepare(root: Path, staging: Path, expected_count: int, scenes_per_volume: int, validation_report: Path | None) -> dict:
    state_path = staging / WORK_STATE
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("root") != str(root) or state.get("expected_count") != expected_count or state.get("scenes_per_volume") != scenes_per_volume:
            raise RuntimeError("existing archive work state targets a different input or scene count")
        return state

    if validation_report is not None:
        source = _fast_source_from_validation(root, expected_count, validation_report)
    else:
        source = collect_batch(root, expected_count, strict=True)
    staging.mkdir(parents=True, exist_ok=True)
    scene_root = staging / "scenes"
    for scene_id in source["scene_ids"]:
        _copy_scene_to_staging(source["scenes"][scene_id], staging, hardlink_inputs=True)
    if validation_report is not None:
        staged_scenes = [
            _fast_staged_scene_summary(scene_id, scene_root / scene_id, source["scenes"][scene_id])
            for scene_id in source["scene_ids"]
        ]
    else:
        staged_scenes = [
            _staged_scene_summary(scene_id, scene_root / scene_id, source["scenes"][scene_id])
            for scene_id in source["scene_ids"]
        ]
    source_manifest_records = source.get("source_manifest_records") or {
        name: _record(root / name, root)
        for name in ("batch_manifest.json", "manifest.jsonl")
        if (root / name).is_file()
    }
    state = {
        "schema_version": 1,
        "root": str(root),
        "staging": str(staging),
        "expected_count": expected_count,
        "scenes_per_volume": scenes_per_volume,
        "scene_ids": source["scene_ids"],
        "source_manifest_rows_sha256": sha256_bytes(canonical_json_bytes(source["rows"])),
        "source_manifest_records": source_manifest_records,
        "reference_count": source["reference_count"],
        "sample_count": source["sample_count"],
        "checkpoint_count": len(source["scene_ids"]),
        "validation_report": str(validation_report) if validation_report is not None else None,
        "staged_scenes": staged_scenes,
        "volumes": [],
    }
    _write_json(state_path, state)
    return state


def _volume_manifest(volume: dict, source_rows_digest: str, scenes_per_volume: int) -> dict:
    return {
        "schema_version": 1,
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


def _validate_zip(zip_path: Path, records: list[dict]) -> None:
    expected = {str(item["path"]): item for item in records}
    with zipfile.ZipFile(zip_path, "r") as archive:
        names = []
        for info in archive.infolist():
            if info.is_dir():
                continue
            names.append(info.filename)
            record = expected.get(info.filename)
            if record is None:
                raise RuntimeError(f"unexpected ZIP member: {info.filename}")
            if info.file_size != int(record["bytes"]):
                raise RuntimeError(f"ZIP member byte mismatch: {info.filename}")
        if sorted(names) != sorted(expected):
            raise RuntimeError("ZIP member inventory mismatch")
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP CRC failure: {bad_member}")
def build_volume(root: Path, staging: Path, expected_count: int, scenes_per_volume: int, volume_index: int, validation_report: Path | None) -> dict:
    state = _load_or_prepare(root, staging, expected_count, scenes_per_volume, validation_report)
    volumes = _volume_base(state["staged_scenes"], scenes_per_volume)
    if not 1 <= volume_index <= len(volumes):
        raise ValueError(f"volume index must be within 1..{len(volumes)}")
    volume = dict(volumes[volume_index - 1])
    start = int(volume["scene_ids"][0].split("_")[1])
    end = int(volume["scene_ids"][-1].split("_")[1])
    volume["zip_name"] = f"dataset_part{volume_index:02d}_scene_{start:04d}_{end:04d}.zip"
    zip_path = staging / volume["zip_name"]
    _write_deterministic_zip(zip_path, volume["files"], staging / "scenes")
    volume["zip_bytes"] = int(zip_path.stat().st_size)
    volume["zip_sha256"] = sha256_file(zip_path)
    _validate_zip(zip_path, volume["files"])
    volume["manifest_name"] = f"dataset_part{volume_index:02d}_scene_{start:04d}_{end:04d}_manifest.json"
    manifest = _volume_manifest(volume, state["source_manifest_rows_sha256"], scenes_per_volume)
    _write_json(staging / volume["manifest_name"], manifest)
    volume["manifest_sha256"] = sha256_file(staging / volume["manifest_name"])

    state["volumes"] = [item for item in state.get("volumes", []) if item.get("volume_index") != volume_index]
    state["volumes"].append(
        {
            key: volume[key]
            for key in (
                "volume_id", "volume_index", "scene_ids", "scene_count", "scene_start", "scene_end",
                "file_count", "uncompressed_bytes", "zip_name", "zip_bytes", "zip_sha256",
                "manifest_name", "manifest_sha256",
            )
        }
    )
    state["volumes"].sort(key=lambda item: int(item["volume_index"]))
    _write_json(staging / WORK_STATE, state)
    return {"status": "pass", "volume_index": volume_index, "scene_count": volume["scene_count"], "zip_name": volume["zip_name"], "zip_bytes": volume["zip_bytes"], "zip_sha256": volume["zip_sha256"], "manifest_name": volume["manifest_name"]}


def finalize(staging: Path) -> dict:
    state = json.loads((staging / WORK_STATE).read_text(encoding="utf-8"))
    expected_count = int(state["expected_count"])
    scenes_per_volume = int(state["scenes_per_volume"])
    volumes = sorted(state.get("volumes", []), key=lambda item: int(item["volume_index"]))
    expected_volumes = (expected_count + scenes_per_volume - 1) // scenes_per_volume
    if len(volumes) != expected_volumes or [int(item["volume_index"]) for item in volumes] != list(range(1, expected_volumes + 1)):
        raise RuntimeError("not all archive volumes have been built and locally validated")
    archive_manifest = {
        "schema_version": 1,
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
        "scene_count": expected_count,
        "scene_ids": state["scene_ids"],
        "scenes_per_volume": scenes_per_volume,
        "volume_count": len(volumes),
        "source": {
            "manifest_rows_sha256": state["source_manifest_rows_sha256"],
            "input_manifests": state["source_manifest_records"],
            "reference_count": state["reference_count"],
            "sample_count": state["sample_count"],
            "checkpoint_count": state["checkpoint_count"],
        },
        "files": [record for scene in state["staged_scenes"] for record in scene["files"]],
        "scenes": state["staged_scenes"],
        "volumes": volumes,
        "determinism": {
            "file_order": "POSIX lexical path",
            "zip_timestamp": "1980-01-01T00:00:00",
            "zip_compression": "deflate level 9",
            "zip_comments": False,
        },
    }
    archive_name = "archive_manifest_v3_400.json"
    _write_json(staging / archive_name, archive_manifest)
    checksum_names = [archive_name, *[item["manifest_name"] for item in volumes], *[item["zip_name"] for item in volumes]]
    checksum_lines = []
    for name in sorted(checksum_names):
        existing = staging / name
        if existing.is_file():
            digest = sha256_file(existing)
        else:
            digest = next(item["zip_sha256"] for item in volumes if item["zip_name"] == name)
        checksum_lines.append(f"{digest}  {name}")
    (staging / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    result = {
        "status": "pass",
        "archive_manifest": archive_name,
        "archive_manifest_sha256": sha256_file(staging / archive_name),
        "checksum_file": "SHA256SUMS.txt",
        "scene_count": expected_count,
        "volume_count": len(volumes),
        "volumes": volumes,
    }
    _write_json(staging / "archive_finalize_report.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--expected-scenes", type=int, default=400)
    parser.add_argument("--scenes-per-volume", type=int, default=SCENES_PER_VOLUME)
    parser.add_argument("--volume-index", type=int)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--validated-report", type=Path, help="passing full flat-batch validation report; skips a second source-file hash scan")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    staging = args.staging.expanduser().resolve()
    if args.finalize:
        result = finalize(staging)
    elif args.volume_index is not None:
        result = build_volume(root, staging, args.expected_scenes, args.scenes_per_volume, args.volume_index, args.validated_report.expanduser().resolve() if args.validated_report else None)
    else:
        raise SystemExit("specify --volume-index or --finalize")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
