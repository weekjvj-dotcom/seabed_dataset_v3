#!/usr/bin/env python3
"""Rename v3 batch Clear reference bundles to scene-based paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


def write_json(path: Path, value) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rename(root: Path, expected_scenes: int) -> dict:
    root = Path(root).expanduser().resolve()
    manifest_path = root / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_ids = [f"scene_{index:04d}" for index in range(1, expected_scenes + 1)]
    rows_by_scene: dict[str, list[dict]] = {}
    for row in rows:
        rows_by_scene.setdefault(str(row["scene_id"]), []).append(row)
    if sorted(rows_by_scene) != expected_ids:
        raise RuntimeError("manifest does not contain the expected contiguous scene range")
    if any(len(scene_rows) != 3 for scene_rows in rows_by_scene.values()):
        raise RuntimeError("every scene must have exactly three degraded rows")

    mapping: dict[str, tuple[str, str]] = {}
    for scene_id in expected_ids:
        scene_rows = rows_by_scene[scene_id]
        reference_paths = {str(row["reference_path"]) for row in scene_rows}
        reference_ids = {str(row["reference_id"]) for row in scene_rows}
        if len(reference_paths) != 1 or len(reference_ids) != 1:
            raise RuntimeError(f"{scene_id}: reference identity is not shared")
        old_relative = next(iter(reference_paths))
        reference_id = next(iter(reference_ids))
        old_path = root / old_relative
        new_relative = f"references/{scene_id}_clear"
        new_path = root / new_relative
        mapping[scene_id] = (old_relative, new_relative)
        if old_path != new_path and old_path.exists() and new_path.exists():
            raise RuntimeError(f"both old and new reference paths exist: {old_path} / {new_path}")
        if not old_path.exists() and not new_path.exists():
            raise RuntimeError(f"reference bundle is missing: {old_path}")
        if old_path.exists() and old_path.name not in {reference_id, f"{scene_id}_clear"}:
            raise RuntimeError(f"unexpected reference directory: {old_path}")

    # Rename only within the same references directory.  The operation is
    # idempotent so a rerun can finish metadata updates after interruption.
    for scene_id in expected_ids:
        old_relative, new_relative = mapping[scene_id]
        old_path = root / old_relative
        new_path = root / new_relative
        if old_path != new_path and old_path.exists():
            old_path.rename(new_path)

    updated_rows: list[dict] = []
    updated_samples = 0
    for row in rows:
        scene_id = str(row["scene_id"])
        _, new_relative = mapping[scene_id]
        row = dict(row)
        row["reference_path"] = new_relative
        updated_rows.append(row)
        sample_path = root / str(row["sample_path"])
        for filename in ("metadata.json", "success.json"):
            path = sample_path / filename
            value = json.loads(path.read_text(encoding="utf-8"))
            changed = False
            if filename == "metadata.json" and value.get("reference_path") != new_relative:
                value["reference_path"] = new_relative
                changed = True
            if filename == "success.json" and isinstance(value.get("metadata"), dict) and value["metadata"].get("reference_path") != new_relative:
                value["metadata"]["reference_path"] = new_relative
                changed = True
            if filename == "success.json" and isinstance(value.get("files"), dict):
                metadata_file = sample_path / "metadata.json"
                expected_seal = {"bytes": metadata_file.stat().st_size, "sha256": file_hash(metadata_file)}
                if value["files"].get("metadata.json") != expected_seal:
                    value["files"]["metadata.json"] = expected_seal
                    changed = True
            if changed:
                write_json(path, value)
                updated_samples += 1

    for scene_id in expected_ids:
        checkpoint_path = root / "states" / scene_id / "accepted.json"
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        checkpoint_rows = checkpoint.get("manifest_rows")
        if not isinstance(checkpoint_rows, list) or len(checkpoint_rows) != 3:
            raise RuntimeError(f"{scene_id}: invalid accepted checkpoint rows")
        for row in checkpoint_rows:
            row["reference_path"] = mapping[scene_id][1]
        write_json(checkpoint_path, checkpoint)

    updated_rows.sort(key=lambda row: str(row["sample_id"]))
    batch_path = root / "batch_manifest.json"
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    batch["rows"] = updated_rows
    batch["pairs"] = len(updated_rows)
    batch["references"] = len({row["reference_id"] for row in updated_rows})
    write_json(batch_path, batch)
    write_jsonl(manifest_path, updated_rows)
    report = {
        "status": "pass",
        "expected_scenes": expected_scenes,
        "renamed_references": expected_scenes,
        "updated_sample_metadata_files": updated_samples,
        "updated_checkpoint_files": expected_scenes,
        "reference_path_pattern": "references/scene_{number:04d}_clear",
        "reference_id_preserved": True,
    }
    write_json(root.parent.parent / "reports" / "v3_400_clear_rename.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--expected-scenes", required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(rename(args.root, args.expected_scenes), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
