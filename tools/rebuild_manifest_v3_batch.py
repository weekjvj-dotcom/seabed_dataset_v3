#!/usr/bin/env python3
"""Rebuild flat v3 batch manifests from accepted checkpoints only."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def _write_atomic(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def rebuild(root: Path, expected_scenes: int) -> dict:
    root = Path(root).expanduser().resolve()
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    expected_ids = [f"scene_{index:04d}" for index in range(1, expected_scenes + 1)]
    states = {str(item["scene_id"]): item for item in plan.get("geometry_states", [])}
    rows = []
    accepted = []
    for scene_id in expected_ids:
        checkpoint_path = root / "states" / scene_id / "accepted.json"
        if not checkpoint_path.is_file():
            raise RuntimeError(f"missing accepted checkpoint: {checkpoint_path}")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("status") != "accepted":
            raise RuntimeError(f"checkpoint is not accepted: {checkpoint_path}")
        checkpoint_rows = checkpoint.get("manifest_rows")
        if not isinstance(checkpoint_rows, list) or len(checkpoint_rows) != 3:
            raise RuntimeError(f"checkpoint does not contain exactly three rows: {checkpoint_path}")
        if scene_id not in states:
            raise RuntimeError(f"scene is absent from formal plan: {scene_id}")
        rows.extend(item for item in checkpoint_rows if isinstance(item, dict))
        accepted.append(scene_id)
    rows.sort(key=lambda item: str(item["sample_id"]))
    if len(rows) != expected_scenes * 3 or len({item["sample_id"] for item in rows}) != len(rows):
        raise RuntimeError("rebuilt manifest does not have unique expected sample rows")
    batch = {
        "schema_version": 1,
        "rows": rows,
        "pairs": len(rows),
        "references": len({item["reference_id"] for item in rows}),
    }
    _write_atomic(root / "batch_manifest.json", json.dumps(batch, ensure_ascii=False, indent=2) + "\n")
    _write_atomic(
        root / "manifest.jsonl",
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in rows),
    )
    return {"status": "pass", "scenes": len(accepted), "pairs": len(rows), "references": batch["references"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--expected-scenes", required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(rebuild(args.root, args.expected_scenes), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
