#!/usr/bin/env python3
"""Create one compact, training-ready v3 400-scene ZIP volume.

The package keeps PNG RGB pairs and shared GT needed by a training loader,
plus a small JSONL pair index.  It intentionally omits the large raw EXR RGB
copies, verbose render metadata, and failed-attempt caches.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path


STATES = ("clear", "mild", "medium", "strong")
WATERS = ("mild", "medium", "strong")
GT_NAMES = ("depth_range_m.exr", "depth_camera_z_m.exr", "semantic_id.png", "instance_id.png", "valid_mask.png")
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
CHUNK_SIZE = 1024 * 1024


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (0o100644 & 0xFFFF) << 16
    return info


def add_file(archive: zipfile.ZipFile, source: Path, member: str) -> None:
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(f"missing source file: {source}")
    with archive.open(zip_info(member), "w") as destination, source.open("rb") as origin:
        shutil.copyfileobj(origin, destination, length=CHUNK_SIZE)


def add_json(archive: zipfile.ZipFile, value, member: str) -> None:
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with archive.open(zip_info(member), "w") as destination:
        destination.write(data)


def load_rows(root: Path, expected_count: int) -> dict[str, list[dict]]:
    rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_ids = [f"scene_{index:04d}" for index in range(1, expected_count + 1)]
    by_scene: dict[str, list[dict]] = {}
    for row in rows:
        by_scene.setdefault(str(row["scene_id"]), []).append(row)
    if sorted(by_scene) != expected_ids or any(len(by_scene[scene_id]) != 3 for scene_id in expected_ids):
        raise RuntimeError("manifest does not contain exactly three samples for every requested scene")
    return by_scene


def build_volume(root: Path, output_dir: Path, expected_count: int, scenes_per_volume: int, volume_index: int) -> dict:
    if expected_count != 400 or scenes_per_volume != 100:
        raise ValueError("this training package is fixed at 400 scenes and 100 scenes per volume")
    by_scene = load_rows(root, expected_count)
    start = (volume_index - 1) * scenes_per_volume + 1
    end = min(start + scenes_per_volume - 1, expected_count)
    if not 1 <= volume_index <= 4:
        raise ValueError("volume index must be 1..4")
    scene_ids = [f"scene_{index:04d}" for index in range(start, end + 1)]
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_name = f"v3_400_training_part{volume_index:02d}_scene_{start:04d}_{end:04d}.zip"
    zip_path = output_dir / zip_name
    temporary = None
    pair_rows = []
    member_names: set[str] = set()
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=output_dir, prefix=f".{zip_name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, allowZip64=True) as archive:
            readme = (
                "v3 400-scene training-ready volume\n"
                "Each scene contains one clear target, three degraded inputs, and shared GT.\n"
                "Lighting mode is fixed per scene in this preliminary single-light batch.\n"
            )
            with archive.open(zip_info("README.txt"), "w") as destination:
                destination.write(readme.encode("utf-8"))
            member_names.add("README.txt")
            for scene_id in scene_ids:
                scene_rows = by_scene[scene_id]
                first = scene_rows[0]
                scene_prefix = f"{scene_id}/"
                clear_member = f"{scene_prefix}{scene_id}_clear.png"
                reference = root / str(first["reference_path"])
                add_file(archive, reference / "clear_rgb.png", clear_member)
                member_names.add(clear_member)
                for row in sorted(scene_rows, key=lambda item: str(item["water_id"])):
                    water_id = str(row["water_id"])
                    sample_member = f"{scene_prefix}{scene_id}_{water_id}.png"
                    add_file(archive, root / str(row["sample_path"]) / "degraded_rgb.png", sample_member)
                    member_names.add(sample_member)
                    pair_rows.append(
                        {
                            "scene_id": scene_id,
                            "lighting_mode": row["lighting_mode"],
                            "water_id": water_id,
                            "input": sample_member,
                            "target": clear_member,
                            "gt": {name: f"{scene_prefix}{name}" for name in GT_NAMES},
                        }
                    )
                for name in GT_NAMES:
                    member = f"{scene_prefix}{name}"
                    add_file(archive, reference / name, member)
                    member_names.add(member)
                metadata = {
                    "scene_id": scene_id,
                    "scene_number": int(scene_id.split("_")[1]),
                    "lighting_mode": first["lighting_mode"],
                    "layout_seed": first["layout_seed"],
                    "layout_family": first["layout_family"],
                    "split": first["split"],
                    "state_order": list(STATES),
                    "clear": clear_member,
                    "degraded": {str(row["water_id"]): f"{scene_prefix}{scene_id}_{row['water_id']}.png" for row in scene_rows},
                    "shared_gt": {name: f"{scene_prefix}{name}" for name in GT_NAMES},
                }
                metadata_member = f"{scene_prefix}{scene_id}_metadata.json"
                add_json(archive, metadata, metadata_member)
                member_names.add(metadata_member)
            pairs_text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in pair_rows)
            with archive.open(zip_info("pairs.jsonl"), "w") as destination:
                destination.write(pairs_text.encode("utf-8"))
            member_names.add("pairs.jsonl")
            manifest = {
                "dataset": "seabed_dataset_v3_400_training_ready",
                "schema_version": 1,
                "scene_count": len(scene_ids),
                "scene_start": f"scene_{start:04d}",
                "scene_end": f"scene_{end:04d}",
                "pair_count": len(pair_rows),
                "states": list(STATES),
                "rgb_format": "16-bit PNG, RGB",
                "gt_policy": "one shared GT bundle per scene",
                "lighting_policy": "one lighting mode per scene; not a three-light same-scene comparison",
                "scene_ids": scene_ids,
            }
            add_json(archive, manifest, "dataset_manifest.json")
            member_names.add("dataset_manifest.json")
        os.replace(temporary, zip_path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    with zipfile.ZipFile(zip_path, "r") as archive:
        actual = {info.filename for info in archive.infolist() if not info.is_dir()}
        if actual != member_names or archive.testzip() is not None:
            raise RuntimeError("training ZIP inventory or CRC check failed")
    return {
        "status": "pass",
        "volume_index": volume_index,
        "scene_count": len(scene_ids),
        "pair_count": len(pair_rows),
        "zip_name": zip_name,
        "zip_bytes": zip_path.stat().st_size,
        "zip_path": str(zip_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--volume-index", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(build_volume(args.root.resolve(), args.output_dir.expanduser().resolve(), 400, 100, args.volume_index), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
