"""Independent validation of additive v3 batch flat bundles.

The checker deliberately does not import bpy.  It reads label EXR/PNG values,
checks RGB EXR headers, success-record hashes, metadata identity, shared GT,
single-light pairing, and accepted checkpoints.  RGB EXR pixel decoding is
left to Blender/OpenEXR because the v3 renderer uses ZIP-compressed RGB EXR.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.contracts import REFERENCE_FILES, SAMPLE_FILES, canonical_hash, complete_record, file_hash  # noqa: E402
from src.storage import rgb_exr_header_valid  # noqa: E402
from src.labels.formats import read_exr, read_png  # noqa: E402


def png(path: Path, bits: int, colour_type: int) -> np.ndarray:
    raw = path.read_bytes()
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError(f"invalid PNG signature: {path}")
    width, height, bit_depth, actual_type = struct.unpack(">IIBB", raw[16:26])
    if bit_depth != bits or actual_type != colour_type:
        raise AssertionError(f"PNG header mismatch: {path} {bit_depth}/{actual_type}")
    with Image.open(path) as image:
        image.load()
        array = np.asarray(image)
    if array.shape[:2] != (height, width):
        raise AssertionError(f"PNG dimensions mismatch: {path}")
    return array


def label_exr(path: Path, width: int, height: int) -> np.ndarray:
    decoded = read_exr(path)
    if decoded["channels"] != ["Y"] or decoded["width"] != width or decoded["height"] != height:
        raise AssertionError(f"label EXR header mismatch: {path}")
    array = np.asarray(decoded["pixels"], dtype=np.float32)
    if not np.isfinite(array).all():
        raise AssertionError(f"non-finite label EXR: {path}")
    return array


def check_success(path: Path, required: tuple[str, ...]) -> None:
    if not complete_record(path, required):
        raise AssertionError(f"invalid success record: {path}")


def check_rgb_exr(path: Path, width: int, height: int) -> None:
    if not rgb_exr_header_valid(path, (width, height)):
        raise AssertionError(f"invalid RGB EXR header: {path}")


def validate(root: Path, expected_scenes: int) -> dict:
    root = Path(root).resolve()
    plan = json.loads((root / "plan.json").read_text())
    manifest_rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines() if line.strip()]
    all_states = plan["geometry_states"]
    if len(all_states) != 500:
        raise AssertionError(f"formal plan scene count {len(all_states)} != 500")
    if len(manifest_rows) != expected_scenes * 3:
        raise AssertionError(f"manifest row count {len(manifest_rows)} != {expected_scenes * 3}")
    if len({row["sample_id"] for row in manifest_rows}) != len(manifest_rows):
        raise AssertionError("duplicate sample IDs")
    width, height = plan["config"]["render"]["resolution"]
    rows_by_scene: dict[str, list[dict]] = {}
    for row in manifest_rows:
        rows_by_scene.setdefault(row["scene_id"], []).append(row)
    states = [state for state in all_states if state["scene_id"] in rows_by_scene]
    if len(states) != expected_scenes or set(rows_by_scene) != {state["scene_id"] for state in states}:
        raise AssertionError("manifest scene IDs differ from plan")

    reference_hashes: dict[str, dict[str, str]] = {}
    mode_counts = Counter()
    accepted_attempts = Counter()
    for state in states:
        scene_id = state["scene_id"]
        rows = rows_by_scene[scene_id]
        if len(rows) != 3:
            raise AssertionError(f"{scene_id}: expected three rows")
        modes = {row["lighting_mode"] for row in rows}
        if modes != {state["lighting_mode"]}:
            raise AssertionError(f"{scene_id}: more than one lighting mode")
        mode_counts.update(modes)
        water_ids = {row["water_id"] for row in rows}
        if water_ids != {"mild", "medium", "strong"}:
            raise AssertionError(f"{scene_id}: incomplete water states")
        references = {row["reference_id"] for row in rows}
        if len(references) != 1:
            raise AssertionError(f"{scene_id}: reference is not shared")
        accepted_attempts.update({rows[0]["accepted_attempt"]})
        reference_id = next(iter(references))
        reference_path = root / rows[0]["reference_path"]
        check_success(reference_path, REFERENCE_FILES)
        ref_meta = json.loads((reference_path / "metadata.json").read_text())
        for key, value in {"scene_id": scene_id, "state_key": scene_id, "reference_id": reference_id, "lighting_mode": state["lighting_mode"]}.items():
            if ref_meta.get(key) != value:
                raise AssertionError(f"{scene_id}: reference metadata identity mismatch {key}")
        if ref_meta["water_objects_in_scene"] or ref_meta["clear_definition"] != "No seawater volume or air-water surface":
            raise AssertionError(f"{scene_id}: Clear contains water")
        if not ref_meta["geometry_quality"]["passed"] or not ref_meta["rgb_quality"]["passed"]:
            raise AssertionError(f"{scene_id}: Clear quality failed")
        clear_png = png(reference_path / "clear_rgb.png", 16, 2)
        if clear_png.shape[:2] != (height, width):
            raise AssertionError(f"{scene_id}: Clear RGB dimensions mismatch")
        check_rgb_exr(reference_path / "clear_rgb.exr", width, height)
        depth = label_exr(reference_path / "depth_range_m.exr", width, height)
        camera_z = label_exr(reference_path / "depth_camera_z_m.exr", width, height)
        semantic = png(reference_path / "semantic_id.png", 16, 0)
        instance = png(reference_path / "instance_id.png", 16, 0)
        valid_mask = png(reference_path / "valid_mask.png", 8, 0)
        valid = valid_mask == 255
        if not np.array_equal(valid, semantic > 0) or not np.array_equal(valid, instance > 0):
            raise AssertionError(f"{scene_id}: label/mask mismatch")
        if not set(np.unique(semantic)).issubset(set(range(6))) or not set(np.unique(valid_mask)).issubset({0, 255}):
            raise AssertionError(f"{scene_id}: invalid label IDs")
        if np.any(depth[valid] <= 0) or np.any(camera_z[valid] <= 0) or np.any(depth[~valid] != 0) or np.any(camera_z[~valid] != 0):
            raise AssertionError(f"{scene_id}: invalid depth/miss values")
        for row in rows:
            sample_path = root / row["sample_path"]
            check_success(sample_path, SAMPLE_FILES)
            sample_meta = json.loads((sample_path / "metadata.json").read_text())
            for key, value in {"scene_id": scene_id, "state_key": scene_id, "reference_id": reference_id, "lighting_mode": state["lighting_mode"], "water_id": row["water_id"], "sample_id": row["sample_id"]}.items():
                if sample_meta.get(key) != value:
                    raise AssertionError(f"{row['sample_id']}: metadata identity mismatch {key}")
            if sample_meta["state_sha256"] != sample_meta["state_after_sha256"] or not sample_meta["rgb_quality"]["passed"]:
                raise AssertionError(f"{row['sample_id']}: sample state/quality failed")
            png(sample_path / "degraded_rgb.png", 16, 2)
            check_rgb_exr(sample_path / "degraded_rgb.exr", width, height)
            if sample_meta["state_sha256"] != ref_meta["state_sha256"]:
                raise AssertionError(f"{row['sample_id']}: geometry/camera/light state is not paired")
            if sample_meta["water"]["preset"]["id"] != row["water_id"]:
                raise AssertionError(f"{row['sample_id']}: water preset mismatch")
        checkpoint = json.loads((root / "states" / scene_id / "accepted.json").read_text())
        if checkpoint.get("status") != "accepted" or len(checkpoint.get("manifest_rows", [])) != 3:
            raise AssertionError(f"{scene_id}: invalid accepted checkpoint")
        if checkpoint.get("lighting_mode") != state["lighting_mode"] or len(checkpoint.get("reference_ids", [])) != 1:
            raise AssertionError(f"{scene_id}: checkpoint single-light contract failed")
        reference_hashes[scene_id] = {name: file_hash(reference_path / name) for name in ("depth_range_m.exr", "depth_camera_z_m.exr", "semantic_id.png", "instance_id.png", "valid_mask.png")}

    expected_modes = {"natural": (expected_scenes + 2) // 3, "artificial": (expected_scenes + 1) // 3, "mixed": expected_scenes // 3}
    if dict(mode_counts) != expected_modes:
        raise AssertionError(f"lighting mode counts {dict(mode_counts)} != {expected_modes}")
    result = {"status": "pass", "scenes": expected_scenes, "references": len(reference_hashes), "pairs": len(manifest_rows), "lighting_mode_counts": dict(mode_counts), "accepted_attempt_counts": dict(accepted_attempts), "shared_gt_checked": True, "rgb_exr_headers_checked": True}
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--expected-scenes", type=int, default=500)
    parser.add_argument("--report")
    args = parser.parse_args()
    result = validate(Path(args.root), args.expected_scenes)
    if args.report:
        Path(args.report).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
