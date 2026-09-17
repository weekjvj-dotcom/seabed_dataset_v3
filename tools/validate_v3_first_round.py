#!/usr/bin/env python3
"""Read-only integrity validation for the completed v3 first-round output."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.contracts import (  # noqa: E402
    REFERENCE_FILES,
    SAMPLE_FILES,
    canonical_hash,
    complete_record,
    file_hash,
    load_config,
    make_plan,
    source_hash,
    validate_checkpoint,
)
from src.labels.formats import read_exr, read_png  # noqa: E402
from src.storage import rgb_exr_header_valid  # noqa: E402


def _all_rows(rows):
    for row in rows:
        for value in row:
            yield value


def _validate_reference(folder: Path, expected: dict) -> dict:
    if not complete_record(folder, REFERENCE_FILES):
        raise ValueError(f"Incomplete reference seal: {folder}")
    metadata = json.loads((folder / "metadata.json").read_text())
    for key, value in expected.items():
        if canonical_hash(metadata.get(key)) != canonical_hash(value):
            raise ValueError(f"Reference metadata mismatch for {key}: {folder}")
    if metadata.get("clear_definition") != "No seawater volume or air-water surface" or metadata.get("water_objects_in_scene"):
        raise ValueError(f"Reference is not clear water-free state: {folder}")
    if not metadata.get("rgb_quality", {}).get("passed") or not metadata.get("geometry_quality", {}).get("passed"):
        raise ValueError(f"Reference quality failed: {folder}")
    width, height = metadata["render"]["resolution"]
    if not rgb_exr_header_valid(folder / "clear_rgb.exr", (width, height)):
        raise ValueError(f"Invalid RGB EXR header: {folder}")
    semantic = read_png(folder / "semantic_id.png")
    instance = read_png(folder / "instance_id.png")
    valid = read_png(folder / "valid_mask.png")
    range_depth = read_exr(folder / "depth_range_m.exr")
    z_depth = read_exr(folder / "depth_camera_z_m.exr")
    if [semantic["width"], semantic["height"]] != [width, height] or semantic["bit_depth"] != 16:
        raise ValueError(f"Semantic dimensions/bit depth invalid: {folder}")
    if [instance["width"], instance["height"]] != [width, height] or instance["bit_depth"] != 16:
        raise ValueError(f"Instance dimensions/bit depth invalid: {folder}")
    if [valid["width"], valid["height"]] != [width, height] or valid["bit_depth"] != 8:
        raise ValueError(f"Valid-mask dimensions/bit depth invalid: {folder}")
    if range_depth["channels"] != ["Y"] or z_depth["channels"] != ["Y"]:
        raise ValueError(f"Depth channel layout invalid: {folder}")
    registry = metadata["full_instance_registry"]
    visible_instances: set[int] = set()
    valid_pixels = 0
    for semantic_row, instance_row, valid_row, range_row, z_row in zip(
        semantic["pixels"], instance["pixels"], valid["pixels"], range_depth["pixels"], z_depth["pixels"]
    ):
        for sem, inst, mask, depth_range, depth_z in zip(semantic_row, instance_row, valid_row, range_row, z_row):
            if mask not in (0, 255) or sem not in range(6):
                raise ValueError(f"Illegal label value: {folder}")
            is_valid = mask == 255
            if is_valid != (sem > 0) or is_valid != (inst > 0):
                raise ValueError(f"Mask/semantic/instance disagreement: {folder}")
            if is_valid:
                valid_pixels += 1
                visible_instances.add(int(inst))
                if depth_range <= 0 or depth_z <= 0:
                    raise ValueError(f"Nonpositive valid depth: {folder}")
                record = registry.get(str(inst))
                if record is None or int(record["semantic_id"]) != sem:
                    raise ValueError(f"Instance registry disagreement: {folder}")
            elif depth_range != 0 or depth_z != 0:
                raise ValueError(f"Invalid-pixel depth is nonzero: {folder}")
    external = metadata.get("asset_summary", {}).get("external_assets", [])
    if len(external) != 3 or metadata.get("asset_summary", {}).get("triangle_count", 0) > 1_500_000:
        raise ValueError(f"External asset or triangle-budget summary invalid: {folder}")
    return {
        "reference_id": metadata["reference_id"],
        "state_key": metadata["state_key"],
        "lighting_mode": metadata["lighting_mode"],
        "valid_pixels": valid_pixels,
        "visible_instances": sorted(visible_instances),
        "triangle_count": metadata["asset_summary"]["triangle_count"],
        "external_asset_uids": [row["asset_uid"] for row in external],
    }


def _validate_sample(folder: Path, expected: dict) -> dict:
    if not complete_record(folder, SAMPLE_FILES):
        raise ValueError(f"Incomplete sample seal: {folder}")
    metadata = json.loads((folder / "metadata.json").read_text())
    for key, value in expected.items():
        if canonical_hash(metadata.get(key)) != canonical_hash(value):
            raise ValueError(f"Sample metadata mismatch for {key}: {folder}")
    if metadata.get("state_after_sha256") != metadata.get("state_sha256"):
        raise ValueError(f"Sample state changed after render: {folder}")
    if not metadata.get("rgb_quality", {}).get("passed") or not metadata.get("geometry_quality", {}).get("passed"):
        raise ValueError(f"Sample quality failed: {folder}")
    width, height = metadata["render"]["resolution"]
    if not rgb_exr_header_valid(folder / "degraded_rgb.exr", (width, height)):
        raise ValueError(f"Invalid degraded RGB EXR header: {folder}")
    return {"sample_id": metadata["sample_id"], "reference_id": metadata["reference_id"], "water_id": metadata["water"]["preset"]["id"]}


def main() -> None:
    config = load_config(ROOT / "configs/v3_first_round.json")
    output = ROOT / "outputs/v3_first_round"
    plan = json.loads((output / "plan.json").read_text())
    if canonical_hash(plan["config"]) != canonical_hash(config) or plan["source_hash"] != source_hash():
        raise ValueError("Formal plan no longer matches current v3 config/source")
    rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines() if line]
    expected_plan = make_plan(config)
    if len(rows) != len(expected_plan["pairs"]) or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Manifest row count or uniqueness failed")
    by_state = defaultdict(list)
    for row in rows:
        by_state[row["state_key"]].append(row)
    reference_reports = []
    sample_reports = []
    for state in expected_plan["geometry_states"]:
        checkpoint_path = output / "states" / state["state_key"] / "accepted.json"
        checkpoint = validate_checkpoint(json.loads(checkpoint_path.read_text()), state, plan)
        if not (output / checkpoint["blend"]).is_file() or (output / checkpoint["blend"]).stat().st_size != checkpoint["blend_bytes"]:
            raise ValueError(f"Native blend record invalid for {state['state_key']}")
        for row in checkpoint["manifest_rows"]:
            reference_folder = output / row["reference_path"]
            common = {
                "state_key": row["state_key"],
                "state_index": row["state_index"],
                "geometry_id": row["geometry_id"],
                "reference_id": row["reference_id"],
                "lighting_mode": row["lighting_mode"],
            }
            if not any(report["reference_id"] == row["reference_id"] for report in reference_reports):
                reference_reports.append(_validate_reference(reference_folder, common))
            sample_reports.append(
                _validate_sample(
                    output / row["sample_path"],
                    {
                        **common,
                        "sample_id": row["sample_id"],
                        "seabed_depth_m": row["seabed_depth_m"],
                        "reference_path": row["reference_path"],
                    },
                )
            )
    gate = json.loads((ROOT / config["calibration_gate"]).read_text())
    report = {
        "schema_version": 1,
        "status": "pass",
        "formal_root": str(output.relative_to(ROOT)),
        "source_sha256": source_hash(),
        "config_sha256": canonical_hash(config),
        "calibration_gate_sha256": file_hash(ROOT / config["calibration_gate"]),
        "manifest_rows": len(rows),
        "reference_count": len(reference_reports),
        "state_count": len(by_state),
        "accepted_attempts": {
            state["state_key"]: json.loads((output / "states" / state["state_key"] / "accepted.json").read_text())["accepted_attempt"]
            for state in expected_plan["geometry_states"]
        },
        "references": reference_reports,
        "samples": sample_reports,
        "gate_status": gate["status"],
    }
    target = ROOT / "reports/v3_first_round_validation.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "manifest_rows", "reference_count", "state_count", "accepted_attempts")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
