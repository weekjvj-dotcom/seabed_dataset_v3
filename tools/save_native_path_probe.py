#!/usr/bin/env python3
"""Save a no-RGB v3 native blend at its final path for relocation testing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from src.contracts import atomic_json, canonical_hash, load_config  # noqa: E402
from src.pipeline import water_plan  # noqa: E402
from src.scene import build_geometry, build_lighting_groups, save_native  # noqa: E402


def main() -> None:
    config = load_config(ROOT / "configs/v3_asset_preview.json")
    state = {"layout_seed": config["layouts"][0]["seed"], "state_index": 0}
    handle = build_geometry(config, config["layouts"][0], state_index=0, attempt_index=0)
    water = water_plan(config, state, 0)
    build_lighting_groups(handle, config, "v3_path_probe", water["effective_presets"])
    target = ROOT / "reports/v3_native_path_probe.blend"
    save_native(handle, target, config, {"purpose": "external-image-path probe"})
    report = {
        "schema_version": 1,
        "blend": str(target.relative_to(ROOT)),
        "config_sha256": canonical_hash(config),
        "asset_path_policy": "relative_to_final_native_blend",
    }
    atomic_json(ROOT / "reports/v3_native_path_probe.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
