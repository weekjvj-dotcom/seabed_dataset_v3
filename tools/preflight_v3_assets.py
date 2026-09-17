#!/usr/bin/env python3
"""Build one v3 state without RGB rendering and audit its labels/material state."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from src.contracts import atomic_json, canonical_hash, load_config  # noqa: E402
from src.labels import export_labels  # noqa: E402
from src.scene import build_geometry  # noqa: E402
from src.state import geometry_digest, state_digest  # noqa: E402


def main() -> None:
    config = load_config(ROOT / "configs/v3_asset_preview.json")
    handle = build_geometry(config, config["layouts"][0], state_index=0, attempt_index=0)
    state_sha, _ = state_digest(handle["base"], handle["objects"])
    geometry_sha, _ = geometry_digest(handle["base"], handle["objects"])
    label_root = ROOT / "reports/preflight_v3_assets_labels"
    labels = export_labels(handle["base"], label_root, handle["active_objects"])
    external = handle["ecology"]["summary"].get("external_assets", [])
    report = {
        "schema_version": 1,
        "purpose": "v3 external asset preflight without RGB render",
        "config_sha256": canonical_hash(config),
        "state_sha256": state_sha,
        "geometry_sha256": geometry_sha,
        "triangle_count": int(handle["ecology"]["summary"]["triangle_count"]),
        "triangle_budget": int(handle["ecology"]["summary"]["mesh_budget"]),
        "active_object_count": int(len(handle["active_objects"])),
        "labels": labels,
        "external_assets": external,
        "passed": bool(
            handle["ecology"]["summary"]["triangle_count"] <= handle["ecology"]["summary"]["mesh_budget"]
            and len(external) == 3
        ),
    }
    atomic_json(ROOT / "reports/preflight_v3_assets.json", report)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "triangle_count": report["triangle_count"],
                "triangle_budget": report["triangle_budget"],
                "active_object_count": report["active_object_count"],
                "external_assets": [row["target_name"] for row in external],
                "state_sha256": report["state_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
