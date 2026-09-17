#!/usr/bin/env python3
"""Seal a passing v3 calibration without inheriting v2 historical evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.contracts import atomic_json, canonical_hash, file_hash, load_config, source_hash  # noqa: E402


def main() -> None:
    config = load_config(ROOT / "configs/v3_calibration.json")
    calibration_dir = ROOT / "reports/calibration_v3"
    result_path = calibration_dir / "result.json"
    result = json.loads(result_path.read_text())
    modes = {"natural", "artificial", "mixed"}
    expected = {(mode, condition) for mode in modes for condition in ("clear", "mild", "medium", "strong")}
    if result.get("status") != "pass":
        raise SystemExit("Calibration result is not pass")
    if result.get("source_sha256") != source_hash():
        raise SystemExit("Calibration source signature differs from current v3 source")
    if canonical_hash(result.get("config")) != canonical_hash(config):
        raise SystemExit("Calibration config differs from v3_calibration.json")
    if not result.get("geometry_quality", {}).get("passed"):
        raise SystemExit("Calibration geometry quality did not pass")
    renders = result.get("renders", [])
    if {(row.get("mode"), row.get("condition")) for row in renders} != expected:
        raise SystemExit("Calibration render grid is incomplete")
    if not all(row.get("quality", {}).get("passed") for row in renders):
        raise SystemExit("Calibration has a failed RGB quality row")
    if set(result.get("noise", {})) != modes or not all(row.get("passed") for row in result["noise"].values()):
        raise SystemExit("Calibration noise gate did not pass all lighting modes")
    if set(result.get("noise_convergence", {})) != modes or not all(row.get("passed") for row in result["noise_convergence"].values()):
        raise SystemExit("Calibration convergence gate did not pass all lighting modes")
    blend = calibration_dir / "calibration.blend"
    if not blend.is_file() or blend.stat().st_size < 1024:
        raise SystemExit("Calibration native blend is missing or implausibly small")
    evidence = [result_path, blend, *sorted(calibration_dir.rglob("*.exr"))]
    gate = {
        "status": "pass",
        "scope": "v3 Stage A: CC0 PBR/scan assets, one state, three lighting modes, 512 samples",
        "source_sha256": source_hash(),
        "config_sha256": canonical_hash(config),
        "runtime_sha256": canonical_hash(result["runtime"]),
        "lighting_modes": sorted(modes),
        "samples": int(config["render"]["samples"]),
        "asset_policy": config["v3_asset_policy"],
        "geometry_quality": result["geometry_quality"],
        "noise": result["noise"],
        "noise_convergence": result["noise_convergence"],
        "evidence": [
            {"path": str(path.relative_to(ROOT)), "sha256": file_hash(path)}
            for path in evidence
        ],
        "limits": [
            "Only the listed CC0 asset sources are approved for this calibration.",
            "PBR height maps are Bump-only; they do not alone change depth GT.",
            "Imported Smithsonian scans replace three stable instance IDs and do change geometry/GT.",
            "This gate does not certify a full 27-pair formal dataset or real-ocean optical calibration.",
        ],
    }
    target = ROOT / config["calibration_gate"]
    atomic_json(target, gate)
    print(json.dumps({"status": "pass", "gate": str(target.relative_to(ROOT)), "evidence_count": len(evidence)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
