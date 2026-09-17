"""Seal a v3 batch calibration gate from independently checked smoke evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-root", required=True)
    parser.add_argument("--content-report", required=True)
    parser.add_argument("--archive-report", required=True)
    parser.add_argument("--output", default=str(ROOT / "reports" / "calibration_v3_500" / "gate.json"))
    args = parser.parse_args()
    smoke = Path(args.smoke_root).resolve()
    content = json.loads(Path(args.content_report).read_text())
    archive = json.loads(Path(args.archive_report).read_text())
    batch_run = json.loads((smoke / "batch_run.json").read_text())
    plan = json.loads((smoke / "plan.json").read_text())
    if content.get("status") != "pass" or content.get("scenes") != 3 or content.get("pairs") != 9:
        raise SystemExit("content smoke validation did not pass")
    if archive.get("status") != "pass" or archive.get("observed_flat_scene_count") != 3:
        raise SystemExit("archive smoke validation did not pass")
    expected_modes = {"natural": 167, "artificial": 167, "mixed": 166}
    if plan.get("lighting_mode_counts") != expected_modes:
        raise SystemExit("formal lighting mode schedule is not balanced")
    if batch_run.get("source_sha256") != plan.get("source_hash") or batch_run.get("config_sha256") != plan.get("config_hash"):
        raise SystemExit("smoke provenance does not match its plan")
    evidence_paths = [
        smoke / "plan.json",
        smoke / "batch_run.json",
        smoke / "batch_manifest.json",
        smoke / "manifest.jsonl",
        Path(args.content_report).resolve(),
        Path(args.archive_report).resolve(),
    ]
    evidence = []
    for path in evidence_paths:
        if not path.is_file() or not path.is_relative_to(ROOT):
            raise SystemExit(f"calibration evidence must be a project file: {path}")
        evidence.append({"path": str(path.relative_to(ROOT)), "sha256": file_hash(path), "bytes": path.stat().st_size})
    gate = {
        "schema_version": 1,
        "status": "pass",
        "scene_contract": "single_lighting_four_state",
        "scope": "v3 additive 500-scene batch calibration; no new algorithm version",
        "source_sha256": plan["source_hash"],
        "config_sha256": plan["config_hash"],
        "runtime_sha256": batch_run["runtime_sha256"],
        "lighting_modes": ["natural", "artificial", "mixed"],
        "lighting_mode_counts": expected_modes,
        "samples": plan["config"]["render"]["samples"],
        "resolution": plan["config"]["render"]["resolution"],
        "smoke_scene_count": 3,
        "smoke_pairs": 9,
        "smoke_archive_status": archive["status"],
        "evidence": evidence,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": gate["status"], "output": str(output), "sha256": file_hash(output), "source_sha256": gate["source_sha256"], "config_sha256": gate["config_sha256"], "runtime_sha256": gate["runtime_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
