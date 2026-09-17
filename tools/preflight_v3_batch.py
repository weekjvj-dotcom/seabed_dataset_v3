"""Geometry-only acceptance preflight for the v3 500-scene batch."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import bpy  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
from run_500_scenes import build_geometry_in_place, cleanup_cached_handle, configure_template_asset_cache, batch_plan  # noqa: E402
from src.contracts import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--report", default=str(ROOT / "reports" / "v3_batch_preflight.json"))
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    cfg = load_config(ROOT / "configs" / "v3_500_scenes.json")
    configure_template_asset_cache()
    rows = []
    started = time.time()
    for state in batch_plan(cfg)["geometry_states"][: args.limit]:
        accepted = None
        failures = []
        for attempt in range(int(cfg.get("batch_max_attempts", cfg["max_attempts"]))):
            handle = None
            try:
                handle = build_geometry_in_place(cfg, {"seed": state["layout_seed"], "split": state["split"]}, 0, attempt)
                accepted = attempt
                cleanup_cached_handle(handle)
                break
            except BaseException as exc:
                failures.append({"attempt": attempt, "error": repr(exc)})
                cleanup_cached_handle(handle)
        rows.append({"scene_id": state["scene_id"], "layout_seed": state["layout_seed"], "layout_family": state["layout_family"], "accepted_attempt": accepted, "failures": failures})
        print("PREFLIGHT " + state["scene_id"] + " " + str(accepted), flush=True)
    result = {"status": "pass" if all(row["accepted_attempt"] is not None for row in rows) else "fail", "scenes_checked": len(rows), "failed_scenes": [row["scene_id"] for row in rows if row["accepted_attempt"] is None], "attempt_histogram": {str(i): sum(row["accepted_attempt"] == i for row in rows) for i in range(int(cfg.get("batch_max_attempts", cfg["max_attempts"])))}, "seconds": time.time() - started, "rows": rows}
    Path(args.report).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "scenes_checked", "failed_scenes", "attempt_histogram", "seconds")}, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
