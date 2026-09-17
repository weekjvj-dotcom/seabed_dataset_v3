"""Schedule the v3 batch runner inside the already-open Blender GUI."""

from __future__ import annotations

import runpy
import sys
import traceback
from pathlib import Path

import bpy  # type: ignore


ROOT = Path(__file__).resolve().parents[1]


def _run_batch():
    try:
        runpy.run_path(str(ROOT / "tools" / "run_500_scenes.py"), run_name="__main__")
    except BaseException:
        (ROOT / "reports" / "last_batch_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    return None


def schedule(start: int, end: int, preview: bool = True, save_native: bool = True) -> None:
    sys.argv = ["blender", "--disable-autoexec", "--", "--config", str(ROOT / "configs" / "v3_500_scenes.json"), "--start", str(start), "--end", str(end)]
    if preview:
        sys.argv.append("--preview")
    if save_native:
        sys.argv.append("--save-native")
    for key in list(sys.modules):
        if key == "src" or key.startswith("src."):
            sys.modules.pop(key, None)
    bpy.app.timers.register(_run_batch, first_interval=0.1)


if __name__ == "__main__":
    schedule(1, 3, preview=True, save_native=True)
