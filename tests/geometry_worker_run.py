"""Run v2 geometry tests in one isolated Blender process.

The script is intentionally separate from the parent test runner so worker
evidence stays under ``reports/geometry_worker_*`` and does not overwrite the
main agent's aggregate reports.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

import bpy  # type: ignore


suite = unittest.defaultTestLoader.loadTestsFromName("seabed_dataset_v2.tests.test_geometry_v2")
result = unittest.TextTestRunner(verbosity=2).run(suite)
report = {
    "module": "seabed_dataset_v2.tests.test_geometry_v2",
    "tests": result.testsRun,
    "errors": [(str(test), traceback) for test, traceback in result.errors],
    "failures": [(str(test), traceback) for test, traceback in result.failures],
    "skipped": [(str(test), reason) for test, reason in result.skipped],
    "success": result.wasSuccessful(),
    "blender": bpy.app.version_string,
}
(ROOT / "reports" / "geometry_worker_runtime.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
if not result.wasSuccessful():
    raise SystemExit(1)
