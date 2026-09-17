import sys,unittest,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import bpy
sys.path.insert(0,str(ROOT/'seabed_dataset_v2/.deps/py313'))
suite=unittest.TestSuite()
selected=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else ['assets','labels','water','render_config']
for name in ['seabed_dataset_v2.tests.test_'+x for x in selected]:
    suite.addTests(unittest.defaultTestLoader.loadTestsFromName(name))
result=unittest.TextTestRunner(verbosity=2).run(suite)
report={'modules':selected,'tests':result.testsRun,'errors':[(str(t),e) for t,e in result.errors],'failures':[(str(t),e) for t,e in result.failures],'skipped':[(str(t),e) for t,e in result.skipped],'success':result.wasSuccessful(),'blender':bpy.app.version_string}
(ROOT/'seabed_dataset_v2/reports/blender_module_tests.json').write_text(json.dumps(report,indent=2))
if not result.wasSuccessful():raise SystemExit(1)
