import sys,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'seabed_dataset/.deps/py313'))
modules=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else ['lighting_v2','quality_v2','geometry_v2']
suite=unittest.TestSuite()
for name in modules:suite.addTests(unittest.defaultTestLoader.loadTestsFromName('seabed_dataset_v2.tests.test_'+name))
r=unittest.TextTestRunner(verbosity=2).run(suite)
report={'tests':r.testsRun,'success':r.wasSuccessful(),'failures':[(str(t),e) for t,e in r.failures],'errors':[(str(t),e) for t,e in r.errors],'skipped':[(str(t),e) for t,e in r.skipped],'modules':modules}
(ROOT/'seabed_dataset_v2/reports/module_tests.json').write_text(json.dumps(report,indent=2))
if not r.wasSuccessful():raise SystemExit(1)
