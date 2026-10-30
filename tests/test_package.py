import importlib.util
import ast
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('reproduce', ROOT / 'reproduce.py')
reproduce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproduce)


class PlannerTests(unittest.TestCase):
    def test_jacobian_rejects_conditioning(self):
        # Exercise the argument guard without importing GPU dependencies.
        tree = ast.parse((ROOT / 'eval/local_jacobian_power.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'require_unconditional')
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<guard>', 'exec'), namespace)
        guard = namespace['require_unconditional']
        guard()
        for kwargs in ({'class_label': 0}, {'class_label': 7}, {'num_class_embeds': 1000}):
            with self.assertRaisesRegex(ValueError, 'not supported'):
                guard(**kwargs)

    def args(self, stage='generate', run='lsun_primary', **kwargs):
        return SimpleNamespace(stage=stage, run=run, workdir='/tmp/repro-plan-only',
                               reference=None, model_path=None, steps=kwargs.get('steps'), device='cuda')

    def test_all_stages(self):
        for run in reproduce.RUNS:
            for stage in ('generate', 'official', 'features', 'bootstrap', 'pixel'):
                plan = reproduce.make_plan(self.args(stage, run))
                self.assertTrue(plan)
                for cmd, _, _ in plan:
                    self.assertTrue(Path(cmd[1]).is_file())

    def test_generation_config(self):
        plan = reproduce.make_plan(self.args())
        self.assertEqual(len(plan), 8)
        cmd = plan[0][0]
        for flag, value in [('--seed', '20260724'), ('--batch-size', '64'), ('--num-samples', '10000')]:
            self.assertEqual(cmd[cmd.index(flag) + 1], value)
        self.assertIn('--revision', cmd)

    def test_worker_flags_exist(self):
        for run in reproduce.RUNS:
            stages = ['generate', 'official', 'features', 'bootstrap', 'pixel']
            if run == 'lsun_primary':
                stages.append('jacobian')
            for stage in stages:
                for cmd, _, _ in reproduce.make_plan(self.args(stage, run)):
                    tree = ast.parse(Path(cmd[1]).read_text())
                    declared = set()
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
                            declared.update(a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str))
                    for value in cmd[2:]:
                        if value.startswith('--'):
                            self.assertIn(value, declared, (stage, value))

    def test_invalid_steps(self):
        for s in ('0', '6,6', '999'):
            with self.assertRaises(ValueError):
                reproduce.make_plan(self.args(steps=s))

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = subprocess.run([sys.executable, str(ROOT / 'reproduce.py'), 'generate',
                                '--workdir', tmp], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn('DRY RUN', p.stdout)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_missing_inputs_fail_before_worker_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = subprocess.run([sys.executable, str(ROOT / 'reproduce.py'), 'official',
                                '--workdir', tmp, '--execute'], capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn('Missing prerequisite', p.stderr)
            self.assertNotIn('tensorflow', p.stderr)

    def test_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'lsun_primary/samples/N1.npz'
            path.parent.mkdir(parents=True)
            path.touch()
            p = subprocess.run([sys.executable, str(ROOT / 'reproduce.py'), 'generate',
                                '--workdir', tmp, '--execute'], capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn('Refusing to overwrite', p.stderr)


if __name__ == '__main__':
    unittest.main()
