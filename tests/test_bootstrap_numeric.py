"""Small synthetic CPU test, not validation against scientific ground truth."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HAS_NUMERIC = all(importlib.util.find_spec(m) for m in ('numpy', 'scipy'))


@unittest.skipUnless(HAS_NUMERIC, 'requires numpy and scipy')
class BootstrapNumericTest(unittest.TestCase):
    def test_repeatability_and_paired_identity(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            rng = np.random.default_rng(17)
            ref = rng.normal(size=(64, 8))
            gen = rng.normal(size=(64, 8))
            np.savez(path / 'ref.npz', features=ref)
            np.savez(path / 'gen.npz', features=gen)
            results = []
            for i in range(2):
                out = path / f'run{i}.json'
                cmd = [sys.executable, str(ROOT / 'eval/paired_bootstrap.py'),
                       '--reference-features', str(path / 'ref.npz'),
                       '--generated-features', f'N1={path / "gen.npz"}',
                       '--generated-features', f'N2={path / "gen.npz"}',
                       '--projection-dim', '4', '--sample-size', '32', '--rounds', '12',
                       '--seed', '17', '--out', str(out)]
                p = subprocess.run(cmd, capture_output=True, text=True)
                self.assertEqual(p.returncode, 0, p.stderr)
                results.append(json.loads(out.read_text()))
            self.assertEqual(results[0], results[1])
            self.assertEqual(results[0]['difference_vs_best']['N2']['ci95'], [0.0, 0.0])


if __name__ == '__main__':
    unittest.main()
