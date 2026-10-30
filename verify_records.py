#!/usr/bin/env python3
"""Offline consistency checks of published records, NOT a fresh metric evaluation."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    manifest = ROOT / 'SHA256SUMS.json'
    if manifest.exists():
        for name, digest in json.loads(manifest.read_text()).items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
        print('Package hashes: PASS')
    runs = json.loads((ROOT / 'runs.json').read_text())
    count = 0
    for name, cfg in runs.items():
        with (ROOT / 'records' / cfg['record']).open() as handle:
            rows = list(csv.DictReader(handle))
        assert [int(r['N']) for r in rows] == cfg['steps']
        for r in rows:
            n = int(r['N'])
            raw = ROOT / 'records/raw' / name
            metric = json.loads((raw / 'official' / f'N{n}.json').read_text())
            meta = json.loads((raw / 'samples' / f'N{n}.json').read_text())
            assert abs(float(r['official_fid']) - metric['fid']) < 1e-8
            for field, key in [('seed', 'seed'), ('num_samples', 'samples'), ('batch_size', 'batch'), ('dtype', 'dtype')]:
                assert meta[field] == cfg[key], (name, n, field)
            assert metric['num_samples'] == cfg['samples']
            expected_latency = meta['elapsed_seconds'] * 1000 / cfg['samples']
            assert abs(float(r['latency_ms_per_image']) - expected_latency) < .001
            count += 1
        best = min(rows, key=lambda r: float(r['official_fid']))
        print(f"{name}: {len(rows)} points, best tested FID={float(best['official_fid']):.3f} at N={best['N']}")
    bootstrap = json.loads((ROOT / 'records/paired_bootstrap_all.json').read_text())
    assert bootstrap['rounds'] == 2000
    assert bootstrap['best_point_label'] == 'N6'
    ci = bootstrap['difference_vs_best']['N8']['ci95']
    assert ci[0] < 0 < ci[1]
    pixel = json.loads((ROOT / 'records/lsun_primary_pixel_sliced_w2.json').read_text())
    for n, expected in [(1, .0521), (6, .0408), (10, .0360), (20, .0443)]:
        assert round(pixel['results']['32'][f'N{n}']['sliced_w2'], 4) == expected
    print(f'{count} official FID values and sampling configurations: PASS')
    print('LSUN projected-bootstrap interval and Table 2 RGB sliced-W2 values: PASS')
    print('These checks validate archived records; no samples or metrics were recomputed.')


if __name__ == '__main__':
    main()
