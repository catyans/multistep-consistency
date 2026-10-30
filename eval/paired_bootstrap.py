#!/usr/bin/env python3
"""Paired bootstrap FID differences using fixed feature caches.

The same reference and sample indices are reused across step counts. Features
are projected to a seeded low-dimensional subspace so hundreds of covariance
square roots remain tractable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import linalg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--reference-features", required=True)
    p.add_argument(
        "--generated-features",
        action="append",
        required=True,
        help="LABEL=PATH; repeat for each step count.",
    )
    p.add_argument("--projection-dim", type=int, default=64)
    p.add_argument("--sample-size", type=int, default=2_000)
    p.add_argument("--rounds", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260724)
    p.add_argument("--out", required=True)
    return p.parse_args()


def load_features(path: str) -> np.ndarray:
    with np.load(path) as data:
        return np.asarray(data["features"], dtype=np.float64)


def fid(x: np.ndarray, y: np.ndarray) -> float:
    mx, my = x.mean(0), y.mean(0)
    cx, cy = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    covmean, _ = linalg.sqrtm(cx @ cy, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(max((mx - my) @ (mx - my) + np.trace(cx + cy - 2 * covmean), 0))


def summarize(values: np.ndarray) -> dict[str, float | list[float]]:
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "ci95": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    reference = load_features(args.reference_features)

    generated: dict[str, np.ndarray] = {}
    for item in args.generated_features:
        label, path = item.split("=", 1)
        generated[label] = load_features(path)

    common_count = min([len(reference), *[len(x) for x in generated.values()]])
    sample_size = min(args.sample_size, common_count)
    feature_dim = reference.shape[1]
    projection = rng.normal(size=(feature_dim, args.projection_dim))
    projection /= np.linalg.norm(projection, axis=0, keepdims=True) + 1e-12

    ref_projected = reference[:common_count] @ projection
    gen_projected = {
        label: values[:common_count] @ projection
        for label, values in generated.items()
    }

    point = {
        label: fid(ref_projected, values)
        for label, values in gen_projected.items()
    }
    best_label = min(point, key=point.get)
    boot = {label: np.empty(args.rounds) for label in generated}

    for round_index in range(args.rounds):
        ref_idx = rng.integers(0, common_count, size=sample_size)
        gen_idx = rng.integers(0, common_count, size=sample_size)
        ref_sample = ref_projected[ref_idx]
        for label, values in gen_projected.items():
            boot[label][round_index] = fid(ref_sample, values[gen_idx])

    payload = {
        "projection_dim": args.projection_dim,
        "sample_size": sample_size,
        "rounds": args.rounds,
        "seed": args.seed,
        "best_point_label": best_label,
        "point_fid": point,
        "bootstrap": {label: summarize(values) for label, values in boot.items()},
        "difference_vs_best": {
            label: summarize(values - boot[best_label])
            for label, values in boot.items()
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

