#!/usr/bin/env python3
"""Multiscale pixel-space sliced-W2 diagnostics for generated NPZ batches.

This intentionally operates on downsampled RGB pixels rather than learned
features. It is still a finite-projection, reduced-resolution diagnostic, not
the full data-space Wasserstein distance appearing in the theory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument(
        "--generated",
        action="append",
        required=True,
        help="Labeled NPZ in the form LABEL=PATH; repeat for each step count.",
    )
    p.add_argument("--resolutions", default="16,32")
    p.add_argument("--projections", type=int, default=256)
    p.add_argument("--max-samples", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--bootstrap-directions", type=int, default=2_000)
    p.add_argument("--seed", type=int, default=20260724)
    p.add_argument("--out", required=True)
    return p.parse_args()


def parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected LABEL=PATH, got {value!r}")
    label, path = value.split("=", 1)
    if not label or not path:
        raise ValueError(f"Expected non-empty LABEL=PATH, got {value!r}")
    return label, Path(path)


def image_count(path: Path) -> int:
    with np.load(path, mmap_mode="r") as data:
        return int(len(data["arr_0"]))


def make_directions(
    resolution: int, projections: int, rng: np.random.Generator
) -> np.ndarray:
    dim = 3 * resolution * resolution
    directions = rng.normal(size=(dim, projections)).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=0, keepdims=True) + 1e-12
    return directions


def project_npz(
    path: Path,
    indices: np.ndarray,
    resolution: int,
    directions: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    direction_tensor = torch.from_numpy(directions)
    with np.load(path, mmap_mode="r") as data:
        images = data["arr_0"]
        for start in range(0, len(indices), batch_size):
            selected = np.asarray(images[indices[start : start + batch_size]])
            x = torch.from_numpy(selected)
            if x.ndim != 4:
                raise ValueError(f"Expected 4-D image array in {path}, got {x.shape}")
            if x.shape[-1] in (1, 3):
                x = x.permute(0, 3, 1, 2)
            x = x.float().div_(255.0)
            if x.shape[-2:] != (resolution, resolution):
                x = F.interpolate(x, (resolution, resolution), mode="area")
            outputs.append((x.flatten(1) @ direction_tensor).numpy())
    return np.concatenate(outputs, axis=0)


def summarize_distance(
    reference: np.ndarray,
    generated: np.ndarray,
    bootstrap_rounds: int,
    rng: np.random.Generator,
) -> dict[str, float | list[float]]:
    if len(reference) != len(generated):
        raise ValueError("Reference and generated projections must have equal size")
    ref_sorted = np.sort(reference, axis=0)
    gen_sorted = np.sort(generated, axis=0)
    per_direction_w2_sq = np.mean((ref_sorted - gen_sorted) ** 2, axis=0)
    estimate = float(np.sqrt(np.mean(per_direction_w2_sq)))

    boot = np.empty(bootstrap_rounds, dtype=np.float64)
    n_directions = len(per_direction_w2_sq)
    for index in range(bootstrap_rounds):
        selected = rng.integers(0, n_directions, size=n_directions)
        boot[index] = np.sqrt(np.mean(per_direction_w2_sq[selected]))

    return {
        "sliced_w2": estimate,
        "direction_bootstrap_95ci": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "per_direction_median_w2": float(
            np.median(np.sqrt(per_direction_w2_sq))
        ),
        "per_direction_p95_w2": float(
            np.quantile(np.sqrt(per_direction_w2_sq), 0.95)
        ),
    }


def main() -> None:
    args = parse_args()
    if args.projections < 1 or args.max_samples < 1 or args.batch_size < 1:
        raise ValueError("projections, max-samples, and batch-size must be positive")
    resolutions = [int(value) for value in args.resolutions.split(",")]
    if any(value < 1 for value in resolutions):
        raise ValueError("All resolutions must be positive")

    generated = [parse_labeled_path(value) for value in args.generated]
    counts = [image_count(Path(args.reference))]
    counts.extend(image_count(path) for _, path in generated)
    count = min(args.max_samples, *counts)

    rng = np.random.default_rng(args.seed)
    reference_count = counts[0]
    reference_indices = rng.choice(reference_count, size=count, replace=False)
    generated_indices = np.arange(count)
    payload: dict[str, object] = {
        "reference": args.reference,
        "generated": {label: str(path) for label, path in generated},
        "num_samples": count,
        "resolutions": resolutions,
        "projections": args.projections,
        "bootstrap_directions": args.bootstrap_directions,
        "seed": args.seed,
        "note": (
            "Finite-projection sliced-W2 on downsampled RGB pixels. Confidence "
            "intervals resample projection directions only. This is a "
            "reduced-resolution diagnostic, not the full data-space W2."
        ),
        "results": {},
    }

    for resolution in resolutions:
        direction_rng = np.random.default_rng(args.seed + resolution)
        directions = make_directions(
            resolution, args.projections, direction_rng
        )
        reference_projection = project_npz(
            Path(args.reference),
            reference_indices,
            resolution,
            directions,
            args.batch_size,
        )
        resolution_results = {}
        for label, path in generated:
            projected = project_npz(
                path,
                generated_indices,
                resolution,
                directions,
                args.batch_size,
            )
            result = summarize_distance(
                reference_projection,
                projected,
                args.bootstrap_directions,
                rng,
            )
            resolution_results[label] = result
            print(
                json.dumps(
                    {"resolution": resolution, "label": label, **result}
                ),
                flush=True,
            )
        payload["results"][str(resolution)] = resolution_results

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
