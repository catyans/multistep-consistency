#!/usr/bin/env python3
"""Compute internally consistent FID, KID, and sliced-W2 from NPZ batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import linalg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True, help="Real-data NPZ with arr_0.")
    p.add_argument(
        "--generated",
        action="append",
        required=True,
        help="LABEL=PATH; repeat for each step count.",
    )
    p.add_argument("--max-samples", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20260724)
    p.add_argument("--kid-subsets", type=int, default=50)
    p.add_argument("--kid-subset-size", type=int, default=1_000)
    p.add_argument("--swd-projections", type=int, default=256)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected LABEL=PATH, got {value!r}")
    label, path = value.split("=", 1)
    return label, Path(path)


def load_images(path: Path, max_samples: int) -> np.ndarray:
    with np.load(path, mmap_mode="r") as data:
        if "arr_0" not in data:
            raise KeyError(f"{path} does not contain arr_0")
        arr = np.asarray(data["arr_0"][:max_samples])
    if arr.ndim != 4:
        raise ValueError(f"Expected four-dimensional images in {path}, got {arr.shape}")
    if arr.shape[-1] in (1, 3):
        return arr
    if arr.shape[1] in (1, 3):
        return np.transpose(arr, (0, 2, 3, 1))
    raise ValueError(f"Cannot infer image channel axis for {path}: {arr.shape}")


def extract_features(
    image_path: Path,
    cache_path: Path,
    max_samples: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if cache_path.exists():
        with np.load(cache_path) as data:
            return np.asarray(data["features"])

    from pytorch_fid.inception import InceptionV3

    images = load_images(image_path, max_samples)
    inception = InceptionV3(
        [InceptionV3.BLOCK_INDEX_BY_DIM[2048]]
    ).to(device).eval()
    features: list[np.ndarray] = []

    for start in range(0, len(images), batch_size):
        batch_np = images[start : start + batch_size]
        batch = (
            torch.from_numpy(batch_np)
            .permute(0, 3, 1, 2)
            .float()
            .div(255)
            .to(device)
        )
        batch = torch.nn.functional.interpolate(
            batch, (299, 299), mode="bilinear", align_corners=False
        )
        with torch.inference_mode():
            feat = inception(batch)[0]
        if feat.shape[-2:] != (1, 1):
            feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
        features.append(feat.flatten(1).cpu().numpy())

    result = np.concatenate(features, axis=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, features=result, source=str(image_path))
    return result


def fid(x: np.ndarray, y: np.ndarray) -> float:
    mx, my = x.mean(0), y.mean(0)
    cx, cy = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    covmean, _ = linalg.sqrtm(cx @ cy, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    value = (mx - my) @ (mx - my) + np.trace(cx + cy - 2 * covmean)
    return float(max(value, 0))


def polynomial_mmd(x: np.ndarray, y: np.ndarray) -> float:
    dim = x.shape[1]
    kxx = (x @ x.T / dim + 1) ** 3
    kyy = (y @ y.T / dim + 1) ** 3
    kxy = (x @ y.T / dim + 1) ** 3
    n, m = len(x), len(y)
    xx = (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
    yy = (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
    return float(xx + yy - 2 * kxy.mean())


def kid(
    x: np.ndarray,
    y: np.ndarray,
    subsets: int,
    subset_size: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    size = min(subset_size, len(x), len(y))
    values = []
    for _ in range(subsets):
        xi = rng.choice(len(x), size=size, replace=False)
        yi = rng.choice(len(y), size=size, replace=False)
        values.append(polynomial_mmd(x[xi], y[yi]))
    return float(np.mean(values)), float(np.std(values, ddof=1))


def sliced_w2(
    x: np.ndarray,
    y: np.ndarray,
    projections: int,
    rng: np.random.Generator,
) -> float:
    count = min(len(x), len(y))
    if len(x) != count:
        x = x[rng.choice(len(x), count, replace=False)]
    if len(y) != count:
        y = y[rng.choice(len(y), count, replace=False)]
    directions = rng.normal(size=(x.shape[1], projections))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True) + 1e-12
    xp = np.sort(x @ directions, axis=0)
    yp = np.sort(y @ directions, axis=0)
    return float(np.sqrt(np.mean((xp - yp) ** 2)))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    reference_path = Path(args.reference)
    reference = extract_features(
        reference_path,
        cache_dir / "reference_features.npz",
        args.max_samples,
        args.batch_size,
        device,
    )

    results = {}
    for value in args.generated:
        label, path = parse_labeled_path(value)
        generated = extract_features(
            path,
            cache_dir / f"{label}_features.npz",
            args.max_samples,
            args.batch_size,
            device,
        )
        kid_mean, kid_sd = kid(
            reference,
            generated,
            args.kid_subsets,
            args.kid_subset_size,
            rng,
        )
        results[label] = {
            "source": str(path),
            "num_reference": len(reference),
            "num_generated": len(generated),
            "fid": fid(reference, generated),
            "kid_mean": kid_mean,
            "kid_sd": kid_sd,
            "sliced_w2": sliced_w2(
                reference, generated, args.swd_projections, rng
            ),
        }
        print(label, json.dumps(results[label], sort_keys=True), flush=True)

    payload = {
        "reference": str(reference_path),
        "max_samples": args.max_samples,
        "seed": args.seed,
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

