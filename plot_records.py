#!/usr/bin/env python3
"""Plot the completed cross-model rebuttal sweeps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def panel(ax, rows, field: str, title: str, ylabel: str) -> None:
    x = [row["N"] for row in rows]
    y = [row[field] for row in rows]
    best = min(range(len(y)), key=y.__getitem__)
    ax.plot(x, y, "o-", color="#31688e", linewidth=2, markersize=5)
    ax.scatter([x[best]], [y[best]], color="#d1495b", s=55, zorder=3)
    ax.annotate(
        f"best tested: N={x[best]:g}",
        (x[best], y[best]),
        xytext=(6, 8),
        textcoords="offset points",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )
    ax.set_title(title)
    ax.set_xlabel("Sampling steps N")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def main() -> None:
    args = parse_args()
    lsun = read_csv(args.input_dir / "lsun_replicate_metrics.csv")
    imagenet = read_csv(args.input_dir / "imagenet64_metrics.csv")

    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    panel(axes[0, 0], lsun, "official_fid", "LSUN Bedroom-256, seed 2", "Official FID ↓")
    panel(axes[0, 1], imagenet, "official_fid", "ImageNet-64", "Official FID ↓")
    panel(
        axes[1, 0],
        lsun,
        "pixel_sliced_w2_32",
        "LSUN Bedroom-256, seed 2",
        r"RGB sliced-$W_2$ (32×32) ↓",
    )
    panel(
        axes[1, 1],
        imagenet,
        "pixel_sliced_w2_32",
        "ImageNet-64",
        r"RGB sliced-$W_2$ (32×32) ↓",
    )
    fig.suptitle("Step-count sweeps on LSUN and ImageNet", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    fig.savefig(args.output.with_suffix(".svg"))


if __name__ == "__main__":
    main()
