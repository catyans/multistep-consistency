#!/usr/bin/env python3
"""Create reusable OpenAI guided-diffusion FID/sFID reference statistics."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--evaluator-dir", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    evaluator_dir = Path(args.evaluator_dir).resolve()
    reference = Path(args.reference).resolve()
    out = Path(args.out).resolve()

    os.chdir(evaluator_dir)
    sys.path.insert(0, str(evaluator_dir))
    import tensorflow.compat.v1 as tf
    from evaluator import Evaluator

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    evaluator = Evaluator(tf.Session(config=config), batch_size=args.batch_size)
    evaluator.warmup()
    activations = evaluator.read_activations(str(reference))
    fid, sfid = evaluator.read_statistics(str(reference), activations)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        mu=fid.mu,
        sigma=fid.sigma,
        mu_s=sfid.mu,
        sigma_s=sfid.sigma,
        num_samples=np.asarray(len(activations[0])),
        source=np.asarray(str(reference)),
    )
    print(f"saved {out} from {len(activations[0])} reference images")


if __name__ == "__main__":
    main()
