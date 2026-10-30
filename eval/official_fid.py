#!/usr/bin/env python3
"""Compute official guided-diffusion FID/sFID from compact reference stats."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator-dir", required=True)
    parser.add_argument("--reference-stats", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluator_dir = Path(args.evaluator_dir).resolve()
    reference_path = Path(args.reference_stats).resolve()
    sample_path = Path(args.samples).resolve()
    out_path = Path(args.out).resolve()

    os.chdir(evaluator_dir)
    sys.path.insert(0, str(evaluator_dir))
    import tensorflow.compat.v1 as tf
    from evaluator import Evaluator, FIDStatistics

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    evaluator = Evaluator(tf.Session(config=config), batch_size=args.batch_size)
    evaluator.warmup()

    with np.load(reference_path) as reference:
        ref_fid = FIDStatistics(reference["mu"], reference["sigma"])
        ref_sfid = FIDStatistics(reference["mu_s"], reference["sigma_s"])

    sample_acts = evaluator.read_activations(str(sample_path))
    sample_fid, sample_sfid = evaluator.read_statistics(str(sample_path), sample_acts)
    payload = {
        "reference_stats": str(reference_path),
        "samples": str(sample_path),
        "num_samples": int(len(sample_acts[0])),
        "inception_score": evaluator.compute_inception_score(sample_acts[0]),
        "fid": sample_fid.frechet_distance(ref_fid),
        "sfid": sample_sfid.frechet_distance(ref_sfid),
        "evaluator": "openai/guided-diffusion",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
