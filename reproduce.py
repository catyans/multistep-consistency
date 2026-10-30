#!/usr/bin/env python3
"""Print reproducible commands; run expensive stages only with --execute.

Uses only Python's standard library. Numerical worker scripts have separate
dependencies. Output paths are per-run and are never silently overwritten.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RUNS = json.loads((ROOT / "runs.json").read_text())


def make_plan(args):
    run = RUNS[args.run]
    work = Path(args.workdir).resolve()
    base = work / args.run
    ref = Path(args.reference).resolve() if args.reference else work / "data" / run["reference"]
    evaluator = ROOT / "third_party" / "guided_diffusion"
    model = str(Path(args.model_path).resolve()) if args.model_path else run["model"]
    steps = [int(x) for x in args.steps.split(",")] if args.steps else run["steps"]
    if not steps or len(set(steps)) != len(steps) or any(n not in run["steps"] for n in steps):
        raise ValueError("--steps must be unique members of the recorded step counts")
    commands = []

    def add(script, options, inputs, outputs):
        commands.append(([sys.executable, str(ROOT / "eval" / script),
                          *map(str, options)], inputs, outputs))

    samples = [base / "samples" / f"N{n}.npz" for n in steps]
    if args.stage == "generate":
        for n, sample in zip(steps, samples):
            grid = base / "grids" / f"N{n}.png"
            opts = ["--model", model, "--steps", n, "--num-samples", run["samples"],
                    "--batch-size", run["batch"], "--seed", run["seed"],
                    "--dtype", run["dtype"], "--device", args.device,
                    "--out", sample, "--grid-out", grid]
            if not args.model_path and run["recorded_revision"]:
                opts += ["--revision", run["recorded_revision"]]
            add("generate_npz.py", opts, [], [sample, sample.with_suffix(".json"), grid])
    elif args.stage == "official":
        stats = base / "reference_stats.npz"
        if not stats.exists():
            add("official_reference_stats.py", ["--evaluator-dir", evaluator,
                "--reference", ref, "--out", stats], [ref], [stats])
        for n, sample in zip(steps, samples):
            out = base / "official" / f"N{n}.json"
            add("official_fid.py", ["--evaluator-dir", evaluator,
                "--reference-stats", stats, "--samples", sample, "--out", out],
                [stats, sample], [out])
    elif args.stage == "features":
        # Preserve step ordering: it affects the historical RNG stream.
        out = base / "feature_metrics.json"
        opts = ["--reference", ref, "--max-samples", run["feature_samples"],
                "--seed", run["seed"], "--batch-size", 32,
                "--device", args.device, "--cache-dir", base / "features", "--out", out]
        for n, sample in zip(steps, samples):
            opts += ["--generated", f"N{n}={sample}"]
        add("feature_metrics.py", opts, [ref, *samples], [out])
    elif args.stage == "bootstrap":
        reference = base / "features" / "reference_features.npz"
        inputs = [reference]
        out = base / "paired_bootstrap.json"
        opts = ["--reference-features", reference, "--projection-dim", 64,
                "--sample-size", 3000, "--rounds", 2000,
                "--seed", run["seed"], "--out", out]
        for n in steps:
            path = base / "features" / f"N{n}_features.npz"
            inputs.append(path)
            opts += ["--generated-features", f"N{n}={path}"]
        add("paired_bootstrap.py", opts, inputs, [out])
    elif args.stage == "pixel":
        out = base / "pixel_sliced_w2.json"
        opts = ["--reference", ref, "--max-samples", run["pixel_samples"],
                "--resolutions", "16,32", "--projections", 256,
                "--bootstrap-directions", 2000, "--seed", run["seed"], "--out", out]
        for n, sample in zip(steps, samples):
            opts += ["--generated", f"N{n}={sample}"]
        add("pixel_sliced_w2.py", opts, [ref, *samples], [out])
    elif args.stage == "jacobian":
        if args.run != "lsun_primary":
            raise ValueError("The published local-gain diagnostic is LSUN-only")
        out = base / "jacobian_fp32.json"
        opts = ["--model", model, "--reference", ref, "--steps", 8,
                "--trials", 8, "--batch-size", 1, "--iterations", 4,
                "--eta", "0.001", "--dtype", "float32", "--seed", run["seed"],
                "--device", args.device, "--out", out]
        add("local_jacobian_power.py", opts, [ref], [out])
    return commands


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["generate", "official", "features", "bootstrap", "pixel", "jacobian"])
    p.add_argument("--run", choices=list(RUNS), default="lsun_primary")
    p.add_argument("--workdir", default=str(ROOT / "outputs"))
    p.add_argument("--reference", help="User-supplied official reference NPZ")
    p.add_argument("--model-path", help="Local snapshot; required for a revision-pinned Jacobian run")
    p.add_argument("--steps", help="Subset for testing; omit for full paper configuration")
    p.add_argument("--device", default="cuda")
    p.add_argument("--execute", action="store_true", help="Actually run; otherwise only print commands")
    args = p.parse_args()
    try:
        commands = make_plan(args)
        # Preflight the entire stage before starting a costly job.
        produced = set()
        for cmd, inputs, outputs in commands:
            print(shlex.join(cmd), flush=True)
            if args.execute:
                for path in inputs:
                    if not path.is_file() and path not in produced:
                        raise FileNotFoundError(f"Missing prerequisite: {path}")
                for path in outputs:
                    if path.exists():
                        raise FileExistsError(f"Refusing to overwrite: {path}; use a new --workdir")
            produced.update(outputs)
        if not args.execute:
            print("DRY RUN: no jobs launched, no downloads performed.")
            return
        for cmd, _, _ in commands:
            subprocess.run(cmd, cwd=ROOT, check=True, env=os.environ.copy())
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
