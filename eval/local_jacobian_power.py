#!/usr/bin/env python3
"""Local adversarial sensitivity via finite-difference power iteration.

The diagnostic alternates a central finite-difference approximation of Jv
with an exact reverse-mode J^T u update for the complete denoised consistency
transition. The result estimates a local worst-direction amplification on
sampled states. It is not a certified global Lipschitz bound.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from transition_diagnostics import (
    denoised_map,
    load_pipeline,
    load_reference,
    summary,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--eta", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=20260724)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float16",
    )
    p.add_argument("--class-label", type=int, default=None)
    p.add_argument("--out", required=True)
    return p.parse_args()


def normalize_per_sample(value: torch.Tensor) -> torch.Tensor:
    flat = value.flatten(1)
    return (flat / (flat.norm(dim=1, keepdim=True) + 1e-12)).view_as(value)


def require_unconditional(class_label=None, num_class_embeds=None) -> None:
    if class_label is not None or num_class_embeds is not None:
        raise ValueError("Class-conditioned Jacobian diagnostics are not supported.")


def main() -> None:
    args = parse_args()
    require_unconditional(class_label=args.class_label)
    if min(args.steps, args.trials, args.batch_size, args.iterations) < 1:
        raise ValueError("steps, trials, batch-size, and iterations must be positive")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    pipe = load_pipeline(args.model, dtype, device)
    require_unconditional(num_class_embeds=getattr(pipe.unet.config, "num_class_embeds", None))
    pipe.unet.eval()
    for parameter in pipe.unet.parameters():
        parameter.requires_grad_(False)
    pipe.scheduler.set_timesteps(args.steps, device=device)

    image_size = int(pipe.unet.config.sample_size)
    total = args.trials * args.batch_size
    real = load_reference(Path(args.reference), total, image_size, rng).to(
        device=device, dtype=dtype
    )
    rows = []

    for step_index, timestep in enumerate(pipe.scheduler.timesteps):
        sigma = pipe.scheduler.sigmas[step_index].to(device=device, dtype=dtype)
        sigma_min = torch.as_tensor(
            pipe.scheduler.config.sigma_min, device=device, dtype=dtype
        )
        noise_scale = torch.sqrt(torch.clamp(sigma**2 - sigma_min**2, min=0))
        initial_values: list[float] = []
        final_values: list[float] = []
        convergence: list[list[float]] = [[] for _ in range(args.iterations)]

        for trial in range(args.trials):
            start = trial * args.batch_size
            x0 = real[start : start + args.batch_size]
            generator = torch.Generator(device=device).manual_seed(
                args.seed + 10_000 * step_index + trial
            )
            noise = torch.randn(
                x0.shape, generator=generator, device=device, dtype=dtype
            )
            state = (x0 + noise_scale * noise).detach()
            direction = normalize_per_sample(
                torch.randn(
                    state.shape,
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )
            )
            perturb_norm = (
                args.eta
                * state.flatten(1).norm(dim=1, keepdim=True)
            ).clamp_min(1e-6)

            for iteration in range(args.iterations):
                perturbation = (
                    direction.flatten(1) * perturb_norm
                ).view_as(state)
                with torch.no_grad():
                    output_plus = denoised_map(
                        pipe,
                        state + perturbation,
                        timestep,
                        sigma,
                        None,
                    )
                    output_minus = denoised_map(
                        pipe,
                        state - perturbation,
                        timestep,
                        sigma,
                        None,
                    )
                jv = (output_plus - output_minus) * 0.5
                ratios = (
                    jv.flatten(1).norm(dim=1)
                    / (perturbation.flatten(1).norm(dim=1) + 1e-12)
                )
                values = ratios.float().cpu().tolist()
                convergence[iteration].extend(values)
                if iteration == 0:
                    initial_values.extend(values)

                output_direction = normalize_per_sample(jv).detach()
                state_for_grad = state.detach().requires_grad_(True)
                output = denoised_map(
                    pipe,
                    state_for_grad,
                    timestep,
                    sigma,
                    None,
                )
                scalar = torch.sum(output * output_direction)
                gradient = torch.autograd.grad(
                    scalar, state_for_grad, create_graph=False
                )[0]
                direction = normalize_per_sample(gradient.detach())

            final_values.extend(values)

        row = {
            "step_index": step_index,
            "timestep": float(timestep.item()),
            "sigma": float(sigma.float().item()),
            "random_initial_direction": summary(initial_values),
            "power_iteration_final": summary(final_values),
            "iteration_median": [
                float(np.median(values)) for values in convergence
            ],
            "iteration_p95": [
                float(np.quantile(values, 0.95)) for values in convergence
            ],
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    payload = {
        "model": args.model,
        "reference": args.reference,
        "steps": args.steps,
        "trials": args.trials,
        "batch_size": args.batch_size,
        "iterations": args.iterations,
        "eta_relative_to_state_norm": args.eta,
        "seed": args.seed,
        "note": (
            "Finite-difference local power iteration on sampled states; "
            "not a certified global Lipschitz upper bound."
        ),
        "timesteps": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
