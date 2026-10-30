#!/usr/bin/env python3
"""Timestep-wise diagnostics for the actual denoised CM transition.

This measures sampled directional sensitivity of

    g_t(x) = clip(c_out(t) * UNet(c_in(t) * x, t) + c_skip(t) * x)

on forward-noised real states, and reports endpoint/local-transition error
proxies. The sensitivity values are diagnostics, not certified global
Lipschitz upper bounds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--reference", required=True, help="Real-data NPZ with arr_0.")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--eta", type=float, default=1e-3)
    p.add_argument("--swd-projections", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260724)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    p.add_argument("--class-label", type=int, default=None)
    p.add_argument("--out", required=True)
    return p.parse_args()


def load_pipeline(model_id: str, dtype: torch.dtype, device: torch.device):
    from diffusers import (
        CMStochasticIterativeScheduler,
        ConsistencyModelPipeline,
        UNet2DModel,
    )

    try:
        pipe = ConsistencyModelPipeline.from_pretrained(
            model_id, torch_dtype=dtype
        )
    except Exception:
        unet = UNet2DModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=dtype
        )
        pipe = ConsistencyModelPipeline(
            unet=unet,
            scheduler=CMStochasticIterativeScheduler(
                num_train_timesteps=40,
                sigma_min=0.002,
                sigma_max=80.0,
                sigma_data=0.5,
                rho=7.0,
            ),
        )
    return pipe.to(device)


def load_reference(
    path: Path, count: int, image_size: int, rng: np.random.Generator
) -> torch.Tensor:
    with np.load(path, mmap_mode="r") as data:
        arr = data["arr_0"]
        indices = rng.choice(len(arr), size=min(count, len(arr)), replace=False)
        images = np.asarray(arr[indices])
    if images.shape[-1] in (1, 3):
        tensor = torch.from_numpy(images).permute(0, 3, 1, 2)
    else:
        tensor = torch.from_numpy(images)
    tensor = tensor.float().div(127.5).sub(1)
    if tensor.shape[-2:] != (image_size, image_size):
        tensor = torch.nn.functional.interpolate(
            tensor, (image_size, image_size), mode="bilinear", align_corners=False
        )
    return tensor


def denoised_map(
    pipe,
    sample: torch.Tensor,
    timestep: torch.Tensor,
    sigma: torch.Tensor,
    class_labels: torch.Tensor | None,
) -> torch.Tensor:
    scheduler = pipe.scheduler
    scaled = sample / torch.sqrt(
        sigma**2 + torch.as_tensor(
            scheduler.config.sigma_data, device=sample.device, dtype=sample.dtype
        ) ** 2
    )
    model_output = pipe.unet(
        scaled,
        timestep.expand(sample.shape[0]),
        class_labels=class_labels,
        return_dict=False,
    )[0]
    c_skip, c_out = scheduler.get_scalings_for_boundary_condition(sigma)
    denoised = c_out * model_output + c_skip * sample
    if scheduler.config.clip_denoised:
        denoised = denoised.clamp(-1, 1)
    return denoised


def sliced_w2(
    x: np.ndarray,
    y: np.ndarray,
    projections: int,
    rng: np.random.Generator,
) -> float:
    directions = rng.normal(size=(x.shape[1], projections))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True) + 1e-12
    xp = np.sort(x @ directions, axis=0)
    yp = np.sort(y @ directions, axis=0)
    return float(np.sqrt(np.mean((xp - yp) ** 2)))


def summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values)
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    pipe = load_pipeline(args.model, dtype, device)
    pipe.unet.eval()
    pipe.scheduler.set_timesteps(args.steps, device=device)

    image_size = int(pipe.unet.config.sample_size)
    total = args.trials * args.batch_size
    real = load_reference(Path(args.reference), total, image_size, rng)
    real = real.to(device=device, dtype=dtype)
    class_labels = None
    if args.class_label is not None:
        class_labels = torch.full(
            (args.batch_size,), args.class_label, device=device, dtype=torch.long
        )

    rows = []
    for step_index, timestep in enumerate(pipe.scheduler.timesteps):
        sigma = pipe.scheduler.sigmas[step_index].to(device=device, dtype=dtype)
        sigma_min = torch.as_tensor(
            pipe.scheduler.config.sigma_min, device=device, dtype=dtype
        )
        noise_scale = torch.sqrt(torch.clamp(sigma**2 - sigma_min**2, min=0))
        sensitivities: list[float] = []
        endpoint_rmse: list[float] = []
        denoised_all: list[torch.Tensor] = []
        real_all: list[torch.Tensor] = []

        for trial in range(args.trials):
            start = trial * args.batch_size
            x0 = real[start : start + args.batch_size]
            generator = torch.Generator(device=device).manual_seed(
                args.seed + 10_000 * step_index + trial
            )
            noise = torch.randn(
                x0.shape, generator=generator, device=device, dtype=dtype
            )
            state = x0 + noise_scale * noise
            direction = torch.randn(
                state.shape, generator=generator, device=device, dtype=dtype
            )
            flat = direction.flatten(1)
            state_norm = state.flatten(1).norm(dim=1, keepdim=True)
            direction = (
                flat / (flat.norm(dim=1, keepdim=True) + 1e-12)
                * (args.eta * state_norm)
            ).view_as(state)

            with torch.inference_mode():
                output = denoised_map(
                    pipe, state, timestep, sigma, class_labels
                )
                output_perturbed = denoised_map(
                    pipe, state + direction, timestep, sigma, class_labels
                )
            ratio = (
                (output_perturbed - output).flatten(1).norm(dim=1)
                / (direction.flatten(1).norm(dim=1) + 1e-12)
            )
            rmse = torch.sqrt(torch.mean((output - x0) ** 2, dim=(1, 2, 3)))
            sensitivities.extend(ratio.float().cpu().tolist())
            endpoint_rmse.extend(rmse.float().cpu().tolist())
            denoised_all.append(output.float().cpu())
            real_all.append(x0.float().cpu())

        denoised = torch.cat(denoised_all)
        target = torch.cat(real_all)
        denoised_small = torch.nn.functional.interpolate(
            denoised, (32, 32), mode="bilinear", align_corners=False
        ).flatten(1).numpy()
        target_small = torch.nn.functional.interpolate(
            target, (32, 32), mode="bilinear", align_corners=False
        ).flatten(1).numpy()
        row = {
            "step_index": step_index,
            "timestep": float(timestep.item()),
            "sigma": float(sigma.float().item()),
            "noise_scale": float(noise_scale.float().item()),
            "directional_sensitivity": summary(sensitivities),
            "endpoint_rmse": summary(endpoint_rmse),
            "pixel_sliced_w2_32": sliced_w2(
                denoised_small,
                target_small,
                args.swd_projections,
                rng,
            ),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    payload = {
        "model": args.model,
        "reference": args.reference,
        "steps": args.steps,
        "trials": args.trials,
        "batch_size": args.batch_size,
        "eta_relative_to_state_norm": args.eta,
        "seed": args.seed,
        "note": (
            "Sampled local directional sensitivity of the complete denoised "
            "transition; not a certified global Lipschitz upper bound."
        ),
        "timesteps": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

