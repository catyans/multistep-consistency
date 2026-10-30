#!/usr/bin/env python3
"""Generate a fixed-step consistency-model sample batch for official evaluation.

Each invocation handles exactly one step count so independent jobs can be
scheduled across GPUs. The output follows the OpenAI evaluator convention:
an ``arr_0`` uint8 NHWC array stored in an NPZ file.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Hugging Face model ID or local path.")
    p.add_argument("--revision", default=None, help="Optional recorded Hugging Face commit.")
    p.add_argument("--steps", required=True, type=int)
    p.add_argument("--num-samples", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260724)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    p.add_argument("--out", required=True)
    p.add_argument("--grid-out", default=None)
    p.add_argument("--grid-count", type=int, default=16)
    p.add_argument("--class-label", type=int, default=None)
    return p.parse_args()


def load_pipeline(model_id: str, dtype: torch.dtype, device: torch.device, revision=None):
    from diffusers import (
        CMStochasticIterativeScheduler,
        ConsistencyModelPipeline,
        UNet2DModel,
    )

    try:
        pipe = ConsistencyModelPipeline.from_pretrained(
            model_id, torch_dtype=dtype, local_files_only=False, revision=revision
        )
    except Exception:
        unet = UNet2DModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=dtype, local_files_only=False, revision=revision
        )
        scheduler = CMStochasticIterativeScheduler(
            num_train_timesteps=40,
            sigma_min=0.002,
            sigma_max=80.0,
            sigma_data=0.5,
            rho=7.0,
        )
        pipe = ConsistencyModelPipeline(unet=unet, scheduler=scheduler)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def to_uint8_nhwc(images: torch.Tensor) -> np.ndarray:
    images = images.detach().float().clamp(0, 1)
    return (
        images.mul(255)
        .round()
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.num_samples < 1 or args.batch_size < 1:
        raise ValueError("steps, num-samples, and batch-size must be positive")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    pipe = load_pipeline(args.model, dtype, device, args.revision)

    chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    grid_images: torch.Tensor | None = None
    started = time.time()
    n_batches = math.ceil(args.num_samples / args.batch_size)

    for batch_index in range(n_batches):
        batch_size = min(
            args.batch_size, args.num_samples - batch_index * args.batch_size
        )
        generator = torch.Generator(device="cpu").manual_seed(
            args.seed + batch_index
        )
        call_args = {
            "batch_size": batch_size,
            "num_inference_steps": args.steps,
            "output_type": "pt",
            "generator": generator,
        }
        if args.class_label is not None:
            call_args["class_labels"] = args.class_label
            label_chunks.append(
                np.full(batch_size, args.class_label, dtype=np.int64)
            )
        elif getattr(pipe.unet.config, "num_class_embeds", None) is not None:
            label_generator = torch.Generator(device="cpu").manual_seed(
                args.seed + 1_000_000 + batch_index
            )
            labels = torch.randint(
                0,
                int(pipe.unet.config.num_class_embeds),
                (batch_size,),
                generator=label_generator,
                dtype=torch.long,
            )
            call_args["class_labels"] = labels
            label_chunks.append(labels.numpy())
        with torch.inference_mode():
            images = pipe(**call_args).images
        if grid_images is None and args.grid_out:
            grid_images = images[: args.grid_count].detach().float().cpu()
        chunks.append(to_uint8_nhwc(images))

        completed = min((batch_index + 1) * args.batch_size, args.num_samples)
        if batch_index == 0 or completed == args.num_samples or batch_index % 25 == 0:
            elapsed = time.time() - started
            print(
                f"N={args.steps} {completed}/{args.num_samples} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    samples = np.concatenate(chunks, axis=0)[: args.num_samples]
    if label_chunks:
        labels = np.concatenate(label_chunks, axis=0)[: args.num_samples]
        np.savez(out, arr_0=samples, arr_1=labels)
    else:
        labels = None
        np.savez(out, arr_0=samples)

    if args.grid_out and grid_images is not None:
        from torchvision.utils import make_grid, save_image

        grid_path = Path(args.grid_out)
        grid_path.parent.mkdir(parents=True, exist_ok=True)
        nrow = max(1, int(math.sqrt(len(grid_images))))
        save_image(make_grid(grid_images, nrow=nrow, padding=2), grid_path)

    metadata = {
        "model": args.model,
        "revision": args.revision,
        "steps": args.steps,
        "num_samples": int(samples.shape[0]),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "dtype": args.dtype,
        "device": str(device),
        "shape": list(samples.shape),
        "class_conditioning": (
            "fixed"
            if args.class_label is not None
            else "deterministic_uniform"
            if labels is not None
            else "unconditional"
        ),
        "class_label": args.class_label,
        "elapsed_seconds": round(time.time() - started, 3),
        "output": str(out),
        "grid_output": args.grid_out,
    }
    out.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
