#!/usr/bin/env python3
"""Export one Projectile model failure as ground truth and a side-by-side video."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml


def load_helpers() -> object:
    path = Path(__file__).with_name("layer_residual_replacement.py")
    spec = importlib.util.spec_from_file_location("layer_residual_replacement", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--short-history", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed-offset", type=int, default=17_000_000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Refusing non-empty output: {args.out}")
    args.out.mkdir(parents=True)

    with (args.dataset_dir / "metadata.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    row = next((item for item in rows if item["sample_id"] == args.sample_id), None)
    if row is None:
        raise ValueError(f"Unknown sample id: {args.sample_id}")

    helpers = load_helpers()
    data = yaml.safe_load(args.data_config.read_text())
    cfg = helpers.SimpleNamespace(
        width=int(data["render"]["width"]), height=int(data["render"]["height"]), frames=int(data["render"]["num_frames"]),
        radius=int(data["render"]["ball_radius_px"]), background=data["render"]["background_rgb"],
        simulation_fps=int(data["physics"]["simulation_fps"]), prediction_start=int(data["history"]["prediction_start"]),
        short_masked_prefix=int(data["history"]["short_masked_prefix_pixels"]), latent_frames=33, condition_latents=17,
    )
    from sshv2.wan.config import StandardTrainingConfig
    from sshv2.wan.trainer import WanTrainingModule
    training = StandardTrainingConfig.from_file(args.training_config)
    training.model.dit.ckpt_file = args.checkpoint
    module = WanTrainingModule(
        dit_config=training.model.dit, vae_config=training.model.vae, no_encoding=False,
        num_condition_frames=training.model.num_condition_frames, num_inference_steps=args.steps,
        pipeline_type=training.model.pipe, pipeline_kwargs=training.model.pipe_kwargs,
    )
    pipe = module.pipe
    pipe.to(args.device)
    pipe.load_models_to_device(("dit", "vae"))
    pipe.pre_encoded_(False)
    cfg.condition_latents = int(training.model.num_condition_frames)

    original = np.asarray(imageio.mimread(args.dataset_dir / "videos" / row["video"]), dtype=np.uint8)
    condition = helpers.load_condition(row, args.dataset_dir, cfg, pipe.torch_dtype, args.device, args.short_history)
    seed = int(row["base_seed"]) + args.seed_offset
    future = helpers.generate(pipe, condition, cfg, args.steps, seed)
    generated = np.concatenate((original[:cfg.prediction_start], future), axis=0)
    separator = np.full((cfg.frames, cfg.height, 4, 3), 255, dtype=np.uint8)
    comparison = np.concatenate((original, separator, generated), axis=2)
    imageio.mimsave(args.out / "ground_truth.mp4", original, fps=int(data["render"]["fps"]))
    imageio.mimsave(args.out / "model_output.mp4", generated, fps=int(data["render"]["fps"]))
    imageio.mimsave(args.out / "ground_truth_vs_model.mp4", comparison, fps=int(data["render"]["fps"]))
    details = {
        "sample": row,
        "generation_seed": seed,
        "prediction_start": cfg.prediction_start,
        "video_layout": "ground truth on the left; model output on the right; frames 0-64 are identical observed input and frames 65-128 are ground truth versus generated future",
        "measurement": helpers.measure(future, row, cfg),
    }
    (args.out / "metadata.json").write_text(json.dumps(helpers.finite(details), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
