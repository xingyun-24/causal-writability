#!/usr/bin/env python3
"""Generate one output per fixed-g RGB-sweep input, sharded by state index."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from layer_residual_replacement import finite, generate, load_condition, measure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--observed-window", type=int, default=16)
    parser.add_argument("--seed-offset", type=int, default=17_000_000)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing non-empty output: {args.out}")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")
    args.out.mkdir(parents=True)
    data = yaml.safe_load(args.data_config.read_text(encoding="utf-8"))
    cfg = SimpleNamespace(
        width=int(data["render"]["width"]), height=int(data["render"]["height"]), frames=int(data["render"]["num_frames"]),
        radius=int(data["render"]["ball_radius_px"]), background=data["render"]["background_rgb"],
        simulation_fps=int(data["physics"]["simulation_fps"]), prediction_start=int(data["history"]["prediction_start"]),
        short_masked_prefix=int(data["history"]["short_masked_prefix_pixels"]), latent_frames=33, condition_latents=17,
        gravity_bands={"low": tuple(data["physics"]["low_gravity_range"]), "high": tuple(data["physics"]["high_gravity_range"])},
        gravity_accuracy_threshold=float(data.get("evaluation", {}).get("E3_threshold", 0.002)),
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
    pipe.to(args.device); pipe.load_models_to_device(("dit", "vae")); pipe.pre_encoded_(False)
    cfg.condition_latents = int(training.model.num_condition_frames)
    with (args.dataset_dir / "metadata.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["variant"] == "aligned"]
    selected = [row for row in rows if int(row["base_seed"]) % args.num_shards == args.shard_index]
    outcomes = []
    old_prefix = cfg.short_masked_prefix
    cfg.short_masked_prefix = cfg.prediction_start - args.observed_window
    try:
        for index, row in enumerate(selected, 1):
            condition = load_condition(row, args.dataset_dir, cfg, pipe.torch_dtype, args.device, True)
            frames = generate(pipe, condition, cfg, args.steps, int(row["base_seed"]) + args.seed_offset)
            outcomes.append({"pair_id": row["pair_id"], "condition": "color_sweep", **measure(frames, row, cfg)})
            (args.out / "outcomes.json").write_text(json.dumps(finite(outcomes), indent=2) + "\n", encoding="utf-8")
            print(f"shard={args.shard_index} samples={index}/{len(selected)}", flush=True)
    finally:
        cfg.short_masked_prefix = old_prefix
    (args.out / "summary.json").write_text(json.dumps({"shard": args.shard_index, "num_shards": args.num_shards, "samples": len(outcomes)}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
