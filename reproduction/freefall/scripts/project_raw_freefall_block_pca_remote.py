#!/usr/bin/env python3
"""Project raw aligned/conflict residuals onto the PCA directions of their difference."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from full_token_pca_recovery import load_matrix
from layer_residual_replacement import load_condition
from projectile_block_pca_fit import capture_residual, cfg_from


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("dataset-dir", "data-config", "training-config", "checkpoint", "pair-manifest", "pca-root", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--observed-window", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    # D^T u_k / sqrt(lambda_k) is the feature-space PC direction.
    matrix, index_by_pair, vectors, eigenvalues, _ = load_matrix(args.pca_root, device)
    directions = (vectors[:, :2].T @ matrix) / torch.sqrt(eigenvalues[:2]).view(-1, 1).float()
    directions = directions.float()
    data = yaml.safe_load(args.data_config.read_text(encoding="utf-8"))
    cfg = cfg_from(args.data_config)
    cfg.radius = int(data["render"]["ball_radius_px"])
    from sshv2.wan.config import StandardTrainingConfig
    from sshv2.wan.trainer import WanTrainingModule
    training = StandardTrainingConfig.from_file(args.training_config)
    training.model.dit.ckpt_file = args.checkpoint
    module = WanTrainingModule(dit_config=training.model.dit, vae_config=training.model.vae, no_encoding=False,
        num_condition_frames=training.model.num_condition_frames, num_inference_steps=args.steps,
        pipeline_type=training.model.pipe, pipeline_kwargs=training.model.pipe_kwargs)
    pipe = module.pipe
    pipe.to(args.device); pipe.load_models_to_device(("dit", "vae")); pipe.pre_encoded_(False)
    cfg.condition_latents = int(training.model.num_condition_frames)
    metadata = {(row["pair_id"], row["variant"]): row for row in csv.DictReader((args.dataset_dir / "metadata.csv").open(newline=""))}
    pairs = json.loads(args.pair_manifest.read_text(encoding="utf-8"))["pairs"]
    results = []
    for count, item in enumerate(pairs, 1):
        pair_id = item["pair_id"]
        for variant in ("aligned", "conflict"):
            row = metadata[(pair_id, variant)]
            old_prefix = cfg.short_masked_prefix
            cfg.short_masked_prefix = cfg.prediction_start - args.observed_window
            condition = load_condition(row, args.dataset_dir, cfg, pipe.torch_dtype, args.device, True)
            cfg.short_masked_prefix = old_prefix
            seed = int(row["base_seed"]) + 17_000_000
            activation = torch.from_numpy(capture_residual(pipe, condition, cfg, args.steps, seed, args.block)).to(device=device)
            scores = (activation.reshape(1, -1) @ directions.T)[0]
            results.append({
                "pair_id": pair_id, "condition": variant, "gravity_true": float(item["gravity_true"]),
                "gravity_interval": item.get("gravity_interval", row.get("gravity_band", "")),
                "input_colour": row.get("colour", row.get("input_colour", "")),
                "pc1": float(scores[0]), "pc2": float(scores[1]),
            })
        print(f"pairs={count}/{len(pairs)}", flush=True)
    (args.out / "projections.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pairs": len(pairs), "points": len(results), "components": 2}, indent=2))


if __name__ == "__main__":
    main()
