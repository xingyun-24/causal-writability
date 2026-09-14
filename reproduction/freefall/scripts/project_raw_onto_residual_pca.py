#!/usr/bin/env python3
"""Project raw aligned/conflict block outputs onto residual-PCA directions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from full_token_pca_recovery import load_matrix
from layer_residual_replacement import condition_token_count, load_condition
from projectile_block_pca_fit import capture_residual, cfg_from


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("dataset-dir", "data-config", "training-config", "checkpoint", "eval-rows", "pca-root", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--block", type=int, default=2)
    parser.add_argument("--observed-window", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    matrix, index_by_pair, vectors, eigenvalues, shape = load_matrix(args.pca_root, device)
    if args.block != 2:
        raise ValueError("This PCA archive is for block 2")
    # D has rows indexed by pair. D^T u_k / sqrt(lambda_k) is the unit feature-space PC direction.
    directions = (vectors[:, :3].T @ matrix) / torch.sqrt(eigenvalues[:3]).view(-1, 1).float()
    directions = directions.float()
    rows = json.loads(args.eval_rows.read_text(encoding="utf-8"))
    measured = {(str(row["pair_id"]), str(row["condition"])): row for row in rows}
    pca_ids = [str(pair_id) for pair_id in np.load(args.pca_root / "decomposition" / "sample_pca.npz")["pair_ids"]]
    selected = [pair_id for index, pair_id in enumerate(pca_ids) if index % args.num_shards == args.shard_index]
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
    results = []
    for count, pair_id in enumerate(selected, 1):
        for condition_name in ("aligned", "conflict"):
            row = metadata[(pair_id, condition_name)]
            old_prefix = cfg.short_masked_prefix; cfg.short_masked_prefix = cfg.prediction_start - args.observed_window
            condition = load_condition(row, args.dataset_dir, cfg, pipe.torch_dtype, args.device, True)
            cfg.short_masked_prefix = old_prefix
            seed = int(row["base_seed"]) + 17_000_000
            activation = torch.from_numpy(capture_residual(pipe, condition, cfg, args.steps, seed, args.block)).to(device=device)
            scores = activation.reshape(1, -1) @ directions.T
            scores = scores[0]
            observed = measured[(pair_id, condition_name)]
            results.append({"pair_id": pair_id, "condition": condition_name, "input_colour": observed["input_colour"],
                "detected_colour": observed.get("detected_colour"), "g_E3": float(observed["g_E3"]),
                "pc1": float(scores[0]), "pc2": float(scores[1]), "pc3": float(scores[2])})
        print(f"pairs={count}/{len(selected)}", flush=True)
    (args.out / "projections.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pairs": len(selected), "points": len(results), "components": 3}, indent=2))


if __name__ == "__main__":
    main()
