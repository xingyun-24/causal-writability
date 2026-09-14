#!/usr/bin/env python3
"""Held-out colour-preserving rollout of band-specific PC2=f(1, conflict g_E3)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from full_token_pca_recovery import load_matrix
from layer_residual_replacement import colour_name, condition_token_count, detect_ball, load_condition, measure
from projectile_block_pca_fit import capture_residual, cfg_from, generate


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("dataset-dir", "data-config", "training-config", "checkpoint", "pair-manifest", "eval-rows", "fit-json", "pca-root", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--block", type=int, default=1)
    parser.add_argument("--observed-window", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    fit = json.loads(args.fit_json.read_text(encoding="utf-8"))
    if fit["basis"] != "[1, conflict generated-video g_E3]":
        raise ValueError("unexpected fit basis")
    holdout = set(fit["holdout_pair_ids"])
    pairs = {item["pair_id"]: item for item in json.loads(args.pair_manifest.read_text(encoding="utf-8"))["pairs"]}
    evaluations = {(str(row["pair_id"]), str(row["condition"])): row for row in json.loads(args.eval_rows.read_text(encoding="utf-8"))}
    device = torch.device(args.device)
    matrix, _, vectors, values, shape = load_matrix(args.pca_root, device)
    data = yaml.safe_load(args.data_config.read_text(encoding="utf-8")); cfg = cfg_from(args.data_config)
    cfg.gravity_bands = {"low": tuple(data["physics"]["low_gravity_range"]), "high": tuple(data["physics"]["high_gravity_range"])}
    cfg.radius = int(data["render"]["ball_radius_px"]); cfg.gravity_accuracy_threshold = float(data["evaluation"]["E3_threshold"])
    from sshv2.wan.config import StandardTrainingConfig
    from sshv2.wan.trainer import WanTrainingModule
    training = StandardTrainingConfig.from_file(args.training_config); training.model.dit.ckpt_file = args.checkpoint
    module = WanTrainingModule(dit_config=training.model.dit, vae_config=training.model.vae, no_encoding=False, num_condition_frames=training.model.num_condition_frames, num_inference_steps=args.steps, pipeline_type=training.model.pipe, pipeline_kwargs=training.model.pipe_kwargs)
    pipe = module.pipe; pipe.to(args.device); pipe.load_models_to_device(("dit", "vae")); pipe.pre_encoded_(False); cfg.condition_latents = int(training.model.num_condition_frames)
    metadata = {(row["pair_id"], row["variant"]): row for row in csv.DictReader((args.dataset_dir / "metadata.csv").open(newline=""))}
    results = []
    for count, pair_id in enumerate(sorted(holdout), 1):
        pair = pairs[pair_id]; band = pair["gravity_interval"]
        coefficient = torch.tensor(fit["bands"][band]["coefficients_intercept_conflict_g_E3"], dtype=torch.float32, device=device)
        conflict = metadata[(pair_id, "conflict")]
        g_conflict = float(evaluations[(pair_id, "conflict")]["g_E3"])
        pc2 = (torch.tensor([1.0, g_conflict], dtype=torch.float32, device=device) @ coefficient) * args.scale
        score = torch.stack((torch.zeros((), device=device), pc2))
        weights = vectors[:, :2] @ (score / torch.sqrt(values[:2].float()))
        delta = (weights.to(matrix.dtype) @ matrix).reshape(shape)
        old_prefix = cfg.short_masked_prefix; cfg.short_masked_prefix = cfg.prediction_start - args.observed_window
        condition = load_condition(conflict, args.dataset_dir, cfg, pipe.torch_dtype, args.device, True); cfg.short_masked_prefix = old_prefix
        seed = int(conflict["base_seed"]) + 17_000_000
        residual = torch.from_numpy(capture_residual(pipe, condition, cfg, args.steps, seed, args.block)).to(device=device)
        call = 0
        def hook(_module, _inputs, hidden):
            nonlocal call
            tokens = condition_token_count(int(hidden.shape[1]), cfg)
            edited = hidden.clone(); edited[:, :tokens] = (residual[call].unsqueeze(0) + delta[call].unsqueeze(0)).to(dtype=hidden.dtype); call += 1
            return edited
        handle = pipe.dit.blocks[args.block].register_forward_hook(hook)
        try: frames = generate(pipe, condition, cfg, args.steps, seed)
        finally: handle.remove()
        _, rgb, _ = detect_ball(frames, type("Config", (), {"radius": cfg.radius, "height": cfg.height})())
        detected = colour_name(rgb); measured = measure(frames, conflict, cfg)
        results.append({"pair_id": pair_id, "band": band, "input_colour": conflict["color_label"], "conflict_g_E3": g_conflict, "predicted_pc2": float(pc2), "detected_colour_majority": detected, **measured})
        print(f"pairs={count}/{len(holdout)}", flush=True)
    changed = [row for row in results if row["detected_colour_majority"] != row["input_colour"]]
    payload = {"method": "separate low/high PC2=f(1, conflict g_E3); PC1 fixed zero; heldout colour-preserving rollout", "scale": args.scale, "pairs": len(results), "E3_accuracy": float(np.mean([row["E3_correct"] for row in results])), "E0_accuracy": float(np.mean([row["E0_correct"] for row in results])), "mean_abs_g_error": float(np.mean([row["gravity_error"] for row in results if row["gravity_error"] is not None])), "colour_changed_pairs": len(changed), "colour_preserved_pairs": len(results)-len(changed), "results": results}
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n"); (args.out / "outcomes.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("scale", "pairs", "E3_accuracy", "E0_accuracy", "mean_abs_g_error", "colour_changed_pairs")}, indent=2))


if __name__ == "__main__":
    main()
