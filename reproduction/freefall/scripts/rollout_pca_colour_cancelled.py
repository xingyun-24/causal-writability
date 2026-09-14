#!/usr/bin/env python3
"""Roll out a PCA residual fit with the colour term explicitly cancelled."""
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


def future_colour_stats(frames: np.ndarray, radius: int, height: int) -> dict[str, object]:
    cfg = type("Config", (), {"radius": radius, "height": height})()
    _, rgb, valid = detect_ball(frames, cfg)
    labels = ["unknown" if not valid[index] else ("red" if rgb[index, 0] > rgb[index, 2] else "blue") for index in range(len(frames))]
    return {"detected_colour_majority": colour_name(rgb), "red_frames": labels.count("red"),
            "blue_frames": labels.count("blue"), "unknown_frames": labels.count("unknown")}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("dataset-dir", "data-config", "training-config", "checkpoint", "eval-rows", "fit-json", "pca-root", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--block", type=int, default=2)
    parser.add_argument("--observed-window", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--pc2-energy-scale", action="store_true", help="Scale only PC2 by sqrt(total PCA energy / PC2 energy)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    matrix, _, vectors, eigenvalues, shape = load_matrix(args.pca_root, device)
    fit = json.loads(args.fit_json.read_text(encoding="utf-8"))
    expected = ["aligned_colour_minus_conflict_colour", "aligned_g_E3_minus_conflict_g_E3"]
    if fit["feature_order"] != expected:
        raise ValueError("Unexpected fit feature order")
    coefficients = torch.tensor(fit["coefficients_rows_feature_order"], dtype=torch.float32, device=device)
    pc2_scale = (torch.sqrt(eigenvalues.sum() / eigenvalues[1]).float()
                 if args.pc2_energy_scale else torch.ones((), device=device))
    rows = json.loads(args.eval_rows.read_text(encoding="utf-8"))
    outputs = {(str(row["pair_id"]), str(row["condition"])): row for row in rows}
    selected = [pair_id for index, pair_id in enumerate(fit["holdout_pair_ids"]) if index % args.num_shards == args.shard_index]
    data = yaml.safe_load(args.data_config.read_text(encoding="utf-8"))
    cfg = cfg_from(args.data_config)
    cfg.gravity_bands = {"low": tuple(data["physics"]["low_gravity_range"]), "high": tuple(data["physics"]["high_gravity_range"])}
    cfg.radius = int(data["render"]["ball_radius_px"])
    cfg.gravity_accuracy_threshold = float(data["evaluation"]["E3_threshold"])
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
        aligned = outputs[(pair_id, "aligned")]
        current = outputs[(pair_id, "conflict")]
        observed_delta_g = float(aligned["g_E3"]) - float(current["g_E3"])
        # Same-colour recovery: target colour equals current colour, so delta_colour is exactly zero.
        scores = torch.tensor([0.0, observed_delta_g], device=device) @ coefficients
        scores[1] *= pc2_scale
        weights = vectors[:, :2] @ (scores / torch.sqrt(eigenvalues[:2].float()))
        delta = (weights.to(matrix.dtype) @ matrix).reshape(shape)
        conflict = metadata[(pair_id, "conflict")]
        old_prefix = cfg.short_masked_prefix; cfg.short_masked_prefix = cfg.prediction_start - args.observed_window
        condition = load_condition(conflict, args.dataset_dir, cfg, pipe.torch_dtype, args.device, True)
        cfg.short_masked_prefix = old_prefix
        seed = int(conflict["base_seed"]) + 17_000_000
        current_residual = torch.from_numpy(capture_residual(pipe, condition, cfg, args.steps, seed, args.block)).to(device=device)
        call = 0
        def hook(_module, _inputs, hidden):
            nonlocal call
            tokens = condition_token_count(int(hidden.shape[1]), cfg)
            edited = hidden.clone()
            edited[:, :tokens] = (current_residual[call].unsqueeze(0) + delta[call].unsqueeze(0)).to(dtype=hidden.dtype)
            call += 1
            return edited
        handle = pipe.dit.blocks[args.block].register_forward_hook(hook)
        try:
            frames = generate(pipe, condition, cfg, args.steps, seed)
        finally:
            handle.remove()
        colours = future_colour_stats(frames, cfg.radius, cfg.height)
        results.append({"pair_id": pair_id, "input_colour": current["detected_colour"],
            "observed_delta_g_E3": observed_delta_g, "predicted_scores": scores.detach().cpu().tolist(),
            **colours, **measure(frames, conflict, cfg)})
        print(f"pairs={count}/{len(selected)}", flush=True)
    changed = [row for row in results if row["detected_colour_majority"] != row["input_colour"]]
    payload = {"method": "rank-2 PCA score fit from observed aligned-conflict colour and E3-g differences; rollout fixes delta_colour=0", "pairs": len(results),
        "pc2_energy_scale": float(pc2_scale),
        "E3_accuracy": float(np.mean([row["E3_correct"] for row in results])), "E0_accuracy": float(np.mean([row["E0_correct"] for row in results])),
        "colour_changed_pairs": len(changed), "colour_preserved_pairs": len(results) - len(changed), "results": results}
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (args.out / "outcomes.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("pairs", "E3_accuracy", "E0_accuracy", "colour_changed_pairs", "colour_preserved_pairs")}, indent=2))


if __name__ == "__main__":
    main()
