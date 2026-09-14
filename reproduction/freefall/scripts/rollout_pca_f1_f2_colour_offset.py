#!/usr/bin/env python3
"""Roll out the fitted f(aligned)-f(conflict) PCA residual correction."""
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
    labels = ["unknown" if not valid[index] else ("red" if rgb[index, 0] > rgb[index, 2] else "blue")
              for index in range(len(frames))]
    return {"detected_colour_majority": colour_name(rgb), "red_frames": labels.count("red"),
            "blue_frames": labels.count("blue"), "unknown_frames": labels.count("unknown")}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("dataset-dir", "data-config", "training-config", "checkpoint", "eval-rows", "fit-json", "pca-root", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--block", type=int, default=2)
    parser.add_argument("--observed-window", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--split", choices=("train", "holdout"), required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")

    fit = json.loads(args.fit_json.read_text(encoding="utf-8"))
    if not str(fit.get("basis", "")).endswith("fit f_aligned-f_conflict"):
        raise ValueError("Fit JSON does not use the required f(aligned)-f(conflict) basis")
    pair_ids = [str(value) for value in fit["pair_ids"]]
    fitted_scores = np.asarray(fit["fitted_scores"], dtype=np.float32)
    if fitted_scores.shape != (len(pair_ids), int(fit["components"])):
        raise ValueError("fitted PCA score shape does not match pair IDs")
    holdout_ids = {str(value) for value in fit["holdout_pair_ids"]}
    selected_all = [(pair_id, score) for pair_id, score in zip(pair_ids, fitted_scores)
                    if (pair_id in holdout_ids) == (args.split == "holdout")]
    selected = selected_all[args.shard_index::args.num_shards]

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    matrix, _, vectors, eigenvalues, shape = load_matrix(args.pca_root, device)
    components = int(fit["components"])
    if components > vectors.shape[1]:
        raise ValueError("Fit requests unavailable PCA components")
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
    eval_rows = {(str(row["pair_id"]), str(row["condition"])): row
                 for row in json.loads(args.eval_rows.read_text(encoding="utf-8-sig"))}
    metadata = {(row["pair_id"], row["variant"]): row
                for row in csv.DictReader((args.dataset_dir / "metadata.csv").open(newline=""))}
    results = []
    for count, (pair_id, score) in enumerate(selected, 1):
        conflict = metadata[(pair_id, "conflict")]
        aligned_eval = eval_rows[(pair_id, "aligned")]
        conflict_eval = eval_rows[(pair_id, "conflict")]
        score_tensor = torch.from_numpy(score).to(device)
        weights = vectors[:, :components] @ (score_tensor / torch.sqrt(eigenvalues[:components].float()))
        delta = (weights.to(matrix.dtype) @ matrix).reshape(shape)
        old_prefix = cfg.short_masked_prefix
        cfg.short_masked_prefix = cfg.prediction_start - args.observed_window
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
        edited = measure(frames, conflict, cfg)
        results.append({
            "pair_id": pair_id, "fit_split": args.split, "input_colour": conflict_eval["detected_colour"],
            "target_colour": aligned_eval["detected_colour"], "baseline_conflict_g_E3": conflict_eval["g_E3"],
            "baseline_aligned_g_E3": aligned_eval["g_E3"], "predicted_scores": score.tolist(),
            **colours, **edited,
        })
        print(f"split={args.split} pairs={count}/{len(selected)}", flush=True)
    changed = [row for row in results if row["detected_colour_majority"] != row["input_colour"]]
    payload = {
        "method": "f(aligned)-f(conflict) fitted residual-PCA correction injected into captured conflict block residual",
        "fit_basis": fit["basis"], "fit_split": args.split, "components": components,
        "pairs": len(results), "E3_accuracy": float(np.mean([row["E3_correct"] for row in results])),
        "E0_accuracy": float(np.mean([row["E0_correct"] for row in results])),
        "mean_abs_g_error": float(np.mean([row["gravity_error"] for row in results])),
        "colour_changed_pairs": len(changed), "colour_preserved_pairs": len(results) - len(changed), "results": results,
    }
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (args.out / "outcomes.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("fit_split", "pairs", "E3_accuracy", "E0_accuracy", "mean_abs_g_error", "colour_changed_pairs")}, indent=2))


if __name__ == "__main__":
    main()
