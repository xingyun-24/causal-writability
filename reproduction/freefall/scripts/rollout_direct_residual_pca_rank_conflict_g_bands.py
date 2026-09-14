#!/usr/bin/env python3
"""Held-out direct residual fit, reconstructed through top-k residual PCA directions."""
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
    for name in ("dataset-dir", "data-config", "training-config", "checkpoint", "pair-manifest", "eval-rows", "pca-root", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--components", type=int, required=True)
    parser.add_argument("--block", type=int, default=1)
    parser.add_argument("--observed-window", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    matrix, index_by_pair, vectors, eigenvalues, shape = load_matrix(args.pca_root, device)
    k = args.components
    if not 1 <= k <= vectors.shape[1]: raise ValueError("invalid component count")
    ordered_ids = list(index_by_pair)
    pairs = {item["pair_id"]: item for item in json.loads(args.pair_manifest.read_text(encoding="utf-8"))["pairs"]}
    evaluation = {(str(row["pair_id"]), str(row["condition"])): row for row in json.loads(args.eval_rows.read_text(encoding="utf-8"))}
    # Rows are the unit feature-space PCs of the full aligned-minus-conflict residual matrix.
    directions = (vectors[:, :k].T @ matrix.float()) / torch.sqrt(eigenvalues[:k]).view(-1, 1).float()
    retained_energy = eigenvalues[:k].sum()
    # Match the established PCA-recovery convention: compensate globally for
    # energy omitted by rank truncation, never normalize individual samples.
    energy_scale = torch.ones((), dtype=matrix.dtype, device=device) if k == matrix.shape[0] else torch.sqrt(eigenvalues.sum() / retained_energy).to(matrix.dtype)
    band_indices = {band: [i for i, pid in enumerate(ordered_ids) if pairs[pid]["gravity_interval"] == band] for band in ("low", "high")}
    projected_coefficients: dict[str, torch.Tensor] = {}
    holdout_ids: dict[str, list[str]] = {}
    diagnostics: dict[str, object] = {}
    for band, indices in band_indices.items():
        train, holdout = indices[1::2], indices[::2]
        train_ids = [ordered_ids[i] for i in train]
        x = torch.tensor([[1.0, float(evaluation[(pid, "conflict")]["g_E3"])] for pid in train_ids], dtype=torch.float32, device=device)
        direct_coefficients = torch.linalg.solve(x.T @ x, x.T @ matrix[train].float())
        # Fit is direct in full residual space; only the final applied delta is PCA-truncated.
        projected_coefficients[band] = direct_coefficients @ directions.T
        residual = matrix[train].float() - x @ direct_coefficients
        diagnostics[band] = {"train_pairs": len(train), "holdout_pairs": len(holdout), "full_residual_fit_element_rmse": float(torch.sqrt(torch.mean(residual ** 2))), "retained_pca_energy": float(retained_energy / eigenvalues.sum())}
        holdout_ids[band] = [ordered_ids[i] for i in holdout]
    data = yaml.safe_load(args.data_config.read_text(encoding="utf-8")); cfg = cfg_from(args.data_config)
    cfg.gravity_bands = {"low": tuple(data["physics"]["low_gravity_range"]), "high": tuple(data["physics"]["high_gravity_range"])}
    cfg.radius = int(data["render"]["ball_radius_px"]); cfg.gravity_accuracy_threshold = float(data["evaluation"]["E3_threshold"])
    from sshv2.wan.config import StandardTrainingConfig
    from sshv2.wan.trainer import WanTrainingModule
    training = StandardTrainingConfig.from_file(args.training_config); training.model.dit.ckpt_file = args.checkpoint
    module = WanTrainingModule(dit_config=training.model.dit, vae_config=training.model.vae, no_encoding=False, num_condition_frames=training.model.num_condition_frames, num_inference_steps=args.steps, pipeline_type=training.model.pipe, pipeline_kwargs=training.model.pipe_kwargs)
    pipe = module.pipe; pipe.to(args.device); pipe.load_models_to_device(("dit", "vae")); pipe.pre_encoded_(False); cfg.condition_latents = int(training.model.num_condition_frames)
    metadata = {(row["pair_id"], row["variant"]): row for row in csv.DictReader((args.dataset_dir / "metadata.csv").open(newline=""))}
    selected = [(pid, band) for band in ("low", "high") for pid in holdout_ids[band]]
    results = []
    for count, (pair_id, band) in enumerate(selected, 1):
        conflict = metadata[(pair_id, "conflict")]
        g_conflict = float(evaluation[(pair_id, "conflict")]["g_E3"])
        score = torch.tensor([1.0, g_conflict], dtype=torch.float32, device=device) @ projected_coefficients[band]
        delta = ((score @ directions).to(matrix.dtype) * energy_scale).reshape(shape)
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
        results.append({"pair_id": pair_id, "band": band, "input_colour": conflict["color_label"], "conflict_g_E3": g_conflict, "detected_colour_majority": detected, **measured})
        print(f"pairs={count}/{len(selected)}", flush=True)
    changed = [row for row in results if row["detected_colour_majority"] != row["input_colour"]]
    payload = {"method": "direct full residual fit [1, conflict g_E3] per band, then globally energy-scaled PCA truncation before injection", "components": k, "energy_scale": float(energy_scale), "pairs": len(results), "band_diagnostics": diagnostics, "E3_accuracy": float(np.mean([row["E3_correct"] for row in results])), "E0_accuracy": float(np.mean([row["E0_correct"] for row in results])), "mean_abs_g_error": float(np.mean([row["gravity_error"] for row in results if row["gravity_error"] is not None])), "colour_changed_pairs": len(changed), "colour_preserved_pairs": len(results)-len(changed), "results": results}
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n"); (args.out / "outcomes.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("components", "pairs", "E3_accuracy", "E0_accuracy", "mean_abs_g_error", "colour_changed_pairs")}, indent=2))


if __name__ == "__main__":
    main()
