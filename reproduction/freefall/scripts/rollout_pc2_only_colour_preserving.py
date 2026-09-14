#!/usr/bin/env python3
"""Test PC2-only fitted cross-colour corrections while preserving input colour."""
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


def colour_value(value: str) -> float:
    return {"red": 1.0, "blue": -1.0}[value]


def colour_stats(frames: np.ndarray, radius: int, height: int) -> dict[str, object]:
    probe = type("Config", (), {"radius": radius, "height": height})()
    _, rgb, valid = detect_ball(frames, probe)
    labels = ["unknown" if not valid[i] else ("red" if rgb[i, 0] > rgb[i, 2] else "blue") for i in range(len(frames))]
    return {"detected_colour_majority": colour_name(rgb), "red_frames": labels.count("red"), "blue_frames": labels.count("blue"), "unknown_frames": labels.count("unknown")}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("dataset-dir", "data-config", "training-config", "checkpoint", "pair-manifest", "fit-json", "pca-root", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--block", type=int, default=1)
    parser.add_argument("--observed-window", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    fit = json.loads(args.fit_json.read_text(encoding="utf-8"))
    if fit["feature_order"] != ["delta_color", "delta_color*g", "delta_color*sqrt(g)"]:
        raise ValueError("unexpected fit feature order")
    pairs = json.loads(args.pair_manifest.read_text(encoding="utf-8"))["pairs"]
    device = torch.device(args.device)
    matrix, _, vectors, eigenvalues, shape = load_matrix(args.pca_root, device)
    coefficients = torch.tensor(fit["coefficients_rows_feature_order"], dtype=torch.float32, device=device)
    data = yaml.safe_load(args.data_config.read_text(encoding="utf-8"))
    cfg = cfg_from(args.data_config)
    cfg.gravity_bands = {"low": tuple(data["physics"]["low_gravity_range"]), "high": tuple(data["physics"]["high_gravity_range"])}
    cfg.radius = int(data["render"]["ball_radius_px"]); cfg.gravity_accuracy_threshold = float(data["evaluation"]["E3_threshold"])
    from sshv2.wan.config import StandardTrainingConfig
    from sshv2.wan.trainer import WanTrainingModule
    training = StandardTrainingConfig.from_file(args.training_config); training.model.dit.ckpt_file = args.checkpoint
    module = WanTrainingModule(dit_config=training.model.dit, vae_config=training.model.vae, no_encoding=False, num_condition_frames=training.model.num_condition_frames, num_inference_steps=args.steps, pipeline_type=training.model.pipe, pipeline_kwargs=training.model.pipe_kwargs)
    pipe = module.pipe; pipe.to(args.device); pipe.load_models_to_device(("dit", "vae")); pipe.pre_encoded_(False); cfg.condition_latents = int(training.model.num_condition_frames)
    metadata = {(row["pair_id"], row["variant"]): row for row in csv.DictReader((args.dataset_dir / "metadata.csv").open(newline=""))}
    results = []
    for count, pair in enumerate(pairs, 1):
        pair_id = pair["pair_id"]; conflict = metadata[(pair_id, "conflict")]; aligned = metadata[(pair_id, "aligned")]
        g = float(pair["gravity_true"]); c_current = colour_value(conflict["color_label"]); c_target = colour_value(aligned["color_label"])
        # Predict the fitted cross-colour difference at the same physical g, then retain PC2 only.
        feature = torch.tensor([c_target - c_current, (c_target - c_current) * g, (c_target - c_current) * np.sqrt(g)], dtype=torch.float32, device=device)
        fitted_cross_score = feature @ coefficients
        score = torch.stack((torch.zeros((), device=device), fitted_cross_score[1] * args.scale))
        weights = vectors[:, :2] @ (score / torch.sqrt(eigenvalues[:2].float()))
        delta = (weights.to(matrix.dtype) @ matrix).reshape(shape)
        old_prefix = cfg.short_masked_prefix; cfg.short_masked_prefix = cfg.prediction_start - args.observed_window
        condition = load_condition(conflict, args.dataset_dir, cfg, pipe.torch_dtype, args.device, True); cfg.short_masked_prefix = old_prefix
        seed = int(conflict["base_seed"]) + 17_000_000
        current_residual = torch.from_numpy(capture_residual(pipe, condition, cfg, args.steps, seed, args.block)).to(device=device)
        call = 0
        def hook(_module, _inputs, hidden):
            nonlocal call
            tokens = condition_token_count(int(hidden.shape[1]), cfg)
            edited = hidden.clone(); edited[:, :tokens] = (current_residual[call].unsqueeze(0) + delta[call].unsqueeze(0)).to(dtype=hidden.dtype); call += 1
            return edited
        handle = pipe.dit.blocks[args.block].register_forward_hook(hook)
        try: frames = generate(pipe, condition, cfg, args.steps, seed)
        finally: handle.remove()
        colours = colour_stats(frames, cfg.radius, cfg.height)
        results.append({"pair_id": pair_id, "input_colour": conflict["color_label"], "scale": args.scale, "pc2_score": float(score[1]), **colours, **measure(frames, conflict, cfg)})
        print(f"pairs={count}/{len(pairs)}", flush=True)
    changed = [row for row in results if row["detected_colour_majority"] != row["input_colour"]]
    payload = {"method": "PC2-only component of fitted cross-colour c*[1,g,sqrt(g)] correction; PC1 fixed to zero", "scale": args.scale, "pairs": len(results), "E3_accuracy": float(np.mean([r["E3_correct"] for r in results])), "E0_accuracy": float(np.mean([r["E0_correct"] for r in results])), "mean_abs_g_error": float(np.mean([r["gravity_error"] for r in results if r["gravity_error"] is not None])), "colour_changed_pairs": len(changed), "colour_preserved_pairs": len(results)-len(changed), "results": results}
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2)+"\n"); (args.out / "outcomes.json").write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps({k: payload[k] for k in ("scale","pairs","E3_accuracy","E0_accuracy","mean_abs_g_error","colour_changed_pairs")}, indent=2))


if __name__ == "__main__":
    main()
