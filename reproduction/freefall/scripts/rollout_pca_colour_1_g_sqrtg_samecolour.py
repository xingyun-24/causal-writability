#!/usr/bin/env python3
"""Roll out color*[1,g,sqrt(g)] PCA fit with color held fixed."""
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
    for name in ("dataset-dir", "data-config", "training-config", "checkpoint", "pair-manifest", "eval-rows", "fit-json", "pca-root", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--block", type=int, default=1)
    parser.add_argument("--observed-window", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")
    args.out.mkdir(parents=True, exist_ok=True)
    fit = json.loads(args.fit_json.read_text(encoding="utf-8"))
    if fit["feature_order"] != ["delta_color", "delta_color*g", "delta_color*sqrt(g)"]:
        raise ValueError("unexpected fit feature order")
    pair_ids = [str(v) for v in fit["pair_ids"]]
    coefficients = torch.tensor(fit["coefficients_rows_feature_order"], dtype=torch.float32)
    pairs = {item["pair_id"]: item for item in json.loads(args.pair_manifest.read_text(encoding="utf-8"))["pairs"]}
    eval_rows = {(str(row["pair_id"]), str(row["condition"])): row for row in json.loads(args.eval_rows.read_text(encoding="utf-8"))}
    selected = [(pair_id, pairs[pair_id]) for pair_id in pair_ids][args.shard_index::args.num_shards]

    device = torch.device(args.device)
    matrix, _, vectors, eigenvalues, shape = load_matrix(args.pca_root, device)
    coefficients = coefficients.to(device)
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
    for count, (pair_id, pair) in enumerate(selected, 1):
        conflict_eval = eval_rows[(pair_id, "conflict")]
        conflict = metadata[(pair_id, "conflict")]
        current_colour = colour_value(conflict["color_label"])
        g_current = float(conflict_eval["g_E3"]); g_target = float(pair["gravity_true"])
        feature = torch.tensor([0.0, current_colour * (g_target - g_current), current_colour * (np.sqrt(g_target) - np.sqrt(max(g_current, 0.0)))], dtype=torch.float32, device=device)
        score = feature @ coefficients
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
        measured = measure(frames, conflict, cfg)
        results.append({"pair_id": pair_id, "input_colour": conflict["color_label"], "target_g": g_target, "current_g_E3": g_current, "predicted_scores": score.detach().cpu().tolist(), **colours, **measured})
        print(f"pairs={count}/{len(selected)}", flush=True)
    changed = [row for row in results if row["detected_colour_majority"] != row["input_colour"]]
    payload = {"method": "same-colour rollout of color*[1,g,sqrt(g)] PCA fit; constant term cancelled", "block": args.block, "pairs": len(results), "E3_accuracy": float(np.mean([row["E3_correct"] for row in results])), "E0_accuracy": float(np.mean([row["E0_correct"] for row in results])), "colour_changed_pairs": len(changed), "colour_preserved_pairs": len(results) - len(changed), "results": results}
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n"); (args.out / "outcomes.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("pairs", "E3_accuracy", "E0_accuracy", "colour_changed_pairs", "colour_preserved_pairs")}, indent=2))


if __name__ == "__main__":
    main()
