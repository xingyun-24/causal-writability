#!/usr/bin/env python3
"""Run the selected no-boundary [1,g] controller as a real activation edit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml

from full_token_pca_recovery import load_matrix
from layer_residual_replacement import condition_token_count
from layer_residual_replacement import load_condition, measure
from projectile_block_pca_fit import capture_residual, cfg_from, generate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def colour_stats(frames: np.ndarray, cfg) -> dict:
    from layer_residual_replacement import detect_ball, colour_name
    _, rgb, valid = detect_ball(frames, cfg)
    return {"detected_colour_majority": colour_name(rgb), "detected_frames": int(valid.sum())}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("dataset-dir", "data-config", "training-config", "checkpoint", "eval-rows", "pca-root", "controller-json", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--block", type=int, default=1)
    parser.add_argument("--observed-window", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--split", choices=("train", "holdout"), required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    controller = json.loads(args.controller_json.read_text(encoding="utf-8"))
    if controller.get("controller") != "no_boundary_state_[1,g]":
        raise ValueError("controller JSON is not the frozen [1,g] controller")
    pair_ids = [str(x) for x in controller["pair_ids"]]
    gravity = np.asarray(controller["gravity_true"], dtype=np.float32)
    coeff = np.asarray(controller["coefficients_rows_feature_order"], dtype=np.float32)
    split_map = {pid: str(s) for pid, s in zip(pair_ids, controller["split_labels"])}
    wanted_split = "heldout" if args.split == "holdout" else "train"
    selected = [pid for pid in pair_ids if split_map[pid] == wanted_split][args.shard_index::args.num_shards]
    args.out.mkdir(parents=True, exist_ok=True)
    matrix, _, vectors, eigenvalues, shape = load_matrix(args.pca_root, torch.device(args.device))
    raw = yaml.safe_load(args.data_config.read_text(encoding="utf-8"))
    cfg = cfg_from(args.data_config)
    cfg.gravity_bands = {"low": tuple(raw["physics"]["low_gravity_range"]), "high": tuple(raw["physics"]["high_gravity_range"])}
    cfg.radius = int(raw["render"]["ball_radius_px"]); cfg.gravity_accuracy_threshold = float(raw["evaluation"]["E3_threshold"])
    from sshv2.wan.config import StandardTrainingConfig
    from sshv2.wan.trainer import WanTrainingModule
    training = StandardTrainingConfig.from_file(args.training_config); training.model.dit.ckpt_file = args.checkpoint
    module = WanTrainingModule(dit_config=training.model.dit, vae_config=training.model.vae, no_encoding=False,
        num_condition_frames=training.model.num_condition_frames, num_inference_steps=args.steps,
        pipeline_type=training.model.pipe, pipeline_kwargs=training.model.pipe_kwargs)
    pipe = module.pipe; pipe.to(args.device); pipe.load_models_to_device(("dit", "vae")); pipe.pre_encoded_(False)
    cfg.condition_latents = int(training.model.num_condition_frames)
    eval_rows = {(str(row["pair_id"]), str(row["condition"])): row for row in json.loads(args.eval_rows.read_text(encoding="utf-8"))}
    metadata = {(row["pair_id"], row["variant"]): row for row in csv.DictReader((args.dataset_dir / "metadata.csv").open(newline="", encoding="utf-8"))}
    out_videos = args.out / "videos"; out_videos.mkdir(exist_ok=True)
    results = []
    for count, pid in enumerate(selected, 1):
        index = pair_ids.index(pid); g = float(gravity[index])
        x = torch.tensor([1.0, g], device=args.device, dtype=torch.float32)
        score = x @ torch.from_numpy(coeff).to(args.device)
        weights = vectors[:, :2] @ (score / torch.sqrt(eigenvalues[:2].float()))
        delta = (weights.to(matrix.dtype) @ matrix).reshape(shape)
        conflict = metadata[(pid, "conflict")]; aligned_eval = eval_rows[(pid, "aligned")]; conflict_eval = eval_rows[(pid, "conflict")]
        old_prefix = cfg.short_masked_prefix; cfg.short_masked_prefix = cfg.prediction_start - args.observed_window
        condition = load_condition(conflict, args.dataset_dir, cfg, pipe.torch_dtype, args.device, True); cfg.short_masked_prefix = old_prefix
        seed = int(conflict["base_seed"]) + 17_000_000
        current = torch.from_numpy(capture_residual(pipe, condition, cfg, args.steps, seed, args.block)).to(args.device)
        call = 0
        def hook(_module, _inputs, hidden):
            nonlocal call
            tokens = condition_token_count(int(hidden.shape[1]), cfg)
            edited = hidden.clone(); edited[:, :tokens] = (current[call].to(hidden.device) + delta[call].unsqueeze(0)).to(hidden.dtype)
            call += 1; return edited
        handle = pipe.dit.blocks[args.block].register_forward_hook(hook)
        try: frames = generate(pipe, condition, cfg, args.steps, seed)
        finally: handle.remove()
        measured = measure(frames, conflict, cfg); path = out_videos / f"{pid}.mp4"
        imageio.mimsave(path, frames, fps=20, codec="libx264", ffmpeg_params=["-crf", "0", "-pix_fmt", "yuv444p"])
        denom = float(aligned_eval["g_E3"]) - float(conflict_eval["g_E3"]); edited_g = measured.get("gravity_hat")
        ratio = None if edited_g is None or abs(denom) < 1e-12 else (float(edited_g) - float(conflict_eval["g_E3"])) / denom
        results.append({"receiver_id": pid, "condition": "fit_only_state_controller_[1,g]", "fit_split": args.split,
                        "direction": conflict["gravity_interval"], "gravity_hat_conflict": float(conflict_eval["g_E3"]),
                        "gravity_hat_aligned": float(aligned_eval["g_E3"]), "gravity_hat_edited": edited_g,
                        "recovery_denominator": denom, "normalized_recovery_g": ratio,
                        "valid": measured.get("valid", False), "invalid_reasons": "" if measured.get("valid", False) else "invalid_track",
                        "outcome": "E3_pass" if measured.get("E3_correct", False) else "E3_fail",
                        "future_color_coordinate": measured.get("detected_colour_majority", ""), "intervention_site": args.block,
                        "fm_calls_observed": args.steps, "fm_calls_expected": args.steps,
                        "intervention_scale": 1.0, "activation_delta_l2": float(delta.float().norm().item()),
                        "output_video_path": str(path), "output_video_sha256": sha256(path), **measured})
        print(f"split={args.split} shard={args.shard_index} pairs={count}/{len(selected)}", flush=True)
    (args.out / f"outcomes_{args.split}_shard{args.shard_index}.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"split": args.split, "shard": args.shard_index, "pairs": len(results)}, indent=2))


if __name__ == "__main__":
    main()
