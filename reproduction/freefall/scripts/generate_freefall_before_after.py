#!/usr/bin/env python3
"""Generate matched Natural-conflict / Controller-edit Free-fall video pairs."""
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
from layer_residual_replacement import condition_token_count, load_condition, measure
from projectile_block_pca_fit import capture_residual, cfg_from, generate


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ("dataset-dir", "data-config", "training-config", "checkpoint", "eval-rows", "pca-root", "controller-json", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--block", type=int, default=1)
    p.add_argument("--observed-window", type=int, default=32)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--pair-id", action="append", required=True)
    args = p.parse_args()
    controller = json.loads(args.controller_json.read_text(encoding="utf-8"))
    if controller.get("controller") != "no_boundary_state_[1,g]":
        raise ValueError("controller JSON is not the frozen [1,g] controller")
    pair_ids = [str(x) for x in controller["pair_ids"]]
    gravity = np.asarray(controller["gravity_true"], dtype=np.float32)
    coeff = np.asarray(controller["coefficients_rows_feature_order"], dtype=np.float32)
    wanted = args.pair_id
    if any(pid not in pair_ids for pid in wanted):
        raise ValueError("requested pair is absent from controller manifest")
    raw = yaml.safe_load(args.data_config.read_text(encoding="utf-8"))
    cfg = cfg_from(args.data_config)
    cfg.gravity_bands = {"low": tuple(raw["physics"]["low_gravity_range"]), "high": tuple(raw["physics"]["high_gravity_range"])}
    cfg.radius = int(raw["render"]["ball_radius_px"])
    cfg.gravity_accuracy_threshold = float(raw["evaluation"]["E3_threshold"])
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
    eval_rows = {(str(row["pair_id"]), str(row["condition"])): row for row in json.loads(args.eval_rows.read_text(encoding="utf-8"))}
    metadata = {(row["pair_id"], row["variant"]): row for row in csv.DictReader((args.dataset_dir / "metadata.csv").open(newline="", encoding="utf-8"))}
    out = args.out; out.mkdir(parents=True, exist_ok=True)
    videos = out / "videos"; videos.mkdir(exist_ok=True)
    registry = []
    for pid in wanted:
        index = pair_ids.index(pid); g = float(gravity[index])
        x = torch.tensor([1.0, g], device=args.device, dtype=torch.float32)
        score = x @ torch.from_numpy(coeff).to(args.device)
        matrix, _, vectors, eigenvalues, shape = load_matrix(args.pca_root, torch.device(args.device))
        weights = vectors[:, :2] @ (score / torch.sqrt(eigenvalues[:2].float()))
        delta = (weights.to(matrix.dtype) @ matrix).reshape(shape)
        conflict = metadata[(pid, "conflict")]
        old_prefix = cfg.short_masked_prefix; cfg.short_masked_prefix = cfg.prediction_start - args.observed_window
        condition = load_condition(conflict, args.dataset_dir, cfg, pipe.torch_dtype, args.device, True)
        cfg.short_masked_prefix = old_prefix
        seed = int(conflict["base_seed"]) + 17_000_000

        natural = generate(pipe, condition, cfg, args.steps, seed)
        natural_path = videos / f"{pid}__natural_conflict.mp4"
        imageio.mimsave(natural_path, natural, fps=20, codec="libx264", ffmpeg_params=["-crf", "0", "-pix_fmt", "yuv444p"])
        natural_measure = measure(natural, conflict, cfg)

        current = torch.from_numpy(capture_residual(pipe, condition, cfg, args.steps, seed, args.block)).to(args.device)
        call = 0
        def hook(_module, _inputs, hidden):
            nonlocal call
            tokens = condition_token_count(int(hidden.shape[1]), cfg)
            edited = hidden.clone()
            edited[:, :tokens] = (current[call].to(hidden.device) + delta[call].unsqueeze(0)).to(hidden.dtype)
            call += 1
            return edited
        handle = pipe.dit.blocks[args.block].register_forward_hook(hook)
        try:
            edited = generate(pipe, condition, cfg, args.steps, seed)
        finally:
            handle.remove()
        if call != args.steps:
            raise RuntimeError(f"expected {args.steps} hook calls, got {call}")
        edited_path = videos / f"{pid}__controller_edit.mp4"
        imageio.mimsave(edited_path, edited, fps=20, codec="libx264", ffmpeg_params=["-crf", "0", "-pix_fmt", "yuv444p"])
        edited_measure = measure(edited, conflict, cfg)
        aligned = eval_rows[(pid, "aligned")]
        conflict_eval = eval_rows[(pid, "conflict")]
        denom = float(aligned["g_E3"]) - float(conflict_eval["g_E3"])
        registry.extend([
            {"pair_id": pid, "condition": "natural_conflict", "block1_controller_write": False,
             "generation_seed": seed, "flow_matching_calls": args.steps, "observed_window": args.observed_window,
             "future_frames": 64, "input_video_colour": conflict["color_label"], "gravity_hat_e3": natural_measure.get("gravity_hat"),
             "gravity_hat_e0": natural_measure.get("gravity_hat_e0"), "valid": natural_measure.get("valid"),
             "video": str(natural_path), "sha256": sha256(natural_path)},
            {"pair_id": pid, "condition": "controller_edit", "block1_controller_write": True,
             "generation_seed": seed, "flow_matching_calls": args.steps, "observed_window": args.observed_window,
             "future_frames": 64, "input_video_colour": conflict["color_label"], "gravity_hat_e3": edited_measure.get("gravity_hat"),
             "gravity_hat_e0": edited_measure.get("gravity_hat_e0"), "valid": edited_measure.get("valid"),
             "normalized_recovery_g": None if abs(denom) < 1e-12 else (float(edited_measure["gravity_hat"]) - float(conflict_eval["g_E3"])) / denom,
             "video": str(edited_path), "sha256": sha256(edited_path)},
        ])
    (out / "video_registry.jsonl").write_text("\n".join(json.dumps(row) for row in registry) + "\n", encoding="utf-8")
    (out / "before_after_manifest.json").write_text(json.dumps({"pair_ids": wanted, "same_condition_history": True,
        "same_generation_seed": True, "same_flow_matching_calls": args.steps, "same_future_window": 64,
        "only_difference": "Block 1 [1,g] controller write", "records": registry}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
