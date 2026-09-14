#!/usr/bin/env python3
"""Roll out direction-specific absolute-target [1,g_target] controllers."""
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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-dir", "data-config", "training-config", "checkpoint", "eval-rows", "pca-root", "controller-json", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--block", type=int, default=1)
    parser.add_argument("--observed-window", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pair-id", action="append")
    parser.add_argument("--include-natural", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    controller = json.loads(args.controller_json.read_text(encoding="utf-8"))
    if controller.get("controller") != "direction_specific_absolute_target_[1,g_target]":
        raise ValueError("wrong controller type")
    pair_ids = np.asarray(controller["pair_ids"], dtype=str)
    gravity = np.asarray(controller["gravity_target"], dtype=np.float32)
    directions = np.asarray(controller["direction"], dtype=str)
    splits = np.asarray(controller["split_labels"], dtype=str)
    wanted = np.asarray(args.pair_id if args.pair_id else pair_ids[splits == "heldout"], dtype=str)
    wanted = wanted[args.shard_index::args.num_shards]

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
    pipe.to(args.device)
    pipe.load_models_to_device(("dit", "vae"))
    pipe.pre_encoded_(False)
    cfg.condition_latents = int(training.model.num_condition_frames)
    matrix, _, vectors, eigenvalues, shape = load_matrix(args.pca_root, torch.device(args.device))
    eval_rows = {(str(row["pair_id"]), str(row["condition"])): row for row in json.loads(args.eval_rows.read_text(encoding="utf-8"))}
    metadata = {(row["pair_id"], row["variant"]): row for row in csv.DictReader((args.dataset_dir / "metadata.csv").open(newline="", encoding="utf-8"))}
    (args.out / "videos").mkdir(parents=True, exist_ok=True)
    results = []

    for count, pair_id in enumerate(wanted, 1):
        index = int(np.where(pair_ids == pair_id)[0][0])
        direction, g_target = str(directions[index]), float(gravity[index])
        coefficient = torch.tensor(controller["models"][direction]["coefficients_rows_[1_g_target]"], device=args.device, dtype=torch.float32)
        score = torch.tensor([1.0, g_target], device=args.device) @ coefficient
        weights = vectors[:, :2] @ (score / torch.sqrt(eigenvalues[:2].float()))
        delta = (weights.to(matrix.dtype) @ matrix).reshape(shape)
        conflict = metadata[(pair_id, "conflict")]
        aligned_eval, conflict_eval = eval_rows[(pair_id, "aligned")], eval_rows[(pair_id, "conflict")]
        old_prefix = cfg.short_masked_prefix
        cfg.short_masked_prefix = cfg.prediction_start - args.observed_window
        condition = load_condition(conflict, args.dataset_dir, cfg, pipe.torch_dtype, args.device, True)
        cfg.short_masked_prefix = old_prefix
        seed = int(conflict["base_seed"]) + 17_000_000
        if args.include_natural:
            natural_frames = generate(pipe, condition, cfg, args.steps, seed)
            natural = measure(natural_frames, conflict, cfg)
            natural_video = args.out / "videos" / f"{pair_id}__natural_conflict.mp4"
            imageio.mimsave(natural_video, natural_frames, fps=20, codec="libx264", ffmpeg_params=["-crf", "0", "-pix_fmt", "yuv444p"])
            results.append({
                "receiver_id": pair_id, "condition": "natural_conflict", "target_direction": direction,
                "g_target": g_target, "gravity_hat_conflict": float(conflict_eval["g_E3"]),
                "gravity_hat_aligned": float(aligned_eval["g_E3"]), "gravity_hat_edited": natural.get("gravity_hat"),
                "recovery_denominator": 0.0, "normalized_recovery_g": 0.0,
                "valid": natural.get("valid", False), "outcome": "E3_pass" if natural.get("E3_correct", False) else "E3_fail",
                "future_color_coordinate": natural.get("detected_colour_majority", ""), "intervention_site": args.block,
                "fm_calls": args.steps, "generation_seed": seed, "video": str(natural_video), "sha256": sha256(natural_video), **natural,
            })
        current = torch.from_numpy(capture_residual(pipe, condition, cfg, args.steps, seed, args.block)).to(args.device)
        call = 0

        def hook(_module, _inputs, hidden):
            nonlocal call
            tokens = condition_token_count(int(hidden.shape[1]), cfg)
            edited = hidden.clone()
            edited[:, :tokens] = (current[call] + delta[call].unsqueeze(0)).to(hidden.dtype)
            call += 1
            return edited

        handle = pipe.dit.blocks[args.block].register_forward_hook(hook)
        try:
            frames = generate(pipe, condition, cfg, args.steps, seed)
        finally:
            handle.remove()
        if call != args.steps:
            raise RuntimeError(f"expected {args.steps} hook calls, got {call}")
        measured = measure(frames, conflict, cfg)
        video_name = f"{pair_id}__controller_edit.mp4" if args.include_natural else f"{pair_id}.mp4"
        video = args.out / "videos" / video_name
        imageio.mimsave(video, frames, fps=20, codec="libx264", ffmpeg_params=["-crf", "0", "-pix_fmt", "yuv444p"])
        denominator = float(aligned_eval["g_E3"]) - float(conflict_eval["g_E3"])
        edited_g = measured.get("gravity_hat")
        results.append({
            "receiver_id": pair_id, "condition": "direction_specific_fit_only_[1,g_target]", "target_direction": direction,
            "g_target": g_target, "gravity_hat_conflict": float(conflict_eval["g_E3"]),
            "gravity_hat_aligned": float(aligned_eval["g_E3"]), "gravity_hat_edited": edited_g,
            "recovery_denominator": denominator,
            "normalized_recovery_g": None if edited_g is None else (float(edited_g) - float(conflict_eval["g_E3"])) / denominator,
            "valid": measured.get("valid", False), "outcome": "E3_pass" if measured.get("E3_correct", False) else "E3_fail",
            "future_color_coordinate": measured.get("detected_colour_majority", ""), "intervention_site": args.block,
            "fm_calls": args.steps, "generation_seed": seed, "video": str(video), "sha256": sha256(video), **measured,
        })
        print(f"pairs={count}/{len(wanted)}", flush=True)
    (args.out / f"outcomes_shard{args.shard_index}.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
