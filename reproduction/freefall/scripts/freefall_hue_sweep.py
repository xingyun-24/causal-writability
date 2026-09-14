#!/usr/bin/env python3
"""Run the frozen Free-fall model over an 11-point red-to-blue cue sweep.

The physical trajectory and generation seed are held fixed within each
history. Only the rendered cue RGB is changed. Short and Long use 32 and 64
observed frames respectively.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import yaml

from audit_v1_estimator import audit_track, finite, generate, make_cfg
from layer_residual_replacement import load_condition


def recolour(frames: np.ndarray, rgb: tuple[int, int, int], background: tuple[int, int, int]) -> np.ndarray:
    out = frames.copy()
    bg = np.asarray(background, dtype=np.int16)
    target = np.asarray(rgb, dtype=np.float32)
    for index, frame in enumerate(out):
        distance = np.linalg.norm(frame.astype(np.int16) - bg[None, None, :], axis=-1)
        mask = distance > 10.0
        alpha = np.clip(distance[mask, None] / 80.0, 0.0, 1.0)
        out[index][mask] = np.rint((1.0 - alpha) * bg + alpha * target).astype(np.uint8)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--pair-limit", type=int, default=64)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("invalid shard")
    args.out.mkdir(parents=True, exist_ok=True)
    raw = yaml.safe_load(args.data_config.read_text(encoding="utf-8"))
    cfg = make_cfg(args.data_config)
    cfg.gravity_bands = {"low": tuple(raw["physics"]["low_gravity_range"]), "high": tuple(raw["physics"]["high_gravity_range"])}
    cfg.radius = int(raw["render"]["ball_radius_px"])
    cfg.gravity_accuracy_threshold = float(raw["evaluation"]["E3_threshold"])
    cfg.condition_latents = 17
    with (args.dataset_dir / "metadata.csv").open(newline="", encoding="utf-8") as handle:
        metadata = list(csv.DictReader(handle))
    by_pair = {}
    for row in metadata:
        if row["variant"] == "aligned":
            by_pair.setdefault(row["pair_id"], row)
    physical = sorted(by_pair.values(), key=lambda row: (row["gravity_interval"], int(row["base_seed"])))
    lows = [row for row in physical if row["gravity_interval"] == "low"][: args.pair_limit // 2]
    highs = [row for row in physical if row["gravity_interval"] == "high"][: args.pair_limit - len(lows)]
    physical = lows + highs
    colours = []
    red = np.asarray(raw["render"]["red_rgb"], dtype=np.float32)
    blue = np.asarray(raw["render"]["blue_rgb"], dtype=np.float32)
    for hue_index in range(11):
        u = hue_index / 10.0
        rgb = np.rint((1.0 - u) * red + u * blue).astype(int).tolist()
        colours.append((hue_index, u, tuple(rgb)))
    jobs = [(window, row, hue_index, u, rgb) for window, label in ((32, "Short"), (64, "Long"))
            for row in physical for hue_index, u, rgb in colours]
    jobs = jobs[args.shard_index::args.num_shards]
    temp = args.out / "sweep_dataset"
    videos_dir = temp / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
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
    results = []
    for count, (window, source, hue_index, u, rgb) in enumerate(jobs, 1):
        source_frames = np.asarray(imageio.mimread(args.dataset_dir / "videos" / source["video"]), dtype=np.uint8)
        video_name = f"{source['pair_id']}_u{hue_index:02d}.mp4"
        video_path = videos_dir / video_name
        if not video_path.exists():
            imageio.mimsave(video_path, recolour(source_frames, rgb, tuple(raw["render"]["background_rgb"])), fps=20, codec="libx264", ffmpeg_params=["-crf", "0", "-pix_fmt", "yuv444p"])
        row = dict(source)
        row["video"] = video_name
        row["color_label"] = "red" if u < 0.5 else "blue"
        cfg.short_masked_prefix = cfg.prediction_start - window
        condition = load_condition(row, temp, cfg, pipe.torch_dtype, args.device, True)
        frames = generate(pipe, condition, cfg, args.steps, int(source["base_seed"]) + 17_000_000)
        measured = audit_track(frames, row, cfg)
        measured["E3_correct"] = bool(measured["g_E3"] is not None and abs(measured["g_E3"] - measured["gravity_true"]) < cfg.gravity_accuracy_threshold)
        measured["E0_correct"] = bool(measured["g_E0_strict"] is not None and abs(measured["g_E0_strict"] - measured["gravity_true"]) < cfg.gravity_accuracy_threshold)
        results.append({"task": "free_fall", "history_regime": "Short" if window == 32 else "Long",
                        "pair_id": source["pair_id"], "base_seed": int(source["base_seed"]), "hue_index": hue_index,
                        "hue_u": u, "input_r": rgb[0], "input_g": rgb[1], "input_b": rgb[2],
                        "physical_band": source["gravity_interval"], "generation_seed": int(source["base_seed"]) + 17_000_000,
                        **finite(measured)})
        if count % 8 == 0 or count == len(jobs):
            (args.out / f"rows_shard{args.shard_index}.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
            print(f"shard={args.shard_index} jobs={count}/{len(jobs)}", flush=True)
    (args.out / f"rows_shard{args.shard_index}.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"shard": args.shard_index, "rows": len(results)}, indent=2))


if __name__ == "__main__":
    main()
