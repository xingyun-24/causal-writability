#!/usr/bin/env python3
"""Audit Projectile Gravity fits with nested boundary-state models E0--E3."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import yaml

from layer_residual_replacement import finite, generate, load_condition, read_pairs


def numbers(row: dict[str, str]) -> tuple[float, float, float, float]:
    position = json.loads(row["position_at_boundary"])
    velocity = json.loads(row["velocity_at_boundary"])
    return float(position[0]), float(position[1]), float(velocity[0]), float(velocity[1])


def track_ball(frames: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.full(len(frames), np.nan, dtype=np.float64)
    y = np.full(len(frames), np.nan, dtype=np.float64)
    rgb = np.full((len(frames), 3), np.nan, dtype=np.float64)
    previous: tuple[float, float] | None = None
    for index, frame in enumerate(np.asarray(frames, dtype=np.uint8)):
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask = ((hsv[..., 1] >= 45) & (hsv[..., 2] >= 55)).astype(np.uint8)
        count, labels, stats, centres = cv2.connectedComponentsWithStats(mask, connectivity=8)
        candidates: list[tuple[float, int]] = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if not 28 <= area <= 420:
                continue
            cx, cy = centres[label]
            temporal = 0.0 if previous is None else .2 * math.hypot(cx - previous[0], cy - previous[1])
            candidates.append((.02 * abs(area - math.pi * radius**2) + temporal, label))
        if not candidates:
            continue
        _, label = min(candidates)
        cx, cy = centres[label]
        component = labels == label
        x[index] = cx / (frame.shape[1] - 1)
        y[index] = 1.0 - cy / (frame.shape[0] - 1)
        rgb[index] = frame[component].mean(axis=0)
        previous = (float(cx), float(cy))
    return x, y, rgb


def rmse(observed: np.ndarray, predicted: np.ndarray, valid: np.ndarray) -> float | None:
    if int(valid.sum()) < 6:
        return None
    return float(np.sqrt(np.mean((observed[valid] - predicted[valid]) ** 2)))


def solve(design: np.ndarray, target: np.ndarray, valid: np.ndarray) -> np.ndarray | None:
    if int(valid.sum()) < design.shape[1] + 3:
        return None
    return np.linalg.lstsq(design[valid], target[valid], rcond=None)[0]


def audit_track(frames: np.ndarray, row: dict[str, str], cfg: Any) -> dict[str, Any]:
    x, y, rgb = track_ball(frames, cfg.radius)
    valid = np.isfinite(x) & np.isfinite(y)
    xb, yb, vxb, vyb = numbers(row)
    time = (np.arange(len(y), dtype=np.float64) + 1.0) / cfg.simulation_fps
    quadratic = -.5 * time**2
    y_base = yb + vyb * time
    x_base = xb + vxb * time
    target_y = y - y_base
    target_x = x - x_base

    # E0: fixed boundary state; E1: delta-y; E2: delta-v; E3: both offsets.
    beta0 = solve(quadratic[:, None], target_y, valid)
    beta1 = solve(np.column_stack((np.ones_like(time), quadratic)), target_y, valid)
    beta2 = solve(np.column_stack((time, quadratic)), target_y, valid)
    beta3 = solve(np.column_stack((np.ones_like(time), time, quadratic)), target_y, valid)
    def y_fit(beta: np.ndarray | None, design: np.ndarray) -> tuple[float | None, float | None]:
        if beta is None:
            return None, None
        return float(beta[-1]), rmse(y, y_base + design @ beta, valid)
    g0, e0 = y_fit(beta0, quadratic[:, None])
    g1, e1 = y_fit(beta1, np.column_stack((np.ones_like(time), quadratic)))
    g2, e2 = y_fit(beta2, np.column_stack((time, quadratic)))
    g3, e3 = y_fit(beta3, np.column_stack((np.ones_like(time), time, quadratic)))

    # The same nested continuation audit for x, which has no acceleration term.
    x0 = rmse(x, x_base, valid)
    bx1 = solve(np.ones((len(time), 1)), target_x, valid)
    bx2 = solve(time[:, None], target_x, valid)
    bx3 = solve(np.column_stack((np.ones_like(time), time)), target_x, valid)
    x1 = None if bx1 is None else rmse(x, x_base + bx1[0], valid)
    x2 = None if bx2 is None else rmse(x, x_base + bx2[0] * time, valid)
    x3 = None if bx3 is None else rmse(x, x_base + bx3[0] + bx3[1] * time, valid)
    detected_colour = "unknown" if not np.isfinite(rgb).all(axis=1).any() else ("red" if np.nanmedian(rgb[:, 0]) > np.nanmedian(rgb[:, 2]) else "blue")
    return {
        "detected_frames": int(valid.sum()), "valid_track": bool(valid.sum() >= 58),
        "detected_colour": detected_colour, "gravity_true": float(row["gravity"]),
        "gravity_interval": row["gravity_interval"], "input_colour": row["color_label"],
        "g_E0_strict": g0, "g_E1_position": g1, "g_E2_velocity": g2, "g_E3": g3,
        "delta_y_E1": None if beta1 is None else float(beta1[0]),
        "delta_v_E2": None if beta2 is None else float(beta2[0]),
        "delta_y_E3": None if beta3 is None else float(beta3[0]),
        "delta_v_E3": None if beta3 is None else float(beta3[1]),
        "y_rmse_E0": e0, "y_rmse_E1": e1, "y_rmse_E2": e2, "y_rmse_E3": e3,
        "x_rmse_E0": x0, "x_rmse_E1": x1, "x_rmse_E2": x2, "x_rmse_E3": x3,
        "delta_x_E3": None if bx3 is None else float(bx3[0]),
        "delta_vx_E3": None if bx3 is None else float(bx3[1]),
    }


def summary(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for condition, subset in sorted(defaultdict(list, {key: [row for row in rows if row["condition"] == key] for key in {row["condition"] for row in rows}}).items()):
        item: dict[str, Any] = {"samples": len(subset), "valid_track_rate": float(np.mean([row["valid_track"] for row in subset]))}
        for key, label in (("g_E0_strict", "E0"), ("g_E3", "E3")):
            errors = [abs(row[key] - row["gravity_true"]) for row in subset if row[key] is not None]
            item[label + "_accuracy"] = float(np.mean([error < threshold for error in errors])) if errors else 0.0
            item[label + "_mae"] = None if not errors else float(np.mean(errors))
        for key in ("y_rmse_E0", "y_rmse_E1", "y_rmse_E2", "y_rmse_E3", "x_rmse_E0", "x_rmse_E1", "x_rmse_E2", "x_rmse_E3"):
            values = [row[key] for row in subset if row[key] is not None]
            item[key + "_mean"] = None if not values else float(np.mean(values))
        output[condition] = item
    return output


def make_cfg(path: Path) -> Any:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    physics = data["physics"]
    # V2 uses low/high names; retain the old aliases so archived V1 results
    # can still be audited with their original configuration.
    bands = {
        "low": tuple(physics.get("low_gravity_range", physics.get("red_gravity_range"))),
        "high": tuple(physics.get("high_gravity_range", physics.get("blue_gravity_range"))),
    }
    return SimpleNamespace(
        width=int(data["render"]["width"]), height=int(data["render"]["height"]), frames=int(data["render"]["num_frames"]),
        radius=int(data["render"]["ball_radius_px"]), background=data["render"]["background_rgb"],
        simulation_fps=int(physics["simulation_fps"]), prediction_start=int(data["history"]["prediction_start"]),
        short_masked_prefix=int(data["history"]["short_masked_prefix_pixels"]), latent_frames=33, condition_latents=17,
        bands=bands, evaluation=data.get("evaluation", {}),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", choices=("ground_truth", "model"), required=True)
    parser.add_argument("--pair-limit", type=int, default=128)
    parser.add_argument("--pair-offset", type=int, default=0, help="zero-based offset for interleaved evaluation shards")
    parser.add_argument("--pair-stride", type=int, default=1, help="stride for interleaved evaluation shards")
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--short-history", action="store_true")
    parser.add_argument("--observed-window", type=int, help="number of real frames immediately before prediction_start; overrides --short-history")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.observed_window is not None:
        if not 1 <= args.observed_window <= 64:
            parser.error("--observed-window must be in [1, 64]")
    if args.pair_offset < 0 or args.pair_stride < 1 or args.pair_offset >= args.pair_stride:
        parser.error("require 0 <= --pair-offset < --pair-stride")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Refusing non-empty output: {args.out}")
    if args.source == "model" and (args.training_config is None or args.checkpoint is None):
        parser.error("--training-config and --checkpoint are required for --source model")
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = make_cfg(args.data_config)
    gravity_threshold = float(cfg.evaluation.get("E3_threshold", 0.002))
    pairs = read_pairs(args.dataset_dir, args.pair_limit)
    pairs = pairs[args.pair_offset::args.pair_stride]
    selected = [(condition, row) for aligned, conflict in pairs for condition, row in (("aligned", aligned), ("conflict", conflict))]
    pipe = None
    if args.source == "model":
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
    rows: list[dict[str, Any]] = []
    for index, (condition_name, row) in enumerate(selected, 1):
        if args.source == "ground_truth":
            frames = np.asarray(imageio.mimread(args.dataset_dir / "videos" / row["video"]), dtype=np.uint8)[cfg.prediction_start:]
        else:
            assert pipe is not None
            if args.observed_window is not None:
                # The training latents preserve only this fixed window before frame 65.
                original_prefix = cfg.short_masked_prefix
                cfg.short_masked_prefix = cfg.prediction_start - args.observed_window
                condition = load_condition(row, args.dataset_dir, cfg, pipe.torch_dtype, args.device, True)
                cfg.short_masked_prefix = original_prefix
            else:
                condition = load_condition(row, args.dataset_dir, cfg, pipe.torch_dtype, args.device, args.short_history)
            frames = generate(pipe, condition, cfg, args.steps, int(row["base_seed"]) + 17_000_000)
        measured = audit_track(frames, row, cfg)
        measured["E3_correct"] = bool(measured["g_E3"] is not None and abs(measured["g_E3"] - measured["gravity_true"]) < gravity_threshold)
        measured["E0_correct"] = bool(measured["g_E0_strict"] is not None and abs(measured["g_E0_strict"] - measured["gravity_true"]) < gravity_threshold)
        rows.append({"pair_id": row["pair_id"], "sample_id": row["sample_id"], "condition": condition_name, "source": args.source, **measured})
        if index % 8 == 0 or index == len(selected): print(f"samples={index}/{len(selected)}", flush=True)
    (args.out / "rows.json").write_text(json.dumps(finite(rows), indent=2) + "\n", encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(finite({
        "source": args.source, "pair_limit": len(pairs), "pair_offset": args.pair_offset, "pair_stride": args.pair_stride, "time_axis": "tau=(frame-64)/simulation_fps",
        "evaluation": {**cfg.evaluation, "E3": "|g_hat - g| < E3_threshold"},
        "summary": summary(rows, float(cfg.evaluation.get("E3_threshold", 0.002))),
    }), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
