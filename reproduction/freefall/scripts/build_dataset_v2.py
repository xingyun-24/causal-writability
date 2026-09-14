#!/usr/bin/env python3
"""Build a paired Projectile Gravity dataset with a shared state and gravity quantile."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import yaml


@dataclass(frozen=True)
class Config:
    width: int; height: int; fps: int; frames: int; radius: int
    background: tuple[int, int, int]; colors: dict[str, tuple[int, int, int]]
    sim_fps: int; low_g: tuple[float, float]; high_g: tuple[float, float]; gap: float
    xb: tuple[float, float]; yb: tuple[float, float]; vxb: tuple[float, float]; vyb: tuple[float, float]
    boundary: int; prediction_start: int


def read_config(path: Path) -> tuple[Config, str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")); render, physics, history = raw["render"], raw["physics"], raw["history"]
    return Config(
        int(render["width"]), int(render["height"]), int(render["fps"]), int(render["num_frames"]), int(render["ball_radius_px"]),
        tuple(render["background_rgb"]), {"red": tuple(render["red_rgb"]), "blue": tuple(render["blue_rgb"])},
        int(physics["simulation_fps"]), tuple(physics["low_gravity_range"]), tuple(physics["high_gravity_range"]), float(physics["paired_gravity_gap"]),
        tuple(physics["x_boundary_range"]), tuple(physics["y_boundary_range"]), tuple(physics["vx_boundary_range"]), tuple(physics["vy_boundary_range"]),
        int(history["boundary_frame"]), int(history["prediction_start"]),
    ), str(raw["version"])


def state_and_gravity(seed: int, cfg: Config) -> tuple[dict[str, float], float, float, float]:
    rng = np.random.default_rng(seed)
    state = {"x_boundary": float(rng.uniform(*cfg.xb)), "y_boundary": float(rng.uniform(*cfg.yb)), "vx_boundary": float(rng.uniform(*cfg.vxb)), "vy_boundary": float(rng.uniform(*cfg.vyb))}
    quantile = float(rng.uniform())
    low = cfg.low_g[0] + (cfg.low_g[1] - cfg.low_g[0]) * quantile
    high = cfg.high_g[0] + (cfg.high_g[1] - cfg.high_g[0]) * quantile
    if not np.isclose(high - low, cfg.gap, atol=1e-12): raise ValueError("V2 requires same-quantile gravity gap")
    return state, quantile, float(low), float(high)


def trajectory(state: dict[str, float], gravity: float, cfg: Config) -> np.ndarray:
    tau = (np.arange(cfg.frames, dtype=np.float64) - cfg.boundary) / cfg.sim_fps
    return np.column_stack((state["x_boundary"] + state["vx_boundary"] * tau, state["y_boundary"] + state["vy_boundary"] * tau - .5 * gravity * tau**2))


def render(path: np.ndarray, colour: tuple[int, int, int], cfg: Config) -> np.ndarray:
    frames = np.full((cfg.frames, cfg.height, cfg.width, 3), cfg.background, dtype=np.uint8)
    scale = 256
    for frame, (x, y) in zip(frames, path, strict=True):
        centre = (int(round(x * (cfg.width - 1) * scale)), int(round((1.0 - y) * (cfg.height - 1) * scale)))
        cv2.circle(frame, centre, cfg.radius * scale, colour, thickness=-1, lineType=cv2.LINE_AA, shift=8)
    return frames


def write_video(path: Path, frames: np.ndarray, fps: int) -> None:
    imageio.mimsave(path, frames, fps=fps, codec="libx264", ffmpeg_params=["-crf", "0", "-pix_fmt", "yuv444p"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "eval"), required=True); parser.add_argument("--base-seeds", type=int, required=True); parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args(); cfg, dataset_version = read_config(args.config)
    if cfg.boundary != cfg.prediction_start - 1: raise ValueError("boundary frame must be prediction_start - 1")
    root = args.out / args.split
    for name in ("videos", "trajectories", "metadata"): (root / name).mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, object]] = []
    for seed in range(args.seed_offset, args.seed_offset + args.base_seeds):
        state, quantile, low_g, high_g = state_and_gravity(seed, cfg)
        base_pair_id = f"{args.split}_{seed:05d}"
        for interval, colour, gravity in (("low", "red", low_g), ("high", "blue", high_g)):
            physical_pair = f"{base_pair_id}_{interval}"; path = trajectory(state, gravity, cfg)
            margin = cfg.radius / min(cfg.width, cfg.height)
            if np.any(path < margin) or np.any(path > 1.0 - margin): raise RuntimeError(f"{physical_pair}: trajectory leaves renderable area")
            trajectory_name = f"{physical_pair}.npy"; np.save(root / "trajectories" / trajectory_name, path)
            variants = (("aligned", colour),) if args.split == "train" else (("aligned", colour), ("conflict", "blue" if colour == "red" else "red"))
            for variant, rendered_colour in variants:
                sample_id = f"{physical_pair}_{variant}"; video_name = f"{sample_id}.mp4"
                row = {
                    "benchmark_version": dataset_version, "sample_id": sample_id, "pair_id": physical_pair, "base_pair_id": base_pair_id,
                    "base_seed": seed, "split": args.split, "gravity": gravity, "gravity_interval": interval, "gravity_quantile": quantile,
                    "color_label": rendered_colour, "variant": variant, **state, "fps": cfg.fps, "simulation_fps": cfg.sim_fps,
                    "frames": cfg.frames, "boundary_frame": cfg.boundary, "prediction_start": cfg.prediction_start,
                    "position_at_boundary": path[cfg.boundary].tolist(), "velocity_at_boundary": [state["vx_boundary"], state["vy_boundary"]],
                    "video": video_name, "trajectory": trajectory_name,
                    "render_hash_without_ball": hashlib.sha256(np.full((cfg.height, cfg.width, 3), cfg.background, np.uint8).tobytes()).hexdigest(),
                }
                write_video(root / "videos" / video_name, render(path, cfg.colors[rendered_colour], cfg), cfg.fps)
                (root / "metadata" / f"{sample_id}.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8"); rows.append(row)
    with (root / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (root / "dataset.json").write_text(json.dumps({"format": "sshv2.dataset.v1", "dataset": dataset_version, "count": len(rows), "metadata": "metadata.csv"}, indent=2) + "\n", encoding="utf-8")
    print(f"split={args.split} records={len(rows)} root={root}")


if __name__ == "__main__":
    main()
