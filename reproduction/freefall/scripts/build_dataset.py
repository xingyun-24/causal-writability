#!/usr/bin/env python3
"""Build continuous-gravity projectile videos with paired colour conflicts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import yaml


@dataclass(frozen=True)
class Config:
    width: int; height: int; fps: int; frames: int; radius: int
    background: tuple[int, int, int]; colors: dict[str, tuple[int, int, int]]
    red_g: tuple[float, float]; blue_g: tuple[float, float]; simulation_fps: int
    x0: tuple[float, float]; y0: tuple[float, float]; vx: tuple[float, float]; vy: tuple[float, float]
    prediction_start: int


def config(path: Path) -> Config:
    d = yaml.safe_load(path.read_text(encoding="utf-8")); r, p, h = d["render"], d["physics"], d["history"]
    return Config(int(r["width"]), int(r["height"]), int(r["fps"]), int(r["num_frames"]), int(r["ball_radius_px"]),
        tuple(r["background_rgb"]), {"red": tuple(r["red_rgb"]), "blue": tuple(r["blue_rgb"])},
        tuple(p["red_gravity_range"]), tuple(p["blue_gravity_range"]), int(p["simulation_fps"]), tuple(p["x0_range"]), tuple(p["y0_range"]), tuple(p["vx_range"]), tuple(p["vy_range"]), int(h["prediction_start"]))


def trajectory(state: dict[str, float], gravity: float, c: Config) -> np.ndarray:
    time = np.arange(c.frames, dtype=float) / c.simulation_fps
    return np.column_stack((state["x0"] + state["vx"] * time, state["y0"] + state["vy"] * time - .5 * gravity * time**2))


def sample(seed: int, c: Config, aligned_color: str) -> tuple[dict[str, float], float]:
    rng = np.random.default_rng(seed)
    gravity = float(rng.uniform(*(c.red_g if aligned_color == "red" else c.blue_g)))
    for _ in range(10_000):
        state = {key: float(rng.uniform(*bounds)) for key, bounds in (("x0", c.x0), ("y0", c.y0), ("vx", c.vx), ("vy", c.vy))}
        path = trajectory(state, gravity, c); margin_x, margin_y = c.radius / c.width, c.radius / c.height
        if path[:, 0].min() >= margin_x and path[:, 0].max() <= 1 - margin_x and path[:, 1].min() >= margin_y and path[:, 1].max() <= 1 - margin_y:
            return state, gravity
    raise RuntimeError(f"could not sample visible trajectory for seed={seed}")


def render(path: np.ndarray, color: tuple[int, int, int], c: Config) -> np.ndarray:
    frames = np.full((c.frames, c.height, c.width, 3), c.background, np.uint8); yy, xx = np.ogrid[-c.radius:c.radius + 1, -c.radius:c.radius + 1]; disk = xx * xx + yy * yy <= c.radius * c.radius
    for frame, (xw, yw) in zip(frames, path, strict=True):
        x, y = round(xw * (c.width - 1)), round((1 - yw) * (c.height - 1)); x0, x1 = max(0, x - c.radius), min(c.width, x + c.radius + 1); y0, y1 = max(0, y - c.radius), min(c.height, y + c.radius + 1)
        mask = disk[y0 - (y - c.radius):y1 - (y - c.radius), x0 - (x - c.radius):x1 - (x - c.radius)]; frame[y0:y1, x0:x1][mask] = color
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "eval"), required=True); parser.add_argument("--base-seeds", type=int, required=True); parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args(); c = config(args.config); root = args.out / args.split; (root / "videos").mkdir(parents=True); (root / "trajectories").mkdir(); (root / "metadata").mkdir()
    rows = []
    for seed in range(args.seed_offset, args.seed_offset + args.base_seeds):
        for aligned_color, stream in (("red", 0), ("blue", 1)):
            state, gravity = sample(seed * 2 + stream, c, aligned_color); path = trajectory(state, gravity, c); pair_id = f"{args.split}_{seed:05d}_{aligned_color}"; trajectory_name = f"{pair_id}.npy"; np.save(root / "trajectories" / trajectory_name, path)
            variants = [("aligned", aligned_color)] if args.split == "train" else [("aligned", aligned_color), ("conflict", "blue" if aligned_color == "red" else "red")]
            for variant, color in variants:
                sample_id = f"{pair_id}_{variant}"; video_name = f"{sample_id}.mp4"
                row = {"benchmark_version": "projectile_gravity_continuous_v2", "sample_id": sample_id, "pair_id": pair_id, "base_seed": seed, "split": args.split,
                   "gravity": gravity, "gravity_interval": "low" if aligned_color == "red" else "high", "color_label": color, "variant": variant, **state,
                   "fps": c.fps, "simulation_fps": c.simulation_fps, "frames": c.frames, "prediction_start": c.prediction_start, "position_at_boundary": path[c.prediction_start - 1].tolist(),
                   "velocity_at_boundary": [state["vx"], state["vy"] - gravity * (c.prediction_start - 1) / c.simulation_fps], "video": video_name, "trajectory": trajectory_name,
                   "render_hash_without_ball": hashlib.sha256(np.full((c.height, c.width, 3), c.background, np.uint8).tobytes()).hexdigest()}
                imageio.mimsave(root / "videos" / video_name, render(path, c.colors[color], c), fps=c.fps); (root / "metadata" / f"{sample_id}.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8"); rows.append(row)
    with (root / "metadata.csv").open("w", newline="", encoding="utf-8") as f: writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (root / "samples.jsonl").open("w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps({"sample_id": row["sample_id"], "split": row["split"], "video": f"videos/{row['video']}", "metadata": f"metadata/{row['sample_id']}.json", "attributes": row}) + "\n")
    (root / "dataset.json").write_text(json.dumps({"format": "sshv2.dataset.v1", "dataset": "projectile_gravity_continuous_v2", "count": len(rows), "metadata": "metadata.csv"}, indent=2) + "\n", encoding="utf-8")
    print(f"split={args.split} records={len(rows)} root={root}")


if __name__ == "__main__": main()
