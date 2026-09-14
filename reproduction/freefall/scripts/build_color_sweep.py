#!/usr/bin/env python3
"""Build an 11-point red-to-blue color sweep at one fixed projectile gravity."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-g", type=float, default=0.012)
    parser.add_argument("--seed", type=int, default=50000)
    parser.add_argument("--samples-per-color", type=int, default=32)
    args = parser.parse_args()
    data = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    render = data["render"]; physics = data["physics"]; history = data["history"]
    width, height = int(render["width"]), int(render["height"])
    frames, fps, sim_fps = int(render["num_frames"]), int(render["fps"]), int(physics["simulation_fps"])
    radius, boundary = int(render["ball_radius_px"]), int(history["boundary_frame"])
    prediction_start = int(history["prediction_start"])
    if not (0.003 <= args.target_g <= 0.025):
        raise ValueError("color-sweep target g must be inside the V3 low-gravity interval [0.003, 0.025]")
    if args.samples_per_color < 1:
        raise ValueError("samples-per-color must be positive")

    root = args.out
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing non-empty output: {root}")
    (root / "videos").mkdir(parents=True, exist_ok=True)
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    background = tuple(render["background_rgb"])
    red = np.asarray(render["red_rgb"], dtype=np.float64)
    blue = np.asarray(render["blue_rgb"], dtype=np.float64)
    bg_hash = hashlib.sha256(np.full((height, width, 3), background, np.uint8).tobytes()).hexdigest()
    rows = []
    trajectories = {}
    for sample_index in range(args.samples_per_color):
        rng = np.random.default_rng(args.seed + sample_index)
        state = {
            "x_boundary": float(rng.uniform(*physics["x_boundary_range"])),
            "y_boundary": float(rng.uniform(*physics["y_boundary_range"])),
            "vx_boundary": float(rng.uniform(*physics["vx_boundary_range"])),
            "vy_boundary": float(rng.uniform(*physics["vy_boundary_range"])),
        }
        tau = (np.arange(frames, dtype=np.float64) - boundary) / sim_fps
        path = np.column_stack((
            state["x_boundary"] + state["vx_boundary"] * tau,
            state["y_boundary"] + state["vy_boundary"] * tau - 0.5 * args.target_g * tau**2,
        ))
        margin = radius / min(width, height)
        if np.any(path < margin) or np.any(path > 1.0 - margin):
            raise RuntimeError(f"sample {sample_index}: trajectory leaves renderable area")
        trajectories[sample_index] = (state, path)
    for index, fraction in enumerate(np.linspace(0.0, 1.0, 11)):
        colour = np.rint((1.0 - fraction) * red + fraction * blue).astype(np.uint8)
        for sample_index, (state, path) in trajectories.items():
            pair_id = f"colorsweep_g{args.target_g:.3f}_c{index:02d}_s{sample_index:02d}"
            video_name = f"{pair_id}.mp4"
            frames_rgb = np.full((frames, height, width, 3), background, dtype=np.uint8)
            scale = 256
            for frame, (x, y) in zip(frames_rgb, path, strict=True):
                centre = (int(round(x * (width - 1) * scale)), int(round((1.0 - y) * (height - 1) * scale)))
                cv2.circle(frame, centre, radius * scale, tuple(int(v) for v in colour), thickness=-1, lineType=cv2.LINE_AA, shift=8)
            imageio.mimsave(root / "videos" / video_name, frames_rgb, fps=fps, codec="libx264", ffmpeg_params=["-crf", "0", "-pix_fmt", "yuv444p"])
            for variant in ("aligned", "conflict"):
                sample_id = f"{pair_id}_{variant}"
                row = {
                "benchmark_version": str(data["version"]), "sample_id": sample_id, "pair_id": pair_id,
                "base_pair_id": pair_id, "base_seed": args.seed + sample_index, "split": "eval",
                "gravity": args.target_g, "gravity_interval": "low", "gravity_quantile": 0.5,
                "color_label": f"c{index:02d}", "color_fraction": float(fraction),
                "color_rgb": [int(v) for v in colour], "variant": variant, **state,
                "fps": fps, "simulation_fps": sim_fps, "frames": frames,
                "boundary_frame": boundary, "prediction_start": prediction_start,
                "position_at_boundary": path[boundary].tolist(),
                "velocity_at_boundary": [state["vx_boundary"], state["vy_boundary"]],
                "video": video_name, "trajectory": "fixed_target_trajectory.npy",
                "render_hash_without_ball": bg_hash,
            }
                (root / "metadata" / f"{sample_id}.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
                rows.append(row)
    with (root / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (root / "dataset.json").write_text(json.dumps({"format": "projectile.color_sweep.v1", "target_g": args.target_g, "seed": args.seed, "colors": 11, "samples_per_color": args.samples_per_color, "metadata": "metadata.csv"}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(root), "target_g": args.target_g, "colors": 11, "samples_per_color": args.samples_per_color, "records": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
