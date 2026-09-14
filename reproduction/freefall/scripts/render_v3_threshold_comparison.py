#!/usr/bin/env python3
"""Render four V3 gravity endpoints from a common boundary state."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from build_dataset_v2 import read_config, render, state_and_gravity, trajectory, write_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=50123)
    args = parser.parse_args()

    cfg, _ = read_config(args.config)
    state, _quantile, _low, _high = state_and_gravity(args.seed, cfg)
    gravities = (0.003, 0.025, 0.045, 0.067)
    labels = tuple(f"g={gravity:.3f}" for gravity in gravities)
    colors = (cfg.colors["red"], cfg.colors["red"], cfg.colors["blue"], cfg.colors["blue"])
    videos = []
    args.out.mkdir(parents=True, exist_ok=True)
    margin = cfg.radius / min(cfg.width, cfg.height)
    for gravity, label, color in zip(gravities, labels, colors, strict=True):
        path = trajectory(state, gravity, cfg)
        if np.any(path < margin) or np.any(path > 1.0 - margin):
            raise RuntimeError(f"{label}: trajectory leaves renderable area")
        video = render(path, color, cfg)
        write_video(args.out / f"{label}.mp4", video, cfg.fps)
        videos.append(video)

    frames, height, width, _ = videos[0].shape
    header, separator = 32, 12
    composite = np.full((frames, height + header, 4 * width + 3 * separator, 3), cfg.background, dtype=np.uint8)
    for index, (video, label) in enumerate(zip(videos, labels, strict=True)):
        start = index * (width + separator)
        composite[:, header:, start:start + width] = video
        if index < len(videos) - 1:
            composite[:, :, start + width:start + width + separator] = (235, 235, 235)
        for frame in composite:
            cv2.putText(frame, label, (start + 5, 22), cv2.FONT_HERSHEY_SIMPLEX, .48, (235, 235, 235), 1, cv2.LINE_AA)
    write_video(args.out / "v3_four_thresholds.mp4", composite, cfg.fps)
    print(f"seed={args.seed} state={state} output={args.out}")


if __name__ == "__main__":
    main()
