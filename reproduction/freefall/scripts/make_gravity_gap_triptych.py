#!/usr/bin/env python3
"""Create a labeled same-state gravity-gap comparison video."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs=3, required=True)
    parser.add_argument("--labels", nargs=3, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    videos = [np.asarray(imageio.mimread(path), dtype=np.uint8)[..., :3] for path in args.inputs]
    if len({tuple(video.shape) for video in videos}) != 1:
        raise ValueError("all inputs must have identical frame, height, and width")
    frames, height, width, _ = videos[0].shape
    header, separator = 32, 16
    output = np.full((frames, height + header, 3 * width + 2 * separator, 3), (28, 30, 34), dtype=np.uint8)
    starts = (0, width + separator, 2 * (width + separator))
    for index, (video, _label, start) in enumerate(zip(videos, args.labels, starts, strict=True)):
        output[:, header:, start:start + width] = video
        if index < 2:
            output[:, :, start + width:start + width + separator] = 235
    for frame in output:
        for label, start in zip(args.labels, starts, strict=True):
            cv2.putText(frame, label, (start + 5, 22), cv2.FONT_HERSHEY_SIMPLEX, .42, (235, 235, 235), 1, cv2.LINE_AA)
    imageio.mimsave(args.out, output, fps=args.fps, codec="libx264", ffmpeg_params=["-crf", "0", "-pix_fmt", "yuv444p"])


if __name__ == "__main__":
    main()
