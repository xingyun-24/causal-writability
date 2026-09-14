"""Private video and tensor interface helpers."""

import os
import torch
import imageio
import numpy as np
from PIL import Image
from pathlib import Path

from diffsynth.trainers.unified_dataset import ImageCropAndResize


def save_video(
    frames: list[Image.Image],
    out_path: str | os.PathLike | Path,
    fps: float,
    quality: int = 9,
    ffmpeg_params: dict | None = None
):
    writer = imageio.get_writer(out_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params)
    for frame in frames:
        writer.append_data(np.array(frame))
    writer.close()


def video_to_tensor(
    path: str | os.PathLike | Path,
    *,
    size: tuple[int, int] | None = None,
    height_division_factor: int = 16,
    width_division_factor: int = 16
):
    transform = None
    if size is not None:
        max_pixels = size[0] * size[1]
        transform = ImageCropAndResize(size[0], size[1], max_pixels, height_division_factor, width_division_factor)

    reader = imageio.get_reader(path)
    try:
        frames = []
        for frame in reader:
            if transform is not None:
                frame = np.array(transform(Image.fromarray(frame)))
            frames.append(frame)
    finally:
        reader.close()

    if len(frames) == 0:
        raise ValueError(f"No frames found in video at {path}")

    frames = np.stack(frames).astype(np.float32)
    tensor = torch.from_numpy(frames) * (2.0 / 255.0) - 1.0
    tensor = tensor.permute(3, 0, 1, 2).unsqueeze(0)
    return tensor


def infer_video_size(path: Path, default: tuple[int, int] = (256, 256)):
    try:
        path = Path(path)
        if not path.exists():
            return default

        reader = imageio.get_reader(path)
        try:
            frame = reader.get_data(0)
        finally:
            reader.close()

        if frame is None or len(frame.shape) < 2:
            return default

        height, width = int(frame.shape[0]), int(frame.shape[1])
        if height <= 0 or width <= 0:
            return default
        return (width, height)
    except Exception:
        return default
