"""Small deterministic video and rendering helpers shared by experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import imageio.v2 as imageio
import numpy as np


def deterministic_rng(seed: int, stream: int) -> np.random.Generator:
    """Return the repository's stateless split RNG for one named stream."""
    mixed = (
        int(seed) * 6364136223846793005 + int(stream)
    ) & ((1 << 64) - 1)
    return np.random.default_rng(mixed)


def as_list(value: np.ndarray | Sequence[float]) -> list[float]:
    return np.asarray(value, dtype=np.float64).tolist()


def draw_circle(
    frame: np.ndarray,
    centre: np.ndarray,
    radius: float,
    color: tuple[int, int, int],
) -> None:
    height, width = frame.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    world_x = (xx + 0.5) / width
    world_y = 1.0 - (yy + 0.5) / height
    mask = (
        (world_x - centre[0]) ** 2
        + (world_y - centre[1]) ** 2
        <= radius**2
    )
    frame[mask] = color


def draw_segment(
    frame: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    height, width = frame.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    start = np.asarray((a[0] * width, (1.0 - a[1]) * height))
    end = np.asarray((b[0] * width, (1.0 - b[1]) * height))
    delta = end - start
    denominator = max(float(delta @ delta), 1e-12)
    projection = np.clip(
        (
            (xx - start[0]) * delta[0]
            + (yy - start[1]) * delta[1]
        )
        / denominator,
        0.0,
        1.0,
    )
    distance = np.hypot(
        xx - (start[0] + projection * delta[0]),
        yy - (start[1] + projection * delta[1]),
    )
    frame[distance <= thickness / 2.0] = color


def valid_positions(
    positions: np.ndarray,
    *,
    radius: float,
) -> bool:
    return bool(
        np.all(positions[:, 0] >= radius)
        and np.all(positions[:, 0] <= 1.0 - radius)
        and np.all(positions[:, 1] >= radius)
        and np.all(positions[:, 1] <= 1.0 - radius)
    )


def read_video(
    path: Path,
    *,
    expected_frames: int | None = None,
) -> np.ndarray:
    reader = imageio.get_reader(path)
    try:
        frames = np.stack(
            [frame[..., :3] for frame in reader],
            axis=0,
        ).astype(np.uint8)
    finally:
        reader.close()
    if (
        expected_frames is not None
        and frames.shape[0] != expected_frames
    ):
        raise ValueError(
            f"Expected {expected_frames} frames but read "
            f"{frames.shape[0]} from {path}"
        )
    return frames


def write_video(
    path: Path,
    frames: np.ndarray,
    *,
    fps: int,
    codec: str = "libx264",
    quality: int = 10,
) -> None:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected [T,H,W,3], got {frames.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        path,
        fps=fps,
        codec=codec,
        quality=quality,
        macro_block_size=None,
    ) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
