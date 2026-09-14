"""Minimal utilities for reading ball x-position from real Wan-VAE latents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


FUTURE_LATENT_START = 17
FUTURE_LATENT_STOP = 33
LATENT_CHANNELS = 16
LATENT_HEIGHT = 16
LATENT_WIDTH = 16
FEATURE_DIM = LATENT_CHANNELS * LATENT_HEIGHT * LATENT_WIDTH
FEATURE_LAYOUT = (
    "per-channel normalized [16,16,16] flattened C-major to 4096"
)
TARGET_DESCRIPTION = "mean pixel x over the four frames of the latent"


def future_chunk_bounds(latent_index: int) -> tuple[int, int]:
    """Return the four pixel frames represented by future latent ``latent_index``."""
    if not FUTURE_LATENT_START <= latent_index < FUTURE_LATENT_STOP:
        raise ValueError(
            f"latent_index must be in [{FUTURE_LATENT_START}, "
            f"{FUTURE_LATENT_STOP}), got {latent_index}"
        )
    return 4 * latent_index - 3, 4 * latent_index + 1


def chunk_mean_targets_px(
    displacement: np.ndarray,
    *,
    equilibrium_x: float,
    width: int,
) -> np.ndarray:
    """Return one mean pixel x-position for each future four-frame chunk."""
    values = np.asarray(displacement, dtype=np.float64)
    if values.ndim != 1 or values.size != 129:
        raise ValueError(
            f"Expected a 129-frame one-dimensional trajectory, got {values.shape}"
        )

    scale = float(width - 1)
    targets = []
    for latent_index in range(FUTURE_LATENT_START, FUTURE_LATENT_STOP):
        start, stop = future_chunk_bounds(latent_index)
        chunk = values[start:stop]
        if chunk.shape != (4,):
            raise AssertionError(
                f"Expected four frames for latent {latent_index}, got {chunk.shape}"
            )
        targets.append((equilibrium_x + float(chunk.mean())) * scale)
    return np.asarray(targets, dtype=np.float32)


def load_future_latents(path: Path) -> np.ndarray:
    """Load one [1,16,33,16,16] tensor and return [16,16,16,16]."""
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected tensor in {path}, got {type(tensor)!r}")

    expected = (1, LATENT_CHANNELS, 33, LATENT_HEIGHT, LATENT_WIDTH)
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"Expected latent shape {expected}, got {tuple(tensor.shape)} in {path}"
        )

    future = tensor[0, :, FUTURE_LATENT_START:FUTURE_LATENT_STOP]
    return future.permute(1, 0, 2, 3).float().numpy()


def channel_statistics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute training-only per-channel normalization statistics."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 4 or array.shape[1:] != (
        LATENT_CHANNELS,
        LATENT_HEIGHT,
        LATENT_WIDTH,
    ):
        raise ValueError(f"Expected [N,16,16,16], got {array.shape}")

    mean = array.mean(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    std = array.std(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    std = np.maximum(std, np.float32(1e-6))
    return mean, std


def normalize_and_flatten(
    values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Apply per-channel normalization and flatten to 4096 features."""
    array = np.asarray(values, dtype=np.float32)
    normalized = (array - mean[None, :, None, None]) / std[None, :, None, None]
    return normalized.reshape(normalized.shape[0], FEATURE_DIM)


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | int | None]:
    """Return only the three P1 evaluation metrics."""
    truth = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if truth.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {truth.shape} versus {pred.shape}")
    if truth.size == 0:
        return {"n": 0, "mae_px": None, "rmse_px": None, "r2": None}

    residual = pred - truth
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    r2 = None if denominator == 0.0 else float(
        1.0 - np.sum(residual**2) / denominator
    )
    return {"n": int(truth.size), "mae_px": mae, "rmse_px": rmse, "r2": r2}


def split_trajectory_ids(
    rows: Sequence[dict[str, str]],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    """Create a slow/fast-stratified train/validation split by trajectory."""
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("val_fraction must lie between 0 and 0.5")

    by_band: dict[str, set[str]] = {"slow": set(), "fast": set()}
    for row in rows:
        band = row["true_band"]
        if band not in by_band:
            raise ValueError(f"Unexpected band: {band}")
        by_band[band].add(row["trajectory_id"])

    train_ids: set[str] = set()
    val_ids: set[str] = set()
    for band_index, band in enumerate(("slow", "fast")):
        ids = np.asarray(sorted(by_band[band]), dtype=object)
        rng = np.random.default_rng([seed, band_index])
        rng.shuffle(ids)
        val_count = max(1, int(round(len(ids) * val_fraction)))
        val_ids.update(str(value) for value in ids[:val_count])
        train_ids.update(str(value) for value in ids[val_count:])

    if train_ids & val_ids:
        raise AssertionError("Trajectory leakage between train and validation")
    return train_ids, val_ids


def save_probe(
    path: Path,
    *,
    coef: np.ndarray,
    intercept: float,
    alpha: float,
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    history: str,
    equilibrium_x: float,
    width: int,
) -> None:
    """Save ridge parameters and the conventions needed to reuse them safely."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        coef=np.asarray(coef, dtype=np.float32),
        intercept=np.asarray([intercept], dtype=np.float32),
        alpha=np.asarray([alpha], dtype=np.float64),
        channel_mean=np.asarray(channel_mean, dtype=np.float32),
        channel_std=np.asarray(channel_std, dtype=np.float32),
        history=np.asarray([history]),
        feature_layout=np.asarray([FEATURE_LAYOUT]),
        future_latent_range=np.asarray(
            [FUTURE_LATENT_START, FUTURE_LATENT_STOP],
            dtype=np.int64,
        ),
        target=np.asarray([TARGET_DESCRIPTION]),
        equilibrium_x=np.asarray([equilibrium_x], dtype=np.float64),
        width=np.asarray([width], dtype=np.int64),
    )


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
