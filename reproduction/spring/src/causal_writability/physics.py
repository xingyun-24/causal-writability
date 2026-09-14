from __future__ import annotations
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from einops import rearrange

from diffsynth.pipelines.wan_video_new import sinusoidal_embedding_1d
from sshv2.diffsynth.trainers.wan_video_train import WanTrainingModule
from sshv2.simulation.spring_shortcuts_v1 import (
    apply_short_history_mask_numpy,
    classify_detected_color,
    dataclass_config_from_dict,
    detect_mass_track,
    displacement_to_pixel_x,
    pixel_x_to_displacement,
    render_video,
    route_metrics_against_bands,
    sample_parameters,
    trajectory,
    validity_metrics,
)
from sshv2.utils.spring_configs import SpringTrainingConfig


BANK_NAME = "large_short_shared_strict_bank_v1"
BANK_VERSION = 1
MODEL_SEEDS = (3407, 3408, 3409)
EXPECTED_STEPS = 20
TARGET_PER_DIRECTION = 64
MIN_OMEGA_GAP = 2.0
ALIGNED_RMSE_MAX = 0.035
CANDIDATES_PER_DIRECTION = 4096
CANDIDATE_SEED_OFFSET = 71_000_000
GENERATION_SEED_OFFSET = 83_000_000
NAMESPACE = "large-short-shared-strict-bank-v1"


def opaque_id(prefix: str, *values: Any) -> str:
    token = hashlib.blake2b(":".join(map(str, values)).encode(), digest_size=10).hexdigest()
    return f"{prefix}_{token}"


def render_hash_without_mass_fill(frames: np.ndarray, displacements: np.ndarray, render_cfg: Any) -> str:
    normalized = np.asarray(frames).copy()
    cy = round(render_cfg.center_y * (render_cfg.height - 1))
    radius = render_cfg.mass_radius_px + 3
    for frame, displacement in zip(normalized, displacements, strict=True):
        cx = round(displacement_to_pixel_x(float(displacement), render_cfg))
        frame[max(0, cy - radius):min(render_cfg.height, cy + radius + 1), max(0, cx - radius):min(render_cfg.width, cx + radius + 1)] = 0
    return hashlib.sha256(normalized.tobytes()).hexdigest()


def tensor_to_uint8(video: torch.Tensor) -> np.ndarray:
    if video.ndim == 5:
        if video.shape[0] != 1:
            raise ValueError(f"Expected batch one, got {tuple(video.shape)}")
        video = video[0]
    if video.ndim != 4:
        raise ValueError(f"Expected [C,T,H,W], got {tuple(video.shape)}")
    return video.detach().float().cpu().permute(1, 2, 3, 0).add(1.0).mul(127.5).clamp(0, 255).byte().numpy()


def decode_latent(pipe: Any, latent: torch.Tensor, *, device: str, expected_frames: int) -> np.ndarray:
    parameter = next(pipe.vae.parameters(), None)
    dtype = parameter.dtype if parameter is not None else torch.bfloat16
    with torch.inference_mode():
        decoded = pipe.vae.decode(latent.to(device=device, dtype=dtype), device=device, tiled=False)
    frames = tensor_to_uint8(decoded)
    if frames.shape != (expected_frames, 128, 128, 3):
        raise AssertionError(f"Unexpected decoded shape {frames.shape}")
    return frames


def evaluate_future(future_frames: np.ndarray, *, metadata: dict[str, Any], color_label_for_route: str, cfg: Any) -> dict[str, Any]:
    track = detect_mass_track(future_frames, cfg.render)
    validity = validity_metrics(track, cfg.render, x_star=float(metadata["x_star"]))
    if not validity["valid"]:
        return {**validity, "omega_pixel": float("nan"), "omega_pixel_class": "invalid", "route_label": "invalid", "free_shm_rmse": float("nan")}
    displacement = pixel_x_to_displacement(track.x_px, cfg.render)
    route = route_metrics_against_bands(
        displacement, np.isfinite(displacement), true_band=cfg.fast_band if metadata["true_band"] == "fast" else cfg.slow_band,
        color_implied_band=cfg.fast_band if color_label_for_route == "blue" else cfg.slow_band,
        slow_band=cfg.slow_band, fast_band=cfg.fast_band, omega_true=float(metadata["omega_true"]),
        x_star=float(metadata["x_star"]), v_star=float(metadata["v_star"]), fps=cfg.render.fps,
        amplitude_low=cfg.amplitude_low, amplitude_high=cfg.amplitude_high,
        reference_amplitude_multiplier=cfg.route_reference_amplitude_multiplier,
        frequency_band_tolerance=cfg.frequency_band_tolerance,
        free_shm_rmse_threshold=cfg.free_shm_rmse_threshold,
    )
    return {
        **validity,
        "omega_pixel": float(route["omega_hat_free"]),
        "omega_pixel_class": str(route["omega_hat_free_class"]),
        "route_label": str(route["route_label"]),
        "free_shm_rmse": float(route["free_shm_rmse"]),
        "free_center_offset": float(route["free_center_offset"]),
        "free_fit_amplitude_ood": bool(route["free_fit_amplitude_ood"]),
        "anchored_route_label": str(route["anchored_route_label"]),
    }


def encode_condition(pipe: Any, frames: np.ndarray, *, prediction_start: int, num_condition_frames: int) -> torch.Tensor:
    pixels = (
        torch.from_numpy(np.asarray(frames).copy())
        .permute(3, 0, 1, 2)
        .float()
        .div(127.5)
        .sub(1.0)
        .unsqueeze(0)
        .to(device=pipe.device, dtype=pipe.torch_dtype)
    )
    with torch.inference_mode():
        encoded = pipe.vae.encode(pixels[:, :, :prediction_start].contiguous(), device=pipe.device, tiled=False)
    encoded = encoded[:, :, :num_condition_frames].to(device=pipe.device, dtype=pipe.torch_dtype)
    if tuple(encoded.shape) != (1, 16, num_condition_frames, 16, 16):
        raise AssertionError(f"Unexpected condition shape {tuple(encoded.shape)}")
    return encoded


def direction_spec(direction: str) -> dict[str, str]:
    if direction == "true_fast_red_conflict":
        return {"true_band": "fast", "aligned_color": "blue", "conflict_color": "red", "conflict_class": "slow"}
    if direction == "true_slow_blue_conflict":
        return {"true_band": "slow", "aligned_color": "red", "conflict_color": "blue", "conflict_class": "fast"}
    raise ValueError(direction)


def candidate_definition(*, direction: str, candidate_index: int, cfg: Any) -> dict[str, Any]:
    spec = direction_spec(direction)
    direction_offset = 0 if direction == "true_fast_red_conflict" else 1
    base_seed = int(cfg.seed + CANDIDATE_SEED_OFFSET + 10_000_000 * direction_offset + candidate_index)
    rng = np.random.default_rng(np.random.SeedSequence([base_seed, 128, 7919, BANK_VERSION]))
    band = cfg.fast_band if spec["true_band"] == "fast" else cfg.slow_band
    params = sample_parameters(rng, band, cfg.amplitude_low, cfg.amplitude_high)
    x, v = trajectory(params, fps=cfg.render.fps, num_frames=cfg.render.num_frames)
    trajectory_id = opaque_id("traj", NAMESPACE, direction, base_seed)
    pair_id = opaque_id("pair", NAMESPACE, direction, base_seed)
    generation_seed = base_seed + GENERATION_SEED_OFFSET
    aligned = render_video(x, spec["aligned_color"], cfg.render)
    conflict = render_video(x, spec["conflict_color"], cfg.render)
    render_hash = render_hash_without_mass_fill(aligned, x, cfg.render)
    if render_hash != render_hash_without_mass_fill(conflict, x, cfg.render):
        raise AssertionError("Color counterfactual changed non-mass pixels")
    aligned_short_history = apply_short_history_mask_numpy(aligned, cfg)
    conflict_short_history = apply_short_history_mask_numpy(conflict, cfg)
    if not (
        np.array_equal(aligned_short_history[cfg.short_prefix_start:], aligned[cfg.short_prefix_start:])
        and np.array_equal(conflict_short_history[cfg.short_prefix_start:], conflict[cfg.short_prefix_start:])
    ):
        raise AssertionError("Short history masking modified observed or future frames")
    if not (
        np.all(aligned_short_history[:cfg.short_prefix_start] == np.asarray(cfg.render.background_rgb, dtype=np.uint8))
        and np.all(conflict_short_history[:cfg.short_prefix_start] == np.asarray(cfg.render.background_rgb, dtype=np.uint8))
    ):
        raise AssertionError("Short history mask is not exact background")
    return {
        "direction": direction,
        "candidate_index": candidate_index,
        "trajectory_id": trajectory_id,
        "pair_id": pair_id,
        "base_seed": base_seed,
        "generation_seed": generation_seed,
        "generation_noise_seed_convention": "base_seed + 83000000; identical across model seeds and color variants",
        "true_band": spec["true_band"],
        "omega_true": float(params.omega),
        "amplitude": float(params.amplitude),
        "phase": float(params.phase),
        "x_star": float(x[cfg.prediction_start - 1]),
        "v_star": float(v[cfg.prediction_start - 1]),
        "input_frames": int(cfg.render.num_frames),
        "prediction_start": int(cfg.prediction_start),
        "short_history_policy": "mask_0_56_background_then_real_57_64",
        "short_masked_pixel_frames": [0, int(cfg.short_prefix_start) - 1],
        "short_real_condition_pixel_frames": [int(cfg.short_prefix_start), int(cfg.prediction_start) - 1],
        "aligned_color": spec["aligned_color"],
        "conflict_color": spec["conflict_color"],
        "render_hash_without_mass_fill": render_hash,
        "_aligned_frames": aligned,
        "_conflict_frames": conflict,
        "_aligned_short_history_frames": aligned_short_history,
        "_conflict_short_history_frames": conflict_short_history,
    }

