"""Shared runtime helpers for the Spring 128-pair mean-residual-direction experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from sshv2.interpretability.condition_residual_patching import (
    ResidualPatchController,
    sample_final_latents,
)
from sshv2.simulation.spring_shortcuts_v1 import (
    classify_detected_color,
    detect_mass_track,
    displacement_to_pixel_x,
    finite_json,
    pixel_x_to_displacement,
    route_metrics_against_bands,
    validity_metrics,
)

EXPECTED_LATENT_SHAPE = (1, 16, 33, 16, 16)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(finite_json(row), ensure_ascii=False) + "\n")


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    write_jsonl(temporary, rows)
    temporary.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(finite_json(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Refusing to write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(finite_json(row))
    temporary.replace(path)




def save_frames_npz(path: Path, frames: np.ndarray) -> None:
    array = np.asarray(frames, dtype=np.uint8)
    if array.shape != (129, 128, 128, 3):
        raise AssertionError(f"Raw frame shape {array.shape} is unexpected")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, frames=array)
    temporary.replace(path)


def load_frames_npz(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        frames = np.asarray(payload["frames"], dtype=np.uint8)
    if frames.shape != (129, 128, 128, 3):
        raise AssertionError(f"Raw frame shape {frames.shape} is unexpected in {path}")
    return frames

def unique_index(rows: Iterable[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row[key])
        if value in result:
            raise AssertionError(f"Duplicate {key}={value!r} in {label}")
        result[value] = row
    return result


def encode_long_condition(
    pipe: Any,
    frames: np.ndarray,
    *,
    prediction_start: int,
    num_condition_frames: int,
) -> torch.Tensor:
    pixels = (
        torch.from_numpy(np.asarray(frames).copy())
        .permute(3, 0, 1, 2)
        .float()
        .div(127.5)
        .sub(1.0)
        .unsqueeze(0)
        .to(device=pipe.device, dtype=pipe.torch_dtype)
    )
    pixels = pixels[:, :, :prediction_start].contiguous()
    encoded = pipe.vae.encode(pixels, device=pipe.device, tiled=False)
    encoded = encoded[:, :, :num_condition_frames].to(
        device=pipe.device, dtype=pipe.torch_dtype
    )
    expected = (1, 16, num_condition_frames, 16, 16)
    if tuple(encoded.shape) != expected:
        raise AssertionError(f"Condition latent shape {tuple(encoded.shape)} != {expected}")
    if not torch.isfinite(encoded).all():
        raise AssertionError("Non-finite condition latent")
    return encoded


def sample_latent(
    *,
    pipe: Any,
    condition_latents: torch.Tensor,
    cfg: Any,
    num_condition_frames: int,
    steps: int,
    seed: int,
    controller: Any | None = None,
) -> torch.Tensor:
    if controller is None:
        controller = ResidualPatchController(
            expected_steps=steps,
            num_condition_frames=num_condition_frames,
        )
    return sample_final_latents(
        pipe=pipe,
        condition_latents=condition_latents,
        num_frames=cfg.render.num_frames,
        height=cfg.render.height,
        width=cfg.render.width,
        num_condition_frames=num_condition_frames,
        num_inference_steps=steps,
        seed=seed,
        controller=controller,
        sigma_shift=5.0,
        denoising_strength=1.0,
    )


def save_latent(path: Path, tensor: torch.Tensor) -> None:
    if tuple(tensor.shape) != EXPECTED_LATENT_SHAPE:
        raise AssertionError(f"Latent shape {tuple(tensor.shape)} != {EXPECTED_LATENT_SHAPE}")
    if not torch.isfinite(tensor).all():
        raise AssertionError("Refusing to save non-finite latent")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(tensor.detach().float().cpu(), temporary)
    temporary.replace(path)


def load_latent(path: Path) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected tensor in {path}")
    if tuple(tensor.shape) != EXPECTED_LATENT_SHAPE:
        raise AssertionError(f"Latent shape {tuple(tensor.shape)} != {EXPECTED_LATENT_SHAPE}")
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"Non-finite latent in {path}")
    return tensor.float()


def max_abs_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    if tuple(left.shape) != tuple(right.shape):
        raise AssertionError(f"Shape mismatch {tuple(left.shape)} vs {tuple(right.shape)}")
    return float((left.float() - right.float()).abs().max().item())


def tensor_to_uint8(video: torch.Tensor) -> np.ndarray:
    if video.ndim == 5:
        if video.shape[0] != 1:
            raise ValueError(f"Expected batch one, got {tuple(video.shape)}")
        video = video[0]
    if video.ndim != 4:
        raise ValueError(f"Expected [C,T,H,W], got {tuple(video.shape)}")
    return (
        video.detach().float().cpu().permute(1, 2, 3, 0)
        .add(1.0).mul(127.5).clamp(0, 255).byte().numpy()
    )


def decode_latent(pipe_or_vae: Any, latent: torch.Tensor, *, device: str, expected_frames: int) -> np.ndarray:
    vae = getattr(pipe_or_vae, "vae", pipe_or_vae)
    parameter = next(vae.parameters(), None)
    dtype = parameter.dtype if parameter is not None else torch.bfloat16
    tensor = latent.to(device=device, dtype=dtype)
    with torch.inference_mode():
        decoded = vae.decode(tensor, device=device, tiled=False)
    frames = tensor_to_uint8(decoded)
    if frames.shape != (expected_frames, 128, 128, 3):
        raise AssertionError(f"Decoded frame shape {frames.shape} is unexpected")
    return frames


def detected_color_summary(track: Any, cfg: Any) -> dict[str, Any]:
    rgb = np.asarray(track.mean_rgb)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise AssertionError(f"Expected mean_rgb [T,3], got {rgb.shape}")
    labels = [
        classify_detected_color(value, cfg.render)
        for value in rgb
        if np.isfinite(value).all()
    ]
    counts = {name: labels.count(name) for name in ("red", "blue", "unknown")}
    majority = max(counts, key=counts.get) if labels else "unknown"
    denominator = max(1, len(labels))
    return {
        "detected_color_majority": majority,
        "detected_color_counts": counts,
        "detected_color_rate_red": counts["red"] / denominator,
        "detected_color_rate_blue": counts["blue"] / denominator,
        "detected_color_rate_unknown": counts["unknown"] / denominator,
    }


def evaluate_future(
    future_frames: np.ndarray,
    *,
    metadata: dict[str, Any],
    color_label_for_route: str,
    cfg: Any,
) -> dict[str, Any]:
    if future_frames.shape != (cfg.future_frames, cfg.render.height, cfg.render.width, 3):
        raise AssertionError(f"Future shape {future_frames.shape} is unexpected")
    track = detect_mass_track(future_frames, cfg.render)
    validity = validity_metrics(track, cfg.render, x_star=float(metadata["x_star"]))
    result: dict[str, Any] = {
        **validity,
        **detected_color_summary(track, cfg),
    }
    if not validity["valid"]:
        result.update({
            "omega_pixel": float("nan"),
            "omega_pixel_class": "invalid",
            "route_label": "invalid",
            "free_shm_rmse": float("nan"),
        })
        return result
    displacement = pixel_x_to_displacement(track.x_px, cfg.render)
    true_band_label = str(metadata.get("true_band", "fast"))
    if true_band_label not in {"slow", "fast"}:
        raise ValueError(f"Unknown true band: {true_band_label}")
    true_band = cfg.slow_band if true_band_label == "slow" else cfg.fast_band
    route = route_metrics_against_bands(
        displacement,
        np.isfinite(displacement),
        true_band=true_band,
        color_implied_band=cfg.fast_band if color_label_for_route == "blue" else cfg.slow_band,
        slow_band=cfg.slow_band,
        fast_band=cfg.fast_band,
        omega_true=float(metadata["omega_true"]),
        x_star=float(metadata["x_star"]),
        v_star=float(metadata["v_star"]),
        fps=cfg.render.fps,
        amplitude_low=cfg.amplitude_low,
        amplitude_high=cfg.amplitude_high,
        reference_amplitude_multiplier=cfg.route_reference_amplitude_multiplier,
        frequency_band_tolerance=cfg.frequency_band_tolerance,
        free_shm_rmse_threshold=cfg.free_shm_rmse_threshold,
    )
    result.update({
        "omega_pixel": float(route["omega_hat_free"]),
        "omega_pixel_class": str(route["omega_hat_free_class"]),
        "route_label": str(route["route_label"]),
        "free_shm_rmse": float(route["free_shm_rmse"]),
        "free_center_offset": float(route["free_center_offset"]),
        "free_fit_amplitude_ood": bool(route["free_fit_amplitude_ood"]),
        "anchored_route_label": str(route["anchored_route_label"]),
    })
    return result


def render_hash_without_mass_fill(
    frames: np.ndarray,
    displacements: np.ndarray,
    render_cfg: Any,
) -> str:
    normalized = np.asarray(frames).copy()
    centers_x = np.asarray([
        displacement_to_pixel_x(float(value), render_cfg)
        for value in displacements
    ])
    cy = round(render_cfg.center_y * (render_cfg.height - 1))
    radius = render_cfg.mass_radius_px + 3
    for frame, cx_float in zip(normalized, centers_x, strict=True):
        cx = round(float(cx_float))
        x0, x1 = max(0, cx - radius), min(render_cfg.width, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(render_cfg.height, cy + radius + 1)
        frame[y0:y1, x0:x1] = 0
    return hashlib.sha256(normalized.tobytes()).hexdigest()


def opaque_id(prefix: str, *values: Any) -> str:
    token = hashlib.blake2b(":".join(map(str, values)).encode(), digest_size=10).hexdigest()
    return f"{prefix}_{token}"


def load_wan_pipe(*, config: Path, checkpoint: Path, steps: int, device: str, cfg: Any) -> tuple[Any, Any]:
    from sshv2.diffsynth.trainers.wan_video_train import WanTrainingModule
    from sshv2.utils.spring_configs import SpringTrainingConfig

    train_cfg = SpringTrainingConfig.from_file(config)
    train_cfg.model.dit.ckpt_file = checkpoint
    if train_cfg.model.num_condition_frames != cfg.long_condition_latents:
        raise AssertionError("Long condition latent count changed")
    if int(train_cfg.model.dit.num_layers) != 30 or int(train_cfg.model.dit.dim) != 768:
        raise AssertionError("Frozen DiT architecture changed")
    model = WanTrainingModule(
        dit_config=train_cfg.model.dit,
        vae_config=train_cfg.model.vae,
        no_encoding=False,
        num_condition_frames=train_cfg.model.num_condition_frames,
        num_inference_steps=steps,
        pipeline_type=train_cfg.model.pipe,
        pipeline_kwargs=train_cfg.model.pipe_kwargs,
    )
    pipe = model.pipe
    pipe.to(device)
    pipe.load_models_to_device(("dit", "vae"))
    pipe.pre_encoded_(False)
    if tuple(int(value) for value in pipe.dit.patch_size) != (1, 2, 2):
        raise AssertionError("Frozen patch size changed")
    return pipe, train_cfg


def finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
