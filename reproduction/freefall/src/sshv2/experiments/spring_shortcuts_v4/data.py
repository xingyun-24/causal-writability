"""Dataset generation and audit boundary for spring_shortcuts_v4.

Stage 1 studies one non-causal appearance shortcut: colour.  Red is correlated
with a continuously sampled slow-frequency band and blue with a continuously
sampled fast-frequency band.  Colour therefore identifies a *band*, never one
single oscillator frequency.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from sshv2.common.dataset import (
    samples_from_metadata_csv,
    write_dataset,
)

BandName = Literal["slow", "fast"]
ColorName = Literal["red", "blue"]
VariantName = Literal["aligned", "conflict"]
HistoryName = Literal["short", "long"]

VERSION = "spring_shortcuts_v4"


@dataclass(frozen=True)
class FrequencyBand:
    name: BandName
    low: float
    high: float

    def __post_init__(self) -> None:
        if not (0.0 < self.low < self.high):
            raise ValueError(f"Invalid {self.name} band: [{self.low}, {self.high}]")

    def contains(self, omega: float, *, atol: float = 1e-9) -> bool:
        return self.low - atol <= float(omega) <= self.high + atol


@dataclass(frozen=True)
class SpringRenderConfig:
    width: int = 128
    height: int = 128
    fps: int = 20
    num_frames: int = 129
    equilibrium_x: float = 0.58
    anchor_x: float = 0.16
    center_y: float = 0.50
    mass_radius_px: int = 7
    spring_half_height_px: int = 5
    spring_coils: int = 10
    background_rgb: tuple[int, int, int] = (28, 30, 34)
    spring_rgb: tuple[int, int, int] = (226, 229, 235)
    anchor_rgb: tuple[int, int, int] = (205, 209, 217)
    red_rgb: tuple[int, int, int] = (235, 48, 48)
    blue_rgb: tuple[int, int, int] = (48, 96, 235)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.fps != 20 or self.num_frames != 129:
            raise ValueError("V4 requires exactly 20 fps and 129 master frames")
        if not (0.0 < self.anchor_x < self.equilibrium_x < 1.0):
            raise ValueError("Require 0 < anchor_x < equilibrium_x < 1")
        if not (0.0 < self.center_y < 1.0):
            raise ValueError("center_y must lie in (0,1)")
        if self.mass_radius_px < 3:
            raise ValueError("mass_radius_px is implausibly small")


@dataclass(frozen=True)
class SpringDatasetConfig:
    slow_band: FrequencyBand
    fast_band: FrequencyBand
    amplitude_low: float = 0.10
    amplitude_high: float = 0.17
    render: SpringRenderConfig = field(default_factory=SpringRenderConfig)
    short_prefix_start: int = 57
    prediction_start: int = 65
    short_mask_mode: str = "background_only"
    seed: int = 3407
    train_base_seeds: int = 1024
    eval_base_seeds: int = 64
    sanity_base_seeds: int = 8
    tiny_train_videos: int = 16
    pilot_train_videos: int = 256
    route_reference_amplitude_multiplier: float = 1.0
    frequency_band_tolerance: float = 0.005
    free_shm_rmse_threshold: float = 0.035

    def __post_init__(self) -> None:
        if self.slow_band.name != "slow" or self.fast_band.name != "fast":
            raise ValueError("frequency-band names must be slow and fast")
        if self.slow_band.high >= self.fast_band.low:
            raise ValueError("slow and fast bands must not overlap")
        if not (0.0 < self.amplitude_low < self.amplitude_high):
            raise ValueError("invalid amplitude range")
        if self.short_prefix_start != 57 or self.prediction_start != 65:
            raise ValueError("V4 freezes short real frames=57..64 and future=65..128")
        if self.short_mask_mode != "background_only":
            raise ValueError("stage 1 supports only the fixed background_only short mask")
        if self.render.num_frames - self.prediction_start != 64:
            raise ValueError("V4 requires exactly 64 future pixel frames")
        for name in ("train_base_seeds", "eval_base_seeds", "sanity_base_seeds"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for n in (self.tiny_train_videos, self.pilot_train_videos):
            if n <= 0 or n % 2:
                raise ValueError("tiny/pilot training video counts must be positive and even")
        if self.tiny_train_videos > 2 * self.train_base_seeds:
            raise ValueError("tiny_train_videos exceeds full training split")
        if self.pilot_train_videos > 2 * self.train_base_seeds:
            raise ValueError("pilot_train_videos exceeds full training split")
        if self.route_reference_amplitude_multiplier < 1.0:
            raise ValueError("route_reference_amplitude_multiplier must be at least 1")
        if self.frequency_band_tolerance < 0:
            raise ValueError("frequency_band_tolerance must be non-negative")
        if 2 * self.frequency_band_tolerance >= self.fast_band.low - self.slow_band.high:
            raise ValueError("frequency_band_tolerance makes the expanded bands overlap")
        if self.free_shm_rmse_threshold <= 0:
            raise ValueError("free_shm_rmse_threshold must be positive")

    @property
    def future_frames(self) -> int:
        return self.render.num_frames - self.prediction_start

    @property
    def long_prefix_frames(self) -> int:
        return self.prediction_start

    @property
    def short_prefix_frames(self) -> int:
        return self.prediction_start - self.short_prefix_start

    @property
    def long_latent_frames(self) -> int:
        return (self.render.num_frames - 1) // 4 + 1

    @property
    def short_masked_frames(self) -> int:
        return self.short_prefix_start

    @property
    def short_latent_frames(self) -> int:
        # V4 preserves the full 129-frame timeline for both histories.
        return self.long_latent_frames

    @property
    def long_condition_latents(self) -> int:
        return (self.long_prefix_frames - 1) // 4 + 1

    @property
    def short_condition_latents(self) -> int:
        # The short model conditions on a 65-frame prefix whose first 57
        # frames are a fixed label-independent mask and whose last 8 frames
        # are real observations. Both models therefore use 17 condition
        # latents and identical temporal positions.
        return self.long_condition_latents

    @property
    def short_real_prefix_latents(self) -> int:
        return (self.short_prefix_frames - 1) // 4 + 1

    @property
    def target_latents(self) -> int:
        return self.long_latent_frames - self.long_condition_latents


def apply_short_history_mask_numpy(
    frames: np.ndarray, cfg: SpringDatasetConfig
) -> np.ndarray:
    """Replace pixel frames 0..56 with a fixed background-only mask.

    The operation happens before VAE encoding.  It removes all true motion,
    colour, phase, and frequency information from the hidden history while
    keeping the original 129-frame timeline and the real pixels 57..128.
    Input/output shape is [T,H,W,3] with uint8-like RGB values.
    """
    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"Expected [T,H,W,3] RGB video, got {array.shape}")
    if array.shape[0] < cfg.prediction_start:
        raise ValueError("Video is too short for the frozen 65-frame prefix")
    out = array.copy()
    mask_rgb = np.asarray(cfg.render.background_rgb, dtype=out.dtype)
    out[: cfg.short_prefix_start, ...] = mask_rgb
    return out


def apply_short_history_mask_tensor(
    video: torch.Tensor, cfg: SpringDatasetConfig
) -> torch.Tensor:
    """Torch equivalent for normalized [B,3,T,H,W] videos in [-1,1]."""
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError(f"Expected [B,3,T,H,W], got {tuple(video.shape)}")
    if video.shape[2] < cfg.prediction_start:
        raise ValueError("Video is too short for the frozen 65-frame prefix")
    out = video.clone()
    rgb = torch.as_tensor(
        cfg.render.background_rgb, device=video.device, dtype=torch.float32
    ).div(127.5).sub(1.0).to(dtype=video.dtype)
    out[:, :, : cfg.short_prefix_start] = rgb.view(1, 3, 1, 1, 1)
    return out


@dataclass(frozen=True)
class OscillatorParameters:
    omega: float
    amplitude: float
    phase: float


@dataclass(frozen=True)
class SpringSample:
    sample_id: str
    trajectory_id: str
    pair_id: str
    base_seed: int
    split: str
    true_band: BandName
    color_label: ColorName
    variant: VariantName
    omega_true: float
    amplitude: float
    phase: float
    fps: int
    frames: int
    short_prefix_start: int
    prediction_start: int
    x_star: float
    v_star: float
    video: str
    source: str
    metadata: str
    trajectory: str
    render_hash_without_mass_fill: str


@dataclass
class MassTrack:
    x_px: np.ndarray
    y_px: np.ndarray
    area_px: np.ndarray
    candidate_count: np.ndarray
    mean_rgb: np.ndarray

    @property
    def detected(self) -> np.ndarray:
        return np.isfinite(self.x_px) & np.isfinite(self.y_px)


def canonical_color(band: BandName) -> ColorName:
    return "red" if band == "slow" else "blue"


def opposite_color(color: ColorName) -> ColorName:
    return "blue" if color == "red" else "red"


def color_band(color: ColorName) -> BandName:
    return "slow" if color == "red" else "fast"


def trajectory(params: OscillatorParameters, *, fps: int, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(num_frames, dtype=np.float64) / float(fps)
    x = params.amplitude * np.cos(params.omega * t + params.phase)
    v = -params.amplitude * params.omega * np.sin(params.omega * t + params.phase)
    return x, v


def anchored_trajectory(omega: float, x_star: float, v_star: float, times_after_boundary: np.ndarray) -> np.ndarray:
    """Unique undamped SHM trajectory with state (x*,v*) at time zero."""
    if omega <= 0:
        raise ValueError("omega must be positive")
    times = np.asarray(times_after_boundary, dtype=np.float64)
    return x_star * np.cos(omega * times) + (v_star / omega) * np.sin(omega * times)


def anchored_amplitude(omega: float, x_star: float, v_star: float) -> float:
    """Amplitude required for an omega-oscillator to match boundary state (x*,v*)."""
    if omega <= 0:
        raise ValueError("omega must be positive")
    return float(math.sqrt(float(x_star) ** 2 + (float(v_star) / float(omega)) ** 2))


def sample_parameters(
    rng: np.random.Generator,
    band: FrequencyBand,
    amplitude_low: float,
    amplitude_high: float,
) -> OscillatorParameters:
    return OscillatorParameters(
        omega=float(rng.uniform(band.low, band.high)),
        amplitude=float(rng.uniform(amplitude_low, amplitude_high)),
        phase=float(rng.uniform(0.0, 2.0 * math.pi)),
    )


def normalized_to_pixel_x(x_norm: float, cfg: SpringRenderConfig) -> float:
    return float(x_norm) * (cfg.width - 1)


def displacement_to_pixel_x(displacement: float, cfg: SpringRenderConfig) -> float:
    return normalized_to_pixel_x(cfg.equilibrium_x + float(displacement), cfg)


def pixel_x_to_displacement(xs_px: np.ndarray, cfg: SpringRenderConfig) -> np.ndarray:
    return np.asarray(xs_px, dtype=np.float64) / float(cfg.width - 1) - cfg.equilibrium_x


def _spring_polyline(anchor: tuple[float, float], mass_left: tuple[float, float], cfg: SpringRenderConfig) -> list[tuple[int, int]]:
    ax, ay = anchor
    mx, my = mass_left
    lead = min(8.0, max(2.0, (mx - ax) * 0.08))
    start_x, end_x = ax + lead, mx - lead
    points: list[tuple[int, int]] = [(round(ax), round(ay)), (round(start_x), round(ay))]
    if end_x <= start_x:
        points.append((round(mx), round(my)))
        return points
    segments = max(2, cfg.spring_coils * 2)
    for i in range(segments + 1):
        u = i / segments
        px = start_x + u * (end_x - start_x)
        py = ay if i in (0, segments) else ay + (cfg.spring_half_height_px if i % 2 else -cfg.spring_half_height_px)
        points.append((round(px), round(py)))
    points.extend([(round(end_x), round(ay)), (round(mx), round(my))])
    return points


def render_frame(displacement: float, color: ColorName, cfg: SpringRenderConfig) -> np.ndarray:
    image = Image.new("RGB", (cfg.width, cfg.height), cfg.background_rgb)
    draw = ImageDraw.Draw(image)
    anchor = (normalized_to_pixel_x(cfg.anchor_x, cfg), cfg.center_y * (cfg.height - 1))
    mass_x = displacement_to_pixel_x(displacement, cfg)
    mass_y = anchor[1]
    radius = cfg.mass_radius_px

    track_y = round(mass_y + radius + 5)
    draw.line((round(anchor[0]), track_y, cfg.width - 8, track_y), fill=(88, 92, 102), width=1)
    eq_x = round(normalized_to_pixel_x(cfg.equilibrium_x, cfg))
    draw.line((eq_x, track_y - 3, eq_x, track_y + 3), fill=(130, 135, 146), width=1)
    draw.rectangle(
        (round(anchor[0]) - 3, round(anchor[1]) - 12, round(anchor[0]) + 3, round(anchor[1]) + 12),
        fill=cfg.anchor_rgb,
    )
    draw.line(_spring_polyline(anchor, (mass_x - radius, mass_y), cfg), fill=cfg.spring_rgb, width=2, joint="curve")
    fill = cfg.red_rgb if color == "red" else cfg.blue_rgb
    bbox = (round(mass_x) - radius, round(mass_y) - radius, round(mass_x) + radius, round(mass_y) + radius)
    draw.ellipse(bbox, fill=fill, outline=(245, 245, 245), width=1)
    return np.asarray(image, dtype=np.uint8)


def render_video(displacements: Sequence[float], color: ColorName, cfg: SpringRenderConfig) -> np.ndarray:
    return np.stack([render_frame(float(x), color, cfg) for x in displacements], axis=0)


def write_video(path: Path, frames: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected [T,H,W,3], got {frames.shape}")
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=10, macro_block_size=None) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))


def load_video(path: Path, *, expected_frames: int | None = None) -> np.ndarray:
    reader = imageio.get_reader(path)
    try:
        frames = np.stack([frame[..., :3] for frame in reader], axis=0).astype(np.uint8)
    finally:
        reader.close()
    if expected_frames is not None and frames.shape[0] != expected_frames:
        raise ValueError(f"Expected {expected_frames} frames but read {frames.shape[0]} from {path}")
    return frames


def _hash_without_mass_fill(frames: np.ndarray, centers_x: np.ndarray, cfg: SpringRenderConfig) -> str:
    """Hash the raw render after replacing the full colour-sensitive mass ROI."""
    normalized = frames.copy()
    cy = round(cfg.center_y * (cfg.height - 1))
    radius = cfg.mass_radius_px + 3
    for frame, cx_float in zip(normalized, centers_x, strict=True):
        cx = round(float(cx_float))
        x0, x1 = max(0, cx - radius), min(cfg.width, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(cfg.height, cy + radius + 1)
        frame[y0:y1, x0:x1] = 0
    return hashlib.sha256(normalized.tobytes()).hexdigest()


def _opaque_id(split: str, base_seed: int, band_index: int, variant_index: int) -> str:
    token = hashlib.blake2b(f"{VERSION}:{split}:{base_seed}:{band_index}:{variant_index}".encode(), digest_size=10).hexdigest()
    return f"sample_{token}"


def _write_rows(path: Path, rows: Sequence[SpringSample]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty metadata file")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def generate_split(
    output_dir: Path,
    cfg: SpringDatasetConfig,
    *,
    split: str,
    num_base_seeds: int,
    include_conflicts: bool,
    overwrite: bool = False,
) -> list[SpringSample]:
    """Generate one deterministic split.

    A base seed yields one slow and one fast physical trajectory.  Evaluation
    additionally renders an aligned and a colour-swapped version of each exact
    trajectory.  Training never contains colour conflicts.
    """
    if num_base_seeds <= 0:
        raise ValueError("num_base_seeds must be positive")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    bands: dict[BandName, FrequencyBand] = {"slow": cfg.slow_band, "fast": cfg.fast_band}
    rows: list[SpringSample] = []

    for base_index in range(num_base_seeds):
        base_seed = cfg.seed + {"sanity": 0, "train": 1_000_000, "eval": 2_000_000}.get(split, 3_000_000) + base_index
        for band_index, band_name in enumerate(("slow", "fast")):
            # A separate stream per band prevents changing one branch from
            # perturbing the other branch's nuisance draws.
            rng = np.random.default_rng(np.random.SeedSequence([base_seed, band_index, 7919]))
            params = sample_parameters(rng, bands[band_name], cfg.amplitude_low, cfg.amplitude_high)
            x, v = trajectory(params, fps=cfg.render.fps, num_frames=cfg.render.num_frames)
            trajectory_token = hashlib.blake2b(f"{split}:{base_seed}:{band_index}".encode(), digest_size=10).hexdigest()
            trajectory_id = f"traj_{trajectory_token}"
            pair_id = f"pair_{trajectory_token}"
            trajectory_name = f"{trajectory_id}.npz"
            np.savez_compressed(output_dir / trajectory_name, x=x, v=v)

            canonical = canonical_color(band_name)
            variants: list[tuple[VariantName, ColorName]] = [("aligned", canonical)]
            if include_conflicts:
                variants.append(("conflict", opposite_color(canonical)))

            centers_x = np.asarray([displacement_to_pixel_x(value, cfg.render) for value in x])
            pair_hash: str | None = None
            for variant_index, (variant, color) in enumerate(variants):
                sample_id = _opaque_id(split, base_seed, band_index, variant_index)
                video_name, metadata_name = f"{sample_id}.mp4", f"{sample_id}.json"
                frames = render_video(x, color, cfg.render)
                render_hash = _hash_without_mass_fill(frames, centers_x, cfg.render)
                if pair_hash is None:
                    pair_hash = render_hash
                elif pair_hash != render_hash:
                    raise AssertionError("counterfactual renders differ outside the mass colour ROI")
                write_video(output_dir / video_name, frames, cfg.render.fps)
                record = {
                    "benchmark_version": VERSION,
                    "sample_id": sample_id,
                    "trajectory_id": trajectory_id,
                    "pair_id": pair_id,
                    "base_seed": base_seed,
                    "split": split,
                    "true_band": band_name,
                    "color_label": color,
                    "variant": variant,
                    "omega_true": params.omega,
                    "amplitude": params.amplitude,
                    "phase": params.phase,
                    "fps": cfg.render.fps,
                    "frames": cfg.render.num_frames,
                    "short_prefix_start": cfg.short_prefix_start,
                    "prediction_start": cfg.prediction_start,
                    "x_star": float(x[cfg.prediction_start - 1]),
                    "v_star": float(v[cfg.prediction_start - 1]),
                    "video": video_name,
                    "trajectory": trajectory_name,
                    "render_hash_without_mass_fill": render_hash,
                }
                (output_dir / metadata_name).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
                rows.append(
                    SpringSample(
                        sample_id=sample_id,
                        trajectory_id=trajectory_id,
                        pair_id=pair_id,
                        base_seed=base_seed,
                        split=split,
                        true_band=band_name,
                        color_label=color,
                        variant=variant,
                        omega_true=params.omega,
                        amplitude=params.amplitude,
                        phase=params.phase,
                        fps=cfg.render.fps,
                        frames=cfg.render.num_frames,
                        short_prefix_start=cfg.short_prefix_start,
                        prediction_start=cfg.prediction_start,
                        x_star=float(x[cfg.prediction_start - 1]),
                        v_star=float(v[cfg.prediction_start - 1]),
                        video=video_name,
                        source=video_name,
                        metadata=metadata_name,
                        trajectory=trajectory_name,
                        render_hash_without_mass_fill=render_hash,
                    )
                )

    _write_rows(output_dir / "metadata.csv", rows)
    audit = audit_rows(rows, cfg, expect_conflicts=include_conflicts)
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return rows


def audit_rows(rows: Sequence[SpringSample], cfg: SpringDatasetConfig, *, expect_conflicts: bool) -> dict[str, Any]:
    by_band = {"slow": 0, "fast": 0}
    by_color = {"red": 0, "blue": 0}
    by_variant = {"aligned": 0, "conflict": 0}
    pairs: dict[str, list[SpringSample]] = {}
    errors: list[str] = []
    for row in rows:
        by_band[row.true_band] += 1
        by_color[row.color_label] += 1
        by_variant[row.variant] += 1
        pairs.setdefault(row.pair_id, []).append(row)
        if row.variant == "aligned" and row.color_label != canonical_color(row.true_band):
            errors.append(f"noncanonical aligned row {row.sample_id}")
        if row.variant == "conflict" and row.color_label == canonical_color(row.true_band):
            errors.append(f"nonconflicting conflict row {row.sample_id}")
        band = cfg.slow_band if row.true_band == "slow" else cfg.fast_band
        if not band.contains(row.omega_true):
            errors.append(f"omega outside true band {row.sample_id}")
    for pair_id, pair in pairs.items():
        expected = 2 if expect_conflicts else 1
        if len(pair) != expected:
            errors.append(f"{pair_id} contains {len(pair)} rows, expected {expected}")
        if len({p.render_hash_without_mass_fill for p in pair}) != 1:
            errors.append(f"{pair_id} differs outside colour ROI")
        invariant = {(p.omega_true, p.amplitude, p.phase, p.x_star, p.v_star, p.trajectory) for p in pair}
        if len(invariant) != 1:
            errors.append(f"{pair_id} counterfactual physics mismatch")
    if by_band["slow"] != by_band["fast"]:
        errors.append("slow/fast count imbalance")
    if expect_conflicts and by_variant["aligned"] != by_variant["conflict"]:
        errors.append("aligned/conflict count imbalance")
    if errors:
        raise AssertionError("; ".join(errors[:20]))
    return {
        "benchmark_version": VERSION,
        "num_rows": len(rows),
        "num_physical_trajectories": len(pairs),
        "by_band": by_band,
        "by_color": by_color,
        "by_variant": by_variant,
        "short_masked_pixel_frames": [0, cfg.short_prefix_start - 1],
        "short_real_prefix_pixel_frames": [cfg.short_prefix_start, cfg.prediction_start - 1],
        "long_prefix_pixel_frames": [0, cfg.prediction_start - 1],
        "future_pixel_frames": [cfg.prediction_start, cfg.render.num_frames - 1],
        "long_latent_frames": cfg.long_latent_frames,
        "short_latent_frames": cfg.short_latent_frames,
        "long_condition_latents": cfg.long_condition_latents,
        "short_condition_latents": cfg.short_condition_latents,
        "short_real_prefix_latents_for_reference_only": cfg.short_real_prefix_latents,
        "target_latents": cfg.target_latents,
    }


def config_to_dict(cfg: SpringDatasetConfig) -> dict[str, Any]:
    return {
        "benchmark_version": VERSION,
        "slow_band": asdict(cfg.slow_band),
        "fast_band": asdict(cfg.fast_band),
        "amplitude_low": cfg.amplitude_low,
        "amplitude_high": cfg.amplitude_high,
        "short_prefix_start": cfg.short_prefix_start,
        "prediction_start": cfg.prediction_start,
        "short_mask_mode": cfg.short_mask_mode,
        "seed": cfg.seed,
        "train_base_seeds": cfg.train_base_seeds,
        "eval_base_seeds": cfg.eval_base_seeds,
        "sanity_base_seeds": cfg.sanity_base_seeds,
        "tiny_train_videos": cfg.tiny_train_videos,
        "pilot_train_videos": cfg.pilot_train_videos,
        "route_reference_amplitude_multiplier": cfg.route_reference_amplitude_multiplier,
        "frequency_band_tolerance": cfg.frequency_band_tolerance,
        "free_shm_rmse_threshold": cfg.free_shm_rmse_threshold,
        "render": asdict(cfg.render),
    }


def dataclass_config_from_dict(payload: dict[str, Any]) -> SpringDatasetConfig:
    render_payload = dict(payload.get("render", {}))
    for key in ("background_rgb", "spring_rgb", "anchor_rgb", "red_rgb", "blue_rgb"):
        if key in render_payload:
            render_payload[key] = tuple(render_payload[key])
    slow_payload = dict(payload["slow_band"])
    fast_payload = dict(payload["fast_band"])
    slow_payload.pop("name", None)
    fast_payload.pop("name", None)
    return SpringDatasetConfig(
        slow_band=FrequencyBand("slow", **slow_payload),
        fast_band=FrequencyBand("fast", **fast_payload),
        amplitude_low=float(payload.get("amplitude_low", 0.10)),
        amplitude_high=float(payload.get("amplitude_high", 0.17)),
        render=SpringRenderConfig(**render_payload),
        short_prefix_start=int(payload.get("short_prefix_start", 57)),
        prediction_start=int(payload.get("prediction_start", 65)),
        short_mask_mode=str(payload.get("short_mask_mode", "background_only")),
        seed=int(payload.get("seed", 3407)),
        train_base_seeds=int(payload.get("train_base_seeds", 1024)),
        eval_base_seeds=int(payload.get("eval_base_seeds", 64)),
        sanity_base_seeds=int(payload.get("sanity_base_seeds", 8)),
        tiny_train_videos=int(payload.get("tiny_train_videos", 16)),
        pilot_train_videos=int(payload.get("pilot_train_videos", 256)),
        route_reference_amplitude_multiplier=float(
            payload.get("route_reference_amplitude_multiplier", 1.0)
        ),
        frequency_band_tolerance=float(payload.get("frequency_band_tolerance", 0.005)),
        free_shm_rmse_threshold=float(payload.get("free_shm_rmse_threshold", 0.035)),
    )


def generate_dataset(
    config: SpringDatasetConfig,
    root: Path,
    *,
    splits: Sequence[str] = ("sanity", "train", "eval"),
    overwrite: bool = False,
) -> Path:
    """Generate configured Spring splits and return the dataset root."""
    root.mkdir(parents=True, exist_ok=True)
    counts = {
        "sanity": config.sanity_base_seeds,
        "train": config.train_base_seeds,
        "eval": config.eval_base_seeds,
    }
    summary: dict[str, Any] = {
        "benchmark_version": VERSION,
        "splits": {},
    }
    common_samples: list[dict[str, Any]] = []
    for split in splits:
        if split not in counts:
            raise ValueError(f"Unknown Spring split: {split}")
        out = root / "videos" / split
        if out.exists() and any(out.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"Refusing to overwrite non-empty directory: {out}"
                )
            shutil.rmtree(out)
        rows = generate_split(
            out,
            config,
            split=split,
            num_base_seeds=counts[split],
            include_conflicts=(split != "train"),
            overwrite=False,
        )
        summary["splits"][split] = {
            "directory": str(out),
            "rows": len(rows),
            "physical_trajectories": counts[split] * 2,
        }
        common_samples.extend(
            samples_from_metadata_csv(
                root,
                out / "metadata.csv",
                split=split,
                subset=split,
            )
        )

    metadata_dir = root / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    resolved = config_to_dict(config)
    (metadata_dir / "generation_config_resolved.yaml").write_text(
        yaml.safe_dump(
            resolved,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    (root / "build_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    write_dataset(
        root,
        experiment="spring",
        dataset=VERSION,
        samples=common_samples,
        extra={
            "generation_config": (
                "metadata/generation_config_resolved.yaml"
            ),
            "build_summary": "build_summary.json",
        },
    )
    return root


def audit_dataset(
    root: Path,
    config: SpringDatasetConfig,
    out: Path,
    *,
    video_limit_per_split: int = 0,
) -> dict[str, Any]:
    """Audit a Spring dataset through the experiment-owned data boundary."""
    from sshv2.experiments.spring_shortcuts_v4.evaluation import (
        detect_mass_track,
        route_metrics_against_bands,
        validity_metrics,
    )

    def read_metadata_rows(path: Path) -> list[dict[str, str]]:
        with path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def prefix_band_fit(
        prefix: np.ndarray,
        fps: int,
        bands: dict[str, tuple[float, float]],
        grid_size: int = 501,
    ) -> tuple[str, float, float]:
        t = np.arange(prefix.size, dtype=np.float64) / float(fps)

        def best(interval: tuple[float, float]) -> float:
            value = float("inf")
            for omega in np.linspace(interval[0], interval[1], grid_size):
                design = np.stack(
                    [np.cos(omega * t), np.sin(omega * t), np.ones_like(t)],
                    axis=1,
                )
                coeff, *_ = np.linalg.lstsq(design, prefix, rcond=None)
                pred = design @ coeff
                value = min(
                    value,
                    float(np.sqrt(np.mean((prefix - pred) ** 2))),
                )
            return value

        ds, df = best(bands["slow"]), best(bands["fast"])
        return ("slow" if ds < df else "fast"), ds, df

    bands = {
        "slow": (config.slow_band.low, config.slow_band.high),
        "fast": (config.fast_band.low, config.fast_band.high),
    }
    report: dict[str, Any] = {
        "passed": True,
        "errors": [],
        "warnings": [],
        "frequency_band_tolerance": config.frequency_band_tolerance,
        "splits": {},
        "prefix_identifiability": {"short": [], "long": []},
        "short_input_design": {
            "mask_mode": config.short_mask_mode,
            "masked_frames": [0, config.short_prefix_start - 1],
            "real_observed_frames": [
                config.short_prefix_start,
                config.prediction_start - 1,
            ],
            "future_frames": [
                config.prediction_start,
                config.render.num_frames - 1,
            ],
        },
    }

    for split in ("sanity", "train", "eval"):
        split_dir = root / "videos" / split
        rows = read_metadata_rows(split_dir / "metadata.csv")
        expected_conflicts = split != "train"
        pair_index: dict[str, list[dict[str, str]]] = defaultdict(list)
        count: defaultdict[tuple[str, str, str], int] = defaultdict(int)
        for row in rows:
            pair_index[row["pair_id"]].append(row)
            count[
                (row["true_band"], row["variant"], row["color_label"])
            ] += 1
            canonical = canonical_color(row["true_band"])
            if split == "train" and (
                row["variant"] != "aligned"
                or row["color_label"] != canonical
            ):
                report["errors"].append(
                    f"training shortcut assignment violation: {row['sample_id']}"
                )
            if (
                row["variant"] == "conflict"
                and row["color_label"] == canonical
            ):
                report["errors"].append(
                    f"conflict row has canonical colour: {row['sample_id']}"
                )
            if any(
                token in Path(row["video"]).stem.lower()
                for token in (
                    "slow",
                    "fast",
                    "red",
                    "blue",
                    "aligned",
                    "conflict",
                )
            ):
                report["errors"].append(
                    f"label-revealing filename: {row['video']}"
                )

        for pair_id, pair in pair_index.items():
            expected = 2 if expected_conflicts else 1
            if len(pair) != expected:
                report["errors"].append(
                    f"{split}:{pair_id} has {len(pair)} rows, "
                    f"expected {expected}"
                )
            invariant = {
                (
                    row["trajectory"],
                    row["omega_true"],
                    row["amplitude"],
                    row["phase"],
                    row["x_star"],
                    row["v_star"],
                )
                for row in pair
            }
            if len(invariant) != 1:
                report["errors"].append(
                    f"counterfactual physics mismatch: {split}:{pair_id}"
                )

        selected = rows[: video_limit_per_split or None]
        valid_count = 0
        gt_route_failures = 0
        decoded_pair_diffs: list[float] = []
        videos_by_pair: dict[
            str,
            list[tuple[dict[str, str], np.ndarray]],
        ] = defaultdict(list)
        for row in selected:
            frames = load_video(
                split_dir / row["video"],
                expected_frames=config.render.num_frames,
            )
            videos_by_pair[row["pair_id"]].append((row, frames))
            future = frames[config.prediction_start :]
            track = detect_mass_track(future, config.render)
            validity = validity_metrics(
                track,
                config.render,
                x_star=float(row["x_star"]),
            )
            valid_count += int(validity["valid"])
            if not validity["valid"]:
                report["errors"].append(
                    f"GT detector/validity failure: "
                    f"{split}:{row['sample_id']}:{validity}"
                )
                continue
            pred_x = pixel_x_to_displacement(track.x_px, config.render)
            metrics = route_metrics_against_bands(
                pred_x,
                np.isfinite(pred_x),
                true_band=(
                    config.slow_band
                    if row["true_band"] == "slow"
                    else config.fast_band
                ),
                color_implied_band=(
                    config.slow_band
                    if color_band(row["color_label"]) == "slow"
                    else config.fast_band
                ),
                slow_band=config.slow_band,
                fast_band=config.fast_band,
                omega_true=float(row["omega_true"]),
                x_star=float(row["x_star"]),
                v_star=float(row["v_star"]),
                fps=config.render.fps,
                amplitude_low=config.amplitude_low,
                amplitude_high=config.amplitude_high,
                reference_amplitude_multiplier=(
                    config.route_reference_amplitude_multiplier
                ),
                frequency_band_tolerance=config.frequency_band_tolerance,
                free_shm_rmse_threshold=config.free_shm_rmse_threshold,
            )
            if (
                row["variant"] == "conflict"
                and metrics["route_label"] != "physics_frequency"
            ):
                gt_route_failures += 1
                report["errors"].append(
                    f"GT conflict future not classified as physics: "
                    f"{split}:{row['sample_id']}:{metrics['route_label']}"
                )

            trajectory_values = np.load(
                split_dir / row["trajectory"]
            )["x"].astype(np.float64)
            quantized = np.rint(
                (config.render.equilibrium_x + trajectory_values)
                * (config.render.width - 1)
            )
            quantized = (
                quantized / (config.render.width - 1)
                - config.render.equilibrium_x
            )
            for history, prefix in (
                (
                    "short",
                    quantized[
                        config.short_prefix_start : config.prediction_start
                    ],
                ),
                ("long", quantized[: config.prediction_start]),
            ):
                predicted_band, ds, df = prefix_band_fit(
                    prefix,
                    config.render.fps,
                    bands,
                )
                report["prefix_identifiability"][history].append(
                    {
                        "correct": predicted_band == row["true_band"],
                        "margin": abs(ds - df),
                    }
                )

        for pair in videos_by_pair.values():
            if len(pair) != 2:
                continue
            (_, a), (row_b, b) = pair
            trajectory_values = np.load(
                split_dir / row_b["trajectory"]
            )["x"]
            diffs: list[float] = []
            yy, xx = np.mgrid[
                : config.render.height,
                : config.render.width,
            ]
            cy = config.render.center_y * (config.render.height - 1)
            for frame_id, displacement in enumerate(trajectory_values):
                cx = displacement_to_pixel_x(
                    float(displacement),
                    config.render,
                )
                outside = (
                    (xx - cx) ** 2 + (yy - cy) ** 2
                    > (config.render.mass_radius_px + 6) ** 2
                )
                diffs.append(
                    float(
                        np.mean(
                            np.abs(
                                a[frame_id].astype(float)
                                - b[frame_id].astype(float)
                            )[outside]
                        )
                    )
                )
            decoded_pair_diffs.append(float(np.mean(diffs)))

        report["splits"][split] = {
            "rows": len(rows),
            "physical_trajectories": len(pair_index),
            "audited_videos": len(selected),
            "gt_validity_rate": valid_count / max(1, len(selected)),
            "gt_conflict_route_failures": gt_route_failures,
            "mean_decoded_counterfactual_difference_outside_roi": (
                float(np.mean(decoded_pair_diffs))
                if decoded_pair_diffs
                else None
            ),
            "counts": {
                "|".join(key): value
                for key, value in count.items()
            },
        }

    for history in ("short", "long"):
        values = report["prefix_identifiability"][history]
        report["prefix_identifiability"][history] = {
            "n": len(values),
            "accuracy": (
                float(np.mean([value["correct"] for value in values]))
                if values
                else None
            ),
            "median_margin": (
                float(np.median([value["margin"] for value in values]))
                if values
                else None
            ),
        }
    long_accuracy = report["prefix_identifiability"]["long"]["accuracy"]
    short_accuracy = report["prefix_identifiability"]["short"]["accuracy"]
    if (
        long_accuracy is not None
        and short_accuracy is not None
        and long_accuracy <= short_accuracy
    ):
        report["warnings"].append(
            "long prefix is not more identifiable than short prefix "
            "under the offline fit"
        )
    report["passed"] = not report["errors"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise ValueError(
            "Spring dataset audit failed; inspect the persisted report"
        )
    return report
