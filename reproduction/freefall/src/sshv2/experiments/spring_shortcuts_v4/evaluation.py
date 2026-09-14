"""Prediction detection and metrics owned by the Spring experiment."""

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import yaml

from sshv2.common.results import (
    find_prediction,
    write_metrics,
    write_predictions,
)
from sshv2.experiments.spring_shortcuts_v4.data import (
    FrequencyBand,
    HistoryName,
    MassTrack,
    SpringDatasetConfig,
    SpringRenderConfig,
    apply_short_history_mask_tensor,
    anchored_amplitude,
    anchored_trajectory,
    canonical_color,
    color_band,
    dataclass_config_from_dict,
    displacement_to_pixel_x,
    load_video,
    pixel_x_to_displacement,
    write_video,
)

def detect_mass_track(
    frames: np.ndarray,
    cfg: SpringRenderConfig,
    *,
    saturation_threshold: int = 45,
    value_threshold: int = 55,
    min_area: int = 28,
    max_area: int = 420,
) -> MassTrack:
    """Track the high-saturation mass without assuming red or blue hue."""
    frames = np.asarray(frames, dtype=np.uint8)
    n = frames.shape[0]
    xs = np.full(n, np.nan, dtype=np.float64)
    ys = np.full(n, np.nan, dtype=np.float64)
    areas = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)
    rgbs = np.full((n, 3), np.nan, dtype=np.float64)
    expected_y = cfg.center_y * (cfg.height - 1)
    expected_area = math.pi * cfg.mass_radius_px**2
    previous_x: float | None = None

    kernel = np.ones((3, 3), dtype=np.uint8)
    for frame_id, frame in enumerate(frames):
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask = ((hsv[..., 1] >= saturation_threshold) & (hsv[..., 2] >= value_threshold)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        candidates: list[tuple[float, int]] = []
        for label in range(1, num):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if not (min_area <= area <= max_area):
                continue
            cx, cy = centroids[label]
            if abs(cy - expected_y) > 18:
                continue
            temporal = 0.0 if previous_x is None else 0.30 * abs(cx - previous_x)
            score = abs(cy - expected_y) + 0.035 * abs(area - expected_area) + temporal
            candidates.append((score, label))
        counts[frame_id] = len(candidates)
        if not candidates:
            continue
        _, label = min(candidates)
        cx, cy = centroids[label]
        component = labels == label
        xs[frame_id], ys[frame_id] = float(cx), float(cy)
        areas[frame_id] = float(stats[label, cv2.CC_STAT_AREA])
        rgbs[frame_id] = frame[component].mean(axis=0)
        previous_x = float(cx)
    return MassTrack(xs, ys, areas, counts, rgbs)


def max_consecutive_true(flags: np.ndarray) -> int:
    best = current = 0
    for flag in np.asarray(flags, dtype=bool):
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def _adjacent_jumps(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    good = np.isfinite(values[:-1]) & np.isfinite(values[1:])
    return np.abs(np.diff(values)[good])


def validity_metrics(
    track: MassTrack,
    cfg: SpringRenderConfig,
    *,
    x_star: float | None = None,
    detection_rate_required: float = 0.90,
    max_missing_run_allowed: int = 3,
    max_jump_px: float = 12.0,
    max_boundary_jump_px: float = 12.0,
    max_y_deviation_px: float = 7.0,
    max_multiple_component_rate: float = 0.10,
) -> dict[str, Any]:
    detected = track.detected
    detection_rate = float(detected.mean())
    missing_run = max_consecutive_true(~detected)
    jumps = _adjacent_jumps(track.x_px)
    max_jump = float(jumps.max()) if jumps.size else float("inf")
    expected_y = cfg.center_y * (cfg.height - 1)
    y_deviation = float(np.nanmax(np.abs(track.y_px - expected_y))) if detected.any() else float("inf")
    valid_x = track.x_px[detected]
    in_bounds = bool(valid_x.size and np.all(valid_x >= cfg.mass_radius_px) and np.all(valid_x <= cfg.width - 1 - cfg.mass_radius_px))
    multiple_rate = float(np.mean(track.candidate_count > 1))
    detected_area = track.area_px[detected]
    expected_area = math.pi * cfg.mass_radius_px**2
    area_median = float(np.median(detected_area)) if detected_area.size else float("nan")
    area_ok = bool(detected_area.size and 0.35 * expected_area <= area_median <= 2.10 * expected_area)
    boundary_jump = float("nan")
    if x_star is not None and detected.any():
        first = int(np.flatnonzero(detected)[0])
        star_px = displacement_to_pixel_x(x_star, cfg)
        boundary_jump = float(abs(track.x_px[first] - star_px))
    boundary_ok = x_star is None or (math.isfinite(boundary_jump) and boundary_jump <= max_boundary_jump_px)
    valid = bool(
        detection_rate >= detection_rate_required
        and missing_run <= max_missing_run_allowed
        and max_jump <= max_jump_px
        and y_deviation <= max_y_deviation_px
        and multiple_rate <= max_multiple_component_rate
        and in_bounds
        and area_ok
        and boundary_ok
    )
    return {
        "valid": valid,
        "detection_rate": detection_rate,
        "max_missing_run": missing_run,
        "max_adjacent_jump_px": max_jump,
        "boundary_jump_px": boundary_jump,
        "max_y_deviation_px": y_deviation,
        "multiple_component_rate": multiple_rate,
        "median_area_px": area_median,
        "area_ok": area_ok,
        "in_bounds": in_bounds,
    }


def _future_times(num_future_frames: int, fps: int) -> np.ndarray:
    return (np.arange(num_future_frames, dtype=np.float64) + 1.0) / float(fps)


def family_distance(
    predicted_x: np.ndarray,
    valid_mask: np.ndarray,
    *,
    x_star: float,
    v_star: float,
    band: FrequencyBand,
    fps: int,
    grid_size: int = 501,
) -> tuple[float, float, float, np.ndarray]:
    """Nearest boundary-anchored trajectory in a frequency band.

    Returns RMSE, best omega, required anchored amplitude, and best curve.
    This is an auxiliary continuity-sensitive metric, not the primary route
    classifier.
    """
    predicted_x = np.asarray(predicted_x, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if predicted_x.ndim != 1 or valid_mask.shape != predicted_x.shape:
        raise ValueError("predicted_x and valid_mask must be aligned 1-D arrays")
    if valid_mask.sum() < 6:
        return float("nan"), float("nan"), float("nan"), np.full_like(predicted_x, np.nan)
    times = _future_times(predicted_x.size, fps)
    best_rmse, best_omega, best_amplitude = float("inf"), float("nan"), float("nan")
    best_curve = np.full_like(predicted_x, np.nan)
    for omega in np.linspace(band.low, band.high, grid_size):
        candidate = anchored_trajectory(float(omega), x_star, v_star, times)
        rmse = float(np.sqrt(np.mean((predicted_x[valid_mask] - candidate[valid_mask]) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_omega = float(omega)
            best_amplitude = anchored_amplitude(best_omega, x_star, v_star)
            best_curve = candidate
    return best_rmse, best_omega, best_amplitude, best_curve


def fit_free_shm_frequency(
    predicted_x: np.ndarray,
    valid_mask: np.ndarray,
    *,
    fps: int,
    omega_low: float,
    omega_high: float,
    grid_size: int = 1601,
) -> tuple[float, float, float, float]:
    """Fit x(t)=a cos(wt)+b sin(wt)+c.

    Returns omega, RMSE, center offset, and fitted oscillation amplitude.
    This fit intentionally does not force exact boundary-state continuity and
    therefore serves as the primary frequency-route readout.
    """
    x = np.asarray(predicted_x, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if mask.sum() < 6:
        return float("nan"), float("nan"), float("nan"), float("nan")
    t = _future_times(x.size, fps)
    best = (float("nan"), float("inf"), float("nan"), float("nan"))
    for omega in np.linspace(omega_low, omega_high, grid_size):
        design = np.stack([np.cos(omega * t), np.sin(omega * t), np.ones_like(t)], axis=1)
        coeff, *_ = np.linalg.lstsq(design[mask], x[mask], rcond=None)
        pred = design @ coeff
        rmse = float(np.sqrt(np.mean((x[mask] - pred[mask]) ** 2)))
        if rmse < best[1]:
            amplitude = float(math.sqrt(float(coeff[0]) ** 2 + float(coeff[1]) ** 2))
            best = (float(omega), rmse, float(coeff[2]), amplitude)
    return best


def classify_omega(
    omega: float,
    slow: FrequencyBand,
    fast: FrequencyBand,
    *,
    tolerance: float = 0.0,
) -> str:
    """Classify a fitted frequency with an explicit measurement tolerance."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if 2 * tolerance >= fast.low - slow.high:
        raise ValueError("tolerance makes the expanded frequency bands overlap")
    if not math.isfinite(omega):
        return "invalid"
    if slow.contains(omega, atol=tolerance):
        return "slow"
    if fast.contains(omega, atol=tolerance):
        return "fast"
    if slow.high < omega < fast.low:
        return "between_bands"
    return "outside_bands"


def _primary_frequency_route_label(
    *,
    omega_class: str,
    free_rmse: float,
    true_band: FrequencyBand,
    color_implied_band: FrequencyBand,
    residual_threshold: float,
) -> str:
    if true_band.name == color_implied_band.name:
        return "iid_not_attributed"
    if not math.isfinite(free_rmse) or omega_class == "invalid":
        return "invalid"
    if free_rmse > residual_threshold:
        return "off_frequency_family"
    if omega_class == true_band.name:
        return "physics_frequency"
    if omega_class == color_implied_band.name:
        return "shortcut_frequency"
    if omega_class == "between_bands":
        return "compromise_frequency"
    return "outside_frequency_bands"


def route_metrics_against_bands(
    predicted_x: np.ndarray,
    valid_mask: np.ndarray,
    *,
    true_band: FrequencyBand,
    color_implied_band: FrequencyBand,
    slow_band: FrequencyBand,
    fast_band: FrequencyBand,
    omega_true: float,
    x_star: float,
    v_star: float,
    fps: int,
    amplitude_low: float = 0.10,
    amplitude_high: float = 0.17,
    reference_amplitude_multiplier: float = 1.0,
    frequency_band_tolerance: float = 0.0,
    free_shm_rmse_threshold: float = 0.035,
    ambiguity_threshold: float = 0.05,
    off_family_rmse: float = 0.035,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Compute primary free-frequency and auxiliary anchored-family metrics."""
    x = np.asarray(predicted_x, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    d_true, omega_true_hat, true_amplitude, true_curve = family_distance(
        x, mask, x_star=x_star, v_star=v_star, band=true_band, fps=fps
    )
    d_color, omega_color_hat, color_amplitude, color_curve = family_distance(
        x, mask, x_star=x_star, v_star=v_star, band=color_implied_band, fps=fps
    )
    exact_curve = anchored_trajectory(omega_true, x_star, v_star, _future_times(x.size, fps))
    d_exact = float(np.sqrt(np.mean((x[mask] - exact_curve[mask]) ** 2))) if mask.sum() else float("nan")
    omega_free, free_rmse, center_offset, free_amplitude = fit_free_shm_frequency(
        x,
        mask,
        fps=fps,
        omega_low=max(0.2, slow_band.low * 0.55),
        omega_high=fast_band.high * 1.35,
    )
    omega_free_class = classify_omega(
        omega_free,
        slow_band,
        fast_band,
        tolerance=frequency_band_tolerance,
    )
    primary_label = _primary_frequency_route_label(
        omega_class=omega_free_class,
        free_rmse=free_rmse,
        true_band=true_band,
        color_implied_band=color_implied_band,
        residual_threshold=free_shm_rmse_threshold,
    )

    amplitude_tol = 1e-9
    lower_reasonable = amplitude_low / reference_amplitude_multiplier - amplitude_tol
    upper_reasonable = amplitude_high * reference_amplitude_multiplier + amplitude_tol
    color_reference_ood = bool(
        math.isfinite(color_amplitude)
        and not (lower_reasonable <= color_amplitude <= upper_reasonable)
    )
    true_reference_ood = bool(
        math.isfinite(true_amplitude)
        and not (lower_reasonable <= true_amplitude <= upper_reasonable)
    )

    if not (math.isfinite(d_true) and math.isfinite(d_color)):
        score, anchored_label = float("nan"), "invalid"
    else:
        score = float((d_true - d_color) / (d_true + d_color + eps))
        if true_band.name != color_implied_band.name and color_reference_ood:
            anchored_label = "reference_out_of_distribution"
        elif min(d_true, d_color) > off_family_rmse:
            anchored_label = "off_family"
        elif abs(score) < ambiguity_threshold:
            anchored_label = "ambiguous"
        elif score > 0:
            anchored_label = "shortcut_band"
        else:
            anchored_label = "physics_band"

    return {
        # Primary result: unconstrained SHM frequency fit.
        "route_label": primary_label,
        "primary_route_label": primary_label,
        "primary_route_definition": "free_frequency_band_with_residual_gate",
        "omega_hat_free": omega_free,
        "omega_hat_free_class": omega_free_class,
        "omega_hat_free_amplitude": free_amplitude,
        "omega_abs_error": abs(omega_free - omega_true) if math.isfinite(omega_free) else float("nan"),
        "free_shm_rmse": free_rmse,
        "free_shm_rmse_threshold": free_shm_rmse_threshold,
        "frequency_band_tolerance": frequency_band_tolerance,
        "free_center_offset": center_offset,
        "free_fit_amplitude_ood": bool(
            math.isfinite(free_amplitude)
            and not (amplitude_low - amplitude_tol <= free_amplitude <= amplitude_high + amplitude_tol)
        ),
        # Auxiliary result: exact boundary-state continuity.
        "anchored_route_label": anchored_label,
        "route_score_band": score,
        "d_true_band": d_true,
        "d_color_band": d_color,
        "d_exact_physics": d_exact,
        "omega_hat_true_family": omega_true_hat,
        "omega_hat_color_family": omega_color_hat,
        "true_reference_amplitude": true_amplitude,
        "color_reference_amplitude": color_amplitude,
        "true_reference_ood": true_reference_ood,
        "color_reference_ood": color_reference_ood,
        "reference_amplitude_reasonable_range": [lower_reasonable, upper_reasonable],
        "min_band_residual": min(d_true, d_color) if math.isfinite(d_true + d_color) else float("nan"),
        "best_true_curve": true_curve,
        "best_color_curve": color_curve,
    }

def classify_detected_color(mean_rgb: np.ndarray, cfg: SpringRenderConfig) -> str:
    rgb = np.asarray(mean_rgb, dtype=np.float64)
    if rgb.shape != (3,) or not np.isfinite(rgb).all() or rgb.sum() <= 1e-6:
        return "unknown"
    chroma = rgb / rgb.sum()
    refs = {
        "red": np.asarray(cfg.red_rgb, dtype=np.float64) / np.sum(cfg.red_rgb),
        "blue": np.asarray(cfg.blue_rgb, dtype=np.float64) / np.sum(cfg.blue_rgb),
    }
    distances = {name: float(np.linalg.norm(chroma - ref)) for name, ref in refs.items()}
    name = min(distances, key=distances.get)
    return name if distances[name] <= 0.25 else "unknown"


def finite_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    return value


@dataclass(frozen=True)
class SpringEvaluationConfig:
    data: SpringDatasetConfig
    history: HistoryName
    limit: int = 0


@dataclass(frozen=True)
class SpringPredictionConfig:
    training_config: Path
    checkpoint: Path
    data_config: Path
    history: HistoryName
    steps: int = 20
    limit: int = 0
    variant: str = "all"
    device: str = "cuda"
    seed_offset: int = 17_000_000
    dataset_id: str = "spring_shortcuts_v4"

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "SpringPredictionConfig":
        return cls(
            training_config=Path(value["training_config"]),
            checkpoint=Path(value["checkpoint"]),
            data_config=Path(value["data_config"]),
            history=str(value["history"]),
            steps=int(
                value.get("steps", value.get("diffusion_steps", 20))
            ),
            limit=int(value.get("limit") or 0),
            variant=str(value.get("variant", "all")),
            device=str(value.get("device", "cuda")),
            seed_offset=int(value.get("seed_offset", 17_000_000)),
            dataset_id=str(
                value.get("dataset_id") or "spring_shortcuts_v4"
            ),
        )


def _spring_config_from_mapping(
    value: Mapping[str, Any],
) -> SpringEvaluationConfig:
    data_path = Path(value["data_config"])
    data = dataclass_config_from_dict(
        yaml.safe_load(data_path.read_text(encoding="utf-8"))
    )
    history = str(value["history"])
    if history not in ("short", "long"):
        raise ValueError("history must be short or long")
    return SpringEvaluationConfig(
        data=data,
        history=history,
        limit=int(value.get("limit") or 0),
    )


def predict(
    dataset_dir: Path,
    prediction_dir: Path,
    config: SpringPredictionConfig | Mapping[str, Any],
) -> dict[str, Any]:
    """Generate Spring futures without running trajectory metrics."""
    if isinstance(config, Mapping):
        config = SpringPredictionConfig.from_mapping(config)
    if config.history not in ("short", "long"):
        raise ValueError("history must be short or long")
    if config.variant not in ("all", "aligned", "conflict"):
        raise ValueError("variant must be all, aligned, or conflict")
    if config.steps <= 0:
        raise ValueError("steps must be positive")

    import torch
    from sshv2.wan.config import StandardTrainingConfig
    from sshv2.wan.trainer import WanTrainingModule

    data_config = dataclass_config_from_dict(
        yaml.safe_load(
            config.data_config.read_text(encoding="utf-8")
        )
    )
    training = StandardTrainingConfig.from_file(config.training_config)
    if (
        data_config.short_condition_latents
        != data_config.long_condition_latents
    ):
        raise AssertionError(
            "V4 requires equal short/long condition-latent counts"
        )
    expected_conditions = data_config.long_condition_latents
    if training.model.num_condition_frames != expected_conditions:
        raise ValueError(
            "Training config condition-latent count does not match "
            f"{config.history} evaluation"
        )
    training.model.dit.ckpt_file = config.checkpoint
    model = WanTrainingModule(
        dit_config=training.model.dit,
        vae_config=training.model.vae,
        no_encoding=False,
        num_condition_frames=training.model.num_condition_frames,
        num_inference_steps=config.steps,
        pipeline_type=training.model.pipe,
        pipeline_kwargs=training.model.pipe_kwargs,
    )
    pipe = model.pipe
    pipe.to(config.device)
    pipe.load_models_to_device(("dit", "vae"))
    pipe.pre_encoded_(False)

    rows = _read_rows(dataset_dir / "metadata.csv")
    if config.variant != "all":
        rows = [
            row for row in rows
            if row["variant"] == config.variant
        ]
    rows = rows[: config.limit or None]
    if not rows:
        raise ValueError("No evaluation rows selected")
    pure_dir = prediction_dir / "predictions"
    pure_dir.mkdir(parents=True, exist_ok=True)
    predictions = []
    for row in rows:
        raw = load_video(
            dataset_dir / row["video"],
            expected_frames=data_config.render.num_frames,
        )
        raw_tensor = (
            torch.from_numpy(raw.copy())
            .permute(3, 0, 1, 2)
            .float()
            .div(127.5)
            .sub(1.0)
            .unsqueeze(0)
            .to(device=config.device, dtype=pipe.torch_dtype)
        )
        condition = raw_tensor[
            :, :, :data_config.prediction_start
        ].contiguous()
        if config.history == "short":
            condition = apply_short_history_mask_tensor(
                condition,
                data_config,
            )
            condition_source = (
                "mask_0_56_background_then_real_57_64"
            )
        else:
            condition_source = "real_0_64"
        generation_seed = int(row["base_seed"]) + config.seed_offset
        with torch.inference_mode():
            generated = pipe(
                prompt="",
                negative_prompt="",
                cfg_scale=1.0,
                height=data_config.render.height,
                width=data_config.render.width,
                num_frames=data_config.render.num_frames,
                num_condition_frames=expected_conditions,
                condition_frames=condition,
                num_inference_steps=config.steps,
                tiled=False,
                num_samples=1,
                return_as_tensor=True,
                progress_bar_cmd=lambda values: values,
                seed=generation_seed,
            )[0]
        frames = (
            generated.detach()
            .float()
            .cpu()
            .permute(1, 2, 3, 0)
            .add(1.0)
            .mul(127.5)
            .clamp(0, 255)
            .byte()
            .numpy()
        )
        future = frames[data_config.prediction_start:]
        if future.shape[0] != data_config.future_frames:
            raise AssertionError(
                f"Future has {future.shape[0]} frames, "
                f"expected {data_config.future_frames}"
            )
        destination = pure_dir / f"{row['sample_id']}.mp4"
        write_video(destination, future, data_config.render.fps)
        predictions.append({
            "prediction_id": row["sample_id"],
            "sample_id": row["sample_id"],
            "prediction": destination.relative_to(
                prediction_dir
            ).as_posix(),
            "attributes": {
                "pair_id": row["pair_id"],
                "variant": row["variant"],
                "history": config.history,
                "generation_seed": generation_seed,
                "condition_source": condition_source,
            },
        })
    return write_predictions(
        prediction_dir,
        experiment="spring",
        dataset=config.dataset_id,
        predictions=predictions,
        checkpoint=config.checkpoint,
        config=config.training_config,
        extra={
            "history": config.history,
            "steps": config.steps,
            "seed_offset": config.seed_offset,
        },
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _prediction_path(root: Path, sample_id: str) -> Path:
    indexed = find_prediction(root, sample_id=sample_id)
    if indexed is not None:
        return indexed
    direct = root / f"{sample_id}.mp4"
    nested = root / "predictions" / f"{sample_id}.mp4"
    return direct if direct.exists() else nested


def _safe_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float))
        and math.isfinite(float(row[key]))
    ]
    return float(np.mean(values)) if values else None


def _safe_median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float))
        and math.isfinite(float(row[key]))
    ]
    return float(np.median(values)) if values else None


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("valid") is True]
    conflicts = [
        row
        for row in valid
        if row.get("variant") == "conflict"
    ]
    route_counts: dict[str, int] = defaultdict(int)
    anchored_counts: dict[str, int] = defaultdict(int)
    for row in conflicts:
        route_counts[str(row.get("route_label", "unknown"))] += 1
        anchored_counts[
            str(row.get("anchored_route_label", "unknown"))
        ] += 1
    keys = (
        "d_exact_physics",
        "d_true_band",
        "d_color_band",
        "route_score_band",
        "omega_abs_error",
        "omega_hat_free",
        "free_shm_rmse",
        "omega_hat_free_amplitude",
        "true_reference_amplitude",
        "color_reference_amplitude",
        "min_band_residual",
        "detection_rate",
        "boundary_jump_px",
        "max_adjacent_jump_px",
        "max_y_deviation_px",
        "multiple_component_rate",
        "color_retention_rate",
    )
    return {
        "num_samples": len(rows),
        "num_valid": len(valid),
        "validity_rate": len(valid) / max(1, len(rows)),
        "num_valid_conflicts": len(conflicts),
        "conflict_primary_route_counts": dict(route_counts),
        "conflict_anchored_route_counts": dict(anchored_counts),
        "conflict_physics_follow_rate": (
            route_counts.get("physics_frequency", 0)
            / max(1, len(conflicts))
        ),
        "conflict_shortcut_follow_rate": (
            route_counts.get("shortcut_frequency", 0)
            / max(1, len(conflicts))
        ),
        "conflict_compromise_rate": (
            route_counts.get("compromise_frequency", 0)
            / max(1, len(conflicts))
        ),
        "conflict_off_frequency_rate": (
            route_counts.get("off_frequency_family", 0)
            + route_counts.get("outside_frequency_bands", 0)
            + route_counts.get("invalid", 0)
        )
        / max(1, len(conflicts)),
        "conflict_color_reference_ood_rate": (
            float(
                np.mean(
                    [
                        bool(row.get("color_reference_ood"))
                        for row in conflicts
                    ]
                )
            )
            if conflicts
            else 0.0
        ),
        "means_valid": {
            key: _safe_mean(valid, key)
            for key in keys
        },
        "medians_valid": {
            key: _safe_median(valid, key)
            for key in keys
        },
    }


def _paired_metrics(
    rows: list[dict[str, Any]],
    trajectories: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_pair[str(row["pair_id"])][str(row["variant"])] = row
    paired: list[dict[str, Any]] = []
    for pair_id, variants in by_pair.items():
        if "aligned" not in variants or "conflict" not in variants:
            continue
        aligned, conflict = variants["aligned"], variants["conflict"]
        record: dict[str, Any] = {
            "pair_id": pair_id,
            "trajectory_id": aligned["trajectory_id"],
            "true_band": aligned["true_band"],
            "both_valid": bool(
                aligned.get("valid")
                and conflict.get("valid")
            ),
        }
        if record["both_valid"]:
            xa = trajectories[aligned["sample_id"]]
            xc = trajectories[conflict["sample_id"]]
            common = np.isfinite(xa) & np.isfinite(xc)
            record["prediction_counterfactual_rmse"] = (
                float(
                    np.sqrt(
                        np.mean(
                            (xa[common] - xc[common]) ** 2
                        )
                    )
                )
                if common.sum() >= 6
                else float("nan")
            )
            oa = float(aligned["omega_hat_free"])
            oc = float(conflict["omega_hat_free"])
            record["omega_counterfactual_shift"] = (
                abs(oa - oc)
                if math.isfinite(oa + oc)
                else float("nan")
            )
            record["conflict_route_label"] = conflict.get("route_label")
        paired.append(record)
    return paired


def evaluate(
    dataset_dir: Path,
    prediction_dir: Path,
    config: SpringEvaluationConfig | Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate existing predictions using the common experiment contract."""
    if isinstance(config, Mapping):
        config = _spring_config_from_mapping(config)
    cfg = config.data
    bands = {"slow": cfg.slow_band, "fast": cfg.fast_band}
    rows = _read_rows(dataset_dir / "metadata.csv")
    rows = rows[: config.limit or None]
    results: list[dict[str, Any]] = []
    trajectories: dict[str, np.ndarray] = {}

    for row in rows:
        path = _prediction_path(prediction_dir, row["sample_id"])
        record: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "trajectory_id": row["trajectory_id"],
            "pair_id": row["pair_id"],
            "true_band": row["true_band"],
            "color_label": row["color_label"],
            "variant": row["variant"],
            "history": config.history,
            "omega_true": float(row["omega_true"]),
            "prediction_path": str(path),
        }
        if not path.exists():
            record.update(
                {"valid": False, "failure": "missing_prediction"}
            )
            results.append(record)
            continue
        try:
            frames = load_video(path)
        except Exception as exc:
            record.update(
                {
                    "valid": False,
                    "failure": f"read_error:{type(exc).__name__}",
                }
            )
            results.append(record)
            continue
        if frames.shape[0] != cfg.future_frames:
            record.update(
                {
                    "valid": False,
                    "failure": (
                        f"unexpected_frame_count:{frames.shape[0]}"
                    ),
                }
            )
            results.append(record)
            continue

        track = detect_mass_track(frames, cfg.render)
        valid_stats = validity_metrics(
            track,
            cfg.render,
            x_star=float(row["x_star"]),
        )
        record.update(valid_stats)
        detected_colors = [
            classify_detected_color(rgb, cfg.render)
            for rgb in track.mean_rgb
            if np.isfinite(rgb).all()
        ]
        if detected_colors:
            record["color_retention_rate"] = float(
                np.mean(
                    [
                        value == row["color_label"]
                        for value in detected_colors
                    ]
                )
            )
            record["detected_color_majority"] = max(
                set(detected_colors),
                key=detected_colors.count,
            )
        else:
            record["color_retention_rate"] = float("nan")
            record["detected_color_majority"] = "unknown"

        predicted_x = pixel_x_to_displacement(track.x_px, cfg.render)
        trajectories[row["sample_id"]] = predicted_x
        if not valid_stats["valid"]:
            record["failure"] = "trajectory_invalid"
            results.append(record)
            continue

        metrics = route_metrics_against_bands(
            predicted_x,
            np.isfinite(predicted_x),
            true_band=bands[row["true_band"]],
            color_implied_band=bands[color_band(row["color_label"])],
            slow_band=cfg.slow_band,
            fast_band=cfg.fast_band,
            omega_true=float(row["omega_true"]),
            x_star=float(row["x_star"]),
            v_star=float(row["v_star"]),
            fps=cfg.render.fps,
            amplitude_low=cfg.amplitude_low,
            amplitude_high=cfg.amplitude_high,
            reference_amplitude_multiplier=(
                cfg.route_reference_amplitude_multiplier
            ),
            frequency_band_tolerance=cfg.frequency_band_tolerance,
            free_shm_rmse_threshold=cfg.free_shm_rmse_threshold,
        )
        metrics.pop("best_true_curve", None)
        metrics.pop("best_color_curve", None)
        if row["variant"] == "aligned":
            metrics.update(
                {
                    "route_label": "iid_not_attributed",
                    "primary_route_label": "iid_not_attributed",
                    "anchored_route_label": "iid_not_attributed",
                    "route_score_band": float("nan"),
                }
            )
        record.update(metrics)
        record["true_color_is_canonical"] = (
            row["color_label"] == canonical_color(row["true_band"])
        )
        results.append(record)

    paired = _paired_metrics(results, trajectories)
    summary: dict[str, Any] = {
        "history": config.history,
        "all": _aggregate(results),
        "by_variant": {
            variant: _aggregate(
                [
                    row
                    for row in results
                    if row["variant"] == variant
                ]
            )
            for variant in ("aligned", "conflict")
        },
        "by_true_band": {
            band: _aggregate(
                [
                    row
                    for row in results
                    if row["true_band"] == band
                ]
            )
            for band in ("slow", "fast")
        },
        "conflict_by_true_band": {
            band: _aggregate(
                [
                    row
                    for row in results
                    if row["true_band"] == band
                    and row["variant"] == "conflict"
                ]
            )
            for band in ("slow", "fast")
        },
        "paired_counterfactuals": {
            "num_pairs": len(paired),
            "both_valid_rate": (
                float(np.mean([row["both_valid"] for row in paired]))
                if paired
                else 0.0
            ),
            "mean_prediction_counterfactual_rmse": _safe_mean(
                paired,
                "prediction_counterfactual_rmse",
            ),
            "mean_omega_counterfactual_shift": _safe_mean(
                paired,
                "omega_counterfactual_shift",
            ),
        },
    }
    return {
        "summary": summary,
        "samples": results,
        "paired_counterfactuals": paired,
    }


def write_evaluation(
    result: dict[str, Any],
    out: Path,
    *,
    context: Mapping[str, Any] | None = None,
) -> None:
    result_context = {
        "history": result["summary"]["history"],
        **dict(context or {}),
    }
    write_metrics(
        out,
        experiment="spring",
        dataset="spring_shortcuts_v4",
        summary=result["summary"],
        samples=result["samples"],
        records={
            "paired_counterfactuals": (
                result["paired_counterfactuals"]
            )
        },
        context=result_context,
    )
    with (out / "per_sample.jsonl").open("w", encoding="utf-8") as handle:
        for row in result["samples"]:
            handle.write(
                json.dumps(finite_json(row), ensure_ascii=False) + "\n"
            )
    (out / "summary.json").write_text(
        json.dumps(
            finite_json(result["summary"]),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
