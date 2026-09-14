"""Prediction and evaluation for the Pendulum experiment."""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import yaml

from sshv2.common.results import (
    find_prediction,
    write_metrics,
    write_predictions,
)
from sshv2.experiments.pendulum.data import (
    DATASET_ID,
    EXPERIMENT_ID,
    ColorName,
    PendulumDatasetConfig,
    PendulumRenderConfig,
    ValueRange,
    apply_short_history_mask,
    config_from_mapping,
    load_video,
    write_video,
)


HistoryName = Literal["short", "long"]


@dataclass
class BobTrack:
    x_px: np.ndarray
    y_px: np.ndarray
    area_px: np.ndarray
    mean_rgb: np.ndarray

    @property
    def detected(self) -> np.ndarray:
        return np.isfinite(self.x_px) & np.isfinite(self.y_px)


@dataclass(frozen=True)
class OscillationFit:
    omega: float
    amplitude: float
    center: float
    rmse: float
    fitted: np.ndarray


def _expected_rgb(
    color: ColorName,
    config: PendulumRenderConfig,
) -> np.ndarray:
    return np.asarray(
        {
            "red": config.red_rgb,
            "blue": config.blue_rgb,
            "gray": config.gray_rgb,
            "green": config.green_rgb,
        }[color],
        dtype=np.float64,
    )


def detect_bob_track(
    frames: np.ndarray,
    config: PendulumRenderConfig,
    *,
    expected_color: ColorName,
    color_distance: float = 90.0,
    min_area: int = 28,
) -> BobTrack:
    """Track the expected bob color, including neutral gray OOD renders."""
    video = np.asarray(frames, dtype=np.uint8)
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(f"expected [T,H,W,3], got {video.shape}")
    expected = _expected_rgb(expected_color, config)
    height, width = config.height, config.width
    yy, xx = np.indices((height, width))
    pivot_x = config.pivot_x * (width - 1)
    pivot_y = config.pivot_y * (height - 1)
    reach_x = config.length * (width - 1) + config.bob_radius_px + 6
    reach_y = config.length * (height - 1) + config.bob_radius_px + 6
    pendulum_region = (
        (np.abs(xx - pivot_x) <= reach_x)
        & (yy >= pivot_y + 4)
        & (yy <= pivot_y + reach_y)
    )

    frame_count = video.shape[0]
    xs = np.full(frame_count, np.nan, dtype=np.float64)
    ys = np.full(frame_count, np.nan, dtype=np.float64)
    areas = np.zeros(frame_count, dtype=np.float64)
    rgbs = np.full((frame_count, 3), np.nan, dtype=np.float64)
    for frame_index, frame in enumerate(video):
        distance = np.linalg.norm(
            frame.astype(np.float64) - expected,
            axis=-1,
        )
        mask = (distance <= color_distance) & pendulum_region
        area = int(mask.sum())
        if area < min_area:
            continue
        xs[frame_index] = float(xx[mask].mean())
        ys[frame_index] = float(yy[mask].mean())
        areas[frame_index] = float(area)
        rgbs[frame_index] = frame[mask].mean(axis=0)
    return BobTrack(xs, ys, areas, rgbs)


def _max_true_run(flags: np.ndarray) -> int:
    current = 0
    maximum = 0
    for flag in np.asarray(flags, dtype=bool):
        current = current + 1 if flag else 0
        maximum = max(maximum, current)
    return maximum


def centers_to_theta(
    track: BobTrack,
    config: PendulumRenderConfig,
) -> np.ndarray:
    """Convert detected bob centers to angles measured from vertical-down."""
    dx = track.x_px / (config.width - 1) - config.pivot_x
    dy = track.y_px / (config.height - 1) - config.pivot_y
    return np.arctan2(dx, dy)


def track_validity(
    track: BobTrack,
    config: PendulumRenderConfig,
    *,
    theta_star: float,
    detection_rate_required: float = 0.90,
    max_missing_run_allowed: int = 3,
    max_jump_px: float = 20.0,
    max_length_error: float = 0.05,
    max_boundary_jump_px: float = 20.0,
) -> dict[str, Any]:
    detected = track.detected
    detection_rate = float(detected.mean())
    missing_run = _max_true_run(~detected)
    adjacent = (
        detected[:-1] & detected[1:]
        if len(detected) > 1
        else np.zeros(0, dtype=bool)
    )
    jumps = np.hypot(
        np.diff(track.x_px),
        np.diff(track.y_px),
    )
    max_jump = (
        float(jumps[adjacent].max())
        if adjacent.any()
        else float("inf")
    )

    dx = track.x_px / (config.width - 1) - config.pivot_x
    dy = track.y_px / (config.height - 1) - config.pivot_y
    lengths = np.hypot(dx, dy)
    length_error = (
        float(np.nanmedian(np.abs(lengths - config.length)))
        if detected.any()
        else float("inf")
    )
    expected_first_x = (
        config.pivot_x + config.length * math.sin(theta_star)
    ) * (config.width - 1)
    expected_first_y = (
        config.pivot_y + config.length * math.cos(theta_star)
    ) * (config.height - 1)
    first_detected = np.flatnonzero(detected)
    boundary_jump = (
        float(
            math.hypot(
                track.x_px[first_detected[0]] - expected_first_x,
                track.y_px[first_detected[0]] - expected_first_y,
            )
        )
        if first_detected.size
        else float("inf")
    )
    median_area = (
        float(np.median(track.area_px[detected]))
        if detected.any()
        else float("nan")
    )
    area_valid = bool(
        detected.any()
        and 0.25 * math.pi * config.bob_radius_px**2
        <= median_area
        <= 2.5 * (2 * config.bob_radius_px + 1) ** 2
    )
    valid = bool(
        detection_rate >= detection_rate_required
        and missing_run <= max_missing_run_allowed
        and max_jump <= max_jump_px
        and length_error <= max_length_error
        and boundary_jump <= max_boundary_jump_px
        and area_valid
    )
    return {
        "valid": valid,
        "detection_rate": detection_rate,
        "max_missing_run": missing_run,
        "max_adjacent_jump_px": max_jump,
        "median_length_error": length_error,
        "boundary_jump_px": boundary_jump,
        "median_area_px": median_area,
        "area_valid": area_valid,
    }


def fit_oscillation(
    theta: np.ndarray,
    valid_mask: np.ndarray,
    *,
    fps: int,
    omega_low: float,
    omega_high: float,
    grid_size: int = 1601,
) -> OscillationFit:
    """Fit theta(t)=a*cos(wt)+b*sin(wt)+c over the future frames."""
    values = np.asarray(theta, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(values)
    if values.ndim != 1 or valid.shape != values.shape:
        raise ValueError("theta and valid_mask must be aligned 1-D arrays")
    if valid.sum() < 6:
        return OscillationFit(
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            np.full_like(values, np.nan),
        )
    times = (np.arange(values.size, dtype=np.float64) + 1.0) / fps
    best = OscillationFit(
        float("nan"),
        float("nan"),
        float("nan"),
        float("inf"),
        np.full_like(values, np.nan),
    )
    for omega in np.linspace(omega_low, omega_high, grid_size):
        design = np.stack(
            (
                np.cos(omega * times),
                np.sin(omega * times),
                np.ones_like(times),
            ),
            axis=1,
        )
        coefficients, *_ = np.linalg.lstsq(
            design[valid],
            values[valid],
            rcond=None,
        )
        fitted = design @ coefficients
        rmse = float(
            np.sqrt(np.mean((values[valid] - fitted[valid]) ** 2))
        )
        if rmse < best.rmse:
            best = OscillationFit(
                omega=float(omega),
                amplitude=float(
                    math.hypot(coefficients[0], coefficients[1])
                ),
                center=float(coefficients[2]),
                rmse=rmse,
                fitted=fitted,
            )
    return best


def _distance_to_range(value: float, value_range: ValueRange) -> float:
    if not math.isfinite(value):
        return float("nan")
    if value < value_range.low:
        return value_range.low - value
    if value > value_range.high:
        return value - value_range.high
    return 0.0


def _classify_range(
    value: float,
    first: ValueRange,
    second: ValueRange,
) -> int | None:
    if first.contains(value):
        return 0
    if second.contains(value):
        return 1
    return None


def color_implied_target(color: str) -> int | None:
    if color == "red":
        return 0
    if color == "blue":
        return 1
    return None


def shape_implied_target(shape: str) -> int:
    return 0 if shape == "circle" else 1


@dataclass(frozen=True)
class PendulumEvaluationConfig:
    data: PendulumDatasetConfig
    history: HistoryName
    limit: int = 0
    subset: str = "all"


@dataclass(frozen=True)
class PendulumPredictionConfig:
    training_config: Path
    checkpoint: Path
    data_config: Path
    history: HistoryName
    steps: int = 20
    limit: int = 0
    subset: str = "all"
    device: str = "cuda"
    seed_offset: int = 23_000_000
    dataset_id: str = DATASET_ID

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "PendulumPredictionConfig":
        return cls(
            training_config=Path(value["training_config"]),
            checkpoint=Path(value["checkpoint"]),
            data_config=Path(value["data_config"]),
            history=_history(value["history"]),
            steps=int(
                value.get("steps", value.get("diffusion_steps", 20))
            ),
            limit=int(value.get("limit") or 0),
            subset=str(value.get("subset", "all")),
            device=str(value.get("device", "cuda")),
            seed_offset=int(value.get("seed_offset", 23_000_000)),
            dataset_id=str(value.get("dataset_id") or DATASET_ID),
        )


def _history(value: Any) -> HistoryName:
    history = str(value)
    if history not in ("short", "long"):
        raise ValueError("history must be short or long")
    return history


def _load_data_config(path: Path) -> PendulumDatasetConfig:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError(f"config root must be a mapping: {path}")
    data = document.get("data", document)
    if not isinstance(data, Mapping):
        raise ValueError(f"data config must be a mapping: {path}")
    return config_from_mapping(data)


def _evaluation_config_from_mapping(
    value: Mapping[str, Any],
) -> PendulumEvaluationConfig:
    return PendulumEvaluationConfig(
        data=_load_data_config(Path(value["data_config"])),
        history=_history(value["history"]),
        limit=int(value.get("limit") or 0),
        subset=str(value.get("subset", "all")),
    )


def _metadata_bank(dataset_dir: Path) -> Path:
    direct = dataset_dir / "metadata.csv"
    nested = dataset_dir / "videos" / "eval" / "metadata.csv"
    if direct.is_file():
        return dataset_dir
    if nested.is_file():
        return nested.parent
    raise FileNotFoundError(
        f"metadata.csv not found under {dataset_dir} or videos/eval"
    )


def _read_rows(bank: Path) -> list[dict[str, str]]:
    with (bank / "metadata.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        return list(csv.DictReader(handle))


def _select_rows(
    rows: list[dict[str, str]],
    *,
    subset: str,
    limit: int,
) -> list[dict[str, str]]:
    if subset not in ("all", "id", "ood"):
        raise ValueError("subset must be all, id, or ood")
    if subset == "id":
        rows = [
            row for row in rows if not row["variant"].startswith("ood_")
        ]
    elif subset == "ood":
        rows = [
            row for row in rows if row["variant"].startswith("ood_")
        ]
    selected = rows[: limit or None]
    if not selected:
        raise ValueError("no evaluation rows selected")
    return selected


def _validate_dataset_binding(
    rows: list[dict[str, str]],
    config: PendulumDatasetConfig,
) -> None:
    model_names = {row["model_name"] for row in rows}
    training_ids = {row["training_manifest_id"] for row in rows}
    test_ids = {row["test_manifest_id"] for row in rows}
    if model_names != {config.model_name}:
        raise ValueError(
            f"dataset model names {model_names} do not match "
            f"{config.model_name}"
        )
    if training_ids != {config.training_manifest_id}:
        raise ValueError("dataset training manifest does not match config")
    if test_ids != {config.test_manifest_id}:
        raise ValueError("dataset test manifest does not match config")


def _validate_training_config_binding(
    path: Path,
    data: PendulumDatasetConfig,
    history: HistoryName,
) -> None:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("training config root must be a mapping")
    binding = document.get("pendulum")
    if not isinstance(binding, Mapping):
        raise ValueError(
            "training config is missing the pendulum binding block"
        )
    expected = {
        "model_name": data.model_name,
        "training_manifest_id": data.training_manifest_id,
        "history": history,
    }
    mismatches = {
        key: (binding.get(key), value)
        for key, value in expected.items()
        if binding.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"training config binding mismatch: {mismatches}"
        )


def predict(
    dataset_dir: Path,
    prediction_dir: Path,
    config: PendulumPredictionConfig | Mapping[str, Any],
) -> dict[str, Any]:
    """Generate 64-frame futures through the shared standard Wan path."""
    if isinstance(config, Mapping):
        config = PendulumPredictionConfig.from_mapping(config)
    if config.steps <= 0:
        raise ValueError("steps must be positive")

    import torch
    from sshv2.wan.config import StandardTrainingConfig
    from sshv2.wan.trainer import WanTrainingModule

    data_config = _load_data_config(config.data_config)
    bank = _metadata_bank(Path(dataset_dir))
    rows = _select_rows(
        _read_rows(bank),
        subset=config.subset,
        limit=config.limit,
    )
    _validate_dataset_binding(rows, data_config)
    _validate_training_config_binding(
        config.training_config,
        data_config,
        config.history,
    )

    training = StandardTrainingConfig.from_file(config.training_config)
    condition_latents = (
        data_config.prediction_start - 1
    ) // 4 + 1
    if training.model.num_condition_frames != condition_latents:
        raise ValueError(
            "training condition frames do not match Pendulum history"
        )
    training.model.dit.ckpt_file = config.checkpoint
    model = WanTrainingModule(
        dit_config=training.model.dit,
        vae_config=training.model.vae,
        no_encoding=False,
        num_condition_frames=condition_latents,
        num_inference_steps=config.steps,
        pipeline_type=training.model.pipe,
        pipeline_kwargs=training.model.pipe_kwargs,
    )
    pipe = model.pipe
    pipe.to(config.device)
    pipe.load_models_to_device(("dit", "vae"))
    pipe.pre_encoded_(False)

    output_bank = Path(prediction_dir) / "predictions"
    output_bank.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, Any]] = []
    for row in rows:
        raw = load_video(
            bank / row["video"],
            expected_frames=data_config.render.num_frames,
        )
        if config.history == "short":
            source = apply_short_history_mask(raw, data_config)
            condition_source = (
                "mask_0_56_background_then_real_57_64"
            )
        else:
            source = raw
            condition_source = "real_0_64"
        condition = (
            torch.from_numpy(
                source[: data_config.prediction_start].copy()
            )
            .permute(3, 0, 1, 2)
            .float()
            .div(127.5)
            .sub(1.0)
            .unsqueeze(0)
            .to(device=config.device, dtype=pipe.torch_dtype)
        )
        generation_seed = int(row["base_seed"]) + config.seed_offset
        with torch.inference_mode():
            generated = pipe(
                prompt="",
                negative_prompt="",
                cfg_scale=1.0,
                height=data_config.render.height,
                width=data_config.render.width,
                num_frames=data_config.render.num_frames,
                num_condition_frames=condition_latents,
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
        future = frames[data_config.prediction_start :]
        if future.shape[0] != data_config.future_frames:
            raise AssertionError(
                f"expected {data_config.future_frames} future frames, "
                f"got {future.shape[0]}"
            )
        destination = output_bank / f"{row['sample_id']}.mp4"
        write_video(destination, future, data_config.render.fps)
        predictions.append(
            {
                "prediction_id": row["sample_id"],
                "sample_id": row["sample_id"],
                "prediction": destination.relative_to(
                    prediction_dir
                ).as_posix(),
                "attributes": {
                    "history": config.history,
                    "condition_source": condition_source,
                    "generation_seed": generation_seed,
                    "model_name": data_config.model_name,
                    "training_manifest_id": (
                        data_config.training_manifest_id
                    ),
                    "test_manifest_id": data_config.test_manifest_id,
                    "variant": row["variant"],
                },
            }
        )
    return write_predictions(
        Path(prediction_dir),
        experiment=EXPERIMENT_ID,
        dataset=config.dataset_id,
        predictions=predictions,
        checkpoint=config.checkpoint,
        config=config.training_config,
        extra={
            "history": config.history,
            "steps": config.steps,
            "seed_offset": config.seed_offset,
            "model_name": data_config.model_name,
            "training_manifest_id": data_config.training_manifest_id,
            "test_manifest_id": data_config.test_manifest_id,
        },
    )


def _prediction_path(root: Path, sample_id: str) -> Path:
    indexed = find_prediction(root, sample_id=sample_id)
    if indexed is not None:
        return indexed
    direct = root / f"{sample_id}.mp4"
    nested = root / "predictions" / f"{sample_id}.mp4"
    return direct if direct.is_file() else nested


def _shape_from_area(
    area: float,
    config: PendulumRenderConfig,
) -> str:
    circle_area = math.pi * config.bob_radius_px**2
    square_area = (2 * config.bob_radius_px + 1) ** 2
    threshold = 0.5 * (circle_area + square_area)
    return "circle" if area < threshold else "square"


def _route_metrics(
    row: Mapping[str, str],
    config: PendulumDatasetConfig,
    fit: OscillationFit,
) -> dict[str, Any]:
    true_index = int(row["target_index"])
    color_target = (
        color_implied_target(row["color_label"])
        if config.model_pairing in ("color", "color_shape")
        else None
    )
    shape_target = (
        shape_implied_target(row["shape_label"])
        if config.model_pairing in ("shape", "color_shape")
        else None
    )
    if config.test_intervention == "color":
        intervention_target = color_target
    elif config.test_intervention == "shape":
        intervention_target = shape_target
    else:
        intervention_target = (
            color_target
            if color_target is not None
            and color_target == shape_target
            else None
        )
    if config.target == "frequency":
        measured = fit.omega
        exact_true = float(row["omega_true"])
        first_range = config.low_frequency
        second_range = config.high_frequency
    else:
        measured = fit.amplitude
        exact_true = float(row["amplitude_true"])
        first_range = config.small_amplitude
        second_range = config.large_amplitude
    predicted_index = _classify_range(
        measured,
        first_range,
        second_range,
    )
    distances = (
        _distance_to_range(measured, first_range),
        _distance_to_range(measured, second_range),
    )
    distance_true = distances[true_index]
    distance_intervention = (
        distances[intervention_target]
        if intervention_target is not None
        else float("nan")
    )
    conflict = bool(
        intervention_target is not None
        and intervention_target != true_index
    )
    if predicted_index == true_index:
        route_label = "physics"
    elif conflict and predicted_index == intervention_target:
        route_label = "intervention_shortcut"
    elif predicted_index is None:
        route_label = "off_target_ranges"
    else:
        route_label = "other_target"
    route_score = (
        (distance_true - distance_intervention)
        / (distance_true + distance_intervention + 1e-12)
        if conflict
        and math.isfinite(distance_true + distance_intervention)
        else float("nan")
    )
    return {
        "measured_target_value": measured,
        "true_target_value": exact_true,
        "target_absolute_error": (
            abs(measured - exact_true)
            if math.isfinite(measured)
            else float("nan")
        ),
        "predicted_target_index": predicted_index,
        "true_target_index": true_index,
        "color_implied_target_index": color_target,
        "shape_implied_target_index": shape_target,
        "intervention_implied_target_index": intervention_target,
        "has_defined_intervention_target": (
            intervention_target is not None
        ),
        "is_conflict": conflict,
        "physics_follow": predicted_index == true_index,
        "color_follow": (
            predicted_index == color_target
            if color_target is not None
            else None
        ),
        "shape_follow": (
            predicted_index == shape_target
            if shape_target is not None
            else None
        ),
        "intervention_follow": (
            predicted_index == intervention_target
            if intervention_target is not None
            else None
        ),
        "distance_to_true_range": distance_true,
        "distance_to_intervention_range": distance_intervention,
        "route_score": route_score,
        "route_label": route_label,
    }


def _safe_mean(
    rows: list[dict[str, Any]],
    key: str,
) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float, bool))
        and math.isfinite(float(row[key]))
    ]
    return float(np.mean(values)) if values else None


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("valid") is True]
    conflicts = [row for row in valid if row.get("is_conflict") is True]
    defined_intervention = [
        row
        for row in valid
        if row.get("has_defined_intervention_target") is True
    ]
    route_counts: dict[str, int] = defaultdict(int)
    predicted_counts = {"class_0": 0, "class_1": 0, "unclassified": 0}
    for row in valid:
        route_counts[str(row.get("route_label", "unknown"))] += 1
        predicted = row.get("predicted_target_index")
        key = (
            f"class_{predicted}"
            if predicted in (0, 1)
            else "unclassified"
        )
        predicted_counts[key] += 1
    return {
        "num_samples": len(rows),
        "num_valid": len(valid),
        "validity_rate": len(valid) / max(1, len(rows)),
        "route_counts": dict(route_counts),
        "predicted_target_counts": predicted_counts,
        "physics_follow_rate": _safe_mean(valid, "physics_follow"),
        "color_follow_rate": _safe_mean(valid, "color_follow"),
        "shape_follow_rate": _safe_mean(valid, "shape_follow"),
        "defined_intervention_follow_rate": _safe_mean(
            defined_intervention,
            "intervention_follow",
        ),
        "conflict_shortcut_follow_rate": _safe_mean(
            conflicts,
            "intervention_follow",
        ),
        "means_valid": {
            key: _safe_mean(valid, key)
            for key in (
                "target_absolute_error",
                "omega_hat",
                "amplitude_hat",
                "fit_rmse",
                "route_score",
                "detection_rate",
                "max_adjacent_jump_px",
                "median_length_error",
                "boundary_jump_px",
                "color_distance",
                "shape_retained",
            )
        },
    }


def _counterfactual_records(
    rows: list[dict[str, Any]],
    trajectories: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_trajectory[str(row["trajectory_id"])].append(row)
    records: list[dict[str, Any]] = []
    for trajectory_id, variants in by_trajectory.items():
        for first_index in range(len(variants)):
            for second_index in range(first_index + 1, len(variants)):
                first = variants[first_index]
                second = variants[second_index]
                both_valid = bool(
                    first.get("valid") and second.get("valid")
                )
                record: dict[str, Any] = {
                    "trajectory_id": trajectory_id,
                    "sample_id_a": first["sample_id"],
                    "sample_id_b": second["sample_id"],
                    "variant_a": first["variant"],
                    "variant_b": second["variant"],
                    "both_valid": both_valid,
                }
                if both_valid:
                    theta_a = trajectories[first["sample_id"]]
                    theta_b = trajectories[second["sample_id"]]
                    common = np.isfinite(theta_a) & np.isfinite(theta_b)
                    record.update(
                        {
                            "prediction_theta_rmse": (
                                float(
                                    np.sqrt(
                                        np.mean(
                                            (
                                                theta_a[common]
                                                - theta_b[common]
                                            )
                                            ** 2
                                        )
                                    )
                                )
                                if common.sum() >= 6
                                else float("nan")
                            ),
                            "omega_shift": abs(
                                float(first["omega_hat"])
                                - float(second["omega_hat"])
                            ),
                            "amplitude_shift": abs(
                                float(first["amplitude_hat"])
                                - float(second["amplitude_hat"])
                            ),
                        }
                    )
                records.append(record)
    return records


def evaluate(
    dataset_dir: Path,
    prediction_dir: Path,
    config: PendulumEvaluationConfig | Mapping[str, Any],
) -> dict[str, Any]:
    """Measure physics and appearance-cue following in existing futures."""
    if isinstance(config, Mapping):
        config = _evaluation_config_from_mapping(config)
    bank = _metadata_bank(Path(dataset_dir))
    rows = _select_rows(
        _read_rows(bank),
        subset=config.subset,
        limit=config.limit,
    )
    _validate_dataset_binding(rows, config.data)
    sample_results: list[dict[str, Any]] = []
    predicted_trajectories: dict[str, np.ndarray] = {}

    for row in rows:
        prediction_path = _prediction_path(
            Path(prediction_dir),
            row["sample_id"],
        )
        record: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "trajectory_id": row["trajectory_id"],
            "pair_id": row["pair_id"],
            "variant": row["variant"],
            "subset": row["subset"],
            "target": row["target"],
            "target_label": row["target_label"],
            "model_name": row["model_name"],
            "training_manifest_id": row["training_manifest_id"],
            "test_manifest_id": row["test_manifest_id"],
            "test_intervention": config.data.test_intervention,
            "history": config.history,
            "frequency_label": row["frequency_label"],
            "amplitude_label": row["amplitude_label"],
            "color_label": row["color_label"],
            "shape_label": row["shape_label"],
            "omega_true": float(row["omega_true"]),
            "amplitude_true": float(row["amplitude_true"]),
            "prediction_path": str(prediction_path),
        }
        if not prediction_path.is_file():
            record.update(
                {"valid": False, "failure": "missing_prediction"}
            )
            sample_results.append(record)
            continue
        try:
            frames = load_video(prediction_path)
        except Exception as exc:
            record.update(
                {
                    "valid": False,
                    "failure": f"read_error:{type(exc).__name__}",
                }
            )
            sample_results.append(record)
            continue
        if frames.shape[0] != config.data.future_frames:
            record.update(
                {
                    "valid": False,
                    "failure": (
                        f"unexpected_frame_count:{frames.shape[0]}"
                    ),
                }
            )
            sample_results.append(record)
            continue

        track = detect_bob_track(
            frames,
            config.data.render,
            expected_color=row["color_label"],
        )
        validity = track_validity(
            track,
            config.data.render,
            theta_star=float(row["theta_star"]),
        )
        theta = centers_to_theta(track, config.data.render)
        predicted_trajectories[row["sample_id"]] = theta
        fit = fit_oscillation(
            theta,
            track.detected,
            fps=config.data.render.fps,
            omega_low=max(0.2, config.data.low_frequency.low - 1.0),
            omega_high=config.data.high_frequency.high + 1.0,
        )
        fit_valid = bool(
            math.isfinite(fit.rmse)
            and fit.rmse <= 0.08
            and math.isfinite(fit.omega)
            and math.isfinite(fit.amplitude)
        )
        validity["valid"] = bool(validity["valid"] and fit_valid)
        record.update(validity)
        record.update(
            {
                "fit_valid": fit_valid,
                "omega_hat": fit.omega,
                "amplitude_hat": fit.amplitude,
                "fit_center": fit.center,
                "fit_rmse": fit.rmse,
            }
        )

        detected_rgb = track.mean_rgb[
            np.isfinite(track.mean_rgb).all(axis=1)
        ]
        color_distance = (
            float(
                np.linalg.norm(
                    detected_rgb
                    - _expected_rgb(
                        row["color_label"],
                        config.data.render,
                    ),
                    axis=1,
                ).mean()
            )
            if detected_rgb.size
            else float("nan")
        )
        predicted_shape = _shape_from_area(
            float(record["median_area_px"]),
            config.data.render,
        )
        record.update(
            {
                "color_distance": color_distance,
                "predicted_shape": predicted_shape,
                "shape_retained": (
                    predicted_shape == row["shape_label"]
                ),
            }
        )
        record.update(_route_metrics(row, config.data, fit))
        if not record["valid"]:
            record["failure"] = (
                "trajectory_invalid"
                if validity["detection_rate"] < 0.90
                else "oscillation_fit_invalid"
            )
        sample_results.append(record)

    counterfactuals = _counterfactual_records(
        sample_results,
        predicted_trajectories,
    )
    variants = sorted({row["variant"] for row in sample_results})
    colors = sorted({row["color_label"] for row in sample_results})
    target_labels = sorted(
        {row["target_label"] for row in sample_results}
    )
    summary = {
        "target": config.data.target,
        "model_name": config.data.model_name,
        "training_manifest_id": config.data.training_manifest_id,
        "test_manifest_id": config.data.test_manifest_id,
        "test_intervention": config.data.test_intervention,
        "ood_enabled": config.data.ood_enabled,
        "history": config.history,
        "future_frames": config.data.future_frames,
        "all": _aggregate(sample_results),
        "by_variant": {
            variant: _aggregate(
                [
                    row
                    for row in sample_results
                    if row["variant"] == variant
                ]
            )
            for variant in variants
        },
        "by_color": {
            color: _aggregate(
                [
                    row
                    for row in sample_results
                    if row["color_label"] == color
                ]
            )
            for color in colors
        },
        "by_target_label": {
            label: _aggregate(
                [
                    row
                    for row in sample_results
                    if row["target_label"] == label
                ]
            )
            for label in target_labels
        },
        "counterfactuals": {
            "num_pairs": len(counterfactuals),
            "both_valid_rate": _safe_mean(
                counterfactuals,
                "both_valid",
            ),
            "mean_prediction_theta_rmse": _safe_mean(
                counterfactuals,
                "prediction_theta_rmse",
            ),
            "mean_omega_shift": _safe_mean(
                counterfactuals,
                "omega_shift",
            ),
            "mean_amplitude_shift": _safe_mean(
                counterfactuals,
                "amplitude_shift",
            ),
        },
    }
    return {
        "summary": summary,
        "samples": sample_results,
        "counterfactuals": counterfactuals,
    }


def write_evaluation(
    result: dict[str, Any],
    output_dir: Path,
    *,
    context: Mapping[str, Any] | None = None,
) -> None:
    """Write only the common metrics envelope; avoid duplicate result files."""
    summary = result["summary"]
    write_metrics(
        output_dir,
        experiment=EXPERIMENT_ID,
        dataset=DATASET_ID,
        summary=summary,
        samples=result["samples"],
        records={"counterfactuals": result["counterfactuals"]},
        context={
            "target": summary["target"],
            "model_name": summary["model_name"],
            "training_manifest_id": summary["training_manifest_id"],
            "test_manifest_id": summary["test_manifest_id"],
            "test_intervention": summary["test_intervention"],
            "history": summary["history"],
            **dict(context or {}),
        },
    )
