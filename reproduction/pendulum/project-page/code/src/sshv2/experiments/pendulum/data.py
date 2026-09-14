"""Data construction for the Pendulum experiment.

One configuration describes one independently trained logical model and one
test intervention.  The generator writes only that selected training/eval
dataset; the 32 names in the experiment plan define the complete label space,
not a requirement to materialize every group in every run.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from sshv2.common.dataset import read_dataset, write_dataset, write_json


EXPERIMENT_ID = "pendulum"
DATASET_ID = "pendulum_v1"

TargetName = Literal["frequency", "amplitude"]
PairingName = Literal["color", "shape", "color_shape"]
InterventionName = Literal["color", "shape", "both"]
ColorName = Literal["red", "blue", "gray", "green"]
TrainingColorName = Literal["red", "blue"]
ShapeName = Literal["circle", "square"]
FrequencyLabel = Literal["low_frequency", "high_frequency"]
AmplitudeLabel = Literal["small_amplitude", "large_amplitude"]
SplitName = Literal["train", "eval"]


@dataclass(frozen=True)
class ValueRange:
    low: float
    high: float

    def __post_init__(self) -> None:
        if not (0.0 < self.low < self.high):
            raise ValueError(f"Invalid positive range [{self.low}, {self.high}]")

    def contains(self, value: float, *, atol: float = 1e-12) -> bool:
        return self.low - atol <= float(value) <= self.high + atol


@dataclass(frozen=True)
class PendulumRenderConfig:
    width: int = 128
    height: int = 128
    fps: int = 20
    num_frames: int = 129
    pivot_x: float = 0.50
    pivot_y: float = 0.14
    length: float = 0.34
    bob_radius_px: int = 7
    rope_width_px: int = 2
    background_rgb: tuple[int, int, int] = (28, 30, 34)
    rope_rgb: tuple[int, int, int] = (226, 229, 235)
    pivot_rgb: tuple[int, int, int] = (205, 209, 217)
    red_rgb: tuple[int, int, int] = (235, 48, 48)
    blue_rgb: tuple[int, int, int] = (48, 96, 235)
    gray_rgb: tuple[int, int, int] = (150, 154, 164)
    green_rgb: tuple[int, int, int] = (48, 190, 104)

    def __post_init__(self) -> None:
        if self.width != 128 or self.height != 128:
            raise ValueError("Pendulum v1 requires 128x128 frames")
        if self.fps != 20 or self.num_frames != 129:
            raise ValueError("Pendulum v1 requires 20 fps and 129 frames")
        if not (0.0 < self.pivot_x < 1.0 and 0.0 < self.pivot_y < 1.0):
            raise ValueError("pivot coordinates must lie in (0, 1)")
        if not (0.0 < self.length < 1.0):
            raise ValueError("length must lie in (0, 1)")
        if self.pivot_y + self.length >= 0.95:
            raise ValueError("pendulum does not fit vertically in the frame")
        if self.bob_radius_px < 3 or self.rope_width_px < 1:
            raise ValueError("invalid bob or rope size")


@dataclass(frozen=True)
class PendulumDatasetConfig:
    target: TargetName = "frequency"
    model_pairing: PairingName = "color"
    test_intervention: InterventionName = "color"
    fixed_shape: ShapeName = "circle"
    fixed_color: TrainingColorName = "red"
    ood_enabled: bool = False
    low_frequency: ValueRange = field(
        default_factory=lambda: ValueRange(2.2, 3.0)
    )
    high_frequency: ValueRange = field(
        default_factory=lambda: ValueRange(5.2, 6.4)
    )
    small_amplitude: ValueRange = field(
        default_factory=lambda: ValueRange(0.10, 0.17)
    )
    large_amplitude: ValueRange = field(
        default_factory=lambda: ValueRange(0.23, 0.30)
    )
    render: PendulumRenderConfig = field(
        default_factory=PendulumRenderConfig
    )
    short_prefix_start: int = 57
    prediction_start: int = 65
    seed: int = 3407
    train_base_seeds: int = 1024
    eval_base_seeds: int = 64
    generate_train: bool = True
    generate_eval: bool = True

    def __post_init__(self) -> None:
        if self.target not in ("frequency", "amplitude"):
            raise ValueError(f"Unknown target: {self.target}")
        if self.model_pairing not in ("color", "shape", "color_shape"):
            raise ValueError(f"Unknown model_pairing: {self.model_pairing}")
        if self.test_intervention not in ("color", "shape", "both"):
            raise ValueError(
                f"Unknown test_intervention: {self.test_intervention}"
            )
        if self.fixed_shape not in ("circle", "square"):
            raise ValueError(f"Unknown fixed_shape: {self.fixed_shape}")
        if self.fixed_color not in ("red", "blue"):
            raise ValueError(f"Unknown fixed_color: {self.fixed_color}")
        if self.low_frequency.high >= self.high_frequency.low:
            raise ValueError("frequency ranges must not overlap")
        if self.small_amplitude.high >= self.large_amplitude.low:
            raise ValueError("amplitude ranges must not overlap")
        if self.large_amplitude.high >= math.pi / 2:
            raise ValueError("large-amplitude range is outside renderer scope")
        if self.short_prefix_start != 57 or self.prediction_start != 65:
            raise ValueError(
                "Pendulum v1 fixes masked frames=0..56 and future=65..128"
            )
        if self.render.num_frames - self.prediction_start != 64:
            raise ValueError("both histories must predict exactly 64 frames")
        if self.train_base_seeds <= 0 or self.eval_base_seeds <= 0:
            raise ValueError("train/eval base seed counts must be positive")
        if not (self.generate_train or self.generate_eval):
            raise ValueError("at least one split must be enabled")

        allowed_interventions: dict[PairingName, set[InterventionName]] = {
            "color": {"color"},
            "shape": {"shape"},
            "color_shape": {"color", "shape", "both"},
        }
        if self.test_intervention not in allowed_interventions[
            self.model_pairing
        ]:
            raise ValueError(
                f"{self.model_pairing} training cannot be tested with "
                f"{self.test_intervention!r}"
            )
        if self.ood_enabled and not (
            self.test_intervention == "color"
            and self.model_pairing in ("color", "color_shape")
        ):
            raise ValueError(
                "gray/green OOD is only valid for a color-only intervention"
            )

    @property
    def model_name(self) -> str:
        if self.model_pairing == "color":
            return f"{self.target}_color_{self.fixed_shape}"
        if self.model_pairing == "shape":
            return f"{self.target}_shape_{self.fixed_color}"
        return f"{self.target}_color_shape"

    @property
    def training_manifest_id(self) -> str:
        values = (
            DATASET_ID,
            self.target,
            self.model_pairing,
            self.fixed_shape if self.model_pairing == "color" else "",
            self.fixed_color if self.model_pairing == "shape" else "",
            self.seed,
            self.train_base_seeds,
            self.low_frequency.low,
            self.low_frequency.high,
            self.high_frequency.low,
            self.high_frequency.high,
            self.small_amplitude.low,
            self.small_amplitude.high,
            self.large_amplitude.low,
            self.large_amplitude.high,
            self.short_prefix_start,
            self.prediction_start,
            tuple(sorted(asdict(self.render).items())),
        )
        digest = hashlib.blake2b(
            repr(values).encode("utf-8"),
            digest_size=6,
        ).hexdigest()
        return f"{self.model_name}__train_{digest}"

    @property
    def test_manifest_id(self) -> str:
        suffix = "with_ood" if self.ood_enabled else "id_only"
        return (
            f"{self.model_name}__{self.test_intervention}_swap__{suffix}"
        )

    @property
    def future_frames(self) -> int:
        return self.render.num_frames - self.prediction_start

    @property
    def latent_frames(self) -> int:
        return (self.render.num_frames - 1) // 4 + 1

    @property
    def condition_latents(self) -> int:
        return (self.prediction_start - 1) // 4 + 1

    @property
    def target_latents(self) -> int:
        return self.latent_frames - self.condition_latents

    @property
    def enabled_splits(self) -> tuple[SplitName, ...]:
        values: list[SplitName] = []
        if self.generate_train:
            values.append("train")
        if self.generate_eval:
            values.append("eval")
        return tuple(values)


@dataclass(frozen=True)
class PendulumParameters:
    omega: float
    amplitude: float
    phase: float


@dataclass(frozen=True)
class PhysicalCase:
    target_index: int
    frequency_label: FrequencyLabel
    amplitude_label: AmplitudeLabel
    parameters: PendulumParameters


def pendulum_trajectory(
    parameters: PendulumParameters,
    *,
    fps: int,
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return angular position and velocity for an undamped small-angle pendulum."""
    if parameters.omega <= 0.0 or parameters.amplitude <= 0.0:
        raise ValueError("omega and amplitude must be positive")
    if fps <= 0 or num_frames <= 0:
        raise ValueError("fps and num_frames must be positive")
    times = np.arange(num_frames, dtype=np.float64) / float(fps)
    arguments = parameters.omega * times + parameters.phase
    theta = parameters.amplitude * np.cos(arguments)
    angular_velocity = (
        -parameters.amplitude * parameters.omega * np.sin(arguments)
    )
    return theta, angular_velocity


def _sample_uniform(
    rng: np.random.Generator,
    value_range: ValueRange,
) -> float:
    return float(rng.uniform(value_range.low, value_range.high))


def sample_matched_physics_pair(
    config: PendulumDatasetConfig,
    *,
    split: SplitName,
    base_index: int,
) -> tuple[int, tuple[PhysicalCase, PhysicalCase]]:
    """Sample a balanced target pair with all non-target physics matched."""
    if base_index < 0:
        raise ValueError("base_index must be non-negative")
    split_offset = {"train": 1_000_000, "eval": 2_000_000}[split]
    base_seed = config.seed + split_offset + base_index
    target_code = 0 if config.target == "frequency" else 1
    seed_sequence = np.random.SeedSequence(
        [base_seed, target_code, 104729]
    )
    shared_seed, first_seed, second_seed = seed_sequence.spawn(3)
    shared_rng = np.random.default_rng(shared_seed)
    first_rng = np.random.default_rng(first_seed)
    second_rng = np.random.default_rng(second_seed)
    phase = float(shared_rng.uniform(0.0, 2.0 * math.pi))

    if config.target == "frequency":
        shared_amplitude = _sample_uniform(
            shared_rng,
            config.small_amplitude,
        )
        first = PhysicalCase(
            target_index=0,
            frequency_label="low_frequency",
            amplitude_label="small_amplitude",
            parameters=PendulumParameters(
                omega=_sample_uniform(first_rng, config.low_frequency),
                amplitude=shared_amplitude,
                phase=phase,
            ),
        )
        second = PhysicalCase(
            target_index=1,
            frequency_label="high_frequency",
            amplitude_label="small_amplitude",
            parameters=PendulumParameters(
                omega=_sample_uniform(second_rng, config.high_frequency),
                amplitude=shared_amplitude,
                phase=phase,
            ),
        )
    else:
        shared_frequency = _sample_uniform(
            shared_rng,
            config.low_frequency,
        )
        first = PhysicalCase(
            target_index=0,
            frequency_label="low_frequency",
            amplitude_label="small_amplitude",
            parameters=PendulumParameters(
                omega=shared_frequency,
                amplitude=_sample_uniform(
                    first_rng,
                    config.small_amplitude,
                ),
                phase=phase,
            ),
        )
        second = PhysicalCase(
            target_index=1,
            frequency_label="low_frequency",
            amplitude_label="large_amplitude",
            parameters=PendulumParameters(
                omega=shared_frequency,
                amplitude=_sample_uniform(
                    second_rng,
                    config.large_amplitude,
                ),
                phase=phase,
            ),
        )
    return base_seed, (first, second)


@dataclass(frozen=True)
class Appearance:
    color: ColorName
    shape: ShapeName


def _opposite_training_color(color: ColorName) -> TrainingColorName:
    if color == "red":
        return "blue"
    if color == "blue":
        return "red"
    raise ValueError(f"cannot swap OOD color {color!r}")


def _opposite_shape(shape: ShapeName) -> ShapeName:
    return "square" if shape == "circle" else "circle"


def training_appearance(
    config: PendulumDatasetConfig,
    target_index: int,
) -> Appearance:
    """Return the appearance correlated with target class 0 or 1."""
    if target_index not in (0, 1):
        raise ValueError(f"target_index must be 0 or 1, got {target_index}")
    correlated_color: TrainingColorName = (
        "red" if target_index == 0 else "blue"
    )
    correlated_shape: ShapeName = (
        "circle" if target_index == 0 else "square"
    )
    if config.model_pairing == "color":
        return Appearance(correlated_color, config.fixed_shape)
    if config.model_pairing == "shape":
        return Appearance(config.fixed_color, correlated_shape)
    return Appearance(correlated_color, correlated_shape)


def id_test_appearance(
    config: PendulumDatasetConfig,
    target_index: int,
) -> Appearance:
    """Swap only the configured in-support appearance cue."""
    aligned = training_appearance(config, target_index)
    color = aligned.color
    shape = aligned.shape
    if config.test_intervention in ("color", "both"):
        color = _opposite_training_color(color)
    if config.test_intervention in ("shape", "both"):
        shape = _opposite_shape(shape)
    return Appearance(color, shape)


def appearances_for_case(
    config: PendulumDatasetConfig,
    *,
    split: SplitName,
    target_index: int,
) -> tuple[tuple[str, Appearance], ...]:
    """Return all renders required for one physical trajectory."""
    if split == "train":
        return (("aligned", training_appearance(config, target_index)),)

    id_variant = f"id_{config.test_intervention}_swap"
    variants: list[tuple[str, Appearance]] = [
        (id_variant, id_test_appearance(config, target_index))
    ]
    if config.ood_enabled:
        trained_shape = training_appearance(config, target_index).shape
        variants.extend(
            (
                ("ood_gray", Appearance("gray", trained_shape)),
                ("ood_green", Appearance("green", trained_shape)),
            )
        )
    return tuple(variants)


def group_name(case: PhysicalCase, appearance: Appearance) -> str:
    """Return the human-readable four-label group name from the plan."""
    return "-".join(
        (
            case.frequency_label,
            case.amplitude_label,
            appearance.color,
            appearance.shape,
        )
    )


@dataclass(frozen=True)
class PendulumSample:
    benchmark_version: str
    sample_id: str
    trajectory_id: str
    pair_id: str
    base_seed: int
    split: SplitName
    subset: str
    variant: str
    target: TargetName
    target_index: int
    target_label: str
    model_name: str
    training_manifest_id: str
    test_manifest_id: str
    group_name: str
    frequency_label: FrequencyLabel
    amplitude_label: AmplitudeLabel
    color_label: ColorName
    shape_label: ShapeName
    omega_true: float
    amplitude_true: float
    phase: float
    pendulum_length: float
    fps: int
    frames: int
    short_prefix_start: int
    prediction_start: int
    theta_star: float
    angular_velocity_star: float
    video: str
    source: str
    metadata: str
    trajectory: str
    render_hash_outside_appearance_roi: str


def bob_centers(
    theta: Sequence[float],
    config: PendulumRenderConfig,
) -> np.ndarray:
    """Convert angular positions to floating-point pixel centers."""
    angles = np.asarray(theta, dtype=np.float64)
    x = (
        config.pivot_x + config.length * np.sin(angles)
    ) * (config.width - 1)
    y = (
        config.pivot_y + config.length * np.cos(angles)
    ) * (config.height - 1)
    return np.stack((x, y), axis=-1)


def _color_rgb(
    color: ColorName,
    config: PendulumRenderConfig,
) -> tuple[int, int, int]:
    return {
        "red": config.red_rgb,
        "blue": config.blue_rgb,
        "gray": config.gray_rgb,
        "green": config.green_rgb,
    }[color]


def render_frame(
    theta: float,
    appearance: Appearance,
    config: PendulumRenderConfig,
) -> np.ndarray:
    """Render one pendulum frame; length is fixed across all samples."""
    image = Image.new(
        "RGB",
        (config.width, config.height),
        config.background_rgb,
    )
    draw = ImageDraw.Draw(image)
    pivot = (
        config.pivot_x * (config.width - 1),
        config.pivot_y * (config.height - 1),
    )
    center = bob_centers((theta,), config)[0]
    pivot_xy = (round(float(pivot[0])), round(float(pivot[1])))
    center_xy = (round(float(center[0])), round(float(center[1])))
    draw.line(
        (*pivot_xy, *center_xy),
        fill=config.rope_rgb,
        width=config.rope_width_px,
    )
    draw.ellipse(
        (
            pivot_xy[0] - 3,
            pivot_xy[1] - 3,
            pivot_xy[0] + 3,
            pivot_xy[1] + 3,
        ),
        fill=config.pivot_rgb,
    )

    radius = config.bob_radius_px
    bounds = (
        center_xy[0] - radius,
        center_xy[1] - radius,
        center_xy[0] + radius,
        center_xy[1] + radius,
    )
    fill = _color_rgb(appearance.color, config)
    if appearance.shape == "circle":
        draw.ellipse(bounds, fill=fill)
    else:
        draw.rectangle(bounds, fill=fill)
    return np.asarray(image, dtype=np.uint8)


def render_video(
    theta: Sequence[float],
    appearance: Appearance,
    config: PendulumRenderConfig,
) -> np.ndarray:
    return np.stack(
        [
            render_frame(float(value), appearance, config)
            for value in theta
        ],
        axis=0,
    )


def write_video(path: Path, frames: np.ndarray, fps: int) -> None:
    """Write an RGB uint8 video without changing its temporal length."""
    import imageio.v2 as imageio

    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"expected [T,H,W,3] RGB video, got {array.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=10,
        macro_block_size=None,
    ) as writer:
        for frame in array:
            writer.append_data(np.asarray(frame, dtype=np.uint8))


def load_video(
    path: Path,
    *,
    expected_frames: int | None = None,
) -> np.ndarray:
    import imageio.v2 as imageio

    reader = imageio.get_reader(path)
    try:
        frames = np.stack(
            [np.asarray(frame)[..., :3] for frame in reader],
            axis=0,
        ).astype(np.uint8)
    finally:
        reader.close()
    if expected_frames is not None and frames.shape[0] != expected_frames:
        raise ValueError(
            f"expected {expected_frames} frames, read {frames.shape[0]}"
        )
    return frames


def apply_short_history_mask(
    frames: np.ndarray,
    config: PendulumDatasetConfig,
) -> np.ndarray:
    """Mask frames 0..56 while preserving real frames 57..128 exactly."""
    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"expected [T,H,W,3] RGB video, got {array.shape}")
    if array.shape[0] != config.render.num_frames:
        raise ValueError(
            f"expected {config.render.num_frames} frames, got {array.shape[0]}"
        )
    output = array.copy()
    background = np.asarray(
        config.render.background_rgb,
        dtype=output.dtype,
    )
    output[: config.short_prefix_start] = background
    return output


def apply_short_history_mask_tensor(
    video: Any,
    config: PendulumDatasetConfig,
) -> Any:
    """Torch equivalent for normalized ``[B,3,T,H,W]`` videos."""
    import torch

    if not isinstance(video, torch.Tensor):
        raise TypeError("video must be a torch.Tensor")
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError(
            f"expected [B,3,T,H,W] video, got {tuple(video.shape)}"
        )
    if video.shape[2] != config.render.num_frames:
        raise ValueError(
            f"expected {config.render.num_frames} frames, "
            f"got {video.shape[2]}"
        )
    output = video.clone()
    background = (
        torch.as_tensor(
            config.render.background_rgb,
            dtype=torch.float32,
            device=video.device,
        )
        .div(127.5)
        .sub(1.0)
        .to(dtype=video.dtype)
    )
    output[:, :, : config.short_prefix_start] = background.view(
        1,
        3,
        1,
        1,
        1,
    )
    return output


def _render_hash_outside_appearance_roi(
    frames: np.ndarray,
    centers: np.ndarray,
    config: PendulumRenderConfig,
) -> str:
    """Hash pixels after removing the full color/shape-sensitive bob ROI."""
    normalized = np.asarray(frames, dtype=np.uint8).copy()
    if len(normalized) != len(centers):
        raise ValueError("frame and center counts differ")
    radius = config.bob_radius_px + 2
    for frame, (cx_float, cy_float) in zip(
        normalized,
        centers,
        strict=True,
    ):
        cx, cy = round(float(cx_float)), round(float(cy_float))
        x0, x1 = max(0, cx - radius), min(
            config.width,
            cx + radius + 1,
        )
        y0, y1 = max(0, cy - radius), min(
            config.height,
            cy + radius + 1,
        )
        frame[y0:y1, x0:x1] = 0
    return hashlib.sha256(normalized.tobytes()).hexdigest()


def _opaque_id(*parts: object, prefix: str) -> str:
    source = ":".join(str(part) for part in (DATASET_ID, *parts))
    token = hashlib.blake2b(
        source.encode("utf-8"),
        digest_size=10,
    ).hexdigest()
    return f"{prefix}_{token}"


def _target_label(
    config: PendulumDatasetConfig,
    case: PhysicalCase,
) -> str:
    return (
        case.frequency_label
        if config.target == "frequency"
        else case.amplitude_label
    )


def _write_metadata_csv(
    path: Path,
    rows: Sequence[PendulumSample],
) -> None:
    if not rows:
        raise ValueError("refusing to write an empty metadata file")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(asdict(rows[0]).keys()),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_sample_metadata(path: Path, sample: PendulumSample) -> None:
    path.write_text(
        json.dumps(
            asdict(sample),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def generate_split(
    output_dir: Path,
    config: PendulumDatasetConfig,
    *,
    split: SplitName,
    num_base_seeds: int,
) -> list[PendulumSample]:
    """Materialize one balanced split for the selected logical model."""
    if num_base_seeds <= 0:
        raise ValueError("num_base_seeds must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[PendulumSample] = []

    for base_index in range(num_base_seeds):
        base_seed, physical_pair = sample_matched_physics_pair(
            config,
            split=split,
            base_index=base_index,
        )
        pair_id = _opaque_id(
            config.target,
            split,
            base_seed,
            prefix="pair",
        )
        for case in physical_pair:
            trajectory_id = _opaque_id(
                config.target,
                split,
                base_seed,
                case.target_index,
                prefix="traj",
            )
            trajectory_name = f"{trajectory_id}.npz"
            theta, angular_velocity = pendulum_trajectory(
                case.parameters,
                fps=config.render.fps,
                num_frames=config.render.num_frames,
            )
            np.savez_compressed(
                output_dir / trajectory_name,
                theta=theta,
                angular_velocity=angular_velocity,
                omega=case.parameters.omega,
                amplitude=case.parameters.amplitude,
                phase=case.parameters.phase,
                pendulum_length=config.render.length,
            )
            centers = bob_centers(theta, config.render)
            trajectory_render_hash: str | None = None

            for variant, appearance in appearances_for_case(
                config,
                split=split,
                target_index=case.target_index,
            ):
                manifest_part = (
                    config.training_manifest_id
                    if split == "train"
                    else config.test_manifest_id
                )
                sample_id = _opaque_id(
                    manifest_part,
                    split,
                    base_seed,
                    case.target_index,
                    variant,
                    prefix="sample",
                )
                video_name = f"{sample_id}.mp4"
                metadata_name = f"{sample_id}.json"
                frames = render_video(theta, appearance, config.render)
                render_hash = _render_hash_outside_appearance_roi(
                    frames,
                    centers,
                    config.render,
                )
                if trajectory_render_hash is None:
                    trajectory_render_hash = render_hash
                elif render_hash != trajectory_render_hash:
                    raise AssertionError(
                        f"{trajectory_id} differs outside appearance ROI"
                    )
                write_video(
                    output_dir / video_name,
                    frames,
                    config.render.fps,
                )
                subset = (
                    "train/aligned"
                    if split == "train"
                    else (
                        "eval/ood"
                        if variant.startswith("ood_")
                        else "eval/id_swap"
                    )
                )
                sample = PendulumSample(
                    benchmark_version=DATASET_ID,
                    sample_id=sample_id,
                    trajectory_id=trajectory_id,
                    pair_id=pair_id,
                    base_seed=base_seed,
                    split=split,
                    subset=subset,
                    variant=variant,
                    target=config.target,
                    target_index=case.target_index,
                    target_label=_target_label(config, case),
                    model_name=config.model_name,
                    training_manifest_id=config.training_manifest_id,
                    test_manifest_id=(
                        "" if split == "train" else config.test_manifest_id
                    ),
                    group_name=group_name(case, appearance),
                    frequency_label=case.frequency_label,
                    amplitude_label=case.amplitude_label,
                    color_label=appearance.color,
                    shape_label=appearance.shape,
                    omega_true=case.parameters.omega,
                    amplitude_true=case.parameters.amplitude,
                    phase=case.parameters.phase,
                    pendulum_length=config.render.length,
                    fps=config.render.fps,
                    frames=config.render.num_frames,
                    short_prefix_start=config.short_prefix_start,
                    prediction_start=config.prediction_start,
                    theta_star=float(theta[config.prediction_start - 1]),
                    angular_velocity_star=float(
                        angular_velocity[config.prediction_start - 1]
                    ),
                    video=video_name,
                    source=video_name,
                    metadata=metadata_name,
                    trajectory=trajectory_name,
                    render_hash_outside_appearance_roi=render_hash,
                )
                _write_sample_metadata(
                    output_dir / metadata_name,
                    sample,
                )
                rows.append(sample)

    _write_metadata_csv(output_dir / "metadata.csv", rows)
    return rows


def audit_rows(
    rows: Sequence[PendulumSample],
    config: PendulumDatasetConfig,
    *,
    split: SplitName,
) -> dict[str, Any]:
    """Validate balance, pairing, intervention, and trajectory invariants."""
    if not rows:
        raise ValueError("cannot audit an empty split")
    errors: list[str] = []
    by_subset: dict[str, dict[int, int]] = {}
    by_pair: dict[str, list[PendulumSample]] = {}
    by_trajectory: dict[str, list[PendulumSample]] = {}

    for row in rows:
        if row.split != split:
            errors.append(f"{row.sample_id}: split mismatch")
        if row.target != config.target:
            errors.append(f"{row.sample_id}: target mismatch")
        if row.model_name != config.model_name:
            errors.append(f"{row.sample_id}: model mismatch")
        if row.training_manifest_id != config.training_manifest_id:
            errors.append(f"{row.sample_id}: training ID mismatch")
        expected_test_id = "" if split == "train" else config.test_manifest_id
        if row.test_manifest_id != expected_test_id:
            errors.append(f"{row.sample_id}: test ID mismatch")
        if row.pendulum_length != config.render.length:
            errors.append(f"{row.sample_id}: pendulum length changed")

        expected_variants = dict(
            appearances_for_case(
                config,
                split=split,
                target_index=row.target_index,
            )
        )
        expected_appearance = expected_variants.get(row.variant)
        if expected_appearance is None:
            errors.append(f"{row.sample_id}: unexpected variant {row.variant}")
        elif (
            row.color_label != expected_appearance.color
            or row.shape_label != expected_appearance.shape
        ):
            errors.append(f"{row.sample_id}: wrong intervention appearance")

        expected_group = "-".join(
            (
                row.frequency_label,
                row.amplitude_label,
                row.color_label,
                row.shape_label,
            )
        )
        if row.group_name != expected_group:
            errors.append(f"{row.sample_id}: group name mismatch")
        expected_target_label = (
            row.frequency_label
            if config.target == "frequency"
            else row.amplitude_label
        )
        if row.target_label != expected_target_label:
            errors.append(f"{row.sample_id}: target label mismatch")

        frequency_range = (
            config.low_frequency
            if row.frequency_label == "low_frequency"
            else config.high_frequency
        )
        amplitude_range = (
            config.small_amplitude
            if row.amplitude_label == "small_amplitude"
            else config.large_amplitude
        )
        if not frequency_range.contains(row.omega_true):
            errors.append(f"{row.sample_id}: frequency outside label range")
        if not amplitude_range.contains(row.amplitude_true):
            errors.append(f"{row.sample_id}: amplitude outside label range")

        subset_counts = by_subset.setdefault(row.subset, {0: 0, 1: 0})
        if row.target_index not in (0, 1):
            errors.append(f"{row.sample_id}: invalid target index")
        else:
            subset_counts[row.target_index] += 1
        by_pair.setdefault(row.pair_id, []).append(row)
        by_trajectory.setdefault(row.trajectory_id, []).append(row)

    for subset, counts in by_subset.items():
        if counts[0] != counts[1]:
            errors.append(f"{subset}: target classes are not 50/50")

    expected_variants_per_trajectory = 1
    if split == "eval" and config.ood_enabled:
        expected_variants_per_trajectory = 3
    for trajectory_id, variants in by_trajectory.items():
        if len(variants) != expected_variants_per_trajectory:
            errors.append(
                f"{trajectory_id}: expected "
                f"{expected_variants_per_trajectory} variants, "
                f"found {len(variants)}"
            )
        if len({row.variant for row in variants}) != len(variants):
            errors.append(f"{trajectory_id}: duplicate appearance variant")
        if len(
            {
                (
                    row.omega_true,
                    row.amplitude_true,
                    row.phase,
                    row.theta_star,
                    row.angular_velocity_star,
                    row.trajectory,
                )
                for row in variants
            }
        ) != 1:
            errors.append(f"{trajectory_id}: counterfactual physics changed")
        if len(
            {
                row.render_hash_outside_appearance_roi
                for row in variants
            }
        ) != 1:
            errors.append(f"{trajectory_id}: render changed outside bob ROI")

    for pair_id, pair_rows in by_pair.items():
        representatives: dict[int, PendulumSample] = {}
        for row in pair_rows:
            representatives.setdefault(row.target_index, row)
        if set(representatives) != {0, 1}:
            errors.append(f"{pair_id}: missing target class")
            continue
        first, second = representatives[0], representatives[1]
        if first.phase != second.phase:
            errors.append(f"{pair_id}: phase is not matched")
        if config.target == "frequency":
            if first.amplitude_true != second.amplitude_true:
                errors.append(f"{pair_id}: amplitude is not matched")
            if not (
                first.frequency_label == "low_frequency"
                and second.frequency_label == "high_frequency"
                and first.amplitude_label
                == second.amplitude_label
                == "small_amplitude"
            ):
                errors.append(f"{pair_id}: wrong frequency target labels")
        else:
            if first.omega_true != second.omega_true:
                errors.append(f"{pair_id}: frequency is not matched")
            if not (
                first.amplitude_label == "small_amplitude"
                and second.amplitude_label == "large_amplitude"
                and first.frequency_label
                == second.frequency_label
                == "low_frequency"
            ):
                errors.append(f"{pair_id}: wrong amplitude target labels")

    if errors:
        raise AssertionError("; ".join(errors[:20]))
    return {
        "benchmark_version": DATASET_ID,
        "split": split,
        "model_name": config.model_name,
        "training_manifest_id": config.training_manifest_id,
        "test_manifest_id": (
            "" if split == "train" else config.test_manifest_id
        ),
        "num_rows": len(rows),
        "num_matched_pairs": len(by_pair),
        "num_physical_trajectories": len(by_trajectory),
        "target_counts_by_subset": by_subset,
        "short_masked_frames": [0, config.short_prefix_start - 1],
        "short_real_history_frames": [
            config.short_prefix_start,
            config.prediction_start - 1,
        ],
        "long_real_history_frames": [0, config.prediction_start - 1],
        "future_frames": [
            config.prediction_start,
            config.render.num_frames - 1,
        ],
        "future_frame_count": config.future_frames,
    }


def config_to_dict(config: PendulumDatasetConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "dataset_id": DATASET_ID,
            "model_name": config.model_name,
            "training_manifest_id": config.training_manifest_id,
            "test_manifest_id": config.test_manifest_id,
        }
    )
    return payload


def config_from_mapping(
    payload: Mapping[str, Any],
) -> PendulumDatasetConfig:
    """Build a validated config from a future YAML/JSON mapping."""
    defaults = PendulumDatasetConfig()

    def value_range(name: str, default: ValueRange) -> ValueRange:
        raw = payload.get(name)
        if raw is None:
            return default
        if isinstance(raw, ValueRange):
            return raw
        if not isinstance(raw, Mapping):
            raise ValueError(f"{name} must be a mapping")
        return ValueRange(low=float(raw["low"]), high=float(raw["high"]))

    render_raw = payload.get("render", {})
    if isinstance(render_raw, PendulumRenderConfig):
        render = render_raw
    else:
        if not isinstance(render_raw, Mapping):
            raise ValueError("render must be a mapping")
        render_values = dict(render_raw)
        for key in (
            "background_rgb",
            "rope_rgb",
            "pivot_rgb",
            "red_rgb",
            "blue_rgb",
            "gray_rgb",
            "green_rgb",
        ):
            if key in render_values:
                render_values[key] = tuple(
                    int(channel) for channel in render_values[key]
                )
        render = PendulumRenderConfig(**render_values)

    def boolean(name: str, default: bool) -> bool:
        value = payload.get(name, default)
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be true or false")
        return value

    return PendulumDatasetConfig(
        target=str(payload.get("target", defaults.target)),
        model_pairing=str(
            payload.get("model_pairing", defaults.model_pairing)
        ),
        test_intervention=str(
            payload.get(
                "test_intervention",
                defaults.test_intervention,
            )
        ),
        fixed_shape=str(
            payload.get("fixed_shape", defaults.fixed_shape)
        ),
        fixed_color=str(
            payload.get("fixed_color", defaults.fixed_color)
        ),
        ood_enabled=boolean("ood_enabled", defaults.ood_enabled),
        low_frequency=value_range(
            "low_frequency",
            defaults.low_frequency,
        ),
        high_frequency=value_range(
            "high_frequency",
            defaults.high_frequency,
        ),
        small_amplitude=value_range(
            "small_amplitude",
            defaults.small_amplitude,
        ),
        large_amplitude=value_range(
            "large_amplitude",
            defaults.large_amplitude,
        ),
        render=render,
        short_prefix_start=int(
            payload.get(
                "short_prefix_start",
                defaults.short_prefix_start,
            )
        ),
        prediction_start=int(
            payload.get("prediction_start", defaults.prediction_start)
        ),
        seed=int(payload.get("seed", defaults.seed)),
        train_base_seeds=int(
            payload.get(
                "train_base_seeds",
                defaults.train_base_seeds,
            )
        ),
        eval_base_seeds=int(
            payload.get("eval_base_seeds", defaults.eval_base_seeds)
        ),
        generate_train=boolean(
            "generate_train",
            defaults.generate_train,
        ),
        generate_eval=boolean(
            "generate_eval",
            defaults.generate_eval,
        ),
    )


def _prepare_output_root(root: Path, *, overwrite: bool) -> None:
    if root.is_symlink():
        raise ValueError("dataset root must not be a symbolic link")
    if not root.exists() or not any(root.iterdir()):
        root.mkdir(parents=True, exist_ok=True)
        return
    if not overwrite:
        raise FileExistsError(f"dataset root is not empty: {root}")

    allowed = {
        "videos",
        "metadata",
        "dataset.json",
        "samples.jsonl",
        "build_summary.json",
    }
    unexpected = sorted(path.name for path in root.iterdir() if path.name not in allowed)
    if unexpected:
        raise ValueError(
            "refusing to overwrite a directory containing unknown files: "
            + ", ".join(unexpected)
        )
    resolved = root.resolve()
    if resolved == Path.cwd().resolve() or len(resolved.parts) < 3:
        raise ValueError(f"refusing to replace broad path: {resolved}")
    shutil.rmtree(root)
    root.mkdir(parents=True)


def _common_sample(
    row: PendulumSample,
) -> dict[str, Any]:
    bank = Path("videos") / row.split
    return {
        "sample_id": row.sample_id,
        "split": row.split,
        "subset": row.subset,
        "video": (bank / row.video).as_posix(),
        "metadata": (bank / row.metadata).as_posix(),
        "attributes": {
            "target": row.target,
            "target_index": row.target_index,
            "target_label": row.target_label,
            "model_name": row.model_name,
            "training_manifest_id": row.training_manifest_id,
            "test_manifest_id": row.test_manifest_id,
            "variant": row.variant,
            "group_name": row.group_name,
            "trajectory_id": row.trajectory_id,
            "pair_id": row.pair_id,
            "frequency_label": row.frequency_label,
            "amplitude_label": row.amplitude_label,
            "color_label": row.color_label,
            "shape_label": row.shape_label,
        },
    }


def generate_dataset(
    config: PendulumDatasetConfig | Mapping[str, Any],
    root: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Generate the one subexperiment selected by ``config``."""
    resolved_config = (
        config
        if isinstance(config, PendulumDatasetConfig)
        else config_from_mapping(config)
    )
    root = Path(root)
    _prepare_output_root(root, overwrite=overwrite)

    all_rows: list[PendulumSample] = []
    split_summaries: dict[str, Any] = {}
    counts: dict[SplitName, int] = {
        "train": resolved_config.train_base_seeds,
        "eval": resolved_config.eval_base_seeds,
    }
    for split in resolved_config.enabled_splits:
        rows = generate_split(
            root / "videos" / split,
            resolved_config,
            split=split,
            num_base_seeds=counts[split],
        )
        audit = audit_rows(rows, resolved_config, split=split)
        write_json(root / "videos" / split / "audit.json", audit)
        all_rows.extend(rows)
        split_summaries[split] = audit

    metadata_dir = root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    config_path = metadata_dir / "generation_config_resolved.json"
    write_json(config_path, config_to_dict(resolved_config))
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "dataset_id": DATASET_ID,
        "model_name": resolved_config.model_name,
        "training_manifest_id": resolved_config.training_manifest_id,
        "test_manifest_id": resolved_config.test_manifest_id,
        "splits": split_summaries,
    }
    write_json(root / "build_summary.json", summary)
    write_dataset(
        root,
        experiment=EXPERIMENT_ID,
        dataset=DATASET_ID,
        samples=(_common_sample(row) for row in all_rows),
        extra={
            "model_name": resolved_config.model_name,
            "training_manifest_id": (
                resolved_config.training_manifest_id
            ),
            "test_manifest_id": resolved_config.test_manifest_id,
            "generation_config": config_path.relative_to(root).as_posix(),
            "build_summary": "build_summary.json",
        },
    )
    return root


def audit_dataset(
    root: Path,
    config: PendulumDatasetConfig | Mapping[str, Any],
) -> dict[str, Any]:
    """Re-audit a generated dataset without rerendering it."""
    resolved_config = (
        config
        if isinstance(config, PendulumDatasetConfig)
        else config_from_mapping(config)
    )
    manifest, common_rows = read_dataset(root, check_files=True)
    if manifest.get("experiment") != EXPERIMENT_ID:
        raise ValueError("dataset experiment ID is not pendulum")
    if manifest.get("dataset") != DATASET_ID:
        raise ValueError(f"dataset ID is not {DATASET_ID}")

    by_split: dict[SplitName, list[PendulumSample]] = {
        "train": [],
        "eval": [],
    }
    for common_row in common_rows:
        metadata_path = root / common_row["metadata"]
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        sample = PendulumSample(**payload)
        by_split[sample.split].append(sample)

        trajectory_path = metadata_path.parent / sample.trajectory
        if not trajectory_path.is_file():
            raise FileNotFoundError(
                f"missing trajectory: {trajectory_path}"
            )
        with np.load(trajectory_path) as trajectory:
            expected_theta, expected_velocity = pendulum_trajectory(
                PendulumParameters(
                    omega=sample.omega_true,
                    amplitude=sample.amplitude_true,
                    phase=sample.phase,
                ),
                fps=sample.fps,
                num_frames=sample.frames,
            )
            if not np.array_equal(trajectory["theta"], expected_theta):
                raise AssertionError(
                    f"{sample.trajectory_id}: theta array changed"
                )
            if not np.array_equal(
                trajectory["angular_velocity"],
                expected_velocity,
            ):
                raise AssertionError(
                    f"{sample.trajectory_id}: velocity array changed"
                )

    result: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "dataset_id": DATASET_ID,
        "model_name": resolved_config.model_name,
        "splits": {},
    }
    for split in resolved_config.enabled_splits:
        result["splits"][split] = audit_rows(
            by_split[split],
            resolved_config,
            split=split,
        )
    return result
