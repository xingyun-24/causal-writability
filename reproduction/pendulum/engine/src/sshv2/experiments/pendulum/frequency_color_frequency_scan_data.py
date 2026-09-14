#!/usr/bin/env python3
"""Build resumable color x physical-frequency Pendulum scan datasets."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from sshv2.common.dataset import write_dataset, write_json, read_json
from sshv2.experiments.pendulum.data import (
    DATASET_ID,
    EXPERIMENT_ID,
    Appearance,
    PendulumParameters,
    _opaque_id,
    _render_hash_outside_appearance_roi,
    bob_centers,
    config_from_mapping,
    config_to_dict,
    pendulum_trajectory,
    render_video,
    write_video,
)


DEFAULT_CONFIG = Path(
    "configs/pendulum/frequency_color_shape_frequency_scan.yaml"
)


@dataclass(frozen=True)
class ScanSpec:
    subexperiment_id: str
    color_alphas: tuple[float, ...]
    frequencies_rad_s: tuple[float, ...]
    shapes: tuple[str, ...]
    amplitude_rad: float
    phase_count: int
    diffusion_repeats: int
    future_frames: int
    low_band_rad_s: tuple[float, float]
    high_band_rad_s: tuple[float, float]
    color_frequency_centers_rad_s: tuple[float, float]

    @property
    def repeats_per_cell(self) -> int:
        return self.phase_count * self.diffusion_repeats

    @property
    def samples_per_shape(self) -> int:
        return (
            len(self.color_alphas)
            * len(self.frequencies_rad_s)
            * self.repeats_per_cell
        )


def _strict_float_tuple(values: Sequence[Any], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(not math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite values")
    if tuple(sorted(set(result))) != result:
        raise ValueError(f"{name} must be strictly increasing and unique")
    return result


def load_config(path: Path):
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("scan config root must be a mapping")
    if not isinstance(payload.get("data"), Mapping):
        raise ValueError("scan config requires a data mapping")
    if not isinstance(payload.get("scan"), Mapping):
        raise ValueError("scan config requires a scan mapping")
    config = config_from_mapping(payload["data"])
    raw = payload["scan"]
    spec = ScanSpec(
        subexperiment_id=str(raw["subexperiment_id"]),
        color_alphas=_strict_float_tuple(raw["color_alphas"], "color_alphas"),
        frequencies_rad_s=_strict_float_tuple(
            raw["frequencies_rad_s"], "frequencies_rad_s"
        ),
        shapes=tuple(str(value) for value in raw["shapes"]),
        amplitude_rad=float(raw["amplitude_rad"]),
        phase_count=int(raw["phase_count"]),
        diffusion_repeats=int(raw["diffusion_repeats"]),
        future_frames=int(raw["future_frames"]),
        low_band_rad_s=tuple(float(value) for value in raw["low_band_rad_s"]),
        high_band_rad_s=tuple(float(value) for value in raw["high_band_rad_s"]),
        color_frequency_centers_rad_s=tuple(
            float(value) for value in raw["color_frequency_centers_rad_s"]
        ),
    )
    supported_models = {"frequency_color_circle", "frequency_color_shape"}
    if config.model_name not in supported_models:
        raise ValueError(
            f"scan requires one of {sorted(supported_models)}, got "
            f"{config.model_name}"
        )
    if spec.color_alphas != tuple(round(index / 10, 1) for index in range(11)):
        raise ValueError("scan requires alpha=0.0,0.1,...,1.0")
    expected_shapes = (
        (config.fixed_shape,)
        if config.model_pairing == "color"
        else ("circle", "square")
    )
    if spec.shapes != expected_shapes:
        raise ValueError(
            f"{config.model_name} requires scan shapes {expected_shapes}, "
            f"got {spec.shapes}"
        )
    if spec.phase_count != 8 or spec.diffusion_repeats != 2:
        raise ValueError("scan requires 8 phases x 2 diffusion repeats")
    if spec.future_frames != config.future_frames:
        raise ValueError("scan future-frame count differs from experiment")
    if spec.amplitude_rad <= 0.0:
        raise ValueError("amplitude must be positive")
    if spec.low_band_rad_s != (
        config.low_frequency.low,
        config.low_frequency.high,
    ):
        raise ValueError("low-band metadata differs from training config")
    if spec.high_band_rad_s != (
        config.high_frequency.low,
        config.high_frequency.high,
    ):
        raise ValueError("high-band metadata differs from training config")
    if spec.samples_per_shape != 2288:
        raise ValueError(f"expected 2288 samples per shape, got {spec.samples_per_shape}")
    return config, spec


def interpolate_rgb(
    red_rgb: Sequence[int],
    blue_rgb: Sequence[int],
    alpha: float,
) -> tuple[int, int, int]:
    red = np.asarray(red_rgb, dtype=np.float64)
    blue = np.asarray(blue_rgb, dtype=np.float64)
    if red.shape != (3,) or blue.shape != (3,):
        raise ValueError("RGB endpoints must have three channels")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    return tuple(
        int(value)
        for value in np.rint((1.0 - alpha) * red + alpha * blue)
    )


def _recolor(
    red_frames: np.ndarray,
    red_rgb: Sequence[int],
    target_rgb: Sequence[int],
) -> np.ndarray:
    output = np.asarray(red_frames, dtype=np.uint8).copy()
    mask = np.all(output == np.asarray(red_rgb, dtype=np.uint8), axis=-1)
    if np.any(mask.sum(axis=(1, 2)) == 0):
        raise AssertionError("rendered frame is missing red bob pixels")
    output[mask] = np.asarray(target_rgb, dtype=np.uint8)
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty metadata")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("metadata rows have inconsistent fields")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _frequency_support(omega: float, spec: ScanSpec) -> tuple[int, str, str]:
    if spec.low_band_rad_s[0] <= omega <= spec.low_band_rad_s[1]:
        return 0, "low_frequency", "low_band_id"
    if spec.high_band_rad_s[0] <= omega <= spec.high_band_rad_s[1]:
        return 1, "high_frequency", "high_band_id"
    return 2, "middle_frequency", "middle_frequency_ood"


def _manifest_id(
    model_name: str,
    training_manifest_id: str,
    spec: ScanSpec,
    shape: str,
) -> str:
    payload = (model_name, training_manifest_id, asdict(spec), shape)
    digest = hashlib.blake2b(
        repr(payload).encode("utf-8"), digest_size=6
    ).hexdigest()
    return f"{model_name}__frequency_scan_{shape}_11x13_{digest}"


def _saved_row(
    metadata_path: Path,
    video_path: Path,
    *,
    sample_id: str,
    test_manifest_id: str,
) -> dict[str, Any] | None:
    if not metadata_path.is_file() or not video_path.is_file():
        return None
    if video_path.stat().st_size <= 0:
        return None
    row = read_json(metadata_path)
    if row.get("sample_id") != sample_id:
        raise ValueError(f"saved sample mismatch: {metadata_path}")
    if row.get("test_manifest_id") != test_manifest_id:
        raise ValueError(f"saved manifest mismatch: {metadata_path}")
    return row


def _audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_root: Path,
    config: Any,
    spec: ScanSpec,
    shape: str,
) -> dict[str, Any]:
    if len(rows) != spec.samples_per_shape:
        raise AssertionError(
            f"expected {spec.samples_per_shape} rows, got {len(rows)}"
        )
    if len({str(row["sample_id"]) for row in rows}) != len(rows):
        raise AssertionError("sample IDs are not unique")
    if {str(row["shape_label"]) for row in rows} != {shape}:
        raise AssertionError("unexpected shape in dataset")
    if {float(row["amplitude_true"]) for row in rows} != {spec.amplitude_rad}:
        raise AssertionError("amplitude changed inside the scan")

    alpha_counts = Counter(float(row["test_color_alpha_target"]) for row in rows)
    omega_counts = Counter(float(row["omega_true"]) for row in rows)
    cell_counts = Counter(
        (float(row["omega_true"]), float(row["test_color_alpha_target"]))
        for row in rows
    )
    expected_alpha = len(spec.frequencies_rad_s) * spec.repeats_per_cell
    expected_omega = len(spec.color_alphas) * spec.repeats_per_cell
    if alpha_counts != Counter({alpha: expected_alpha for alpha in spec.color_alphas}):
        raise AssertionError(f"alpha counts differ: {alpha_counts}")
    if omega_counts != Counter({omega: expected_omega for omega in spec.frequencies_rad_s}):
        raise AssertionError(f"frequency counts differ: {omega_counts}")
    if set(cell_counts.values()) != {spec.repeats_per_cell}:
        raise AssertionError("grid cells do not all have 16 repeats")

    by_state: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        alpha = float(row["test_color_alpha_target"])
        expected_rgb = interpolate_rgb(
            config.render.red_rgb, config.render.blue_rgb, alpha
        )
        if tuple(row["test_color_rgb_target"]) != expected_rgb:
            raise AssertionError(f"{row['sample_id']}: RGB mismatch")
        video_path = output_root / "videos" / "eval" / str(row["video"])
        if not video_path.is_file() or video_path.stat().st_size <= 0:
            raise AssertionError(f"missing video: {video_path}")
        by_state[str(row["physical_state_id"])].append(row)

    expected_states = len(spec.frequencies_rad_s) * spec.repeats_per_cell
    if len(by_state) != expected_states:
        raise AssertionError(
            f"expected {expected_states} physical states, got {len(by_state)}"
        )
    for state_id, variants in by_state.items():
        if len(variants) != len(spec.color_alphas):
            raise AssertionError(f"{state_id}: incomplete color sweep")
        if {float(row["test_color_alpha_target"]) for row in variants} != set(
            spec.color_alphas
        ):
            raise AssertionError(f"{state_id}: missing alpha")
        for key in ("omega_true", "amplitude_true", "phase", "base_seed"):
            if len({float(row[key]) for row in variants}) != 1:
                raise AssertionError(f"{state_id}: {key} changed across color")

    return {
        "status": "passed",
        "shape": shape,
        "num_samples": len(rows),
        "num_grid_cells": len(cell_counts),
        "num_physical_states": len(by_state),
        "repeats_per_cell": spec.repeats_per_cell,
        "alpha_counts": {f"{key:.1f}": value for key, value in alpha_counts.items()},
        "frequency_counts": {f"{key:.1f}": value for key, value in omega_counts.items()},
        "fixed_amplitude_rad": spec.amplitude_rad,
        "fixed_length": config.render.length,
        "future_frames": config.future_frames,
    }


def generate(config_path: Path, output_root: Path, shape: str) -> Path:
    config, spec = load_config(config_path)
    if shape not in spec.shapes:
        raise ValueError(f"unsupported shape: {shape}")
    output_root = Path(output_root)
    bank = output_root / "videos" / "eval"
    metadata_dir = output_root / "metadata"
    bank.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    test_manifest_id = _manifest_id(
        config.model_name,
        config.training_manifest_id,
        spec,
        shape,
    )
    state = {
        "subexperiment_id": spec.subexperiment_id,
        "shape": shape,
        "training_manifest_id": config.training_manifest_id,
        "test_manifest_id": test_manifest_id,
        "scan": asdict(spec),
        "expected_samples": spec.samples_per_shape,
        "config": str(config_path),
    }
    state_path = metadata_dir / "generation_state.json"
    if state_path.is_file():
        if read_json(state_path) != json.loads(json.dumps(state)):
            raise ValueError("existing generation state differs")
    else:
        write_json(state_path, state)

    rows: list[dict[str, Any]] = []
    completed = 0
    for omega_index, omega in enumerate(spec.frequencies_rad_s):
        target_index, frequency_label, frequency_support = _frequency_support(
            omega, spec
        )
        for phase_index in range(spec.phase_count):
            phase = 2.0 * math.pi * phase_index / spec.phase_count
            parameters = PendulumParameters(
                omega=omega,
                amplitude=spec.amplitude_rad,
                phase=phase,
            )
            theta, angular_velocity = pendulum_trajectory(
                parameters,
                fps=config.render.fps,
                num_frames=config.render.num_frames,
            )
            trajectory_id = _opaque_id(
                spec.subexperiment_id,
                omega_index,
                phase_index,
                prefix="traj",
            )
            trajectory_name = f"{trajectory_id}.npz"
            trajectory_path = bank / trajectory_name
            if not trajectory_path.is_file():
                _atomic_npz(
                    trajectory_path,
                    theta=theta,
                    angular_velocity=angular_velocity,
                    omega=omega,
                    amplitude=spec.amplitude_rad,
                    phase=phase,
                    pendulum_length=config.render.length,
                )
            red_frames = render_video(
                theta, Appearance("red", shape), config.render
            )
            centers = bob_centers(theta, config.render)

            for repeat_index in range(spec.diffusion_repeats):
                base_seed = 41_000_000 + omega_index * 100 + phase_index * 2 + repeat_index
                physical_state_id = _opaque_id(
                    spec.subexperiment_id,
                    omega_index,
                    phase_index,
                    repeat_index,
                    prefix="state",
                )
                pair_id = physical_state_id
                for alpha_index, alpha in enumerate(spec.color_alphas):
                    alpha_code = int(round(alpha * 100))
                    omega_code = int(round(omega * 100))
                    variant = f"w{omega_code:03d}_a{alpha_code:03d}"
                    sample_id = _opaque_id(
                        test_manifest_id,
                        omega_index,
                        phase_index,
                        repeat_index,
                        alpha_index,
                        prefix="sample",
                    )
                    video_name = f"{sample_id}.mp4"
                    metadata_name = f"{sample_id}.json"
                    video_path = bank / video_name
                    metadata_path = bank / metadata_name
                    saved = _saved_row(
                        metadata_path,
                        video_path,
                        sample_id=sample_id,
                        test_manifest_id=test_manifest_id,
                    )
                    if saved is not None:
                        rows.append(saved)
                        completed += 1
                        continue

                    target_rgb = interpolate_rgb(
                        config.render.red_rgb,
                        config.render.blue_rgb,
                        alpha,
                    )
                    frames = _recolor(
                        red_frames, config.render.red_rgb, target_rgb
                    )
                    render_hash = _render_hash_outside_appearance_roi(
                        frames, centers, config.render
                    )
                    write_video(video_path, frames, config.render.fps)
                    row: dict[str, Any] = {
                        "benchmark_version": DATASET_ID,
                        "subexperiment_id": spec.subexperiment_id,
                        "sample_id": sample_id,
                        "physical_state_id": physical_state_id,
                        "trajectory_id": trajectory_id,
                        "pair_id": pair_id,
                        "grid_cell_id": f"{shape}_w{omega_code:03d}_a{alpha_code:03d}",
                        "base_seed": base_seed,
                        "split": "eval",
                        "subset": "eval/frequency_color_2d_scan",
                        "variant": variant,
                        "target": config.target,
                        "target_index": target_index,
                        "target_label": frequency_label,
                        "model_name": config.model_name,
                        "training_manifest_id": config.training_manifest_id,
                        "test_manifest_id": test_manifest_id,
                        "frequency_label": frequency_label,
                        "amplitude_label": "small_amplitude_fixed",
                        "color_label": (
                            "red" if alpha == 0.0 else "blue" if alpha == 1.0
                            else "interpolated_red_blue"
                        ),
                        "shape_label": shape,
                        "test_color_alpha_target": alpha,
                        "test_color_rgb_target": target_rgb,
                        "test_color_support": (
                            "endpoint_id" if alpha in (0.0, 1.0)
                            else "interpolated_color_ood"
                        ),
                        "test_frequency_index": omega_index,
                        "test_frequency_support": frequency_support,
                        "omega_true": omega,
                        "amplitude_true": spec.amplitude_rad,
                        "phase": phase,
                        "phase_index": phase_index,
                        "diffusion_repeat": repeat_index,
                        "pendulum_length": config.render.length,
                        "fps": config.render.fps,
                        "frames": config.render.num_frames,
                        "short_prefix_start": config.short_prefix_start,
                        "prediction_start": config.prediction_start,
                        "theta_star": float(theta[config.prediction_start - 1]),
                        "angular_velocity_star": float(
                            angular_velocity[config.prediction_start - 1]
                        ),
                        "video": video_name,
                        "source": video_name,
                        "metadata": metadata_name,
                        "trajectory": trajectory_name,
                        "render_hash_outside_appearance_roi": render_hash,
                    }
                    write_json(metadata_path, row)
                    rows.append(row)
                    completed += 1
                    write_json(
                        output_root / "progress.json",
                        {
                            "status": "running",
                            "shape": shape,
                            "completed_samples": completed,
                            "expected_samples": spec.samples_per_shape,
                            "last_sample_id": sample_id,
                        },
                    )

    audit = _audit(
        rows,
        output_root=output_root,
        config=config,
        spec=spec,
        shape=shape,
    )
    _write_csv(bank / "metadata.csv", rows)
    write_json(bank / "audit.json", audit)
    write_json(metadata_dir / "generation_config_resolved.json", config_to_dict(config))
    write_json(metadata_dir / "scan_spec_resolved.json", asdict(spec))
    write_json(
        output_root / "build_summary.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "dataset_id": DATASET_ID,
            "model_name": config.model_name,
            "training_manifest_id": config.training_manifest_id,
            "test_manifest_id": test_manifest_id,
            "eval": audit,
        },
    )
    write_dataset(
        output_root,
        experiment=EXPERIMENT_ID,
        dataset=DATASET_ID,
        samples=(
            {
                "sample_id": row["sample_id"],
                "split": "eval",
                "subset": row["subset"],
                "video": (Path("videos") / "eval" / str(row["video"])).as_posix(),
                "metadata": (Path("videos") / "eval" / str(row["metadata"])).as_posix(),
                "attributes": {
                    "target": row["target"],
                    "target_index": row["target_index"],
                    "model_name": row["model_name"],
                    "training_manifest_id": row["training_manifest_id"],
                    "test_manifest_id": row["test_manifest_id"],
                    "physical_state_id": row["physical_state_id"],
                    "pair_id": row["pair_id"],
                    "grid_cell_id": row["grid_cell_id"],
                    "variant": row["variant"],
                    "test_color_alpha_target": row["test_color_alpha_target"],
                    "test_color_rgb_target": row["test_color_rgb_target"],
                    "omega_true": row["omega_true"],
                    "shape_label": row["shape_label"],
                    "phase_index": row["phase_index"],
                    "diffusion_repeat": row["diffusion_repeat"],
                },
            }
            for row in rows
        ),
        extra={
            "subexperiment_id": spec.subexperiment_id,
            "model_name": config.model_name,
            "training_manifest_id": config.training_manifest_id,
            "test_manifest_id": test_manifest_id,
            "generation_state": "metadata/generation_state.json",
            "build_summary": "build_summary.json",
        },
    )
    write_json(
        output_root / "progress.json",
        {
            "status": "complete",
            "shape": shape,
            "completed_samples": len(rows),
            "expected_samples": spec.samples_per_shape,
            "test_manifest_id": test_manifest_id,
        },
    )
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--shape", choices=("circle", "square"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(generate(args.config, args.output_root, args.shape))


if __name__ == "__main__":
    main()
