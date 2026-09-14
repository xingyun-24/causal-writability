"""Residual-patching diagnostics for the Spring V4 raw-video model.

This is deliberately an *optional* analysis module.  It does not change the
shared Wan sampler: a forward hook on an existing DiT block records its output
after every denoising call, or replaces/adds its condition-token residual.

The experiment answers two questions for every generated intervention video:

* Which colour is rendered for the mass in the generated future?
* Does an instantaneous position/velocity basis fit learned residual
  coordinates better than the trigonometric phase basis?

The latter is an observational model comparison.  For a harmonic
oscillator these quantities are mathematically coupled, so the output must not
be interpreted as proving that the model represents one and not the other.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from sshv2.experiments.spring_shortcuts_v4.data import (
    SpringDatasetConfig,
    color_band,
    dataclass_config_from_dict,
    load_video,
    pixel_x_to_displacement,
    write_video,
)
from sshv2.experiments.spring_shortcuts_v4.evaluation import (
    classify_detected_color,
    detect_mass_track,
    route_metrics_against_bands,
    validity_metrics,
)


def read_metadata(dataset_dir: Path) -> list[dict[str, str]]:
    with (dataset_dir / "metadata.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def paired_rows(rows: Iterable[dict[str, str]], *, limit: int = 0) -> list[tuple[dict[str, str], dict[str, str]]]:
    """Return deterministically ordered (aligned, conflict) Spring pairs."""
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        if row.get("variant") not in {"aligned", "conflict"}:
            continue
        grouped.setdefault(row["pair_id"], {})[row["variant"]] = row
    pairs = [
        (group["aligned"], group["conflict"])
        for _, group in sorted(grouped.items())
        if {"aligned", "conflict"}.issubset(group)
    ]
    return pairs[:limit or None]


def boundary_state(row: dict[str, str]) -> dict[str, float]:
    """Return the known analytic state at the last observed Spring frame."""
    omega = float(row["omega_true"])
    x = float(row["x_star"])
    velocity = float(row["v_star"])
    acceleration = -(omega**2) * x
    phase = math.atan2(-velocity / omega, x) % (2.0 * math.pi)
    return {
        "theta_star": phase,
        "theta_cos": math.cos(phase),
        "theta_sin": math.sin(phase),
        "x_star": x,
        "velocity_star": velocity,
        "acceleration_star": acceleration,
        "omega_true": omega,
        "amplitude": float(row["amplitude"]),
    }


def condition_token_count(total_tokens: int, cfg: SpringDatasetConfig) -> int:
    """Derive the condition-token prefix from the observed token layout."""
    if total_tokens % cfg.long_latent_frames:
        raise ValueError(
            f"Token count {total_tokens} is not divisible by "
            f"{cfg.long_latent_frames} latent frames"
        )
    tokens_per_latent = total_tokens // cfg.long_latent_frames
    return cfg.long_condition_latents * tokens_per_latent


def _frames_from_tensor(generated: Any) -> np.ndarray:
    return (
        generated.detach().float().cpu().permute(1, 2, 3, 0).add(1.0)
        .mul(127.5).clamp(0, 255).byte().numpy()
    )


def measure_generated(frames: np.ndarray, row: dict[str, str], cfg: SpringDatasetConfig) -> dict[str, Any]:
    """Measure final colour and dynamics from an already generated future."""
    track = detect_mass_track(frames, cfg.render)
    validity = validity_metrics(track, cfg.render, x_star=float(row["x_star"]))
    colours = [
        classify_detected_color(rgb, cfg.render)
        for rgb in track.mean_rgb
        if np.isfinite(rgb).all()
    ]
    counts = Counter(colours)
    observed = len(colours)
    result: dict[str, Any] = {
        "valid": bool(validity["valid"]),
        "detected_colour_majority": (
            counts.most_common(1)[0][0] if counts else "unknown"
        ),
        "detected_colour_last": colours[-1] if colours else "unknown",
        "detected_colour_counts": dict(sorted(counts.items())),
        "red_frame_rate": counts["red"] / observed if observed else float("nan"),
        "blue_frame_rate": counts["blue"] / observed if observed else float("nan"),
        "unknown_colour_frame_rate": counts["unknown"] / observed if observed else float("nan"),
        **validity,
    }
    if not validity["valid"]:
        result["route_label"] = "invalid"
        return result
    trajectory = pixel_x_to_displacement(track.x_px, cfg.render)
    route = route_metrics_against_bands(
        trajectory,
        np.isfinite(trajectory),
        true_band=cfg.slow_band if row["true_band"] == "slow" else cfg.fast_band,
        color_implied_band=(
            cfg.slow_band if color_band(row["color_label"]) == "slow" else cfg.fast_band
        ),
        slow_band=cfg.slow_band,
        fast_band=cfg.fast_band,
        omega_true=float(row["omega_true"]),
        x_star=float(row["x_star"]),
        v_star=float(row["v_star"]),
        fps=cfg.render.fps,
        amplitude_low=cfg.amplitude_low,
        amplitude_high=cfg.amplitude_high,
        reference_amplitude_multiplier=cfg.route_reference_amplitude_multiplier,
        frequency_band_tolerance=cfg.frequency_band_tolerance,
        free_shm_rmse_threshold=cfg.free_shm_rmse_threshold,
    )
    route.pop("best_true_curve", None)
    route.pop("best_color_curve", None)
    result.update(route)
    return result


def _finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _loo_predictions(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Leave-one-out predictions for a small linear model."""
    if features.ndim != 2 or targets.ndim != 2 or features.shape[0] != targets.shape[0]:
        raise ValueError("features and targets must have equal sample counts")
    if features.shape[0] < features.shape[1] + 3:
        raise ValueError("Too few samples for leave-one-out linear fitting")
    predicted = np.empty_like(targets, dtype=np.float64)
    for held_out in range(features.shape[0]):
        keep = np.arange(features.shape[0]) != held_out
        beta, *_ = np.linalg.lstsq(features[keep], targets[keep], rcond=None)
        predicted[held_out] = features[held_out] @ beta
    return predicted


def trigonometric_kinematics_report(rows: list[dict[str, Any]], coordinates: np.ndarray) -> dict[str, Any]:
    """Compare phase's ``(cos, sin)`` basis against the ``(x, v)`` basis.

    The baseline is ``beta0 + beta_c cos(theta*) + beta_s sin(theta*)``.  The
    alternative replaces ``cos(theta*)`` and ``sin(theta*)`` with the raw,
    instantaneous ``x*`` and ``v*`` values.  Both fits are evaluated
    leave-one-out, so the reported error change is not a training-set effect.
    This is deliberately a comparison of equal-size bases, not an unrestricted
    state regression.
    """
    if len(rows) != coordinates.shape[0] or coordinates.ndim != 2:
        raise ValueError("one coordinate row is required for every metadata row")
    states = [boundary_state(row) for row in rows]
    phase = np.asarray([[1.0, s["theta_cos"], s["theta_sin"]] for s in states])
    position_velocity = np.asarray(
        [[1.0, s["x_star"], s["velocity_star"]] for s in states]
    )
    phase_prediction = _loo_predictions(phase, coordinates)
    position_velocity_prediction = _loo_predictions(position_velocity, coordinates)
    phase_mse = ((coordinates - phase_prediction) ** 2).mean(axis=0)
    position_velocity_mse = ((coordinates - position_velocity_prediction) ** 2).mean(axis=0)
    names = [f"pc_{index + 1}" for index in range(coordinates.shape[1])]
    per_coordinate = {
        name: {
            "trigonometric_cos_sin_rmse": float(math.sqrt(baseline)),
            "position_velocity_rmse": float(math.sqrt(replacement)),
            "rmse_change_from_trigonometric": float(math.sqrt(baseline) - math.sqrt(replacement)),
            "mse_reduction_fraction_from_trigonometric": (
                None if baseline <= 1e-12 else float(1.0 - replacement / baseline)
            ),
        }
        for name, baseline, replacement in zip(names, phase_mse, position_velocity_mse, strict=True)
    }
    return {
        "interpretation": (
            "Leave-one-out comparison of a trigonometric (cos, sin) basis against "
            "an instantaneous (position, velocity) basis. Phase and kinematics "
            "are coupled in SHM, so a positive reduction is comparative predictive "
            "evidence, not proof of a separate representation."
        ),
        "targets": names,
        "trigonometric_basis": "1 + cos(theta_star) + sin(theta_star)",
        "position_velocity_basis": "1 + x_star + velocity_star",
        "per_coordinate": per_coordinate,
    }


def analyse_saved_coordinates(coordinates_path: Path, output_path: Path) -> dict[str, Any]:
    """Re-evaluate a saved held-out PCA coordinate table without recomputing PCA."""
    saved = json.loads(coordinates_path.read_text(encoding="utf-8"))
    entries = [entry for entry in saved.get("coordinate_rows", []) if entry.get("split") == "held_out"]
    if not entries:
        raise ValueError("Saved coordinate file has no held-out coordinate rows")
    coordinate_names = sorted(
        (name for name in entries[0] if name.startswith("pc_")),
        key=lambda name: int(name.removeprefix("pc_")),
    )
    if not coordinate_names:
        raise ValueError("Saved coordinate file has no PCA coordinate columns")
    rows = [
        {
            "omega_true": entry["omega_true"],
            "x_star": entry["x_star"],
            "v_star": entry["velocity_star"],
            "amplitude": entry["amplitude"],
        }
        for entry in entries
    ]
    coordinates = np.asarray(
        [[float(entry[name]) for name in coordinate_names] for entry in entries],
        dtype=np.float64,
    )
    report = trigonometric_kinematics_report(rows, coordinates)
    report["source_coordinate_file"] = str(coordinates_path)
    report["held_out_pairs"] = len(entries)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_finite_json(report), indent=2) + "\n", encoding="utf-8")
    return report


def _load_delta(path: Path) -> list[Any]:
    """Load one saved condition-residual direction without trusting code."""
    import torch

    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, list) or not value or not all(isinstance(item, torch.Tensor) for item in value):
        raise TypeError(f"Expected a non-empty list of tensors in {path}")
    return value


def _direction_dot(left: Path, right: Path, *, device: str) -> float:
    """Exact dot product while keeping only one denoising step on device."""
    import torch

    a, b = _load_delta(left), _load_delta(right)
    if len(a) != len(b):
        raise ValueError("Residual directions have different denoising-step counts")
    total = torch.zeros((), device=device, dtype=torch.float64)
    for first, second in zip(a, b, strict=True):
        if tuple(first.shape) != tuple(second.shape):
            raise ValueError("Residual directions have incompatible tensor shapes")
        total += (first.to(device=device, dtype=torch.float32) * second.to(device=device, dtype=torch.float32)).sum(dtype=torch.float64)
    return float(total.cpu())


def analyse_saved_deltas(
    mechanism_dir: Path,
    output_path: Path,
    *,
    fit_count: int = 0,
    components: int = 2,
    device: str = "cpu",
) -> dict[str, Any]:
    """Fit an exact dual PCA, then measure kinematic correction of phase fit.

    This avoids stacking the approximately 16.7-million-dimensional vectors.
    PCA is computed from their Gram matrix, then each held-out coordinate is
    obtained from dot products with the fit directions.  It is therefore the
    same uncentred PCA geometry as a direct SVD, but has bounded host memory.
    """
    entries = json.loads((mechanism_dir / "pairs.json").read_text(encoding="utf-8"))
    if not isinstance(entries, list) or len(entries) < 8:
        raise ValueError("Need at least eight captured aligned/conflict pairs")
    entries = sorted(entries, key=lambda item: str(item["pair_id"]))
    n = len(entries)
    fit_count = fit_count or n // 2
    if not 2 <= components < fit_count < n:
        raise ValueError("Require 2 <= components < fit_count < captured pair count")
    paths = [mechanism_dir / "residual_deltas" / str(item["delta"]) for item in entries]
    if not all(path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError(f"Missing residual directions: {missing[:3]}")

    # K[j, i] = <d_j, d_i> for every sample j and fit sample i.  Compute
    # every entry explicitly: a row-major traversal cannot safely reuse a
    # symmetric entry before the corresponding later row has been written.
    cross = np.empty((n, fit_count), dtype=np.float64)
    for row, left in enumerate(paths):
        for column, right in enumerate(paths[:fit_count]):
            cross[row, column] = _direction_dot(left, right, device=device)
    gram = (cross[:fit_count] + cross[:fit_count].T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    positive = eigenvalues > max(1e-10, float(eigenvalues[0]) * 1e-10)
    if int(positive.sum()) < components:
        raise ValueError("Fit residual directions do not have the requested numerical rank")
    scales = np.sqrt(eigenvalues[:components])
    coordinates = cross @ (eigenvectors[:, :components] / scales)
    held_entries = entries[fit_count:]
    report = trigonometric_kinematics_report(
        [dict(item["conflict"]) for item in held_entries], coordinates[fit_count:],
    )
    report.update({
        "pca": {
            "centering": "none",
            "fit_pairs": fit_count,
            "held_out_pairs": n - fit_count,
            "components": components,
            "explained_energy_fraction": [float(value / eigenvalues.sum()) for value in eigenvalues[:components]],
        },
        "coordinate_rows": [
            {
                "pair_id": entry["pair_id"],
                "split": "fit" if index < fit_count else "held_out",
                **boundary_state(dict(entry["conflict"])),
                **{f"pc_{component + 1}": float(value) for component, value in enumerate(coordinates[index])},
            }
            for index, entry in enumerate(entries)
        ],
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_finite_json(report), indent=2) + "\n", encoding="utf-8")
    return report


@dataclass
class CapturedRun:
    frames: np.ndarray
    residuals: list[Any]


class BlockRun:
    """Run a public Wan pipeline while observing or editing a single block."""

    def __init__(self, pipe: Any, block_index: int, cfg: SpringDatasetConfig, steps: int) -> None:
        self.pipe = pipe
        self.block_index = block_index
        self.cfg = cfg
        self.steps = steps
        blocks = pipe.dit.blocks
        if not 0 <= block_index < len(blocks):
            raise ValueError(f"block_index must be in [0, {len(blocks) - 1}]")
        self.block = blocks[block_index]

    def run(
        self,
        condition: Any,
        *,
        seed: int,
        capture: bool = False,
        edits: list[Any] | None = None,
        mode: str = "replace",
    ) -> CapturedRun:
        """Generate once; edit only condition tokens after the target block."""
        import torch

        if mode not in {"replace", "add"}:
            raise ValueError("mode must be replace or add")
        residuals: list[Any] = []
        call_index = 0

        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
            nonlocal call_index
            if not isinstance(output, torch.Tensor) or output.ndim != 3:
                raise TypeError("Expected a [batch,tokens,hidden] DiT block output")
            count = condition_token_count(int(output.shape[1]), self.cfg)
            if capture:
                # Preserve the model's native dtype for an exact replacement;
                # the derived deltas are saved separately in float32 for PCA.
                residuals.append(output[:, :count].detach().to("cpu").clone())
            if edits is None:
                call_index += 1
                return output
            if call_index >= len(edits):
                raise RuntimeError("The sampler invoked the hooked block more times than recorded")
            edit = edits[call_index].to(device=output.device, dtype=output.dtype)
            if tuple(edit.shape) != tuple(output[:, :count].shape):
                raise ValueError(
                    f"Edit shape {tuple(edit.shape)} does not match condition residual "
                    f"{tuple(output[:, :count].shape)}"
                )
            edited = output.clone()
            edited[:, :count] = edit if mode == "replace" else output[:, :count] + edit
            call_index += 1
            return edited

        handle = self.block.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                generated = self.pipe(
                    prompt="", negative_prompt="", cfg_scale=1.0,
                    height=self.cfg.render.height, width=self.cfg.render.width,
                    num_frames=self.cfg.render.num_frames,
                    num_condition_frames=self.cfg.long_condition_latents,
                    condition_frames=condition,
                    num_inference_steps=self.steps, tiled=False, num_samples=1,
                    return_as_tensor=True, progress_bar_cmd=lambda values: values, seed=seed,
                )[0]
        finally:
            handle.remove()
        if edits is not None and call_index != len(edits):
            raise RuntimeError(f"Recorded {len(edits)} edits but hook ran {call_index} times")
        if capture and len(residuals) != self.steps:
            raise RuntimeError(f"Expected {self.steps} residuals, captured {len(residuals)}")
        return CapturedRun(_frames_from_tensor(generated)[self.cfg.prediction_start :], residuals)


def _condition_tensor(row: dict[str, str], dataset_dir: Path, cfg: SpringDatasetConfig, pipe: Any, device: str) -> Any:
    import torch

    raw = load_video(dataset_dir / row["video"], expected_frames=cfg.render.num_frames)
    tensor = torch.from_numpy(raw.copy()).permute(3, 0, 1, 2).float().div(127.5).sub(1.0).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=pipe.torch_dtype)
    return tensor[:, :, :cfg.prediction_start].contiguous()


def load_spring_pipe(training_config: Path, checkpoint: Path, *, device: str, steps: int) -> Any:
    """Load the same standard Wan pipeline used by Spring evaluation."""
    from sshv2.wan.config import StandardTrainingConfig
    from sshv2.wan.trainer import WanTrainingModule

    training = StandardTrainingConfig.from_file(training_config)
    training.model.dit.ckpt_file = checkpoint
    module = WanTrainingModule(
        dit_config=training.model.dit, vae_config=training.model.vae,
        no_encoding=False, num_condition_frames=training.model.num_condition_frames,
        num_inference_steps=steps, pipeline_type=training.model.pipe,
        pipeline_kwargs=training.model.pipe_kwargs,
    )
    pipe = module.pipe
    pipe.to(device)
    pipe.load_models_to_device(("dit", "vae"))
    pipe.pre_encoded_(False)
    return pipe


def run_replacement_experiment(
    dataset_dir: Path,
    output_dir: Path,
    training_config: Path,
    checkpoint: Path,
    data_config: Path,
    *,
    block: int,
    steps: int = 20,
    seed_offset: int = 17_000_000,
    limit: int = 0,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run the paired replacement experiment and write inspectable artefacts."""
    import torch
    import yaml

    cfg = dataclass_config_from_dict(yaml.safe_load(data_config.read_text(encoding="utf-8")))
    pipe = load_spring_pipe(training_config, checkpoint, device=device, steps=steps)
    runner = BlockRun(pipe, block, cfg, steps)
    pairs = paired_rows(read_metadata(dataset_dir), limit=limit)
    if not pairs:
        raise ValueError("No aligned/conflict pairs found")
    video_dir = output_dir / "videos"
    vector_dir = output_dir / "residual_deltas"
    video_dir.mkdir(parents=True, exist_ok=True)
    vector_dir.mkdir(parents=True, exist_ok=True)
    outcomes: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for aligned, conflict in pairs:
        seed = int(aligned["base_seed"]) + seed_offset
        donor = runner.run(_condition_tensor(aligned, dataset_dir, cfg, pipe, device), seed=seed, capture=True)
        receiver = runner.run(_condition_tensor(conflict, dataset_dir, cfg, pipe, device), seed=seed, capture=True)
        deltas = [left.float() - right.float() for left, right in zip(donor.residuals, receiver.residuals, strict=True)]
        delta_path = vector_dir / f"{aligned['pair_id']}.pt"
        torch.save(deltas, delta_path)
        edited = runner.run(_condition_tensor(conflict, dataset_dir, cfg, pipe, device), seed=seed, edits=donor.residuals, mode="replace")
        for label, row, run in (("aligned", aligned, donor), ("conflict", conflict, receiver), ("replacement", conflict, edited)):
            destination = video_dir / f"{aligned['pair_id']}__{label}.mp4"
            write_video(destination, run.frames, cfg.render.fps)
            outcome = {
                "pair_id": aligned["pair_id"], "sample_id": row["sample_id"], "condition": label,
                "block": block, "generation_seed": seed, **boundary_state(row),
                **measure_generated(run.frames, row, cfg),
            }
            outcomes.append(outcome)
        manifest.append({"pair_id": aligned["pair_id"], "aligned": aligned, "conflict": conflict, "delta": delta_path.name})
    (output_dir / "outcomes.json").write_text(json.dumps(_finite_json(outcomes), indent=2) + "\n", encoding="utf-8")
    (output_dir / "pairs.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary = {
        "block": block, "steps": steps, "pairs": len(pairs),
        "condition_tokens_only": True,
        "replacement_note": "At the hooked block, only condition-token outputs are replaced by the paired aligned run.",
        "colour_summary": {
            label: dict(Counter(str(row["detected_colour_majority"]) for row in outcomes if row["condition"] == label))
            for label in ("aligned", "conflict", "replacement")
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--block", type=int, default=13, help="Zero-based DiT block to patch after")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed-offset", type=int, default=17_000_000)
    parser.add_argument("--limit", type=int, default=0, help="Paired-sample cap; 0 uses all pairs")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Refusing to mix results into non-empty output directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    result = run_replacement_experiment(
        args.dataset_dir, args.out, args.training_config, args.checkpoint, args.data_config,
        block=args.block, steps=args.steps, seed_offset=args.seed_offset,
        limit=args.limit, device=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
