#!/usr/bin/env python3
"""Causal condition-residual layer scan for Pendulum shortcut models.

For each frozen, qualified aligned/conflict receiver, capture the aligned
donor condition-token residual at the embedding output and after every DiT
block.  Re-run the natural conflict receiver with exactly one location
replaced on every flow-matching step, then measure frequency and appearance.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from sshv2.experiments.pendulum.mechanism_pca import (
    _condition_latents,
    _denoise,
    _load_runtime,
    _measure_one,
    _read_json,
    _sha256_file,
    _write_csv,
    _write_future_video,
    _write_json,
    manifest_sha256,
    read_jsonl,
    validate_frozen_receiver_manifest,
)


PROGRAM_VERSION = "pendulum_condition_residual_layer_scan_v1"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def _freeze_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    materialized = [dict(row) for row in rows]
    if path.is_file():
        existing = read_jsonl(path)
        if existing != materialized:
            raise ValueError(f"frozen file differs from current selection: {path}")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in materialized
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return materialized


def _qualified_receivers(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_path = args.source_root / "receiver_manifest.jsonl"
    metrics_path = args.source_root / "metrics.csv"
    rows = read_jsonl(manifest_path)
    validate_frozen_receiver_manifest(
        rows,
        target="frequency",
        expected_fit=24,
        expected_heldout=8,
    )
    by_receiver: dict[str, dict[str, dict[str, str]]] = {}
    for metric in _read_csv(metrics_path):
        if metric["condition"] not in {"aligned", "conflict"}:
            continue
        by_receiver.setdefault(metric["receiver_id"], {})[
            metric["condition"]
        ] = metric

    selected: list[dict[str, Any]] = []
    qualification: list[dict[str, Any]] = []
    for row in rows:
        if row["split"] != "heldout":
            continue
        receiver_id = str(row["receiver_id"])
        pair = by_receiver.get(receiver_id, {})
        if set(pair) != {"aligned", "conflict"}:
            raise ValueError(f"missing natural baselines for {receiver_id}")
        aligned = pair["aligned"]
        conflict = pair["conflict"]
        omega_true = float(row["omega_true"])
        omega_aligned = float(aligned["omega_hat"])
        omega_conflict = float(conflict["omega_hat"])
        gap = abs(omega_aligned - omega_conflict)
        aligned_error = abs(omega_aligned - omega_true)
        conflict_error = abs(omega_conflict - omega_true)
        reasons: list[str] = []
        if not (_as_bool(aligned["valid"]) and _as_bool(conflict["valid"])):
            reasons.append("invalid_natural_rollout")
        if aligned_error > args.max_aligned_error:
            reasons.append("aligned_not_physical")
        if gap < args.min_frequency_gap:
            reasons.append("insufficient_recovery_space")
        if conflict_error - aligned_error < args.min_conflict_excess_error:
            reasons.append("conflict_not_worse_than_aligned")
        if aligned["detected_color"] != str(row["aligned_color"]):
            reasons.append("aligned_color_mismatch")
        if conflict["detected_color"] != str(row["conflict_color"]):
            reasons.append("conflict_color_mismatch")
        qualified = not reasons
        qualification.append(
            {
                "receiver_id": receiver_id,
                "target_label": row["target_label"],
                "phase_index": row["phase_index"],
                "diffusion_repeat": row["diffusion_repeat"],
                "omega_true": omega_true,
                "omega_aligned": omega_aligned,
                "omega_conflict": omega_conflict,
                "frequency_gap": gap,
                "aligned_absolute_error": aligned_error,
                "conflict_absolute_error": conflict_error,
                "qualified": qualified,
                "exclusion_reasons": ";".join(reasons),
            }
        )
        if qualified:
            selected.append(dict(row))

    if not selected:
        raise ValueError("no receiver passed frozen natural-rollout qualification")
    if {str(row["target_label"]) for row in selected} != {"low", "high"}:
        raise ValueError("qualified receiver bank must contain both targets")
    return selected, qualification


def _check_residual(value: Any, condition_tokens: int) -> None:
    import torch

    if not isinstance(value, torch.Tensor) or value.ndim != 3:
        raise RuntimeError("residual must be [batch, token, hidden]")
    if value.shape[0] != 1 or value.shape[1] < condition_tokens:
        raise RuntimeError(
            f"incompatible residual shape {tuple(value.shape)} for "
            f"{condition_tokens} condition tokens"
        )


class _AllLocationCapture:
    def __init__(self, *, condition_tokens: int, expected_steps: int, blocks: int):
        self.condition_tokens = condition_tokens
        self.expected_steps = expected_steps
        self.values: list[list[Any]] = [[] for _ in range(blocks + 1)]

    def _capture(self, location: int, value: Any) -> None:
        _check_residual(value, self.condition_tokens)
        if len(self.values[location]) >= self.expected_steps:
            raise RuntimeError(f"location {location} exceeded expected FM calls")
        self.values[location].append(
            value[0, : self.condition_tokens].detach().to(device="cpu").clone()
        )

    def embedding_pre_hook(self, _module: Any, inputs: tuple[Any, ...]) -> None:
        self._capture(0, inputs[0])

    def block_hook(self, location: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            self._capture(location, output)

        return hook

    def stacked(self) -> list[Any]:
        import torch

        for location, values in enumerate(self.values):
            if len(values) != self.expected_steps:
                raise RuntimeError(
                    f"location {location} captured {len(values)}/"
                    f"{self.expected_steps} FM calls"
                )
        return [torch.stack(values, dim=0) for values in self.values]


def _denoise_capture_all(
    pipe: Any,
    data_config: Any,
    condition: Any,
    *,
    seed: int,
    steps: int,
    condition_tokens: int,
) -> list[Any]:
    import torch

    capture = _AllLocationCapture(
        condition_tokens=condition_tokens,
        expected_steps=steps,
        blocks=len(pipe.dit.blocks),
    )
    handles = [
        pipe.dit.blocks[0].register_forward_pre_hook(capture.embedding_pre_hook)
    ]
    handles.extend(
        block.register_forward_hook(capture.block_hook(index + 1))
        for index, block in enumerate(pipe.dit.blocks)
    )
    try:
        _denoise_loop(pipe, data_config, condition, seed=seed, steps=steps)
    finally:
        for handle in handles:
            handle.remove()
    return capture.stacked()


class _LocationReplacement:
    def __init__(
        self,
        donor: Any,
        *,
        condition_tokens: int,
        expected_steps: int,
    ) -> None:
        if tuple(donor.shape[:2]) != (expected_steps, condition_tokens):
            raise ValueError(
                f"donor shape {tuple(donor.shape)} does not match "
                f"[{expected_steps}, {condition_tokens}, hidden]"
            )
        self.donor = donor
        self.condition_tokens = condition_tokens
        self.expected_steps = expected_steps
        self.calls = 0

    def _replace(self, value: Any) -> Any:
        _check_residual(value, self.condition_tokens)
        if self.calls >= self.expected_steps:
            raise RuntimeError("replacement exceeded expected FM calls")
        if self.donor.shape[2] != value.shape[2]:
            raise RuntimeError("donor/receiver hidden dimensions differ")
        result = value.clone()
        result[:, : self.condition_tokens] = self.donor[self.calls].unsqueeze(0)
        self.calls += 1
        return result

    def pre_hook(self, _module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        return (self._replace(inputs[0]), *inputs[1:])

    def output_hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        return self._replace(output)


def _denoise_replace(
    pipe: Any,
    data_config: Any,
    condition: Any,
    *,
    seed: int,
    steps: int,
    condition_tokens: int,
    location: int,
    donor: Any,
) -> Any:
    replacement = _LocationReplacement(
        donor,
        condition_tokens=condition_tokens,
        expected_steps=steps,
    )
    if location == 0:
        handle = pipe.dit.blocks[0].register_forward_pre_hook(
            replacement.pre_hook
        )
    else:
        handle = pipe.dit.blocks[location - 1].register_forward_hook(
            replacement.output_hook
        )
    try:
        latents = _denoise_loop(
            pipe,
            data_config,
            condition,
            seed=seed,
            steps=steps,
        )
    finally:
        handle.remove()
    if replacement.calls != steps:
        raise RuntimeError(
            f"location {location} replacement ran "
            f"{replacement.calls}/{steps} times"
        )
    return latents


def _denoise_loop(
    pipe: Any,
    data_config: Any,
    condition: Any,
    *,
    seed: int,
    steps: int,
) -> Any:
    import torch

    pipe.scheduler.set_timesteps(steps, denoising_strength=1.0, shift=5.0)
    latent_frames = (data_config.render.num_frames - 1) // 4 + 1
    shape = (
        1,
        pipe.vae.z_dim,
        latent_frames,
        data_config.render.height // pipe.vae.upsampling_factor,
        data_config.render.width // pipe.vae.upsampling_factor,
    )
    latents = pipe.generate_noise(
        shape, seed=seed, rand_device=pipe.device
    ).to(device=pipe.device, dtype=pipe.torch_dtype)
    with torch.inference_mode():
        for step_index, timestep in enumerate(pipe.scheduler.timesteps):
            timestep_input = timestep.unsqueeze(0).to(
                device=pipe.device, dtype=pipe.torch_dtype
            )
            latents[:, :, : condition.shape[2]] = condition
            noise_pred = pipe.model_fn(
                dit=pipe.dit, latents=latents, timestep=timestep_input
            )
            latents = pipe.scheduler.step(
                noise_pred, pipe.scheduler.timesteps[step_index], latents
            )
            latents[:, :, : condition.shape[2]] = condition
    return latents.detach()


def _location_label(location: int) -> str:
    return "embedding output" if location == 0 else f"after block {location - 1}"


def _scan_video_path(root: Path, receiver_id: str, location: int) -> Path:
    return root / "videos" / receiver_id / f"location_{location:02d}.mp4"


def _generate_scan(
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    pipe: Any,
    data_config: Any,
) -> None:
    import torch

    locations = tuple(range(len(pipe.dit.blocks) + 1))
    started = time.monotonic()
    total = len(rows) * len(locations)
    completed = sum(
        _scan_video_path(args.out_root, str(row["receiver_id"]), location).is_file()
        for row in rows
        for location in locations
    )
    for receiver_position, row in enumerate(rows):
        receiver_id = str(row["receiver_id"])
        missing = [
            location
            for location in locations
            if not _scan_video_path(args.out_root, receiver_id, location).is_file()
        ]
        if missing:
            aligned_condition = _condition_latents(
                pipe,
                data_config,
                Path(str(row["aligned_video"])),
                history=args.history,
            )
            print(
                f"capture donor {receiver_position + 1}/{len(rows)} "
                f"{receiver_id}",
                flush=True,
            )
            donor_locations = _denoise_capture_all(
                pipe,
                data_config,
                aligned_condition,
                seed=int(row["generation_seed"]),
                steps=args.steps,
                condition_tokens=args.condition_tokens,
            )
            del aligned_condition
            conflict_condition = _condition_latents(
                pipe,
                data_config,
                Path(str(row["conflict_video"])),
                history=args.history,
            )
            for location in missing:
                donor = donor_locations[location].to(
                    device=pipe.device, dtype=pipe.torch_dtype
                )
                latents = _denoise_replace(
                    pipe,
                    data_config,
                    conflict_condition,
                    seed=int(row["generation_seed"]),
                    steps=args.steps,
                    condition_tokens=args.condition_tokens,
                    location=location,
                    donor=donor,
                )
                destination = _scan_video_path(
                    args.out_root, receiver_id, location
                )
                _write_future_video(pipe, data_config, latents, destination)
                completed += 1
                elapsed = max(time.monotonic() - started, 1e-9)
                _write_json(
                    args.out_root / "progress.json",
                    {
                        "status": "generating",
                        "completed_interventions": completed,
                        "expected_interventions": total,
                        "last_receiver_id": receiver_id,
                        "last_location": location,
                        "last_location_label": _location_label(location),
                        "elapsed_seconds": elapsed,
                    },
                )
                print(
                    f"scan {completed}/{total} {receiver_id} "
                    f"{_location_label(location)} "
                    f"elapsed={elapsed / 60.0:.1f}m",
                    flush=True,
                )
                del donor, latents
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del conflict_condition, donor_locations

        noop_path = args.out_root / "videos" / receiver_id / "self_replay.mp4"
        if not noop_path.is_file():
            conflict_condition = _condition_latents(
                pipe,
                data_config,
                Path(str(row["conflict_video"])),
                history=args.history,
            )
            _unused, conflict_activation = _denoise(
                pipe,
                data_config,
                conflict_condition,
                seed=int(row["generation_seed"]),
                steps=args.steps,
                block_index=args.noop_block_index,
                condition_tokens=args.condition_tokens,
                capture=True,
            )
            if conflict_activation is None:
                raise AssertionError("self-replay capture returned no activation")
            donor = torch.from_numpy(conflict_activation).to(
                device=pipe.device, dtype=pipe.torch_dtype
            )
            latents = _denoise_replace(
                pipe,
                data_config,
                conflict_condition,
                seed=int(row["generation_seed"]),
                steps=args.steps,
                condition_tokens=args.condition_tokens,
                location=args.noop_block_index + 1,
                donor=donor,
            )
            _write_future_video(pipe, data_config, latents, noop_path)
            del conflict_condition, conflict_activation, donor, latents, _unused
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _natural_metrics(
    source_metrics: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, Mapping[str, str]]]:
    result: dict[str, dict[str, Mapping[str, str]]] = {}
    for row in source_metrics:
        if row["condition"] in {"aligned", "conflict"}:
            result.setdefault(row["receiver_id"], {})[row["condition"]] = row
    return result


def _measure_scan(
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    blocks: int,
    data_config: Any,
) -> list[dict[str, Any]]:
    natural = _natural_metrics(_read_csv(args.source_root / "metrics.csv"))
    metrics: list[dict[str, Any]] = []
    for row in rows:
        receiver_id = str(row["receiver_id"])
        baseline = natural[receiver_id]
        omega_aligned = float(baseline["aligned"]["omega_hat"])
        omega_conflict = float(baseline["conflict"]["omega_hat"])
        denominator = omega_aligned - omega_conflict
        if abs(denominator) <= 1e-8:
            raise ValueError(f"zero baseline recovery space for {receiver_id}")
        expected_colors = (
            str(row["aligned_color"]),
            str(row["conflict_color"]),
        )
        for location in range(blocks + 1):
            path = _scan_video_path(args.out_root, receiver_id, location)
            measured = _measure_one(
                path, row, data_config, expected_colors=expected_colors
            )
            omega_hat = float(measured["omega_hat"])
            metrics.append(
                {
                    "receiver_id": receiver_id,
                    "target_label": row["target_label"],
                    "phase_index": row["phase_index"],
                    "diffusion_repeat": row["diffusion_repeat"],
                    "location": location,
                    "location_label": _location_label(location),
                    "block_index_zero_based": (
                        "" if location == 0 else location - 1
                    ),
                    "omega_true": row["omega_true"],
                    "omega_aligned": omega_aligned,
                    "omega_conflict": omega_conflict,
                    "omega_hat": omega_hat,
                    "frequency_recovery": (
                        (omega_hat - omega_conflict) / denominator
                        if math.isfinite(omega_hat)
                        else float("nan")
                    ),
                    "aligned_color_recovered": (
                        measured["detected_color"] == row["aligned_color"]
                    ),
                    "aligned_shape_recovered": (
                        measured["detected_shape"] == row["aligned_shape"]
                    ),
                    "video": str(path),
                    **measured,
                }
            )

        noop_path = args.out_root / "videos" / receiver_id / "self_replay.mp4"
        measured = _measure_one(
            noop_path, row, data_config, expected_colors=expected_colors
        )
        metrics.append(
            {
                "receiver_id": receiver_id,
                "target_label": row["target_label"],
                "phase_index": row["phase_index"],
                "diffusion_repeat": row["diffusion_repeat"],
                "location": "self_replay",
                "location_label": f"self replay after block {args.noop_block_index}",
                "block_index_zero_based": args.noop_block_index,
                "omega_true": row["omega_true"],
                "omega_aligned": omega_aligned,
                "omega_conflict": omega_conflict,
                "omega_hat": measured["omega_hat"],
                "frequency_recovery": (
                    (float(measured["omega_hat"]) - omega_conflict) / denominator
                ),
                "aligned_color_recovered": (
                    measured["detected_color"] == row["aligned_color"]
                ),
                "aligned_shape_recovered": (
                    measured["detected_shape"] == row["aligned_shape"]
                ),
                "video": str(noop_path),
                **measured,
            }
        )
    _write_csv(args.out_root / "metrics.csv", metrics)
    return metrics


def _finite(values: Iterable[float]) -> np.ndarray:
    result = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    return result


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    blocks: int,
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for target in ("low", "high", "pooled"):
        for location in range(blocks + 1):
            selected = [
                row
                for row in rows
                if row["location"] == location
                and (target == "pooled" or row["target_label"] == target)
            ]
            valid_selected = [row for row in selected if bool(row["valid"])]
            recovery = _finite(
                row["frequency_recovery"] for row in valid_selected
            )
            omega = _finite(row["omega_hat"] for row in valid_selected)
            valid = [bool(row["valid"]) for row in selected]
            recovery_statistics = (
                (
                    float(np.median(recovery)),
                    float(np.quantile(recovery, 0.25)),
                    float(np.quantile(recovery, 0.75)),
                )
                if recovery.size
                else (float("nan"), float("nan"), float("nan"))
            )
            summary.append(
                {
                    "target_label": target,
                    "location": location,
                    "location_label": _location_label(location),
                    "samples": len(selected),
                    "valid_recovery_samples": int(recovery.size),
                    "valid_fraction": float(np.mean(valid)),
                    "median_frequency_recovery": recovery_statistics[0],
                    "q25_frequency_recovery": recovery_statistics[1],
                    "q75_frequency_recovery": recovery_statistics[2],
                    "median_omega_hat": (
                        float(np.median(omega)) if omega.size else float("nan")
                    ),
                    "aligned_color_fraction": float(
                        np.mean(
                            [bool(row["aligned_color_recovered"]) for row in selected]
                        )
                    ),
                    "aligned_shape_fraction": float(
                        np.mean(
                            [bool(row["aligned_shape_recovered"]) for row in selected]
                        )
                    ),
                }
            )
    return summary


def _infer_transition(
    summary: Sequence[Mapping[str, Any]],
    target: str,
) -> dict[str, Any]:
    pooled = [row for row in summary if row["target_label"] == target]
    values = np.asarray(
        [float(row["median_frequency_recovery"]) for row in pooled]
    )
    changes = np.diff(values)
    finite_changes = np.flatnonzero(np.isfinite(changes))
    steepest = (
        int(finite_changes[np.argmin(changes[finite_changes])]) + 1
        if finite_changes.size
        else None
    )
    effective = [
        int(row["location"])
        for row in pooled
        if float(row["median_frequency_recovery"]) >= 0.5
    ]
    return {
        "target_label": target,
        "steepest_drop_into_location": steepest,
        "steepest_drop_location_label": (
            _location_label(steepest) if steepest is not None else None
        ),
        "steepest_drop": (
            float(changes[steepest - 1]) if steepest is not None else None
        ),
        "last_location_with_median_recovery_at_least_0p5": (
            max(effective) if effective else None
        ),
        "last_effective_location_label": (
            _location_label(max(effective)) if effective else None
        ),
    }


def _plot(
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    summary: Sequence[Mapping[str, Any]],
    blocks: int,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = args.out_root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    x = np.arange(blocks + 1)
    colors = {"low": "#d95f02", "high": "#1b6ca8"}
    labels = {"low": "low-frequency receivers", "high": "high-frequency receivers"}
    transitions = {
        target: _infer_transition(summary, target)
        for target in ("low", "high", "pooled")
    }
    transition = transitions["pooled"]

    figure, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    for target in ("low", "high"):
        aggregated = [row for row in summary if row["target_label"] == target]
        median = np.asarray(
            [float(row["median_frequency_recovery"]) for row in aggregated]
        )
        q25 = np.asarray(
            [float(row["q25_frequency_recovery"]) for row in aggregated]
        )
        q75 = np.asarray(
            [float(row["q75_frequency_recovery"]) for row in aggregated]
        )
        axes[0].plot(
            x, median, marker="o", markersize=3, color=colors[target],
            label=labels[target],
        )
        axes[0].fill_between(x, q25, q75, color=colors[target], alpha=0.15)
        axes[1].plot(
            x,
            [float(row["aligned_color_fraction"]) for row in aggregated],
            marker="o",
            markersize=3,
            color=colors[target],
            label=(
                f"{target}-frequency: color"
                if args.model_name == "frequency_color_shape"
                else labels[target]
            ),
        )
        if args.model_name == "frequency_color_shape":
            axes[1].plot(
                x,
                [float(row["aligned_shape_fraction"]) for row in aggregated],
                marker="s",
                markersize=3,
                linestyle="--",
                color=colors[target],
                label=f"{target}-frequency: shape",
            )
        axes[2].plot(
            x,
            [float(row["valid_fraction"]) for row in aggregated],
            marker="o",
            markersize=3,
            color=colors[target],
            label=labels[target],
        )
    axes[0].axhline(0.0, color="0.35", linestyle="--", linewidth=1, label="natural conflict")
    axes[0].axhline(1.0, color="0.35", linestyle=":", linewidth=1, label="natural aligned")
    axes[0].set_ylabel("frequency recovery R")
    axes[0].set_title(
        f"{args.model_name}: aligned donor residual → conflict receiver"
    )
    axes[0].legend(ncol=2, frameon=False)
    axes[1].set_ylabel(
        "aligned appearance fraction"
        if args.model_name == "frequency_color_shape"
        else "aligned-color fraction"
    )
    axes[1].set_ylim(-0.03, 1.03)
    if args.model_name == "frequency_color_shape":
        axes[1].legend(ncol=2, frameon=False)
    axes[2].set_ylabel("valid rollout fraction")
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].set_xlabel("single patched residual location")
    for axis in axes:
        axis.grid(alpha=0.2)
        if transition["steepest_drop_into_location"] is not None:
            axis.axvline(
                int(transition["steepest_drop_into_location"]),
                color="0.45",
                linestyle="--",
                linewidth=1,
            )
    tick_positions = [0, *range(3, blocks + 1, 3)]
    if blocks not in tick_positions:
        tick_positions.append(blocks)
    axes[-1].set_xticks(
        tick_positions,
        ["embedding", *[f"B{position - 1}" for position in tick_positions[1:]]],
    )
    figure.tight_layout()
    figure.savefig(plots / "layer_scan_summary.png", dpi=200)
    figure.savefig(plots / "layer_scan_summary.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(13, 5.5))
    for target in ("low", "high"):
        selected = [
            row
            for row in rows
            if row["target_label"] == target
            and isinstance(row["location"], int)
            and bool(row["valid"])
        ]
        grouped = [
            [row for row in selected if row["location"] == location]
            for location in range(blocks + 1)
        ]
        for location, group in enumerate(grouped):
            axis.scatter(
                np.full(len(group), location)
                + np.linspace(-0.10, 0.10, len(group)),
                [float(row["frequency_recovery"]) for row in group],
                color=colors[target], alpha=0.25, s=15,
            )
        aggregated = [row for row in summary if row["target_label"] == target]
        axis.plot(
            x,
            [float(row["median_frequency_recovery"]) for row in aggregated],
            color=colors[target], marker="o", markersize=3, label=labels[target],
        )
    axis.axhline(0.0, color="0.35", linestyle="--", linewidth=1)
    axis.axhline(1.0, color="0.35", linestyle=":", linewidth=1)
    axis.set(
        xlabel="single patched residual location",
        ylabel="frequency recovery R",
        title=f"{args.model_name}: full condition-residual layer scan",
    )
    axis.set_xticks(
        tick_positions,
        ["embedding", *[f"B{position - 1}" for position in tick_positions[1:]]],
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(plots / "frequency_recovery_by_layer.png", dpi=200)
    figure.savefig(plots / "frequency_recovery_by_layer.pdf")
    plt.close(figure)
    return transitions


def run(args: argparse.Namespace) -> None:
    required = (
        args.experiment_config,
        args.training_config,
        args.checkpoint,
        args.source_root / "receiver_manifest.jsonl",
        args.source_root / "metrics.csv",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.out_root.mkdir(parents=True, exist_ok=True)
    qualified, qualification = _qualified_receivers(args)
    rows = _freeze_jsonl(
        args.out_root / "qualified_receiver_manifest.jsonl", qualified
    )
    _write_csv(args.out_root / "qualification.csv", qualification)

    state = {
        "protocol": PROGRAM_VERSION,
        "model_name": args.model_name,
        "history": args.history,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256_file(args.checkpoint),
        "source_root": str(args.source_root),
        "source_receiver_manifest_sha256": manifest_sha256(
            read_jsonl(args.source_root / "receiver_manifest.jsonl")
        ),
        "qualified_receiver_manifest_sha256": manifest_sha256(rows),
        "qualified_receivers": len(rows),
        "qualified_low_receivers": sum(
            row["target_label"] == "low" for row in rows
        ),
        "qualified_high_receivers": sum(
            row["target_label"] == "high" for row in rows
        ),
        "steps": args.steps,
        "condition_tokens": args.condition_tokens,
        "hidden_size": args.hidden_size,
        "max_aligned_error": args.max_aligned_error,
        "min_frequency_gap": args.min_frequency_gap,
        "min_conflict_excess_error": args.min_conflict_excess_error,
        "noop_block_index": args.noop_block_index,
        "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    state_path = args.out_root / "run_state.json"
    if state_path.is_file():
        previous = _read_json(state_path)
        comparable = dict(previous)
        comparable.pop("status", None)
        comparable.pop("blocks", None)
        comparable.pop("locations", None)
        comparable.pop("transition", None)
        comparable.pop("transition_by_target", None)
        comparable.pop("self_replay_max_absolute_omega_error", None)
        comparable.pop("self_replay_median_absolute_omega_error", None)
        if comparable != state:
            raise ValueError("existing layer-scan state differs from request")
    else:
        _write_json(state_path, {**state, "status": "running"})

    pipe, data_config = _load_runtime(args, rows)
    blocks = len(pipe.dit.blocks)
    if not 0 <= args.noop_block_index < blocks:
        raise ValueError("--noop-block-index is outside the model")
    if not args.measure_only:
        _generate_scan(args, rows, pipe, data_config)
    metrics = _measure_scan(args, rows, blocks, data_config)
    summary = _aggregate(metrics, blocks)
    _write_csv(args.out_root / "summary_by_layer.csv", summary)
    transitions = _plot(args, metrics, summary, blocks)
    noop = [row for row in metrics if row["location"] == "self_replay"]
    noop_abs_error = _finite(
        abs(float(row["omega_hat"]) - float(row["omega_conflict"]))
        for row in noop
    )
    result = {
        **state,
        "blocks": blocks,
        "locations": blocks + 1,
        "transition": transitions["pooled"],
        "transition_by_target": transitions,
        "self_replay_max_absolute_omega_error": float(noop_abs_error.max()),
        "self_replay_median_absolute_omega_error": float(
            np.median(noop_abs_error)
        ),
    }
    _write_json(args.out_root / "summary.json", result)
    _write_json(state_path, {**result, "status": "complete"})
    _write_json(
        args.out_root / "progress.json",
        {
            "status": "complete",
            "completed_interventions": len(rows) * (blocks + 1),
            "expected_interventions": len(rows) * (blocks + 1),
        },
    )
    print(f"complete {args.out_root}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        choices=("frequency_color_circle", "frequency_color_shape"),
        required=True,
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--history", choices=("short", "long"), default="long")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--condition-tokens", type=int, default=1088)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--block-index", type=int, default=13)
    parser.add_argument("--noop-block-index", type=int, default=13)
    parser.add_argument("--max-aligned-error", type=float, default=0.75)
    parser.add_argument("--min-frequency-gap", type=float, default=1.0)
    parser.add_argument(
        "--min-conflict-excess-error", type=float, default=0.5
    )
    parser.add_argument("--measure-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.steps < 1 or args.condition_tokens < 1 or args.hidden_size < 1:
        raise ValueError("steps, condition tokens, and hidden size must be positive")
    for name in (
        "max_aligned_error",
        "min_frequency_gap",
        "min_conflict_excess_error",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative")
    run(args)


if __name__ == "__main__":
    main()
