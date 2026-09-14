#!/usr/bin/env python3
"""Fit the Pendulum top-4, direction-specific harmonic edit-coordinate law.

The uncentered PCA basis is frozen from fit matched differences by Stage 3.
This program never refits PCA and never reads a held-out coordinate while
fitting.  For each target direction it regresses the four causal edit
coordinates on the first boundary-phase harmonic using the 32 fit receivers,
freezes the resulting 2-D coefficient plane, and projects the disjoint 32
held-out receivers for reporting and intervention synthesis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sshv2.experiments.pendulum.data import config_from_mapping
from sshv2.experiments.pendulum.stage4_coordinate_difference import (
    angular_velocity_star,
    coordinates,
)


RANK = 4


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def r2_score(target: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(np.sum((target - predicted) ** 2))
    centered = target - target.mean(axis=0, keepdims=True)
    total = float(np.sum(centered**2))
    return 1.0 - residual / total if total > 0 else float("nan")


def orient_columns(plane: np.ndarray) -> np.ndarray:
    result = np.array(plane, dtype=np.float64, copy=True)
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0:
            result[:, column] *= -1
    if np.linalg.det(result[:2, :]) < 0:
        result[:, 1] *= -1
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-bank", type=Path, required=True)
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = read_csv(args.strict_bank)
    if len(rows) != 128:
        raise ValueError("expected 128 strict-bank receivers")
    experiment = yaml.safe_load(args.experiment_config.read_text(encoding="utf-8"))
    config = config_from_mapping(experiment["data"])
    directions = np.load(args.directions, mmap_mode="r")
    if directions.shape[0] != len(rows):
        raise ValueError("directions/manifest length mismatch")
    components = np.load(args.components, mmap_mode="r").reshape(8, -1)[:RANK]
    z_true = coordinates(directions, components, range(len(rows)))

    boundary_phase = []
    angular_velocity = []
    for row in rows:
        velocity = angular_velocity_star(row, config.prediction_start, config.render.fps)
        theta_norm = float(row["theta_star"]) / float(row["amplitude_true"])
        velocity_norm = velocity / (float(row["amplitude_true"]) * float(row["omega_true"]))
        boundary_phase.append(math.atan2(-velocity_norm, theta_norm) % (2 * math.pi))
        angular_velocity.append(velocity)
    phase = np.asarray(boundary_phase, dtype=np.float64)
    design = np.stack([np.ones(len(rows)), np.cos(phase), np.sin(phase)], axis=1)

    z_hat = np.full_like(z_true, np.nan)
    plane_coordinates = np.full((len(rows), 2), np.nan, dtype=np.float64)
    plane_predictions = np.full((len(rows), 2), np.nan, dtype=np.float64)
    summaries: dict[str, Any] = {}
    planes: dict[str, np.ndarray] = {}
    for direction, target_label in (("A", "high"), ("B", "low")):
        fit_indices = [i for i, row in enumerate(rows) if row["direction"] == direction and row["split"] == "fit"]
        held_indices = [i for i, row in enumerate(rows) if row["direction"] == direction and row["split"] == "heldout"]
        if len(fit_indices) != 32 or len(held_indices) != 32:
            raise ValueError(f"{direction}: expected 32 fit + 32 held-out")
        weight, *_ = np.linalg.lstsq(design[fit_indices], z_true[fit_indices], rcond=None)
        z_hat[fit_indices + held_indices] = design[fit_indices + held_indices] @ weight

        # The first-harmonic coefficient span is a fit-only 2-D plane in the
        # frozen top-4 causal edit-coordinate system.
        u, _, _ = np.linalg.svd(weight[1:3, :].T, full_matrices=False)
        plane = orient_columns(u[:, :2])
        planes[direction] = plane
        group = fit_indices + held_indices
        plane_coordinates[group] = z_true[group] @ plane
        plane_predictions[group] = z_hat[group] @ plane
        summaries[target_label] = {
            "direction": direction,
            "fit_n": len(fit_indices),
            "heldout_n": len(held_indices),
            "heldout_top4_r2": r2_score(z_true[held_indices], z_hat[held_indices]),
            "heldout_phase_plane_r2": r2_score(
                plane_coordinates[held_indices], plane_predictions[held_indices]
            ),
            "harmonic_weight": weight.tolist(),
            "phase_plane": plane.tolist(),
        }

    release_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        receiver_id = row["receiver_id"]
        release_rows.append(
            {
                "receiver_id": receiver_id,
                "pair_id": row.get("pair_id", ""),
                "split": row["split"],
                "direction": row["direction"],
                "target_label": row["target_label"],
                "boundary_phase": phase[index],
                "theta_star": float(row["theta_star"]),
                "angular_velocity_star": angular_velocity[index],
                **{f"z_true_{i + 1}": z_true[index, i] for i in range(RANK)},
                **{f"z_hat_{i + 1}": z_hat[index, i] for i in range(RANK)},
                "phase_coordinate_1": plane_coordinates[index, 0],
                "phase_coordinate_2": plane_coordinates[index, 1],
                "phase_prediction_1": plane_predictions[index, 0],
                "phase_prediction_2": plane_predictions[index, 1],
            }
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "coordinate_predictions.csv", release_rows)
    (args.out_dir / "coordinate_summary.json").write_text(
        json.dumps(
            {
                "rank": RANK,
                "pca_fit": "raw uncentered fit matched differences only",
                "state_law": "direction-specific first harmonic of boundary phase",
                "fit_heldout_disjoint": True,
                "directions": summaries,
                "predicted_edit_reads_heldout_donor_activation": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, sort_keys=True))


if __name__ == "__main__":
    main()
