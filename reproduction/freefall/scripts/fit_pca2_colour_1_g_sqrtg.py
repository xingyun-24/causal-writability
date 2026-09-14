#!/usr/bin/env python3
"""Fit the first two residual-PCA scores with color * [1, g, sqrt(g)]."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def color_value(value: str) -> float:
    return {"red": 1.0, "blue": -1.0}[value]


def quality(actual: np.ndarray, predicted: np.ndarray, mean: np.ndarray) -> dict[str, object]:
    residual = actual - predicted
    mse = np.mean(residual ** 2, axis=0)
    baseline = np.mean((actual - mean[None, :]) ** 2, axis=0)
    return {
        "joint_rmse": float(np.sqrt(np.mean(residual ** 2))),
        "component_rmse": [float(v) for v in np.sqrt(mse)],
        "component_R2": [float(1 - e / b) if b > 0 else None for e, b in zip(mse, baseline)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca-npz", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    packed = np.load(args.pca_npz)
    pair_ids = [str(v) for v in packed["pair_ids"]]
    eigenvalues = packed["eigenvalues"].astype(np.float64)
    scores = packed["vectors"][:, :2].astype(np.float64) * np.sqrt(eigenvalues[:2])[None, :]
    pairs = {item["pair_id"]: item for item in json.loads(args.pair_manifest.read_text(encoding="utf-8"))["pairs"]}
    rows = {(row["pair_id"], row["variant"]): row for row in csv.DictReader((args.dataset_dir / "metadata.csv").open(newline=""))}
    gravity = np.asarray([float(pairs[pair_id]["gravity_true"]) for pair_id in pair_ids])
    intervals = np.asarray([pairs[pair_id]["gravity_interval"] for pair_id in pair_ids])
    delta_color = np.asarray([
        color_value(rows[(pair_id, "aligned")]["color_label"]) - color_value(rows[(pair_id, "conflict")]["color_label"])
        for pair_id in pair_ids
    ])
    design = np.column_stack((delta_color, delta_color * gravity, delta_color * np.sqrt(gravity)))
    holdout = np.zeros(len(pair_ids), dtype=bool)
    for interval in ("low", "high"):
        holdout[np.flatnonzero(intervals == interval)[::2]] = True
    train = ~holdout
    coefficients, *_ = np.linalg.lstsq(design[train], scores[train], rcond=None)
    predicted = design @ coefficients
    full_coefficients, *_ = np.linalg.lstsq(design, scores, rcond=None)
    payload = {
        "target": "first two PCA score coordinates of aligned-minus-conflict residuals",
        "basis": "f(color,g)=color*[1,g,sqrt(g)]*B; fit f(aligned)-f(conflict); no intercept",
        "color_encoding": "red=+1, blue=-1", "feature_order": ["delta_color", "delta_color*g", "delta_color*sqrt(g)"],
        "pairs": len(pair_ids), "train_pairs": int(train.sum()), "holdout_pairs": int(holdout.sum()),
        "split": "every other pair after pair-id ordering, separately within low/high intervals",
        "pca_energy_fraction": float(eigenvalues[:2].sum() / eigenvalues.sum()),
        "coefficients_rows_feature_order": coefficients.tolist(), "full_coefficients_rows_feature_order": full_coefficients.tolist(),
        "train_quality": quality(scores[train], predicted[train], scores[train].mean(axis=0)),
        "holdout_quality": quality(scores[holdout], predicted[holdout], scores[train].mean(axis=0)),
        "pair_ids": pair_ids, "gravity_true": gravity.tolist(), "delta_color": delta_color.tolist(),
        "actual_scores": scores.tolist(), "fitted_scores": predicted.tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train": payload["train_quality"], "holdout": payload["holdout_quality"]}, indent=2))


if __name__ == "__main__":
    main()
