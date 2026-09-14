#!/usr/bin/env python3
"""Fit residual-delta PCA scores with f(colour, g) = colour * [1, g] B."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def colour_value(value: str) -> float:
    if value == "red":
        return 1.0
    if value == "blue":
        return -1.0
    raise ValueError(f"Expected red/blue color_label, found {value!r}")


def quality(actual: np.ndarray, predicted: np.ndarray, reference_mean: np.ndarray) -> dict[str, object]:
    residual = actual - predicted
    mse = np.mean(residual ** 2, axis=0)
    baseline = np.mean((actual - reference_mean[None, :]) ** 2, axis=0)
    return {
        "joint_rmse": float(np.sqrt(np.mean(residual ** 2))),
        "component_rmse": [float(value) for value in np.sqrt(mse)],
        "component_R2": [float(1.0 - error / base) if base > 0 else None for error, base in zip(mse, baseline)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca-npz", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    packed = np.load(args.pca_npz)
    pair_ids = [str(value) for value in packed["pair_ids"]]
    eigenvalues = packed["eigenvalues"].astype(np.float64)
    scores = packed["vectors"][:, :2].astype(np.float64) * np.sqrt(eigenvalues[:2])[None, :]
    pairs = {item["pair_id"]: item for item in json.loads(args.pair_manifest.read_text(encoding="utf-8"))["pairs"]}
    rows = {(row["pair_id"], row["variant"]): row for row in csv.DictReader((args.dataset_dir / "metadata.csv").open(newline=""))}
    gravity = np.asarray([float(pairs[pair_id]["gravity_true"]) for pair_id in pair_ids], dtype=np.float64)
    intervals = np.asarray([pairs[pair_id]["gravity_interval"] for pair_id in pair_ids])
    delta_colour = np.asarray([
        colour_value(rows[(pair_id, "aligned")]["color_label"]) - colour_value(rows[(pair_id, "conflict")]["color_label"])
        for pair_id in pair_ids
    ], dtype=np.float64)
    design = np.column_stack((delta_colour, delta_colour * gravity))
    holdout = np.zeros(len(pair_ids), dtype=bool)
    for interval in ("low", "high"):
        holdout[np.flatnonzero(intervals == interval)[::2]] = True
    train = ~holdout
    coefficients, *_ = np.linalg.lstsq(design[train], scores[train], rcond=None)
    predicted = design @ coefficients
    full_coefficients, *_ = np.linalg.lstsq(design, scores, rcond=None)
    payload = {
        "target": "first two PCA score coordinates of aligned-minus-conflict residuals",
        "basis": "f(color, g)=color*[1, g]*B; fit f(aligned)-f(conflict); no intercept",
        "color_encoding": "red=+1, blue=-1", "feature_order": ["aligned_color-conflict_color", "(aligned_color-conflict_color)*actual_generated_g"],
        "components": 2, "pairs": len(pair_ids), "train_pairs": int(train.sum()), "holdout_pairs": int(holdout.sum()),
        "split": "every other pair after pair-id ordering, separately within low/high intervals",
        "pca_energy_fraction": float(eigenvalues[:2].sum() / eigenvalues.sum()),
        "coefficients_rows_feature_order": coefficients.tolist(), "full_coefficients_rows_feature_order": full_coefficients.tolist(),
        "train_quality": quality(scores[train], predicted[train], scores[train].mean(axis=0)),
        "holdout_quality": quality(scores[holdout], predicted[holdout], scores[train].mean(axis=0)),
        "pair_ids": pair_ids, "gravity_true": gravity.tolist(), "delta_colour": delta_colour.tolist(),
        "actual_scores": scores.tolist(), "fitted_scores": predicted.tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(6.5, 5.4), dpi=190)
    scatter = ax.scatter(scores[:, 0], scores[:, 1], c=gravity, cmap="viridis", s=30, alpha=.9, label="actual PCA score")
    ax.scatter(predicted[:, 0], predicted[:, 1], c=gravity, cmap="viridis", marker="x", s=32, linewidths=.9, label="color * (1, g) fit")
    for actual, fitted in zip(scores, predicted):
        ax.plot((actual[0], fitted[0]), (actual[1], fitted[1]), color="#505050", alpha=.15, linewidth=.5)
    fig.colorbar(scatter, ax=ax).set_label("Actual generated g")
    ax.set(xlabel="PC1 score", ylabel="PC2 score", title="PCA score fit: color * (1, g)")
    ax.legend(frameon=False); ax.grid(alpha=.22); fig.tight_layout(); fig.savefig(args.out.with_suffix(".png"), bbox_inches="tight"); plt.close(fig)
    print(json.dumps({"train_quality": payload["train_quality"], "holdout_quality": payload["holdout_quality"]}, indent=2))


if __name__ == "__main__":
    main()
