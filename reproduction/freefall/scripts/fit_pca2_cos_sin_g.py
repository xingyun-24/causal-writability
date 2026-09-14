#!/usr/bin/env python3
"""Fit the first two uncentered residual-PCA scores from cos(alpha*g), sin(alpha*g)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def quality(actual: np.ndarray, predicted: np.ndarray, reference_mean: np.ndarray) -> dict[str, object]:
    residual = actual - predicted
    component_mse = np.mean(residual ** 2, axis=0)
    baseline_mse = np.mean((actual - reference_mean[None, :]) ** 2, axis=0)
    return {
        "joint_rmse": float(np.sqrt(np.mean(residual ** 2))),
        "component_rmse": [float(value) for value in np.sqrt(component_mse)],
        "component_R2": [float(1.0 - error / base) if base > 0 else None for error, base in zip(component_mse, baseline_mse)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca-npz", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alpha-min", type=float, default=0.1)
    parser.add_argument("--alpha-max", type=float, default=3000.0)
    parser.add_argument("--alpha-count", type=int, default=181)
    args = parser.parse_args()
    packed = np.load(args.pca_npz)
    pair_ids = [str(value) for value in packed["pair_ids"]]
    eigenvalues = packed["eigenvalues"].astype(np.float64)
    scores = packed["vectors"][:, :2].astype(np.float64) * np.sqrt(eigenvalues[:2])[None, :]
    pairs = {item["pair_id"]: item for item in json.loads(args.pair_manifest.read_text(encoding="utf-8"))["pairs"]}
    gravity = np.asarray([float(pairs[pair_id]["gravity_true"]) for pair_id in pair_ids], dtype=np.float64)
    intervals = np.asarray([pairs[pair_id]["gravity_interval"] for pair_id in pair_ids])
    holdout = np.zeros(len(pair_ids), dtype=bool)
    for interval in ("low", "high"):
        holdout[np.flatnonzero(intervals == interval)[::2]] = True
    train = ~holdout
    alphas = np.geomspace(args.alpha_min, args.alpha_max, args.alpha_count)
    sweep = []
    for alpha in alphas:
        design = np.column_stack((np.cos(alpha * gravity), np.sin(alpha * gravity)))
        coefficients, *_ = np.linalg.lstsq(design[train], scores[train], rcond=None)
        fitted = design @ coefficients
        sweep.append({
            "alpha": float(alpha),
            "train_mse": float(np.mean((scores[train] - fitted[train]) ** 2)),
            "holdout_mse": float(np.mean((scores[holdout] - fitted[holdout]) ** 2)),
        })
    selected_index = int(np.argmin([row["train_mse"] for row in sweep]))
    alpha = float(alphas[selected_index])
    design = np.column_stack((np.cos(alpha * gravity), np.sin(alpha * gravity)))
    coefficients, *_ = np.linalg.lstsq(design[train], scores[train], rcond=None)
    predicted = design @ coefficients
    full_coefficients, *_ = np.linalg.lstsq(design, scores, rcond=None)
    payload = {
        "target": "first two PCA score coordinates of aligned-minus-conflict residuals",
        "basis": "cos(alpha * actual generated g), sin(alpha * actual generated g); no separate intercept",
        "alpha_selection": "minimize train MSE over a log-spaced alpha grid; heldout pairs are not used for selection",
        "alpha": alpha, "alpha_grid": {"min": args.alpha_min, "max": args.alpha_max, "count": args.alpha_count}, "alpha_sweep": sweep,
        "components": 2, "pairs": len(pair_ids), "train_pairs": int(train.sum()), "holdout_pairs": int(holdout.sum()),
        "split": "every other pair after pair-id ordering, separately within low/high intervals",
        "pca_energy_fraction": float(eigenvalues[:2].sum() / eigenvalues.sum()),
        "coefficients_rows_cos_sin": coefficients.tolist(), "full_coefficients_rows_cos_sin": full_coefficients.tolist(),
        "train_quality": quality(scores[train], predicted[train], scores[train].mean(axis=0)),
        "holdout_quality": quality(scores[holdout], predicted[holdout], scores[train].mean(axis=0)),
        "pair_ids": pair_ids, "gravity_true": gravity.tolist(), "actual_scores": scores.tolist(), "fitted_scores": predicted.tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(6.5, 5.4), dpi=190)
    scatter = ax.scatter(scores[:, 0], scores[:, 1], c=gravity, cmap="viridis", s=30, alpha=.9, label="actual PCA score")
    ax.scatter(predicted[:, 0], predicted[:, 1], c=gravity, cmap="viridis", marker="x", s=32, linewidths=.9, label="cos(g), sin(g) fit")
    for actual, fitted in zip(scores, predicted):
        ax.plot((actual[0], fitted[0]), (actual[1], fitted[1]), color="#505050", alpha=.15, linewidth=.5)
    fig.colorbar(scatter, ax=ax).set_label("Actual generated g")
    ax.set(xlabel="PC1 score", ylabel="PC2 score", title=f"PCA score fit: cos({alpha:.3g}g), sin({alpha:.3g}g)")
    ax.legend(frameon=False); ax.grid(alpha=.22); fig.tight_layout(); fig.savefig(args.out.with_suffix(".png"), bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=190)
    ax.plot(alphas, [row["train_mse"] for row in sweep], label="train MSE")
    ax.plot(alphas, [row["holdout_mse"] for row in sweep], label="holdout MSE")
    ax.axvline(alpha, color="#c44e52", linestyle="--", label=f"selected alpha={alpha:.3g}")
    ax.set(xscale="log", xlabel="alpha", ylabel="2D PCA-score MSE", title="Frequency sweep for cos(alpha*g), sin(alpha*g)")
    ax.grid(alpha=.22); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(args.out.with_name(args.out.stem + "_alpha_sweep.png"), bbox_inches="tight"); plt.close(fig)
    print(json.dumps({"alpha": alpha, "train_quality": payload["train_quality"], "holdout_quality": payload["holdout_quality"]}, indent=2))


if __name__ == "__main__":
    main()
