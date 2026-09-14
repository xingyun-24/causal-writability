#!/usr/bin/env python3
"""Fit rank-2 PCA score differences from observed colour and observed E3 gravity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def colour_value(colour: str) -> float:
    if colour == "red":
        return 1.0
    if colour == "blue":
        return -1.0
    raise ValueError(f"Unsupported colour {colour!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca-npz", type=Path, required=True)
    parser.add_argument("--eval-rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    packed = np.load(args.pca_npz)
    pair_ids = [str(value) for value in packed["pair_ids"]]
    eigenvalues = packed["eigenvalues"].astype(np.float64)
    scores = packed["vectors"][:, :2].astype(np.float64) * np.sqrt(eigenvalues[:2])[None, :]
    rows = json.loads(args.eval_rows.read_text(encoding="utf-8"))
    by_key = {(str(row["pair_id"]), str(row["condition"])): row for row in rows}
    aligned = [by_key[(pair_id, "aligned")] for pair_id in pair_ids]
    conflict = [by_key[(pair_id, "conflict")] for pair_id in pair_ids]
    delta_colour = np.asarray([colour_value(row["detected_colour"]) - colour_value(other["detected_colour"])
                               for row, other in zip(aligned, conflict)], dtype=np.float64)
    delta_g = np.asarray([float(row["g_E3"]) - float(other["g_E3"])
                          for row, other in zip(aligned, conflict)], dtype=np.float64)
    features = np.column_stack((delta_colour, delta_g))
    intervals = np.asarray([row["gravity_interval"] for row in conflict])
    holdout = np.zeros(len(pair_ids), dtype=bool)
    for interval in ("low", "high"):
        holdout[np.flatnonzero(intervals == interval)[::2]] = True
    train = ~holdout
    coefficients, *_ = np.linalg.lstsq(features[train], scores[train], rcond=None)
    fitted = features @ coefficients
    full_coefficients, *_ = np.linalg.lstsq(features, scores, rcond=None)
    def quality(mask: np.ndarray, coefficients_: np.ndarray) -> dict[str, float]:
        residual = scores[mask] - features[mask] @ coefficients_
        baseline = np.mean((scores[mask] - scores[train].mean(axis=0, keepdims=True)) ** 2)
        mse = float(np.mean(residual ** 2))
        return {"rmse": float(np.sqrt(mse)), "R2": float(1.0 - mse / baseline)}
    payload = {
        "basis": "observed colour difference and observed E3 gravity difference; no intercept",
        "direction": "aligned minus conflict",
        "feature_order": ["aligned_colour_minus_conflict_colour", "aligned_g_E3_minus_conflict_g_E3"],
        "components": 2,
        "pairs": len(pair_ids), "train_pairs": int(train.sum()), "holdout_pairs": int(holdout.sum()),
        "holdout_pair_ids": [pair_id for pair_id, selected in zip(pair_ids, holdout) if selected],
        "coefficients_rows_feature_order": coefficients.tolist(),
        "full_coefficients_rows_feature_order": full_coefficients.tolist(),
        "train_quality": quality(train, coefficients), "holdout_quality": quality(holdout, coefficients),
        "pca_energy_fraction": float(eigenvalues[:2].sum() / eigenvalues.sum()),
        "observed_features": {"pair_ids": pair_ids, "delta_colour": delta_colour.tolist(), "delta_g_E3": delta_g.tolist()},
        "fitted_scores": fitted.tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("train_quality", "holdout_quality", "pca_energy_fraction")}, indent=2))


if __name__ == "__main__":
    main()
