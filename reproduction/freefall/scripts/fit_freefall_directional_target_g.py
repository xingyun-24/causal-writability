#!/usr/bin/env python3
"""Fit separate low/high absolute-target gravity controllers in PCA space."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def r2(observed: np.ndarray, predicted: np.ndarray) -> list[float]:
    total = np.sum((observed - observed.mean(axis=0)) ** 2, axis=0)
    error = np.sum((observed - predicted) ** 2, axis=0)
    return [float(1.0 - item_error / item_total) for item_error, item_total in zip(error, total, strict=True)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-fit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.source_fit.read_text(encoding="utf-8"))
    pair_ids = np.asarray(source["pair_ids"], dtype=str)
    gravity = np.asarray(source["gravity_true"], dtype=np.float64)
    scores = np.asarray(source["actual_scores"], dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != 2:
        raise ValueError("Expected exactly two PCA coordinates")

    bands = np.asarray(["low" if pair_id.endswith("_low") else "high" for pair_id in pair_ids])
    split = np.full(len(pair_ids), "", dtype="U7")
    models: dict[str, object] = {}
    for band in ("low", "high"):
        indices = np.flatnonzero(bands == band)
        indices = indices[np.argsort(pair_ids[indices])]
        holdout, train = indices[::2], indices[1::2]
        split[train] = "train"
        split[holdout] = "heldout"
        design_train = np.column_stack((np.ones(len(train)), gravity[train]))
        coefficient, *_ = np.linalg.lstsq(design_train, scores[train], rcond=None)
        design_holdout = np.column_stack((np.ones(len(holdout)), gravity[holdout]))
        predicted_holdout = design_holdout @ coefficient
        holdout_r2 = r2(scores[holdout], predicted_holdout)
        models[band] = {
            "train_pair_ids": pair_ids[train].tolist(),
            "heldout_pair_ids": pair_ids[holdout].tolist(),
            "coefficients_rows_[1_g_target]": coefficient.tolist(),
            "train_condition_number": float(np.linalg.cond(design_train)),
            "heldout_coordinate_R2": holdout_r2,
            "heldout_joint_R2": float(1.0 - np.sum((scores[holdout] - predicted_holdout) ** 2) /
                                       np.sum((scores[holdout] - scores[holdout].mean(axis=0)) ** 2)),
            "heldout_rmse": float(np.sqrt(np.mean((scores[holdout] - predicted_holdout) ** 2))),
        }

    predictions = []
    for index, pair_id in enumerate(pair_ids):
        band = bands[index]
        coefficient = np.asarray(models[band]["coefficients_rows_[1_g_target]"], dtype=np.float64)
        predicted = np.array([1.0, gravity[index]]) @ coefficient
        for component in range(2):
            predictions.append({
                "receiver_id": str(pair_id), "split": str(split[index]), "target_direction": str(band),
                "g_target": float(gravity[index]), "coordinate": component + 1,
                "observed_score": float(scores[index, component]), "predicted_score": float(predicted[component]),
                "residual": float(scores[index, component] - predicted[component]),
            })
    payload = {
        "controller": "direction_specific_absolute_target_[1,g_target]",
        "direction_specific": True,
        "coordinate_difference_model": False,
        "input_contract": "target direction plus absolute target gravity only; no generated gravity input",
        "target": "two PCA coordinates of aligned-minus-conflict residual",
        "feature_order": ["1", "g_target"],
        "pairs": int(len(pair_ids)), "train_pairs": 64, "heldout_pairs": 64,
        "split": "every other pair after pair-id ordering, independently within low/high directions",
        "pair_ids": pair_ids.tolist(), "gravity_target": gravity.tolist(), "direction": bands.tolist(),
        "split_labels": split.tolist(), "models": models, "coordinate_predictions": predictions,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({band: models[band]["heldout_joint_R2"] for band in ("low", "high")}, indent=2))


if __name__ == "__main__":
    main()
