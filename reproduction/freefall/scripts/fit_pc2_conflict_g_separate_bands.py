#!/usr/bin/env python3
"""Fit residual PC2 from [1, conflict generated-video g_E3] per gravity band."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca-npz", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--eval-rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    packed = np.load(args.pca_npz)
    pair_ids = [str(value) for value in packed["pair_ids"]]
    values = packed["eigenvalues"].astype(np.float64)
    pc2_scores = packed["vectors"][:, 1].astype(np.float64) * np.sqrt(values[1])
    pairs = {item["pair_id"]: item for item in json.loads(args.pair_manifest.read_text(encoding="utf-8"))["pairs"]}
    evaluation = {(str(row["pair_id"]), str(row["condition"])): row for row in json.loads(args.eval_rows.read_text(encoding="utf-8"))}
    bands = np.asarray([pairs[pair_id]["gravity_interval"] for pair_id in pair_ids])
    conflict_g = np.asarray([float(evaluation[(pair_id, "conflict")]["g_E3"]) for pair_id in pair_ids])
    result: dict[str, object] = {}
    holdout_ids: list[str] = []
    for band in ("low", "high"):
        indices = np.flatnonzero(bands == band)
        holdout_local = np.zeros(len(indices), dtype=bool)
        holdout_local[::2] = True
        design = np.column_stack((np.ones(len(indices)), conflict_g[indices]))
        coefficient, *_ = np.linalg.lstsq(design[~holdout_local], pc2_scores[indices][~holdout_local], rcond=None)
        predicted = design @ coefficient
        holdout = indices[holdout_local]
        holdout_ids.extend(pair_ids[index] for index in holdout)
        result[band] = {
            "train_pair_ids": [pair_ids[index] for index in indices[~holdout_local]],
            "holdout_pair_ids": [pair_ids[index] for index in holdout],
            "coefficients_intercept_conflict_g_E3": coefficient.tolist(),
            "train_rmse": float(np.sqrt(np.mean((pc2_scores[indices][~holdout_local] - predicted[~holdout_local]) ** 2))),
            "holdout_rmse": float(np.sqrt(np.mean((pc2_scores[holdout] - predicted[holdout_local]) ** 2))),
        }
    payload = {
        "target": "PC2 score of aligned-minus-conflict residual",
        "basis": "[1, conflict generated-video g_E3]",
        "components": 2,
        "split": "every other pair after pair-id ordering, separately within low/high intervals",
        "holdout_pair_ids": holdout_ids,
        "bands": result,
        "pc2_energy_fraction": float(values[1] / values.sum()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
