#!/usr/bin/env python3
"""Fit uncentered residual-PCA scores from color*[1,g,sqrt(g)]."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def color_value(value: str) -> float:
    return {"red": 1.0, "blue": -1.0}[value]


def quality(actual: np.ndarray, predicted: np.ndarray, mean: np.ndarray) -> dict[str, object]:
    mse = np.mean((actual - predicted) ** 2, axis=0)
    baseline = np.mean((actual - mean[None, :]) ** 2, axis=0)
    return {"joint_rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))), "component_R2": [float(1-e/b) if b > 0 else None for e,b in zip(mse,baseline)]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca-npz", type=Path, required=True); parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--components", type=int, required=True)
    args = parser.parse_args()
    packed = np.load(args.pca_npz); pair_ids=[str(x) for x in packed["pair_ids"]]; eig=packed["eigenvalues"].astype(np.float64)
    if not 2 <= args.components <= len(eig): raise ValueError("components must be >=2 and available")
    scores=packed["vectors"][:, :args.components].astype(np.float64)*np.sqrt(eig[:args.components])[None,:]
    pairs={x["pair_id"]:x for x in json.loads(args.pair_manifest.read_text())["pairs"]}
    rows={(x["pair_id"],x["variant"]):x for x in csv.DictReader((args.dataset_dir/"metadata.csv").open(newline=""))}
    g=np.asarray([float(pairs[i]["gravity_true"]) for i in pair_ids]); interval=np.asarray([pairs[i]["gravity_interval"] for i in pair_ids])
    dc=np.asarray([color_value(rows[(i,"aligned")]["color_label"])-color_value(rows[(i,"conflict")]["color_label"]) for i in pair_ids])
    design=np.column_stack((dc,dc*g,dc*np.sqrt(g)))
    holdout=np.zeros(len(pair_ids),bool)
    for name in ("low","high"): holdout[np.flatnonzero(interval==name)[::2]]=True
    train=~holdout; coefficient,*_=np.linalg.lstsq(design[train],scores[train],rcond=None); predicted=design@coefficient
    payload={"target":f"first {args.components} PCA score coordinates of aligned-minus-conflict residuals","basis":"f(color,g)=color*[1,g,sqrt(g)]*B; fit f(aligned)-f(conflict); no intercept","color_encoding":"red=+1, blue=-1","feature_order":["delta_color","delta_color*g","delta_color*sqrt(g)"],"components":args.components,"pair_ids":pair_ids,"coefficients_rows_feature_order":coefficient.tolist(),"train_quality":quality(scores[train],predicted[train],scores[train].mean(0)),"holdout_quality":quality(scores[holdout],predicted[holdout],scores[train].mean(0)),"pca_energy_fraction":float(eig[:args.components].sum()/eig.sum())}
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps({"train":payload["train_quality"],"holdout":payload["holdout_quality"],"pca_energy_fraction":payload["pca_energy_fraction"]},indent=2))


if __name__ == "__main__": main()
