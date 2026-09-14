#!/usr/bin/env python3
"""Plot uncentered residual-delta sample PCA coordinates for a captured run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    metadata = sum(
        (json.loads((path / "metadata.json").read_text(encoding="utf-8")) for path in sorted(args.pca_root.glob("shard*"))), []
    )
    metadata.sort(key=lambda item: item["pair_id"])
    packed = np.load(args.pca_root / "decomposition" / "sample_pca.npz")
    pair_ids = [str(item) for item in packed["pair_ids"]]
    if pair_ids != [item["pair_id"] for item in metadata]:
        raise ValueError("PCA vectors and metadata have different pair ordering")
    values = packed["eigenvalues"].astype(np.float64)
    vectors = packed["vectors"].astype(np.float64)
    coordinates = vectors[:, :2] * np.sqrt(values[:2])[None, :]
    gravity = np.asarray([item["gravity_true"] for item in metadata], dtype=np.float64)
    intervals = np.asarray([item["gravity_interval"] for item in metadata])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.8), dpi=220)
    scatter = ax.scatter(
        coordinates[:, 0], coordinates[:, 1], c=gravity, cmap="viridis", s=28, alpha=.9,
        linewidths=.25, edgecolors="#202020",
    )
    for interval, marker in (("low", "o"), ("high", "s")):
        mask = intervals == interval
        ax.scatter(coordinates[mask, 0], coordinates[mask, 1], marker=marker, s=35, facecolors="none", edgecolors="#202020", linewidths=.6, label=f"{interval} g")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Actual generated g")
    ax.set(xlabel="PC1 coordinate", ylabel="PC2 coordinate", title="Freefall block 1 residual-delta PCA (strict 128 pairs)")
    ax.grid(alpha=.2); ax.legend(frameon=False, loc="best")
    fig.tight_layout(); fig.savefig(args.out, bbox_inches="tight"); plt.close(fig)
    print(json.dumps({
        "pairs": len(metadata), "pc1_energy_fraction": float(values[0] / values.sum()),
        "pc2_energy_fraction": float(values[1] / values.sum()), "pc1_pc2_energy_fraction": float(values[:2].sum() / values.sum()),
        "out": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
