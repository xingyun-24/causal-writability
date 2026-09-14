#!/usr/bin/env python3
"""Plot 3D raw aligned/conflict projections with point colour equal to measured g."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projections", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.projections.read_text(encoding="utf-8"))
    values = np.asarray([[row["pc1"], row["pc2"], row["pc3"]] for row in rows], dtype=float)
    g = np.asarray([row["g_E3"] for row in rows], dtype=float)
    aligned = np.asarray([row["condition"] == "aligned" for row in rows])
    fig = plt.figure(figsize=(9.2, 7.6), dpi=220)
    axis = fig.add_subplot(111, projection="3d")
    norm = plt.Normalize(float(g.min()), float(g.max()))
    cmap = plt.get_cmap("viridis")
    for mask, marker, label in ((aligned, "o", "aligned"), (~aligned, "^", "conflict")):
        axis.scatter(values[mask, 0], values[mask, 1], values[mask, 2], c=g[mask], cmap=cmap, norm=norm,
                     marker=marker, s=34, alpha=.86, edgecolors="#202124", linewidths=.25, label=label)
    axis.set_xlabel("Projection on residual PC1")
    axis.set_ylabel("Projection on residual PC2")
    axis.set_zlabel("Projection on residual PC3")
    axis.set_title("Raw aligned/conflict activations projected onto residual PCA directions")
    axis.legend(frameon=True)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axis, pad=.12, label="Measured generated-video g_E3")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"points": len(rows), "g_min": float(g.min()), "g_max": float(g.max()), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
