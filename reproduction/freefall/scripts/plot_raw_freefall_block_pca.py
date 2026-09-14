#!/usr/bin/env python3
"""Plot aligned/conflict raw residual projections in difference-PCA coordinates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = sum((json.loads(path.read_text(encoding="utf-8")) for path in args.inputs), [])
    all_x = np.asarray([row["pc1"] for row in rows], dtype=float)
    all_y = np.asarray([row["pc2"] for row in rows], dtype=float)
    x_span = max(float(np.ptp(all_x)), 1.0)
    y_span = max(float(np.ptp(all_y)), 1.0)
    x_limits = (float(all_x.min() - .06 * x_span), float(all_x.max() + .06 * x_span))
    y_limits = (float(all_y.min() - .06 * y_span), float(all_y.max() + .06 * y_span))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=220, sharex=True, sharey=True)
    for ax, condition, title in zip(axes, ("aligned", "conflict"), ("Aligned raw residual", "Conflict raw residual")):
        subset = [row for row in rows if row["condition"] == condition]
        x = np.asarray([row["pc1"] for row in subset]); y = np.asarray([row["pc2"] for row in subset])
        c = np.asarray([row["gravity_true"] for row in subset]); band = np.asarray([row["gravity_interval"] for row in subset])
        scatter = ax.scatter(x, y, c=c, cmap="viridis", s=28, alpha=.9, edgecolors="#202020", linewidths=.25)
        for interval, marker in (("low", "o"), ("high", "s")):
            mask = band == interval
            ax.scatter(x[mask], y[mask], marker=marker, s=35, facecolors="none", edgecolors="#202020", linewidths=.6, label=f"{interval} g")
        ax.set(xlabel="Projection on residual PC1", ylabel="Projection on residual PC2", title=title)
        ax.set_xlim(*x_limits); ax.set_ylim(*y_limits)
        ax.grid(alpha=.2); ax.legend(frameon=False, loc="best")
        fig.colorbar(scatter, ax=ax).set_label("Actual generated g")
    fig.suptitle("Freefall block 1: raw residuals projected onto PCA of aligned - conflict")
    fig.tight_layout(); fig.savefig(args.out, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
