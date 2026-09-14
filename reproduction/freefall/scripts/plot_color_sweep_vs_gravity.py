#!/usr/bin/env python3
"""Plot E3 gravity versus an 11-point red-to-blue RGB sweep."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, nargs="+", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    outcomes = []
    for path in args.outcomes:
        outcomes.extend(json.loads(path.read_text(encoding="utf-8")))
    metadata = {row["pair_id"]: row for row in csv.DictReader(args.metadata.open(newline="", encoding="utf-8")) if row["variant"] == "aligned"}
    rows = [
        row for row in outcomes
        if row["condition"] in {"aligned_baseline", "color_sweep"} and row["pair_id"] in metadata
    ]
    rows.sort(key=lambda row: float(metadata[row["pair_id"]]["color_fraction"]))
    if not rows or len(rows) % 11:
        raise ValueError(f"expected a positive multiple of 11 color points, got {len(rows)}")
    target_g = float(metadata[rows[0]["pair_id"]]["gravity"])
    fractions = np.asarray([float(metadata[row["pair_id"]]["color_fraction"]) for row in rows])
    estimates = np.asarray([float(row["gravity_hat"]) for row in rows])
    colors = np.asarray([json.loads(metadata[row["pair_id"]]["color_rgb"]) for row in rows], dtype=float) / 255.0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=180)
    rng = np.random.default_rng(20260821)
    ax.scatter(fractions + rng.normal(0.0, 0.009, len(fractions)), estimates, s=26, c=colors,
               edgecolors="black", linewidths=.3, alpha=.72, zorder=2)
    grid = np.linspace(0.0, 1.0, 11)
    means = [float(np.mean(estimates[np.isclose(fractions, value)])) for value in grid]
    ax.plot(grid, means, color="#333333", marker="o", markersize=3.5, linewidth=1.15, label="mean E3 $\\hat{g}$", zorder=3)
    # V3 reserves this interval as neither the low nor high gravity band.
    ax.axhspan(0.025, 0.045, color="#f0ad4e", alpha=.16, zorder=0)
    ax.axhline(0.025, color="#c47f16", linestyle=":", linewidth=.9)
    ax.axhline(0.045, color="#c47f16", linestyle=":", linewidth=.9)
    ax.text(1.035, 0.035, "middle band\n[0.025, 0.045]", color="#98630e", fontsize=8,
            va="center", ha="left", clip_on=False)
    ax.axhline(target_g, color="#222222", linestyle="--", linewidth=1.0, label=fr"target $g={target_g:.3f}$")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, max(0.075, float(estimates.max()) * 1.12))
    ax.set_xticks(np.linspace(0, 1, 11), [f"{i}/10" for i in range(11)])
    ax.set_xlabel("Color interpolation: red (0) -> blue (10), RGB")
    ax.set_ylabel("E3 $\\hat{g}$")
    ax.set_title(fr"Projectile V3 hist16 100k | fixed target $g={target_g:.3f}$")
    ax.grid(axis="y", alpha=.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out_dir / "color_sweep_vs_gravity.png", bbox_inches="tight")
    plt.close(fig)

    payload = {
        "target_g": target_g,
        "colors": 11,
        "samples_per_color": len(rows) // 11,
        "x": "color_fraction, linear RGB interpolation from configured red_rgb to blue_rgb",
        "y": "E3 gravity_hat",
        "middle_band": [0.025, 0.045],
        "points": [
            {"index": int(round(float(fraction) * 10)), "color_fraction": float(fraction),
             "rgb": [int(value) for value in np.asarray(color * 255).round()], "gravity_hat": float(estimate),
             "gravity_error": float(abs(estimate - target_g))}
            for fraction, color, estimate in zip(fractions, colors, estimates, strict=True)
        ],
        "means_by_color": [{"index": index, "mean_gravity_hat": means[index]} for index in range(11)],
    }
    (args.out_dir / "color_sweep_vs_gravity.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
