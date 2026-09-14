#!/usr/bin/env python3
"""Summarize direct PCA recovery rank sweep outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for path in args.recovery_root.glob("rank*/summary.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append({key: payload[key] for key in (
            "components", "retained_energy_fraction", "energy_scale", "pairs", "E3_accuracy", "E0_accuracy"
        )})
    rows.sort(key=lambda row: row["components"])
    if not rows:
        raise FileNotFoundError(f"No rank*/summary.json under {args.recovery_root}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pca_rank_recovery.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    ranks = [row["components"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=180)
    axes[0].plot(ranks, [row["E3_accuracy"] * 100 for row in rows], marker="o", label="E3")
    axes[0].plot(ranks, [row["E0_accuracy"] * 100 for row in rows], marker="s", label="E0")
    axes[0].set(xlabel="Retained PCA dimensions", ylabel="Recovery accuracy (%)", title="Block 1 direct PCA recovery")
    axes[0].set_xscale("log", base=2); axes[0].set_ylim(-3, 103); axes[0].grid(alpha=.25); axes[0].legend(frameon=False)
    axes[1].plot(ranks, [row["retained_energy_fraction"] * 100 for row in rows], marker="o", color="#4c78a8")
    axes[1].set(xlabel="Retained PCA dimensions", ylabel="Retained energy (%)", title="Uncentered residual PCA energy")
    axes[1].set_xscale("log", base=2); axes[1].set_ylim(-3, 103); axes[1].grid(alpha=.25)
    fig.suptitle("Freefall V3 large | hist32 100k | strict 128 pairs | block 1")
    fig.tight_layout(); fig.savefig(args.out / "pca_rank_recovery.png", bbox_inches="tight"); plt.close(fig)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
