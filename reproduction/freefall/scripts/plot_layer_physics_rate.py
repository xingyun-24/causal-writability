#!/usr/bin/env python3
"""Plot per-layer physics-side rate for the two Projectile conflict types."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


GROUPS = {
    ("low", "blue"): "Low gravity, blue conflict",
    ("high", "red"): "High gravity, red conflict",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = json.loads(args.outcomes.read_text(encoding="utf-8"))
    grouped: dict[tuple[str, str], dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("condition") != "replacement":
            continue
        key = (str(row["true_band"]), str(row["input_colour"]))
        if key in GROUPS:
            grouped[key][int(row["block"])].append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table: list[dict[str, object]] = []
    plt.figure(figsize=(8.2, 4.8), dpi=170)
    colours = ["#D55E00", "#0072B2"]
    for colour, (key, label) in zip(colours, GROUPS.items(), strict=True):
        layers = sorted(grouped[key])
        rates = []
        for layer in layers:
            values = grouped[key][layer]
            physics = sum(value.get("route_label") == "physics_side" for value in values)
            rate = physics / len(values)
            rates.append(rate)
            table.append({
                "layer": layer,
                "pair_type": label,
                "pairs": len(values),
                "physics_side_pairs": physics,
                "physics_side_rate": rate,
            })
        plt.plot(layers, rates, marker="o", markersize=3.5, linewidth=2, color=colour, label=label)

    plt.xlabel("Layer (direct residual replacement after block)")
    plt.ylabel("Physics-side rate after replacement")
    plt.xlim(-0.5, 29.5)
    plt.ylim(-0.02, 1.02)
    plt.xticks(range(0, 30, 2))
    plt.yticks([0, .2, .4, .6, .8, 1.0], ["0%", "20%", "40%", "60%", "80%", "100%"])
    plt.grid(axis="y", alpha=.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(args.output_dir / "layer_physics_side_rate_by_pair_type.png", bbox_inches="tight")
    plt.savefig(args.output_dir / "layer_physics_side_rate_by_pair_type.pdf", bbox_inches="tight")
    plt.close()
    with (args.output_dir / "layer_physics_side_rate_by_pair_type.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


if __name__ == "__main__":
    main()
