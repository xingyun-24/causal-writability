#!/usr/bin/env python3
"""Audit continuous-gravity data before it can enter training or analysis."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml


def trajectory(row: dict[str, str]) -> np.ndarray:
    time = np.arange(int(row["frames"]), dtype=float) / float(row["simulation_fps"])
    return np.column_stack((float(row["x0"]) + float(row["vx"]) * time,
                            float(row["y0"]) + float(row["vy"]) * time - .5 * float(row["gravity"]) * time**2))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", type=Path, required=True); parser.add_argument("--data-config", type=Path, required=True); parser.add_argument("--out", type=Path, default=None); args = parser.parse_args()
    d = yaml.safe_load(args.data_config.read_text(encoding="utf-8")); red, blue = map(tuple, (d["physics"]["red_gravity_range"], d["physics"]["blue_gravity_range"]))
    with (args.dataset / "metadata.csv").open(newline="", encoding="utf-8") as f: rows = list(csv.DictReader(f))
    errors: list[str] = []; pairs: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        pairs[row["pair_id"]].append(row); g = float(row["gravity"]); canonical = "red" if row["gravity_interval"] == "low" else "blue"; bounds = red if row["gravity_interval"] == "low" else blue
        if not bounds[0] <= g <= bounds[1]: errors.append(f"{row['sample_id']}: gravity outside declared interval")
        if row["variant"] == "aligned" and row["color_label"] != canonical: errors.append(f"{row['sample_id']}: aligned colour inconsistent")
        if row["variant"] == "conflict" and row["color_label"] == canonical: errors.append(f"{row['sample_id']}: conflict colour not flipped")
        saved = np.load(args.dataset / "trajectories" / row["trajectory"])
        if saved.shape != (int(row["frames"]), 2) or not np.allclose(saved, trajectory(row), atol=1e-12): errors.append(f"{row['sample_id']}: trajectory mismatch")
        acceleration = np.diff(saved[:, 1], n=2) * float(row["simulation_fps"]) ** 2
        if not np.allclose(acceleration, -g, atol=1e-9): errors.append(f"{row['sample_id']}: nonphysical acceleration")
    for pair_id, group in pairs.items():
        kinds = {r["variant"] for r in group}; expected = {"aligned"} if len(group) == 1 else {"aligned", "conflict"}
        if kinds != expected: errors.append(f"{pair_id}: variants={sorted(kinds)}")
        values = {(r["gravity"], r["gravity_interval"], r["x0"], r["y0"], r["vx"], r["vy"], r["trajectory"]) for r in group}
        if len(values) != 1: errors.append(f"{pair_id}: counterfactual changes physics")
    report = {"records": len(rows), "pairs": len(pairs), "by_interval": dict(Counter(r["gravity_interval"] for r in rows)),
              "by_colour": dict(Counter(r["color_label"] for r in rows)), "by_variant": dict(Counter(r["variant"] for r in rows)),
              "gravity_ranges_observed": {name: [min(float(r["gravity"]) for r in rows if r["gravity_interval"] == name), max(float(r["gravity"]) for r in rows if r["gravity_interval"] == name)] for name in ("low", "high")}, "errors": errors, "passed": not errors}
    out = args.out or args.dataset / "audit.json"; out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, indent=2))
    if errors: raise SystemExit(1)


if __name__ == "__main__": main()
