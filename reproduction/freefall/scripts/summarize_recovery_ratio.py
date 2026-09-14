#!/usr/bin/env python3
"""Score residual replacement with the canonical endpoint-normalized R metric."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def finite_gravity(row: dict) -> float | None:
    value = row.get("gravity_hat")
    return float(value) if value is not None and math.isfinite(float(value)) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--highlight-block", type=int, default=None)
    args = parser.parse_args()

    rows: list[dict] = []
    for path in args.outcomes:
        rows.extend(json.loads(path.read_text(encoding="utf-8")))

    baselines: dict[str, dict[str, float]] = defaultdict(dict)
    replacements: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        condition = row["condition"]
        value = finite_gravity(row)
        if condition in {"aligned_baseline", "conflict_baseline"} and value is not None:
            baselines[row["pair_id"]][condition] = value
        elif condition == "replacement":
            replacements[int(row["block"])].append(row)

    records = []
    ratio_rows = []
    for block in sorted(replacements):
        ratios = []
        invalid = 0
        for row in replacements[block]:
            pair = baselines.get(row["pair_id"], {})
            g_edit = finite_gravity(row)
            g_conflict = pair.get("conflict_baseline")
            g_aligned = pair.get("aligned_baseline")
            if g_edit is None or g_conflict is None or g_aligned is None:
                invalid += 1
                continue
            denominator = g_aligned - g_conflict
            if abs(denominator) <= 1e-12:
                invalid += 1
                continue
            recovery = (g_edit - g_conflict) / denominator
            ratios.append(recovery)
            ratio_rows.append({"pair_id": row["pair_id"], "block": block, "R": recovery,
                               "recovered": 0.9 <= recovery <= 1.1,
                               "g_conflict": g_conflict, "g_aligned": g_aligned,
                               "g_edit": g_edit})
        if not ratios:
            raise ValueError(f"No valid R values for block {block}")
        values = np.asarray(ratios)
        records.append({
            "block": block,
            "edited_pairs": len(replacements[block]),
            "valid_R_pairs": len(values),
            "excluded_pairs": invalid,
            "recovered_pairs": int(np.sum((values >= 0.9) & (values <= 1.1))),
            "recovery_accuracy": float(np.mean((values >= 0.9) & (values <= 1.1))),
            "mean_R": float(np.mean(values)),
            "median_R": float(np.median(values)),
            "mean_absolute_R_error": float(np.mean(np.abs(values - 1.0))),
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    criterion = {
        "name": "endpoint_normalized_recovery_R",
        "formula": "R = (g_edit - g_conflict) / (g_aligned - g_conflict)",
        "state_source": "E3 gravity_hat from generated videos",
        "success_interval": [0.9, 1.1],
        "invalid_denominator_rule": "exclude abs(g_aligned - g_conflict) <= 1e-12",
    }
    payload = {"criterion": criterion, "layers": records}
    (args.out_dir / "recovery_ratio_R.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "recovery_ratio_R_rows.json").write_text(json.dumps(ratio_rows, indent=2) + "\n", encoding="utf-8")

    xs = [item["block"] for item in records]
    ys = [item["recovery_accuracy"] * 100 for item in records]
    plt.figure(figsize=(8, 4.5), dpi=160)
    plt.plot(xs, ys, marker="o", color="#1f77b4", linewidth=1.8, markersize=3.5)
    if args.highlight_block is not None:
        selected = next((item for item in records if item["block"] == args.highlight_block), None)
        if selected is None:
            raise ValueError(f"Highlight block {args.highlight_block} has no result")
        plt.scatter([selected["block"]], [selected["recovery_accuracy"] * 100], color="#d62728", zorder=3)
    plt.xlabel("Modified DiT block")
    plt.ylabel("Recovery rate: 0.9 <= R <= 1.1 (%)")
    plt.xticks(xs)
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=.25)
    plt.tight_layout()
    plt.savefig(args.out_dir / "recovery_ratio_R.png")
    print(json.dumps({"criterion": criterion, "highlight": args.highlight_block,
                      "record": next((item for item in records if item["block"] == args.highlight_block), None)}))


if __name__ == "__main__":
    main()
