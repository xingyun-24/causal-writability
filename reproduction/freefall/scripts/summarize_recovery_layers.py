#!/usr/bin/env python3
"""Merge recovery worker outcomes and plot E3 accuracy by layer."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--selected-block", type=int, help="user-selected final block")
    args = parser.parse_args()
    rows: list[dict] = []
    for path in args.outcomes:
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    replaced = [row for row in rows if row["condition"] == "replacement"]
    baseline_rows = [row for row in rows if row["condition"] == "conflict_baseline"]
    baseline = {row["pair_id"]: bool(row.get("E3_correct", False)) for row in baseline_rows}
    if not baseline:
        raise ValueError("No conflict baselines available for recovery scoring")
    baseline_failed = {pair_id for pair_id, correct in baseline.items() if not correct}
    baseline_correct = set(baseline) - baseline_failed
    by_block: dict[int, list[dict]] = defaultdict(list)
    for row in replaced:
        by_block[int(row["block"])].append(row)
    records = []
    for block in sorted(by_block):
        values = by_block[block]
        errors = [row["gravity_error"] for row in values if row["gravity_error"] is not None]
        by_pair = {row["pair_id"]: bool(row.get("E3_correct", False)) for row in values}
        recovered = sum(by_pair[pair_id] for pair_id in baseline_failed)
        retained = sum(by_pair[pair_id] for pair_id in baseline_correct)
        records.append({
            "block": block,
            "pairs": len(values),
            "baseline_failed_pairs": len(baseline_failed),
            "recovered_pairs": recovered,
            "recovery_accuracy": recovered / len(baseline_failed),
            "baseline_correct_pairs": len(baseline_correct),
            "retained_correct_pairs": retained,
            "retained_accuracy": retained / len(baseline_correct) if baseline_correct else None,
            "E3_accuracy": float(np.mean([bool(row.get("E3_correct", False)) for row in values])),
            "E0_accuracy": float(np.mean([bool(row.get("E0_correct", False)) for row in values])),
            "mean_gravity_error": float(np.mean(errors)) if errors else None,
        })
    if not records:
        raise ValueError("No replacement outcomes")
    # Match the original layer-scan convention: identify how deep the recovery
    # signal remains fully effective, rather than preferring the earliest layer.
    best_accuracy = max(item["recovery_accuracy"] for item in records)
    best = max((item for item in records if item["recovery_accuracy"] == best_accuracy), key=lambda item: item["block"])
    selected = best if args.selected_block is None else next(item for item in records if item["block"] == args.selected_block)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected_reason = "deepest block on the maximum recovery-accuracy plateau" if args.selected_block is None else "user-selected final layer"
    payload = {"metric": "recovery among samples whose conflict E3 is incorrect", "E3": "|g_hat - g| < E3_threshold", "E3_threshold": 0.002, "baseline": {"pairs": len(baseline), "failed_pairs": len(baseline_failed), "correct_pairs": len(baseline_correct), "E0_accuracy": float(np.mean([bool(row.get("E0_correct", False)) for row in baseline_rows]))}, "selection_rule": "deepest block on the maximum recovery-accuracy plateau", "best_layer": best, "selected_layer": selected, "selected_layer_reason": selected_reason, "layers": records}
    (args.out_dir / "layer_E3.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    xs = [item["block"] for item in records]; ys = [item["recovery_accuracy"] * 100 for item in records]
    plt.figure(figsize=(8, 4.5), dpi=160)
    plt.plot(xs, ys, marker="o", color="#1f77b4", linewidth=1.8, markersize=3.5)
    plt.scatter([selected["block"]], [selected["recovery_accuracy"] * 100], color="#d62728", zorder=3)
    plt.xlabel("Modified DiT block")
    plt.ylabel("E3 recovery (%)")
    plt.xticks(xs)
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=.25)
    plt.tight_layout()
    plt.savefig(args.out_dir / "layer_E3.png")
    print(json.dumps(best))


if __name__ == "__main__":
    main()
