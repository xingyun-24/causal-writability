#!/usr/bin/env python3
"""Select balanced aligned-correct/conflict-failed E3 pairs for recovery."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True, help="model evaluator rows.json")
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-interval", type=int, default=4)
    parser.add_argument("--e3-threshold", type=float, default=0.002)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    if args.per_interval < 1:
        parser.error("--per-interval must be positive")

    data = yaml.safe_load(args.data_config.read_text(encoding="utf-8"))
    threshold = float(data.get("evaluation", {}).get("E3_threshold", args.e3_threshold))
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in json.loads(args.rows.read_text(encoding="utf-8")):
        grouped[row["pair_id"]][row["condition"]] = row

    candidates: dict[str, list[dict]] = defaultdict(list)
    for pair_id, conditions in grouped.items():
        aligned, conflict = conditions.get("aligned"), conditions.get("conflict")
        if aligned is None or conflict is None or not aligned["valid_track"] or not conflict["valid_track"]:
            continue
        interval = aligned["gravity_interval"]
        aligned_error = abs(aligned["g_E3"] - aligned["gravity_true"])
        conflict_error = abs(conflict["g_E3"] - conflict["gravity_true"])
        if interval != conflict["gravity_interval"] or not aligned_error < threshold or conflict_error < threshold:
            continue
        candidates[interval].append({
            "pair_id": pair_id,
            "base_seed": int(pair_id.split("_")[1]),
            "gravity_interval": interval,
            "gravity_true": aligned["gravity_true"],
            "aligned_g_E3": aligned["g_E3"],
            "conflict_g_E3": conflict["g_E3"],
            "aligned_y_rmse_E3": aligned["y_rmse_E3"],
            "conflict_y_rmse_E3": conflict["y_rmse_E3"],
            "conflict_abs_g_error": conflict_error,
        })

    selected: list[dict] = []
    available: dict[str, int] = {}
    for interval in ("low", "high"):
        ranked = sorted(candidates[interval], key=lambda item: (-item["conflict_abs_g_error"], item["pair_id"]))
        available[interval] = len(ranked)
        selected.extend(ranked[:args.per_interval])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "selection": "valid tracks; aligned E3 error below threshold; conflict E3 error at or above threshold; ranked by conflict absolute gravity error",
        "E3_accuracy_threshold": threshold,
        "candidates_by_interval": available,
        "per_interval": args.per_interval,
        "pairs": selected,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"selected={len(selected)} out={args.out}")


if __name__ == "__main__":
    main()
