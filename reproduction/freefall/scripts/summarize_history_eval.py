#!/usr/bin/env python3
"""Merge history-length evaluator shards and summarize aligned/conflict metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summary(rows: list[dict]) -> dict:
    output = {}
    for condition in ("aligned", "conflict"):
        selected = [row for row in rows if row["condition"] == condition]
        e3_errors = [abs(float(row["g_E3"]) - float(row["gravity_true"])) for row in selected if row["g_E3"] is not None]
        e0_errors = [abs(float(row["g_E0_strict"]) - float(row["gravity_true"])) for row in selected if row["g_E0_strict"] is not None]
        output[condition] = {
            "samples": len(selected), "valid_track_rate": float(np.mean([row["valid_track"] for row in selected])),
            "E3_accuracy": float(np.mean([row["E3_correct"] for row in selected])), "E3_mae": float(np.mean(e3_errors)),
            "E0_accuracy": float(np.mean([row["E0_correct"] for row in selected])), "E0_mae": float(np.mean(e0_errors)),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--merged-rows", type=Path, help="optional combined rows.json for downstream selection")
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.glob("shard*/rows.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    payload = {"rows": len(rows), "pairs": len({row["pair_id"] for row in rows}), "metrics": summary(rows)}
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.merged_rows is not None:
        args.merged_rows.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
