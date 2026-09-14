#!/usr/bin/env python3
"""Rebuild standard evaluation summaries with E3/E0 accuracy."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.002)
    args = parser.parse_args()
    rows = json.loads(args.rows.read_text(encoding="utf-8"))
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["condition"]].append(row)
    summary = {}
    for condition, values in sorted(groups.items()):
        item = {"samples": len(values), "valid_track_rate": float(np.mean([row["valid_track"] for row in values]))}
        for estimate_key, label in (("g_E3", "E3"), ("g_E0_strict", "E0")):
            errors = [abs(float(row[estimate_key]) - float(row["gravity_true"])) for row in values if row.get(estimate_key) is not None]
            item[f"{label}_accuracy"] = float(np.mean([error < args.threshold for error in errors])) if errors else 0.0
            item[f"{label}_mae"] = float(np.mean(errors)) if errors else None
        summary[condition] = item
    payload = {
        "evaluation": {"E3_threshold": args.threshold, "E3": "|g_hat - g| < E3_threshold"},
        "summary": summary,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
