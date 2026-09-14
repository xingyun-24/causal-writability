#!/usr/bin/env python3
"""Replace obsolete band correctness fields with E3/E0 correctness."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def value(row: dict, error_key: str, estimate_key: str) -> float | None:
    if row.get(error_key) is not None:
        return float(row[error_key])
    estimate = row.get(estimate_key)
    truth = row.get("gravity_true")
    if estimate is None or truth is None:
        return None
    return abs(float(estimate) - float(truth))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--threshold", type=float, default=0.002)
    args = parser.parse_args()
    for path in args.paths:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"Expected a JSON list: {path}")
        for row in rows:
            if "g_E3" not in row and "g_E3_free" in row:
                row["g_E3"] = row["g_E3_free"]
            for old_key, new_key in (("aligned_g_E3_free", "aligned_g_E3"), ("conflict_g_E3_free", "conflict_g_E3")):
                if new_key not in row and old_key in row:
                    row[new_key] = row[old_key]
            row.pop("g_E3_free", None)
            row.pop("aligned_g_E3_free", None)
            row.pop("conflict_g_E3_free", None)
            e3_error = value(row, "gravity_error", "g_E3")
            e0_error = value(row, "gravity_error_e0", "g_E0_strict")
            row["E3_correct"] = bool(e3_error is not None and e3_error < args.threshold)
            row["E0_correct"] = bool(e0_error is not None and e0_error < args.threshold)
            row.pop("true_band_correct", None)
            row.pop("e3_correct", None)
            row.pop("e0_correct", None)
        path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"migrated={len(rows)} path={path}")


if __name__ == "__main__":
    main()
