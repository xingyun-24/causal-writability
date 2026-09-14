#!/usr/bin/env python3
"""Summarize held-out V2 aligned E3 gravity-fit failures."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


BANDS = {"low": (0.009, 0.015), "high": (0.029, 0.035)}


def failure_mode(row: dict[str, Any]) -> str:
    estimate = float(row["g_E3_free"])
    true_band = row["gravity_interval"]
    other = "high" if true_band == "low" else "low"
    if BANDS[other][0] <= estimate <= BANDS[other][1]:
        return "opposite_band"
    if BANDS["low"][1] < estimate < BANDS["high"][0]:
        return "middle_gap"
    return "below_low" if estimate < BANDS["low"][0] else "above_high"


def describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [float(row["g_E3_free"]) - float(row["gravity_true"]) for row in rows]
    absolute = [abs(value) for value in errors]
    return {
        "count": len(rows),
        "mae": statistics.mean(absolute),
        "median_absolute_error": statistics.median(absolute),
        "max_absolute_error": max(absolute),
        "mean_signed_error": statistics.mean(errors),
        "g_hat_range": [min(float(row["g_E3_free"]) for row in rows), max(float(row["g_E3_free"]) for row in rows)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows", type=Path)
    args = parser.parse_args()
    with args.rows.open(encoding="utf-8") as handle:
        aligned = [row for row in json.load(handle) if row["condition"] == "aligned"]
    incorrect = [
        row for row in aligned
        if not BANDS[row["gravity_interval"]][0] <= float(row["g_E3_free"]) <= BANDS[row["gravity_interval"]][1]
    ]
    output = {
        "aligned_samples": len(aligned),
        "incorrect": describe(incorrect),
        "incorrect_by_true_band": {
            band: describe([row for row in incorrect if row["gravity_interval"] == band])
            for band in BANDS
        },
        "failure_modes": dict(Counter(failure_mode(row) for row in incorrect)),
        "colour_mismatches": sum(row["detected_colour"] != row["input_colour"] for row in incorrect),
        "mean_absolute_free_state_offsets": {
            key: statistics.mean(abs(float(row[key])) for row in incorrect)
            for key in ("delta_y_E3", "delta_v_E3", "delta_x_E3", "delta_vx_E3")
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
