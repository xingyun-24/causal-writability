#!/usr/bin/env python3
"""Prepare the audited probability-based Pendulum layer-scan release.

The paper plots the unconditional fraction of frozen receivers satisfying
``valid and 0.75 < R < 1.25``.  Detection, missing-run, jump, boundary, and fit
gates determine validity.  Bob size and inferred pendulum length are retained
as diagnostics because the appearance-tolerant mask changes their pixel
geometry without preventing a complete, well-fitted trajectory.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "evaluator_v2"
FILES = {
    ("short", "high"): "short_A.csv",
    ("short", "low"): "short_B.csv",
    ("long", "high"): "long_A.csv",
    ("long", "low"): "long_B.csv",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def final_valid(row: dict[str, str]) -> bool:
    return bool(
        float(row["detection_rate"]) >= 0.50
        and int(row["max_missing_run"]) <= 12
        and float(row["max_adjacent_jump_px"]) <= 30.0
        and float(row["boundary_jump_px"]) <= 30.0
        and as_bool(row["fit_valid"])
    )


def isotonic_decreasing(values: list[float]) -> list[float]:
    levels: list[float] = []
    weights: list[float] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, value in enumerate(values):
        levels.append(-float(value))
        weights.append(1.0)
        starts.append(index)
        ends.append(index)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            weight = weights[-2] + weights[-1]
            pooled = (
                levels[-2] * weights[-2] + levels[-1] * weights[-1]
            ) / weight
            levels[-2:] = [pooled]
            weights[-2:] = [weight]
            starts[-2:] = [starts[-2]]
            ends[-2:] = [ends[-1]]
    output = [math.nan] * len(values)
    for level, start, end in zip(levels, starts, ends):
        for index in range(start, end + 1):
            output[index] = -level
    return output


def crossing(curve: list[float], threshold: float) -> float:
    for index, value in enumerate(curve):
        if value < threshold:
            if index == 0:
                return 0.0
            previous = curve[index - 1]
            if previous == value:
                return float(index)
            return index - 1 + (previous - threshold) / (previous - value)
    return float(len(curve) - 1)


def main() -> None:
    layer_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "status": "audited_probability_v2",
        "common_strict_count_by_direction": 38,
        "site_convention": (
            "location 0 = embedding output; location b+1 = after zero-based block b"
        ),
        "success_contract": "valid AND 0.75 < normalized recovery < 1.25",
        "validity_contract": {
            "detection_rate_min": 0.50,
            "max_missing_run": 12,
            "max_adjacent_jump_px": 30.0,
            "max_boundary_jump_px": 30.0,
            "fit_valid_required": True,
            "diagnostic_only": ["median_length_error", "area_valid"],
        },
        "directions": {},
        "source_files": sorted(FILES.values()),
    }

    for (history, direction), filename in FILES.items():
        path = SOURCE / filename
        rows = read_rows(path)
        raw_success: list[float] = []
        valid_curve: list[float] = []
        for location in range(31):
            selected = [row for row in rows if int(row["location"]) == location]
            if len(selected) != 38:
                raise AssertionError(
                    f"{history}/{direction}/{location}: expected 38, got {len(selected)}"
                )
            valid = [row for row in selected if final_valid(row)]
            success = [
                row
                for row in valid
                if math.isfinite(float(row["frequency_recovery"]))
                and 0.75 < float(row["frequency_recovery"]) < 1.25
            ]
            probability = len(success) / len(selected)
            raw_success.append(probability)
            valid_curve.append(len(valid) / len(selected))
            layer_rows.append(
                {
                    "history": history,
                    "direction": direction,
                    "location": location,
                    "site_label": selected[0].get("location_label", ""),
                    "n": len(selected),
                    "valid_n": len(valid),
                    "valid_fraction": len(valid) / len(selected),
                    "success_n": len(success),
                    "success_probability": probability,
                }
            )
        fitted = isotonic_decreasing(raw_success)
        record = {
            "D_omega": sum(raw_success),
            "L25": crossing(fitted, 0.25),
            "L50": crossing(fitted, 0.50),
            "L75": crossing(fitted, 0.75),
            "raw_success_curve": raw_success,
            "isotonic_success_curve": fitted,
            "valid_curve": valid_curve,
        }
        summary["directions"].setdefault(direction, {})[history] = record

    for direction in ("low", "high"):
        short = summary["directions"][direction]["short"]
        long = summary["directions"][direction]["long"]
        summary["directions"][direction]["long_minus_short"] = {
            "D_omega": long["D_omega"] - short["D_omega"],
            "L50": long["L50"] - short["L50"],
        }

    expected = {
        "low": (10.526315789473683, 9.461538461538462),
        "high": (8.052631578947373, 7.947712418300654),
    }
    for direction, (expected_d, expected_l) in expected.items():
        delta = summary["directions"][direction]["long_minus_short"]
        if not math.isclose(delta["D_omega"], expected_d, abs_tol=1e-12):
            raise AssertionError((direction, "D_omega", delta["D_omega"]))
        if not math.isclose(delta["L50"], expected_l, abs_tol=1e-12):
            raise AssertionError((direction, "L50", delta["L50"]))

    with (HERE / "layer_scan.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(layer_rows[0]))
        writer.writeheader()
        writer.writerows(layer_rows)
    (HERE / "layer_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        direction: summary["directions"][direction]["long_minus_short"]
        for direction in ("low", "high")
    }, indent=2))


if __name__ == "__main__":
    main()
