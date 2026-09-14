#!/usr/bin/env python3
"""Apply one relaxed evaluator contract to all frozen recovery videos."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sshv2.experiments.pendulum.data import config_from_mapping
from reevaluate_pendulum_detection import measure


CONDITIONS = ("natural_conflict", "full_matched", "top4_oracle", "fit_only_predicted")


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def outcome_for(valid: bool, omega: float | None, target: str) -> str:
    if not valid or omega is None:
        return "invalid"
    low = 2.195 <= omega <= 3.005
    high = 5.195 <= omega <= 6.405
    if (target == "low" and low) or (target == "high" and high):
        return "physics"
    if (target == "low" and high) or (target == "high" and low):
        return "shortcut"
    if 3.005 < omega < 5.195:
        return "compromise"
    return "off_family"


def invalid_reasons(measured: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if float(measured["detection_rate"]) < contract["detection_rate_required"]:
        reasons.append("detection_rate_below_relaxed_threshold")
    if int(measured["max_missing_run"]) > contract["max_missing_run_allowed"]:
        reasons.append("max_missing_run_above_relaxed_threshold")
    if float(measured["max_adjacent_jump_px"]) > contract["max_jump_px"]:
        reasons.append("adjacent_jump_above_relaxed_threshold")
    if float(measured["median_length_error"]) > contract["max_length_error"]:
        reasons.append("median_length_error_above_relaxed_threshold")
    if float(measured["boundary_jump_px"]) > contract["max_boundary_jump_px"]:
        reasons.append("boundary_jump_above_relaxed_threshold")
    if not bool(measured["area_valid"]):
        reasons.append("area_invalid")
    if not bool(measured["fit_valid"]):
        reasons.append("oscillation_fit_invalid")
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--release-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--color-distance", type=float, default=150.0)
    parser.add_argument("--min-area", type=int, default=12)
    parser.add_argument("--detection-rate-required", type=float, default=0.50)
    parser.add_argument("--max-missing-run-allowed", type=int, default=12)
    parser.add_argument("--max-jump-px", type=float, default=30.0)
    parser.add_argument("--max-length-error", type=float, default=0.10)
    parser.add_argument("--max-boundary-jump-px", type=float, default=30.0)
    parser.add_argument("--max-fit-rmse", type=float, default=0.12)
    args = parser.parse_args()

    contract = {
        "color_distance": args.color_distance,
        "min_area": args.min_area,
        "detection_rate_required": args.detection_rate_required,
        "max_missing_run_allowed": args.max_missing_run_allowed,
        "max_jump_px": args.max_jump_px,
        "max_length_error": args.max_length_error,
        "max_boundary_jump_px": args.max_boundary_jump_px,
        "max_fit_rmse": args.max_fit_rmse,
    }
    experiment = yaml.safe_load(args.experiment_config.read_text(encoding="utf-8"))
    data_config = config_from_mapping(experiment["data"])
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        manifests = {row["receiver_id"]: row for row in csv.DictReader(handle)}
    with args.release_csv.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    if len(source) != 256:
        raise ValueError(f"expected 256 release rows, got {len(source)}")

    output: list[dict[str, Any]] = []
    for index, row in enumerate(source, 1):
        manifest = manifests[row["receiver_id"]]
        measured = measure(
            args.repo_root / row["output_video_path"], manifest, data_config, **contract
        )
        omega = finite(measured["omega_hat"])
        denominator = float(row["recovery_denominator"])
        recovery = (
            0.0
            if row["condition"] == "natural_conflict"
            else (omega - float(row["omega_natural"])) / denominator
            if omega is not None
            else None
        )
        reasons = invalid_reasons(measured, contract)
        valid = not reasons
        updated = dict(row)
        updated.update(
            {
                "strict_valid": row["valid"],
                "strict_normalized_recovery": row["normalized_recovery"],
                "omega_edited": omega,
                "normalized_recovery": recovery,
                "valid": valid,
                "outcome": outcome_for(valid, omega, row["target_label"]),
                "invalid_reasons": ";".join(reasons),
                "detection_rate": measured["detection_rate"],
                "max_missing_run": measured["max_missing_run"],
                "max_adjacent_jump_px": measured["max_adjacent_jump_px"],
                "median_length_error": measured["median_length_error"],
                "boundary_jump_px": measured["boundary_jump_px"],
                "median_area_px": measured["median_area_px"],
                "area_valid": measured["area_valid"],
                "fit_valid": measured["fit_valid"],
                "fit_rmse": finite(measured["fit_rmse"]),
                "detected_color": measured["detected_color"],
            }
        )
        output.append(updated)
        if index % 32 == 0:
            print(f"measured {index}/{len(source)}", flush=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)

    summaries: dict[str, Any] = {}
    for condition in CONDITIONS:
        group = [row for row in output if row["condition"] == condition]
        valid = [row for row in group if row["valid"]]
        success = [
            row for row in valid
            if row["normalized_recovery"] is not None
            and 0.75 < float(row["normalized_recovery"]) < 1.25
        ]
        recoveries = [float(row["normalized_recovery"]) for row in valid if row["normalized_recovery"] is not None]
        reasons = Counter(
            reason
            for row in group
            for reason in str(row["invalid_reasons"]).split(";")
            if reason
        )
        summaries[condition] = {
            "n": len(group),
            "strict_valid_n": sum(parse_bool(row["strict_valid"]) for row in group),
            "valid_n": len(valid),
            "valid_rate": len(valid) / len(group),
            "near_full_success_n": len(success),
            "near_full_success_rate_all": len(success) / len(group),
            "conditional_median_recovery": float(np.median(recoveries)) if recoveries else None,
            "invalid_reason_counts": dict(sorted(reasons.items())),
            "outcome_counts": dict(sorted(Counter(row["outcome"] for row in group).items())),
        }
    summary = {
        "status": "appearance_tolerant_sensitivity_analysis",
        "evaluator_contract": contract,
        "conditions": summaries,
        "selected_example": {"receiver_id": "B_082", "selection": "same frozen frame-strip receiver"},
    }
    args.output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
