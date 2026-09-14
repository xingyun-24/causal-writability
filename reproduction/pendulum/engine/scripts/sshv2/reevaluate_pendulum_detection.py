#!/usr/bin/env python3
"""Re-evaluate existing Pendulum videos under an explicit sensitivity contract.

This script never generates frames or touches model checkpoints.  It only
reruns bob detection, trajectory validity, and oscillation fitting on frozen
MP4 files.  Strict source CSVs are preserved; output is written separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sshv2.experiments.pendulum.data import config_from_mapping, load_video
from sshv2.experiments.pendulum.evaluation import (
    centers_to_theta,
    detect_bob_track,
    fit_oscillation,
    track_validity,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def measure(
    video: Path,
    manifest: dict[str, Any],
    data_config: Any,
    *,
    color_distance: float,
    min_area: int,
    detection_rate_required: float,
    max_missing_run_allowed: int,
    max_jump_px: float,
    max_length_error: float,
    max_boundary_jump_px: float,
    max_fit_rmse: float,
) -> dict[str, Any]:
    frames = load_video(video, expected_frames=data_config.future_frames)
    colors = tuple(dict.fromkeys((manifest["aligned_color"], manifest["conflict_color"])))
    candidates = [
        (
            color,
            detect_bob_track(
                frames,
                data_config.render,
                expected_color=color,
                color_distance=color_distance,
                min_area=min_area,
            ),
        )
        for color in colors
    ]
    detected_color, track = max(candidates, key=lambda item: float(item[1].detected.mean()))
    validity = track_validity(
        track,
        data_config.render,
        theta_star=float(manifest["theta_star"]),
        detection_rate_required=detection_rate_required,
        max_missing_run_allowed=max_missing_run_allowed,
        max_jump_px=max_jump_px,
        max_length_error=max_length_error,
        max_boundary_jump_px=max_boundary_jump_px,
    )
    theta = centers_to_theta(track, data_config.render)
    fit = fit_oscillation(
        theta,
        track.detected,
        fps=data_config.render.fps,
        omega_low=max(0.2, data_config.low_frequency.low - 1.0),
        omega_high=data_config.high_frequency.high + 1.0,
    )
    fit_valid = bool(
        math.isfinite(fit.rmse)
        and fit.rmse <= max_fit_rmse
        and math.isfinite(fit.omega)
        and math.isfinite(fit.amplitude)
    )
    return {
        **validity,
        "valid": bool(validity["valid"] and fit_valid),
        "fit_valid": fit_valid,
        "detected_color": detected_color,
        "omega_hat": fit.omega,
        "amplitude_hat": fit.amplitude,
        "fit_center": fit.center,
        "fit_rmse": fit.rmse,
    }


def parameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "color_distance": args.color_distance,
        "min_area": args.min_area,
        "detection_rate_required": args.detection_rate_required,
        "max_missing_run_allowed": args.max_missing_run_allowed,
        "max_jump_px": args.max_jump_px,
        "max_length_error": args.max_length_error,
        "max_boundary_jump_px": args.max_boundary_jump_px,
        "max_fit_rmse": args.max_fit_rmse,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receiver", action="append", default=[])
    parser.add_argument("--location", action="append", type=int, default=[])
    parser.add_argument("--color-distance", type=float, default=150.0)
    parser.add_argument("--min-area", type=int, default=12)
    parser.add_argument("--detection-rate-required", type=float, default=0.50)
    parser.add_argument("--max-missing-run-allowed", type=int, default=12)
    parser.add_argument("--max-jump-px", type=float, default=30.0)
    parser.add_argument("--max-length-error", type=float, default=0.08)
    parser.add_argument("--max-boundary-jump-px", type=float, default=30.0)
    parser.add_argument("--max-fit-rmse", type=float, default=0.12)
    args = parser.parse_args()

    experiment = yaml.safe_load(args.experiment_config.read_text(encoding="utf-8"))
    data_config = config_from_mapping(experiment["data"])
    manifest_rows = {row["receiver_id"]: row for row in load_jsonl(args.manifest)}
    with (args.scan_root / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))

    selected = [
        row for row in source_rows
        if row["location"].isdigit()
        and (not args.receiver or row["receiver_id"] in args.receiver)
        and (not args.location or int(row["location"]) in args.location)
    ]
    results = []
    contract = parameters(args)
    for index, row in enumerate(selected, 1):
        video = args.repo_root / row["video"]
        measured = measure(video, manifest_rows[row["receiver_id"]], data_config, **contract)
        omega_aligned = float(row["omega_aligned"])
        omega_conflict = float(row["omega_conflict"])
        denominator = omega_aligned - omega_conflict
        recovery = (
            (float(measured["omega_hat"]) - omega_conflict) / denominator
            if math.isfinite(float(measured["omega_hat"]))
            else float("nan")
        )
        results.append(
            {
                **row,
                "strict_valid": row["valid"],
                "strict_detection_rate": row["detection_rate"],
                "strict_frequency_recovery": row["frequency_recovery"],
                **measured,
                "frequency_recovery": recovery,
            }
        )
        if index % 100 == 0 or index == len(selected):
            print(f"measured {index}/{len(selected)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0]) if results else []
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    args.output.with_suffix(".contract.json").write_text(
        json.dumps({"contract": contract, "rows": len(results)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
