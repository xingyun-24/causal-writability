"""Evaluate generated spring futures in validity, trajectory, and frequency space."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sshv2.simulation.spring_shortcuts_v1 import (
    canonical_color_for_config,
    classify_detected_color,
    color_band_for_config,
    dataclass_config_from_dict,
    detect_mass_track,
    finite_json,
    load_video,
    pixel_x_to_displacement,
    route_metrics_against_bands,
    validity_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Raw eval split containing metadata.csv")
    parser.add_argument("--predictions", type=Path, required=True, help="Future-only MP4 directory or generation root")
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--history", choices=("short", "long"), required=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def prediction_path(root: Path, sample_id: str) -> Path:
    direct = root / f"{sample_id}.mp4"
    nested = root / "predictions" / f"{sample_id}.mp4"
    if direct.exists():
        return direct
    return nested


def safe_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return float(np.mean(values)) if values else None


def safe_median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))]
    return float(np.median(values)) if values else None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("valid") is True]
    conflicts = [row for row in valid if row.get("variant") == "conflict"]
    route_counts: dict[str, int] = defaultdict(int)
    anchored_counts: dict[str, int] = defaultdict(int)
    for row in conflicts:
        route_counts[str(row.get("route_label", "unknown"))] += 1
        anchored_counts[str(row.get("anchored_route_label", "unknown"))] += 1
    keys = (
        "d_exact_physics",
        "d_true_band",
        "d_color_band",
        "route_score_band",
        "omega_abs_error",
        "omega_hat_free",
        "free_shm_rmse",
        "omega_hat_free_amplitude",
        "true_reference_amplitude",
        "color_reference_amplitude",
        "min_band_residual",
        "detection_rate",
        "boundary_jump_px",
        "max_adjacent_jump_px",
        "max_y_deviation_px",
        "multiple_component_rate",
        "color_retention_rate",
    )
    return {
        "num_samples": len(rows),
        "num_valid": len(valid),
        "validity_rate": len(valid) / max(1, len(rows)),
        "num_valid_conflicts": len(conflicts),
        "conflict_primary_route_counts": dict(route_counts),
        "conflict_anchored_route_counts": dict(anchored_counts),
        "conflict_physics_follow_rate": route_counts.get("physics_frequency", 0) / max(1, len(conflicts)),
        "conflict_shortcut_follow_rate": route_counts.get("shortcut_frequency", 0) / max(1, len(conflicts)),
        "conflict_compromise_rate": route_counts.get("compromise_frequency", 0) / max(1, len(conflicts)),
        "conflict_off_frequency_rate": (
            route_counts.get("off_frequency_family", 0)
            + route_counts.get("outside_frequency_bands", 0)
            + route_counts.get("invalid", 0)
        ) / max(1, len(conflicts)),
        "conflict_color_reference_ood_rate": float(
            np.mean([bool(row.get("color_reference_ood")) for row in conflicts])
        ) if conflicts else 0.0,
        "means_valid": {key: safe_mean(valid, key) for key in keys},
        "medians_valid": {key: safe_median(valid, key) for key in keys},
    }


def paired_counterfactual_metrics(rows: list[dict[str, Any]], trajectories: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_pair[str(row["pair_id"])][str(row["variant"])] = row
    paired: list[dict[str, Any]] = []
    for pair_id, variants in by_pair.items():
        if "aligned" not in variants or "conflict" not in variants:
            continue
        aligned, conflict = variants["aligned"], variants["conflict"]
        record: dict[str, Any] = {
            "pair_id": pair_id,
            "trajectory_id": aligned["trajectory_id"],
            "true_band": aligned["true_band"],
            "both_valid": bool(aligned.get("valid") and conflict.get("valid")),
        }
        if record["both_valid"]:
            xa = trajectories[aligned["sample_id"]]
            xc = trajectories[conflict["sample_id"]]
            common = np.isfinite(xa) & np.isfinite(xc)
            record["prediction_counterfactual_rmse"] = (
                float(np.sqrt(np.mean((xa[common] - xc[common]) ** 2))) if common.sum() >= 6 else float("nan")
            )
            oa, oc = float(aligned["omega_hat_free"]), float(conflict["omega_hat_free"])
            record["omega_counterfactual_shift"] = abs(oa - oc) if math.isfinite(oa + oc) else float("nan")
            record["conflict_route_label"] = conflict.get("route_label")
        paired.append(record)
    return paired


def main() -> None:
    args = parse_args()
    cfg = dataclass_config_from_dict(yaml.safe_load(args.data_config.read_text(encoding="utf-8")))
    bands = {"slow": cfg.slow_band, "fast": cfg.fast_band}
    rows = read_rows(args.dataset / "metadata.csv")
    if args.limit:
        rows = rows[: args.limit]
    results: list[dict[str, Any]] = []
    detected_trajectories: dict[str, np.ndarray] = {}

    for row in rows:
        path = prediction_path(args.predictions, row["sample_id"])
        record: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "trajectory_id": row["trajectory_id"],
            "pair_id": row["pair_id"],
            "true_band": row["true_band"],
            "color_label": row["color_label"],
            "variant": row["variant"],
            "history": args.history,
            "omega_true": float(row["omega_true"]),
            "prediction_path": str(path),
        }
        if not path.exists():
            record.update({"valid": False, "failure": "missing_prediction"})
            results.append(record)
            continue
        try:
            frames = load_video(path)
        except Exception as exc:
            record.update({"valid": False, "failure": f"read_error:{type(exc).__name__}"})
            results.append(record)
            continue
        if frames.shape[0] != cfg.future_frames:
            record.update({"valid": False, "failure": f"unexpected_frame_count:{frames.shape[0]}"})
            results.append(record)
            continue

        track = detect_mass_track(frames, cfg.render)
        valid_stats = validity_metrics(track, cfg.render, x_star=float(row["x_star"]))
        record.update(valid_stats)
        detected_colors = [
            classify_detected_color(rgb, cfg.render)
            for rgb in track.mean_rgb
            if np.isfinite(rgb).all()
        ]
        if detected_colors:
            retained = [value == row["color_label"] for value in detected_colors]
            record["color_retention_rate"] = float(np.mean(retained))
            record["detected_color_majority"] = max(set(detected_colors), key=detected_colors.count)
        else:
            record["color_retention_rate"] = float("nan")
            record["detected_color_majority"] = "unknown"

        predicted_x = pixel_x_to_displacement(track.x_px, cfg.render)
        detected_trajectories[row["sample_id"]] = predicted_x
        if not valid_stats["valid"]:
            record["failure"] = "trajectory_invalid"
            results.append(record)
            continue

        metrics = route_metrics_against_bands(
            predicted_x,
            np.isfinite(predicted_x),
            true_band=bands[row["true_band"]],
            color_implied_band=bands[color_band_for_config(row["color_label"], cfg)],
            slow_band=cfg.slow_band,
            fast_band=cfg.fast_band,
            omega_true=float(row["omega_true"]),
            x_star=float(row["x_star"]),
            v_star=float(row["v_star"]),
            fps=cfg.render.fps,
            amplitude_low=cfg.amplitude_low,
            amplitude_high=cfg.amplitude_high,
            reference_amplitude_multiplier=cfg.route_reference_amplitude_multiplier,
            frequency_band_tolerance=cfg.frequency_band_tolerance,
            free_shm_rmse_threshold=cfg.free_shm_rmse_threshold,
        )
        metrics.pop("best_true_curve", None)
        metrics.pop("best_color_curve", None)
        if row["variant"] == "aligned":
            metrics["route_label"] = "iid_not_attributed"
            metrics["primary_route_label"] = "iid_not_attributed"
            metrics["anchored_route_label"] = "iid_not_attributed"
            metrics["route_score_band"] = float("nan")
        record.update(metrics)
        record["true_color_is_canonical"] = (
            row["color_label"] == canonical_color_for_config(row["true_band"], cfg)
        )
        results.append(record)

    paired = paired_counterfactual_metrics(results, detected_trajectories)
    summary: dict[str, Any] = {
        "history": args.history,
        "all": aggregate(results),
        "by_variant": {},
        "by_true_band": {},
        "conflict_by_true_band": {},
        "paired_counterfactuals": {
            "num_pairs": len(paired),
            "both_valid_rate": float(np.mean([p["both_valid"] for p in paired])) if paired else 0.0,
            "mean_prediction_counterfactual_rmse": safe_mean(paired, "prediction_counterfactual_rmse"),
            "mean_omega_counterfactual_shift": safe_mean(paired, "omega_counterfactual_shift"),
        },
    }
    for variant in ("aligned", "conflict"):
        summary["by_variant"][variant] = aggregate([row for row in results if row["variant"] == variant])
    for band in ("slow", "fast"):
        summary["by_true_band"][band] = aggregate([row for row in results if row["true_band"] == band])
        summary["conflict_by_true_band"][band] = aggregate(
            [row for row in results if row["true_band"] == band and row["variant"] == "conflict"]
        )

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "per_sample.jsonl").open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(finite_json(row), ensure_ascii=False) + "\n")
    with (args.out / "paired_counterfactuals.jsonl").open("w", encoding="utf-8") as handle:
        for row in paired:
            handle.write(json.dumps(finite_json(row), ensure_ascii=False) + "\n")
    (args.out / "summary.json").write_text(
        json.dumps(finite_json(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(finite_json(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
