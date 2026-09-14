#!/usr/bin/env python3
"""Run the fit-only top-4 Pendulum controller under a frozen scale protocol.

Scale selection is restricted to the fit split.  After a scale is frozen, the
held-out split is generated once.  The evaluator and every threshold are shared
with natural, full-matched, and oracle conditions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from sshv2.experiments.pendulum.mechanism_pca import (
    _condition_latents,
    _denoise,
    _load_runtime,
    _measure_one,
    _write_future_video,
    read_jsonl,
)
from sshv2.experiments.pendulum.stage4_coordinate_difference import (
    synthesize_edit,
)


RANK = 4
STEPS = 20
CONDITION_TOKENS = 1088
# The released large-seed3407 Stage-3 artifact records and captures edits at
# commitment_block_index=12.  Injection must use that same intervention site;
# applying the fitted tensor at block 15 is a protocol mismatch and produces a
# superficially finite but causally ineffective edit.
BLOCK_INDEX = 12


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def invalid_reasons(measured: dict[str, Any]) -> list[str]:
    reasons = []
    if float(measured["detection_rate"]) < 0.90:
        reasons.append("detection_rate_below_0.90")
    if int(measured["max_missing_run"]) > 3:
        reasons.append("max_missing_run_above_3")
    if float(measured["max_adjacent_jump_px"]) > 20.0:
        reasons.append("adjacent_jump_above_20px")
    if float(measured["median_length_error"]) > 0.05:
        reasons.append("median_length_error_above_0.05")
    if float(measured["boundary_jump_px"]) > 20.0:
        reasons.append("boundary_jump_above_20px")
    if not bool(measured["area_valid"]):
        reasons.append("area_invalid")
    if not bool(measured["fit_valid"]):
        reasons.append("oscillation_fit_invalid")
    return reasons


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def scale_label(scale: float) -> str:
    return f"{scale:.2f}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--natural-metrics", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--coordinate-predictions", type=Path, required=True)
    parser.add_argument("--strict-bank-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--split", choices=("fit", "heldout"), required=True)
    parser.add_argument("--direction", choices=("A", "B"), required=True)
    parser.add_argument("--scales", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--hidden-size", type=int, default=1152)
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    selected = [row for row in rows if row["split"] == args.split and row["direction"] == args.direction]
    if len(selected) != 32:
        raise ValueError(f"expected 32 selected receivers, got {len(selected)}")
    coordinates = {
        row["receiver_id"]: np.asarray(
            [float(row[f"z_hat_{i}"]) for i in range(1, RANK + 1)],
            dtype=np.float64,
        )
        for row in read_csv(args.coordinate_predictions)
    }
    components = np.load(args.components, mmap_mode="r").reshape(8, -1)[:RANK]
    natural_rows = read_csv(args.natural_metrics)
    natural: dict[str, dict[str, dict[str, str]]] = {}
    for row in natural_rows:
        natural.setdefault(row["receiver_id"], {})[row["condition"]] = row
    scales = tuple(float(value) for value in args.scales.split(","))
    if args.split == "heldout" and len(scales) != 1:
        raise ValueError("held-out generation requires exactly one frozen scale")

    pipe, data_config = _load_runtime(
        SimpleNamespace(
            experiment_config=args.experiment_config,
            training_config=args.training_config,
            model_name="frequency_color_circle",
            history="short",
            checkpoint=args.checkpoint,
            device=args.device,
            steps=STEPS,
            block_index=BLOCK_INDEX,
            condition_tokens=CONDITION_TOKENS,
            hidden_size=args.hidden_size,
        ),
        rows,
    )
    direction_shape = (STEPS, CONDITION_TOKENS, args.hidden_size)
    metrics: list[dict[str, Any]] = []
    for position, row in enumerate(selected, 1):
        receiver_id = row["receiver_id"]
        conflict_tensor = _condition_latents(
            pipe, data_config, Path(row["conflict_video"]), history="short"
        )
        edit_array = synthesize_edit(
            components,
            coordinates[receiver_id].reshape(1, RANK),
            direction_shape,
        )
        edit = torch.from_numpy(np.array(edit_array, copy=True)).to(
            device=pipe.device, dtype=pipe.torch_dtype
        )
        omega_aligned = float(natural[receiver_id]["aligned"]["omega_hat"])
        omega_conflict = float(natural[receiver_id]["conflict"]["omega_hat"])
        denominator = omega_aligned - omega_conflict
        for scale in scales:
            destination = (
                args.out_root
                / args.split
                / args.direction
                / receiver_id
                / f"scale_{scale_label(scale)}.mp4"
            )
            if not destination.is_file():
                latents, _ = _denoise(
                    pipe,
                    data_config,
                    conflict_tensor,
                    seed=int(row["generation_seed"]),
                    steps=STEPS,
                    block_index=BLOCK_INDEX,
                    condition_tokens=CONDITION_TOKENS,
                    edit=edit,
                    strength=scale,
                )
                _write_future_video(pipe, data_config, latents, destination)
                del latents
            measured = _measure_one(
                destination,
                row,
                data_config,
                expected_colors=(row["aligned_color"], row["conflict_color"]),
            )
            reasons = invalid_reasons(measured)
            omega_hat = finite(measured["omega_hat"])
            recovery = (
                (omega_hat - omega_conflict) / denominator
                if omega_hat is not None and abs(denominator) > 1e-8
                else None
            )
            no_op_reference = (
                args.strict_bank_root / "natural_videos" / receiver_id / "conflict.mp4"
            )
            no_op_max_abs = None
            if scale == 0.0:
                from sshv2.experiments.pendulum.data import load_video

                replay = load_video(destination, expected_frames=data_config.future_frames)
                reference = load_video(no_op_reference, expected_frames=data_config.future_frames)
                no_op_max_abs = int(np.max(np.abs(replay.astype(np.int16) - reference.astype(np.int16))))
            metrics.append(
                {
                    "receiver_id": receiver_id,
                    "split": args.split,
                    "direction": row["direction"],
                    "target_label": row["target_label"],
                    "scale": scale,
                    "omega_true": float(row["omega_true"]),
                    "omega_natural": omega_conflict,
                    "omega_aligned": omega_aligned,
                    "omega_edited": omega_hat,
                    "recovery_denominator": denominator,
                    "normalized_recovery": recovery,
                    "valid": not reasons,
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
                    "detected_shape": measured["detected_shape"],
                    "intervention_layer_zero_based": BLOCK_INDEX,
                    "fm_calls_observed": STEPS,
                    "fm_calls_expected": STEPS,
                    "condition_prefix_tokens_edited": CONDITION_TOKENS,
                    "target_suffix_tokens_edited": 0,
                    "activation_delta_l2_unscaled": float(np.linalg.norm(edit_array)),
                    "activation_delta_l2_scaled": float(np.linalg.norm(edit_array)) * scale,
                    "no_op_max_abs_pixel": no_op_max_abs,
                    "output_video_path": str(destination),
                }
            )
        del edit, edit_array, conflict_tensor
        torch.cuda.empty_cache()
        print(f"{args.split} {args.direction} {position}/32 {receiver_id}", flush=True)

    write_csv(args.out_root / f"metrics_{args.split}_{args.direction}.csv", metrics)
    print(json.dumps({"status": "complete", "rows": len(metrics), "split": args.split, "direction": args.direction}))


if __name__ == "__main__":
    main()
