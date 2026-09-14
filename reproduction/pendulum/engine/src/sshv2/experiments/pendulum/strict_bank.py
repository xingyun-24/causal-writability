#!/usr/bin/env python3
"""Build the frozen 128-pair strict matched-counterfactual bank.

Formal Pendulum mechanism bank with the same semantics as Spring:
clean physics endpoint <-> clean shortcut endpoint, with compromise /
off-family samples excluded before any patching.

  * Candidate pool: 384 Direction-A + 384 Direction-B matched physical
    identities, generated programmatically in a fixed order.
    Direction A: true physics = high/fast, aligned = blue, conflict = red.
    Direction B: true physics = low/slow, aligned = red, conflict = blue.
    Every candidate shares physical trajectory / parameters / phase /
    amplitude / generation noise; only the appearance cue differs.
  * Qualification (before patching): natural aligned/conflict rollouts only,
    with the fixed rules
        aligned : valid, trajectory RMSE <= 0.035,
                  frequency class = true endpoint;
        conflict: valid, frequency class = opposite (shortcut) endpoint,
                  route = shortcut_frequency;
        |omega_A - omega_C| >= 2.0.
  * Acceptance follows the fixed candidate order (no ranking by gap or
    patch recovery) until 64 pairs per direction are frozen.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from sshv2.experiments.pendulum.data import (
    Appearance,
    PendulumParameters,
    PendulumRenderConfig,
    config_from_mapping,
    pendulum_trajectory,
    render_video,
    write_video,
)
from sshv2.experiments.pendulum.mechanism_pca import (
    _condition_latents,
    _denoise,
    _load_runtime,
    _measure_one,
    _write_future_video,
)


PROGRAM_VERSION = "pendulum_strict_bank_v2_hidden_size_parameterized"
POOL_PER_DIRECTION = 384
TARGET_PER_DIRECTION = 64
SEED = 3407
SEED_OFFSET = 23_000_000
RMSE_THRESHOLD = 0.035
GAP_THRESHOLD = 2.0
CONDITION_TOKENS = 1088
DEFAULT_HIDDEN_SIZE = 768
STEPS = 20
LOW_BAND = (2.2, 3.0)
HIGH_BAND = (5.2, 6.4)
TEST_MANIFEST_ID = "frequency_color_circle__strict_bank_128_v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _freeze_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    materialized = [dict(row) for row in rows]
    if path.is_file():
        existing = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if existing != materialized:
            raise ValueError(f"frozen file differs from current selection: {path}")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in materialized
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return materialized


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _cls(omega: float) -> str | None:
    if LOW_BAND[0] <= omega <= LOW_BAND[1]:
        return "slow"
    if HIGH_BAND[0] <= omega <= HIGH_BAND[1]:
        return "fast"
    return None


def _color_implied_cls(color: str) -> str:
    return "slow" if color == "red" else "fast"


def candidate_specs(direction: str) -> list[dict[str, Any]]:
    """Fixed-order candidate pool for one direction."""
    rng = np.random.default_rng(SEED + 1_000_000 + (0 if direction == "A" else 1))
    band = HIGH_BAND if direction == "A" else LOW_BAND
    specs: list[dict[str, Any]] = []
    direction_offset = 0 if direction == "A" else 10_000_000
    for index in range(POOL_PER_DIRECTION):
        omega = float(rng.uniform(band[0], band[1]))
        amplitude = float(rng.uniform(0.10, 0.17))
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        base_seed = 50_000_000 + direction_offset + index
        specs.append(
            {
                "candidate_id": f"{direction}_{index:03d}",
                "direction": direction,
                "omega_true": omega,
                "amplitude_true": amplitude,
                "phase": phase,
                "base_seed": base_seed,
                "generation_seed": base_seed + SEED_OFFSET,
                "aligned_color": "blue" if direction == "A" else "red",
                "conflict_color": "red" if direction == "A" else "blue",
            }
        )
    return specs


def render_candidate(
    spec: Mapping[str, Any],
    render: PendulumRenderConfig,
    video_root: Path,
    prediction_start: int,
) -> tuple[Path, Path, float]:
    direction = str(spec["direction"])
    theta, _velocity = pendulum_trajectory(
        PendulumParameters(
            omega=float(spec["omega_true"]),
            amplitude=float(spec["amplitude_true"]),
            phase=float(spec["phase"]),
        ),
        fps=render.fps,
        num_frames=render.num_frames,
    )
    theta_star = float(theta[prediction_start - 1])
    paths = {}
    for condition, color in (
        ("aligned", spec["aligned_color"]),
        ("conflict", spec["conflict_color"]),
    ):
        appearance = Appearance(color=color, shape="circle")
        frames = render_video(theta, appearance, render)
        destination = (
            video_root
            / direction
            / f"{spec['candidate_id']}_{condition}.mp4"
        )
        write_video(destination, frames, render.fps)
        paths[condition] = destination
    return paths["aligned"], paths["conflict"], theta_star


def trajectory_rmse(
    predicted_theta: np.ndarray,
    true_theta: np.ndarray,
) -> float | None:
    finite = np.isfinite(predicted_theta) & np.isfinite(true_theta)
    if finite.sum() < 6:
        return None
    return float(np.sqrt(np.mean((predicted_theta[finite] - true_theta[finite]) ** 2)))


def measure_natural(
    video_path: Path,
    *,
    expected_colors: Sequence[str],
    theta_star: float,
    true_future_theta: np.ndarray,
    data_config: Any,
) -> dict[str, Any]:
    from sshv2.experiments.pendulum.data import load_video
    from sshv2.experiments.pendulum.evaluation import (
        centers_to_theta,
        detect_bob_track,
        fit_oscillation,
        track_validity,
    )

    frames = load_video(video_path, expected_frames=data_config.future_frames)
    colors = tuple(dict.fromkeys(str(color) for color in expected_colors))
    candidates = [
        (
            color,
            detect_bob_track(frames, data_config.render, expected_color=color),
        )
        for color in colors
    ]
    detected_color, track = max(
        candidates, key=lambda item: float(item[1].detected.mean())
    )
    validity = track_validity(
        track, data_config.render, theta_star=theta_star
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
        and fit.rmse <= 0.08
        and math.isfinite(fit.omega)
    )
    valid = bool(validity["valid"] and fit_valid)
    detected = track.detected & np.isfinite(theta)
    dense = (
        np.interp(
            np.arange(theta.size),
            np.flatnonzero(detected),
            theta[detected],
        )
        if detected.sum() >= 2
        else np.full(theta.size, np.nan)
    )
    omega_hat = float(fit.omega) if math.isfinite(fit.omega) else None
    return {
        "valid": valid,
        "fit_valid": fit_valid,
        "omega_hat": omega_hat,
        "omega_class": _cls(omega_hat) if omega_hat is not None else None,
        "trajectory_rmse": trajectory_rmse(dense, true_future_theta),
        "detected_color": detected_color,
        "fit_rmse": float(fit.rmse) if math.isfinite(fit.rmse) else None,
        "detection_rate": float(validity["detection_rate"]),
    }


def qualify_candidate(
    direction: str,
    aligned: Mapping[str, Any],
    conflict: Mapping[str, Any],
) -> tuple[bool, list[str], float]:
    true_cls = "fast" if direction == "A" else "slow"
    shortcut_cls = "slow" if direction == "A" else "fast"
    reasons: list[str] = []
    if not aligned["valid"]:
        reasons.append("aligned_invalid")
    rmse = _finite(aligned.get("trajectory_rmse"))
    if rmse is None or rmse > RMSE_THRESHOLD:
        reasons.append("aligned_trajectory_rmse")
    if aligned.get("omega_class") != true_cls:
        reasons.append("aligned_wrong_class")
    if not conflict["valid"]:
        reasons.append("conflict_invalid")
    if conflict.get("omega_class") != shortcut_cls:
        reasons.append("conflict_wrong_class")
    gap = (
        abs(
            _finite(aligned.get("omega_hat")) - _finite(conflict.get("omega_hat"))
        )
        if _finite(aligned.get("omega_hat")) is not None
        and _finite(conflict.get("omega_hat")) is not None
        else None
    )
    if gap is None or gap < GAP_THRESHOLD:
        reasons.append("gap_below_2")
    return (not reasons), reasons, gap


def _runtime_args(args: argparse.Namespace) -> Any:
    return __import__("types").SimpleNamespace(
        experiment_config=args.experiment_config,
        training_config=args.training_config,
        model_name=args.model_name,
        history=args.history,
        checkpoint=args.checkpoint,
        device=args.device,
        steps=STEPS,
        block_index=0,
        condition_tokens=CONDITION_TOKENS,
        hidden_size=args.hidden_size,
    )


def run_direction(
    args: argparse.Namespace,
    direction: str,
) -> dict[str, Any]:
    data_document = yaml.safe_load(args.experiment_config.read_text(encoding="utf-8"))
    data_config = config_from_mapping(data_document["data"])
    render = data_config.render
    true_theta_cache: dict[str, np.ndarray] = {}

    video_root = args.data_root / "videos"
    natural_root = args.out_root / "natural_videos"
    specs = candidate_specs(direction)

    pipe = None
    records: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    rejection_counter: Counter[str] = Counter()
    started = time.monotonic()
    for position, spec in enumerate(specs):
        candidate_id = str(spec["candidate_id"])
        aligned_video = video_root / direction / f"{candidate_id}_aligned.mp4"
        conflict_video = video_root / direction / f"{candidate_id}_conflict.mp4"
        theta, _velocity = pendulum_trajectory(
            PendulumParameters(
                omega=float(spec["omega_true"]),
                amplitude=float(spec["amplitude_true"]),
                phase=float(spec["phase"]),
            ),
            fps=render.fps,
            num_frames=render.num_frames,
        )
        theta_star = float(theta[data_config.prediction_start - 1])
        future_theta = theta[data_config.prediction_start :]
        true_theta_cache[candidate_id] = future_theta
        if not aligned_video.is_file() or not conflict_video.is_file():
            render_candidate(spec, render, video_root, data_config.prediction_start)

        measured: dict[str, dict[str, Any]] = {}
        for condition, video, expected_color in (
            ("aligned", aligned_video, spec["aligned_color"]),
            ("conflict", conflict_video, spec["conflict_color"]),
        ):
            destination = natural_root / candidate_id / f"{condition}.mp4"
            if pipe is None:
                pipe, _data_config = _load_runtime(
                    _runtime_args(args),
                    [
                        {
                            "training_manifest_id": (
                                "frequency_color_circle__train_e273989068c3"
                            ),
                        }
                    ],
                )
            if not destination.is_file():
                condition_tensor = _condition_latents(
                    pipe,
                    data_config,
                    video,
                    history=args.history,
                )
                latents, _ = _denoise(
                    pipe,
                    data_config,
                    condition_tensor,
                    seed=int(spec["generation_seed"]),
                    steps=STEPS,
                    block_index=0,
                    condition_tokens=CONDITION_TOKENS,
                    capture=False,
                )
                _write_future_video(pipe, data_config, latents, destination)
                del condition_tensor, latents
                if args.device.startswith("cuda"):
                    import torch

                    torch.cuda.empty_cache()
            measured[condition] = measure_natural(
                destination,
                expected_colors=(
                    str(spec["aligned_color"]),
                    str(spec["conflict_color"]),
                ),
                theta_star=theta_star,
                true_future_theta=future_theta,
                data_config=data_config,
            )

        qualified, reasons, gap = qualify_candidate(
            direction, measured["aligned"], measured["conflict"]
        )
        for reason in reasons:
            rejection_counter[reason] += 1
        accepted_flag = bool(qualified and len(accepted) < TARGET_PER_DIRECTION)
        if accepted_flag:
            split = (
                "fit"
                if len(accepted) < TARGET_PER_DIRECTION // 2
                else "heldout"
            )
            accepted.append(
                {
                    "receiver_id": candidate_id,
                    "candidate_id": candidate_id,
                    "direction": direction,
                    "target_label": "high" if direction == "A" else "low",
                    "split": split,
                    "target": "frequency",
                    "omega_true": float(spec["omega_true"]),
                    "amplitude_true": float(spec["amplitude_true"]),
                    "phase": float(spec["phase"]),
                    "phase_index": position,
                    "diffusion_repeat": 0,
                    "generation_seed": int(spec["generation_seed"]),
                    "base_seed": int(spec["base_seed"]),
                    "theta_star": theta_star,
                    "aligned_color": str(spec["aligned_color"]),
                    "conflict_color": str(spec["conflict_color"]),
                    "aligned_shape": "circle",
                    "conflict_shape": "circle",
                    "aligned_video": str(aligned_video),
                    "conflict_video": str(conflict_video),
                    "training_manifest_id": (
                        "frequency_color_circle__train_e273989068c3"
                    ),
                    "test_manifest_id": TEST_MANIFEST_ID,
                    "model_name": args.model_name,
                    "accepted_index": len(accepted),
                }
            )
        records.append(
            {
                "candidate_id": candidate_id,
                "direction": direction,
                "omega_true": float(spec["omega_true"]),
                "amplitude_true": float(spec["amplitude_true"]),
                "phase": float(spec["phase"]),
                "base_seed": int(spec["base_seed"]),
                "generation_seed": int(spec["generation_seed"]),
                "qualified": qualified,
                "accepted": accepted_flag,
                "rejection_reasons": ";".join(reasons),
                "gap_omega": gap,
                "aligned_omega_hat": measured["aligned"]["omega_hat"],
                "aligned_omega_class": measured["aligned"]["omega_class"],
                "aligned_valid": measured["aligned"]["valid"],
                "aligned_trajectory_rmse": measured["aligned"][
                    "trajectory_rmse"
                ],
                "conflict_omega_hat": measured["conflict"]["omega_hat"],
                "conflict_omega_class": measured["conflict"]["omega_class"],
                "conflict_valid": measured["conflict"]["valid"],
                "conflict_trajectory_rmse": measured["conflict"][
                    "trajectory_rmse"
                ],
            }
        )
        print(
            f"{direction} {position + 1}/{len(specs)} {candidate_id} "
            f"qualified={qualified} accepted={accepted_flag} "
            f"accepted_total={len(accepted)} reasons={';'.join(reasons)} "
            f"elapsed={time.monotonic() - started:.0f}s",
            flush=True,
        )
        if len(accepted) == TARGET_PER_DIRECTION:
            # Continue evaluating the rest of the pool for the audit.
            pass

    candidate_path = args.out_root / f"candidate_records_{direction}.jsonl"
    temporary = args.out_root / f".candidate_records_{direction}.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    os.replace(temporary, candidate_path)
    _freeze_jsonl(
        args.out_root / f"accepted_{direction}.jsonl", accepted
    )
    return {
        "direction": direction,
        "candidates": len(specs),
        "accepted": len(accepted),
        "qualified_total": sum(1 for record in records if record["qualified"]),
        "rejection_reasons": dict(rejection_counter),
    }


def merge_bank(args: argparse.Namespace) -> dict[str, Any]:
    a_rows = [
        json.loads(line)
        for line in (
            args.out_root / "accepted_A.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    b_rows = [
        json.loads(line)
        for line in (
            args.out_root / "accepted_B.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(a_rows) != TARGET_PER_DIRECTION or len(b_rows) != TARGET_PER_DIRECTION:
        raise ValueError(
            f"accepted counts A={len(a_rows)} B={len(b_rows)}; "
            f"need {TARGET_PER_DIRECTION} each"
        )
    bank = [*a_rows, *b_rows]
    _freeze_jsonl(args.out_root / "strict_bank_128.jsonl", bank)
    _atomic_write_csv(
        args.out_root / "strict_bank_128.csv",
        [
            {key: row.get(key) for key in bank[0] if key != "model_name"}
            for row in bank
        ],
    )
    def load_records(direction: str) -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for line in (
                args.out_root / f"candidate_records_{direction}.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    natural_rows: list[dict[str, Any]] = []
    for direction in ("A", "B"):
        aligned_color = "blue" if direction == "A" else "red"
        conflict_color = "red" if direction == "A" else "blue"
        for record in load_records(direction):
            receiver_id = str(record["candidate_id"])
            for condition in ("aligned", "conflict"):
                natural_rows.append(
                    {
                        "receiver_id": receiver_id,
                        "condition": condition,
                        "omega_hat": record.get(f"{condition}_omega_hat"),
                        "valid": record.get(f"{condition}_valid"),
                        "detected_color": (
                            aligned_color
                            if condition == "aligned"
                            else conflict_color
                        ),
                        "omega_true": record.get("omega_true"),
                        "trajectory_rmse": record.get(
                            f"{condition}_trajectory_rmse"
                        ),
                        "omega_class": record.get(f"{condition}_omega_class"),
                    }
                )
    _atomic_write_csv(args.out_root / "natural_metrics.csv", natural_rows)

    audit = {
        "program_version": PROGRAM_VERSION,
        "hidden_size": args.hidden_size,
        "pool_per_direction": POOL_PER_DIRECTION,
        "target_per_direction": TARGET_PER_DIRECTION,
        "rmse_threshold": RMSE_THRESHOLD,
        "gap_threshold": GAP_THRESHOLD,
        "candidate_total": POOL_PER_DIRECTION * 2,
    }
    for direction in ("A", "B"):
        records = load_records(direction)
        qualified = sum(1 for row in records if row["qualified"])
        reasons: Counter[str] = Counter()
        for row in records:
            raw = str(row["rejection_reasons"])
            for reason in (raw.split(";") if raw else []):
                if reason:
                    reasons[reason] += 1
        audit[direction] = {
            "candidates": len(records),
            "qualified_total": qualified,
            "acceptance_rate": (
                TARGET_PER_DIRECTION / max(1, len(records))
            ),
            "pass_rate": qualified / max(1, len(records)),
            "rejection_reasons": dict(reasons),
        }
    _atomic_write_json(args.out_root / "strict_bank_audit.json", audit)
    return {
        "accepted_A": len(a_rows),
        "accepted_B": len(b_rows),
        "audit": audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the frozen strict 128-pair mechanism bank."
    )
    parser.add_argument("--model-name", default="frequency_color_circle")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--history", choices=("short", "long"), default="short")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=DEFAULT_HIDDEN_SIZE,
        help="DiT residual width (768 for medium, 1152 for large).",
    )
    parser.add_argument(
        "--direction", choices=("A", "B", "merge"), required=True
    )
    args = parser.parse_args()
    args.out_root = Path(args.out_root)
    args.data_root = Path(args.data_root)
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.direction == "merge":
        result = merge_bank(args)
    else:
        result = run_direction(args, args.direction)
    print(json.dumps({"status": "ok", **result}, sort_keys=True))


if __name__ == "__main__":
    main()
