#!/usr/bin/env python3
"""Stage 4: coordinate-difference synthesis on the strict bank.

Using the frozen strict_bank_128 (l* = after block 15, m* = 2):

    z_i = V_{m*}^T d_i,   d_i = h_{A,i} - h_{C,i},

the program fits a low-dimensional state coordinate

    z_{T,C} = g(q_T) - g(q_C),   g(q) = W phi(q),

with q = (lambda, s*) where lambda is the future-frequency endpoint and
s* = (theta_star, angular_velocity_star) is the physical boundary state at
the end of the conditioning history (fit-set normalized).

Interpretation of the matched-pair fit signal: each pair's directed edit
(conflict -> aligned) is the endpoint-to-endpoint causal edit at boundary
state s*.  Direction A (true high) provides the low -> high edit at high
states; Direction B (true low) provides the high -> low edit at low states.
The source state is therefore the opposite appearance endpoint and the
target state is the true physics endpoint:

    Direction A: q_C = (low_center, s*), q_T = (omega_true, s*)
    Direction B: q_C = (high_center, s*), q_T = (omega_true, s*)

Basis complexity is tried in the fixed order phi0 -> phi1 -> phi2 -> phi3
and the simplest basis with stable heldout coordinate prediction AND causal
recovery close to the oracle top-m ceiling is selected.

Heldout synthesis never uses heldout true d, heldout PCA coordinates, heldout
matched-difference norms, or target/aligned activations.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from sshv2.experiments.pendulum.data import (
    PendulumParameters,
    config_from_mapping,
    pendulum_trajectory,
)
from sshv2.experiments.pendulum.mechanism_pca import (
    _condition_latents,
    _denoise,
    _load_runtime,
    _measure_one,
    _sha256_file,
    _write_future_video,
    read_jsonl,
)


PROGRAM_VERSION = "pendulum_stage4_coordinate_difference_v2_parameterized"
DEFAULT_COMMITMENT_BLOCK_INDEX = 15
DEFAULT_RANK = 2
CONDITION_TOKENS = 1088
DEFAULT_HIDDEN_SIZE = 768
STEPS = 20
LOW_CENTER = 2.6
HIGH_CENTER = 5.8
RANKS = (1, 2, 3, 4, 8)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


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


def _finite(values: Sequence[Any]) -> np.ndarray:
    result = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    return result


def _median(values: Sequence[Any]) -> float | None:
    finite = _finite(values)
    return float(np.median(finite)) if finite.size else None


def angular_velocity_star(row: Mapping[str, Any], prediction_start: int, fps: int) -> float:
    _, velocity = pendulum_trajectory(
        PendulumParameters(
            omega=float(row["omega_true"]),
            amplitude=float(row["amplitude_true"]),
            phase=float(row["phase"]),
        ),
        fps=fps,
        num_frames=prediction_start + 1,
    )
    return float(velocity[prediction_start - 1])


def basis_features(
    q_lambda: np.ndarray,
    s: np.ndarray,
    basis: str,
) -> np.ndarray:
    """Return phi(q) rows for a normalized state feature matrix."""
    lam = q_lambda.reshape(-1, 1)
    s1 = s[:, 0:1]
    s2 = s[:, 1:2]
    if basis == "phi0":
        return lam
    if basis == "phi1":
        return np.hstack([lam, lam * s1, lam * s2])
    if basis == "phi2":
        return np.hstack([lam, lam**2, lam * s1, lam * s2, lam**2 * s1, lam**2 * s2])
    if basis == "phi3":
        return np.hstack(
            [
                lam,
                lam**2,
                lam * s1,
                lam * s2,
                lam**2 * s1,
                lam**2 * s2,
                lam * s1**2,
                lam * s1 * s2,
                lam * s2**2,
            ]
        )
    raise ValueError(f"unknown basis {basis}")


def state_features(
    rows: Sequence[Mapping[str, Any]],
    prediction_start: int,
    fps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (lambda_true per row, s* matrix [theta_star, v_star])."""
    lambdas = np.asarray([float(row["omega_true"]) for row in rows])
    theta = np.asarray([float(row["theta_star"]) for row in rows])
    v = np.asarray(
        [angular_velocity_star(row, prediction_start, fps) for row in rows]
    )
    return lambdas, np.stack([theta, v], axis=1)


def fit_normalize(
    train: np.ndarray,
    apply: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-12] = 1.0
    return (
        (train - mean) / std,
        (apply - mean) / std,
        mean,
        std,
    )


def load_components(path: Path, rank: int) -> np.ndarray:
    return np.load(path, mmap_mode="r").reshape(rank, -1)


def coordinates(
    directions: np.ndarray,
    components: np.ndarray,
    indices: Sequence[int],
) -> np.ndarray:
    """z_i = V_rank^T d_i for the given manifest rows."""
    directions = directions.reshape(directions.shape[0], -1)
    flat = components.shape[1]
    result = np.zeros((len(indices), components.shape[0]), dtype=np.float64)
    chunk = 262_144
    for start in range(0, flat, chunk):
        stop = min(flat, start + chunk)
        d_chunk = np.asarray(
            directions[list(indices), start:stop], dtype=np.float32
        )
        result += d_chunk @ np.asarray(components[:, start:stop], dtype=np.float32).T
    return result


def synthesize_edit(
    components: np.ndarray,
    z_hat: np.ndarray,
    direction_shape: tuple[int, int, int],
) -> np.ndarray:
    """d_hat = V_{m*} z_hat (flattened -> reshaped)."""
    flat = components.shape[1]
    result = np.empty(flat, dtype=np.float32)
    chunk = 262_144
    for start in range(0, flat, chunk):
        stop = min(flat, start + chunk)
        result[start:stop] = z_hat @ np.asarray(
            components[:, start:stop], dtype=np.float32
        )
    return result.reshape(direction_shape)


def _video_path(out_root: Path, basis: str, receiver_id: str) -> Path:
    return out_root / "videos" / receiver_id / f"synthesized_{basis}.mp4"


def synthesize_and_measure(
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    heldout_indices: Sequence[int],
    basis_edits: dict[str, dict[str, np.ndarray]],
    pipe: Any,
    data_config: Any,
) -> list[dict[str, Any]]:
    import torch

    metrics: list[dict[str, Any]] = []
    for position, manifest_index in enumerate(heldout_indices):
        row = rows[int(manifest_index)]
        receiver_id = str(row["receiver_id"])
        conflict_tensor = _condition_latents(
            pipe,
            data_config,
            Path(str(row["conflict_video"])),
            history=args.history,
        )
        for basis, edits in basis_edits.items():
            destination = _video_path(args.out_root, basis, receiver_id)
            if destination.is_file():
                print(f"synthesize {receiver_id} {basis} resume", flush=True)
            else:
                edit = torch.from_numpy(
                    np.array(edits[receiver_id], dtype=np.float32, copy=True)
                ).to(device=pipe.device, dtype=pipe.torch_dtype)
                latents, _ = _denoise(
                    pipe,
                    data_config,
                    conflict_tensor,
                    seed=int(row["generation_seed"]),
                    steps=STEPS,
                    block_index=args.commitment_block_index,
                    condition_tokens=CONDITION_TOKENS,
                    edit=edit,
                    strength=1.0,
                )
                _write_future_video(pipe, data_config, latents, destination)
                print(f"synthesize {receiver_id} {basis} complete", flush=True)
                del edit, latents
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            measured = _measure_one(
                destination,
                row,
                data_config,
                expected_colors=(
                    str(row["aligned_color"]),
                    str(row["conflict_color"]),
                ),
            )
            metrics.append(
                {
                    "receiver_id": receiver_id,
                    "target_label": row["target_label"],
                    "split": row["split"],
                    "basis": basis,
                    "omega_hat": measured["omega_hat"],
                    "valid": measured["valid"],
                    "detected_color": measured["detected_color"],
                    "fit_rmse": measured["fit_rmse"],
                }
            )
        del conflict_tensor
    _atomic_write_csv(args.out_root / "synthesis_metrics.csv", metrics)
    return metrics


def recovery_for(
    metrics: Sequence[Mapping[str, Any]],
    natural_by_id: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, float | None]:
    per_basis: dict[str, list[float]] = {}
    for row in metrics:
        if not bool(row["valid"]):
            continue
        receiver_id = str(row["receiver_id"])
        baselines = natural_by_id.get(receiver_id)
        if baselines is None:
            continue
        omega_a = float(baselines["aligned"]["omega_hat"])
        omega_c = float(baselines["conflict"]["omega_hat"])
        denominator = omega_a - omega_c
        if abs(denominator) <= 1e-8 or row["omega_hat"] is None:
            continue
        recovery = (float(row["omega_hat"]) - omega_c) / denominator
        if math.isfinite(recovery):
            per_basis.setdefault(str(row["basis"]), []).append(recovery)
    return {
        basis: (
            float(np.median(values)) if values else None
        )
        for basis, values in per_basis.items()
    }


def _natural_by_id(args: argparse.Namespace) -> dict[str, dict[str, dict[str, Any]]]:
    def _safe_float(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    result: dict[str, dict[str, dict[str, Any]]] = {}
    for row in _read_csv(args.natural_metrics):
        result.setdefault(row["receiver_id"], {})[row["condition"]] = {
            "omega_hat": _safe_float(row["omega_hat"]),
            "valid": row["valid"] == "True",
        }
    return result


def oracle_recovery(args: argparse.Namespace) -> dict[str, float | None]:
    rows = _read_csv(args.ceiling_metrics)
    by_edit: dict[str, list[float]] = {}
    for row in rows:
        if not bool(row["valid"]):
            continue
        value = float(row["frequency_recovery"])
        if math.isfinite(value):
            by_edit.setdefault(str(row["edit"]), []).append(value)
    return {
        edit: float(np.median(values)) if values else None
        for edit, values in by_edit.items()
    }


def coordinate_r2(predicted: np.ndarray, target: np.ndarray) -> float | None:
    mask = np.isfinite(predicted) & np.isfinite(target)
    if mask.sum() < 2:
        return None
    p = predicted[mask]
    t = target[mask]
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else None


def build_report(
    *,
    args: argparse.Namespace,
    checkpoint_sha: str,
    basis_results: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    oracle: Mapping[str, float | None],
) -> str:
    def _fmt(value: Any) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "n/a"
        return "n/a" if not math.isfinite(number) else f"{number:.3f}"

    lines = [
        f"# Pendulum Stage 4：Coordinate-Difference Synthesis（strict_bank_128, ℓ\*=block {args.commitment_block_index}, m\*={args.rank}）",
        "",
        "Status: complete. 协议版本: " + PROGRAM_VERSION.replace("pendulum_", ""),
        "",
        "## 实验身份",
        "",
        f"- Model: `{args.model_name}` 50k-short；Checkpoint SHA256: `{checkpoint_sha}`",
        f"- Bank: strict_bank_128（fit64 = 32A + 32B，heldout64 = 32A + 32B，预固定）",
        f"- 状态 q = (λ, s\*)；s\* = (θ\*, v\*)（conditioning history 末端物理边界状态），fit-set 归一化",
        f"- 拟合信号：每对 matched pair 的定向 edit（conflict→aligned）= 端点↔端点 causal edit；"
        "Direction A 提供 low→high（q_C=(2.6,s\*), q_T=(ω_true,s\*)），Direction B 提供 high→low",
        f"- z_i = V_{args.rank}ᵀd_i；g(q) = Wφ(q)；heldout 合成只用物理状态特征，禁止使用 heldout 真实 d / PCA 坐标 / 范数 / target activation",
        "",
        "## 各 basis 的 heldout 表现",
        "",
        f"| basis | 特征 | heldout coord R² | synthesized recovery | oracle rank-{args.rank} recovery | 选择 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in basis_results:
        lines.append(
            f"| {row['basis']} | {row['feature_count']} 维 | "
            f"{_fmt(row['heldout_coordinate_r2'])} | "
            f"{_fmt(row['synthesized_recovery'])} | "
            f"{_fmt(oracle.get(f'rank_{args.rank}'))} | "
            f"{'✓' if row['basis'] == selection['selected_basis'] else ''} |"
        )
    lines += [
        "",
        "## 选择",
        "",
        f"- Oracle top-m (rank-{args.rank}) ceiling recovery: {_fmt(oracle.get(f'rank_{args.rank}'))}；"
        f"full-d: {_fmt(oracle.get('full_d'))}",
        f"- 选择规则：最简单的 basis，其 heldout coordinate R² ≥ {selection['min_coordinate_r2']} "
        f"且 synthesized recovery ≥ max(0.5, {selection['full_d_close_fraction']} × oracle rank-{args.rank})",
        f"- **selected basis = {selection['selected_basis']}**",
        "",
        "## Artifacts",
        "",
        "- `basis_results.csv`: 逐 basis 的拟合/预测/recovery",
        "- `synthesis_metrics.csv`: heldout 逐 receiver 逐 basis 的合成注入测量",
        "- `coordinate_predictions.csv`: heldout ẑ vs z",
        "- `plots/stage4_basis_selection.png`",
    ]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.out_root = Path(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha = _sha256_file(checkpoint)
    rows = read_jsonl(args.manifest)
    if len(rows) != 128:
        raise ValueError(f"manifest has {len(rows)} rows; expected 128")
    fit_indices = [
        index for index, row in enumerate(rows) if row["split"] == "fit"
    ]
    heldout_indices = [
        index for index, row in enumerate(rows) if row["split"] == "heldout"
    ]
    if len(fit_indices) != 64 or len(heldout_indices) != 64:
        raise ValueError("strict bank must have 64 fit + 64 heldout")

    import yaml

    experiment = yaml.safe_load(args.experiment_config.read_text(encoding="utf-8"))
    data_config = config_from_mapping(experiment["data"])
    prediction_start = data_config.prediction_start
    fps = data_config.render.fps

    pipe, _data_config = _load_runtime(
        SimpleNamespace(
            experiment_config=args.experiment_config,
            training_config=args.training_config,
            model_name=args.model_name,
            history=args.history,
            checkpoint=args.checkpoint,
            device=args.device,
            steps=STEPS,
            block_index=args.commitment_block_index,
            condition_tokens=CONDITION_TOKENS,
            hidden_size=args.hidden_size,
        ),
        rows,
    )

    directions = np.load(args.directions, mmap_mode="r")
    direction_shape = tuple(int(value) for value in directions.shape[1:])
    components = load_components(args.components, max(RANKS))

    lambdas, s_raw = state_features(rows, prediction_start, fps)
    fit_mask = np.zeros(len(rows), dtype=bool)
    fit_mask[fit_indices] = True
    _train_norm, s_norm_all, _s_mean, _s_std = fit_normalize(
        s_raw[fit_indices], s_raw
    )
    lambda_fit = lambdas[fit_indices]
    lambda_mean = float(lambda_fit.mean())
    lambda_std = float(lambda_fit.std())
    if lambda_std < 1e-12:
        lambda_std = 1.0
    lambda_norm = (lambdas - lambda_mean) / lambda_std

    # Source lambda per row: opposite appearance endpoint center.
    source_lambda = np.asarray(
        [
            LOW_CENTER if row["target_label"] == "high" else HIGH_CENTER
            for row in rows
        ]
    )
    source_lambda_norm = (source_lambda - lambda_mean) / lambda_std
    target_lambda_norm = lambda_norm

    z_all = coordinates(directions, components[: args.rank], range(len(rows)))
    z_fit = z_all[fit_indices]
    z_heldout = z_all[heldout_indices]

    basis_results: list[dict[str, Any]] = []
    basis_edits: dict[str, dict[str, np.ndarray]] = {}
    predictions_rows: list[dict[str, Any]] = []
    for basis in ("phi0", "phi1", "phi2", "phi3"):
        phi_source_fit = basis_features(
            source_lambda_norm[fit_indices],
            s_norm_all[fit_indices],
            basis,
        )
        phi_target_fit = basis_features(
            target_lambda_norm[fit_indices],
            s_norm_all[fit_indices],
            basis,
        )
        dphi_fit = phi_target_fit - phi_source_fit
        weight, *_ = np.linalg.lstsq(dphi_fit, z_fit, rcond=None)
        dphi_heldout = basis_features(
            target_lambda_norm[heldout_indices],
            s_norm_all[heldout_indices],
            basis,
        ) - basis_features(
            source_lambda_norm[heldout_indices],
            s_norm_all[heldout_indices],
            basis,
        )
        z_hat_heldout = dphi_heldout @ weight
        r2 = coordinate_r2(z_hat_heldout, z_heldout)
        edits = {
            str(rows[index]["receiver_id"]): synthesize_edit(
                components[: args.rank],
                z_hat_heldout[position : position + 1],
                direction_shape,
            )
            for position, index in enumerate(heldout_indices)
        }
        basis_edits[basis] = edits
        for position, index in enumerate(heldout_indices):
            predictions_rows.append(
                {
                    "receiver_id": rows[index]["receiver_id"],
                    "basis": basis,
                    **{
                        f"z_true_{coordinate + 1}": z_heldout[position, coordinate]
                        for coordinate in range(args.rank)
                    },
                    **{
                        f"z_hat_{coordinate + 1}": z_hat_heldout[position, coordinate]
                        for coordinate in range(args.rank)
                    },
                }
            )
        basis_results.append(
            {
                "basis": basis,
                "feature_count": dphi_fit.shape[1],
                "heldout_coordinate_r2": r2,
                "synthesized_recovery": None,
                "weight_shape": list(weight.shape),
            }
        )
        print(
            f"basis {basis} fitted, heldout coord R2 = {r2}",
            flush=True,
        )

    metrics = synthesize_and_measure(
        args,
        rows,
        heldout_indices,
        basis_edits,
        pipe,
        data_config,
    )
    natural_by_id = _natural_by_id(args)
    recovery = recovery_for(metrics, natural_by_id)
    oracle = oracle_recovery(args)
    for row in basis_results:
        row["synthesized_recovery"] = recovery.get(row["basis"])

    # Selection: simplest basis with stable coordinate prediction AND causal
    # recovery close to the oracle top-m ceiling.
    min_coordinate_r2 = 0.90
    full_d_close_fraction = 0.90
    oracle_rank = oracle.get(f"rank_{args.rank}")
    recovery_target = (
        max(0.5, full_d_close_fraction * oracle_rank)
        if oracle_rank is not None
        else 0.5
    )
    selected_basis = None
    for row in basis_results:
        r2 = row["heldout_coordinate_r2"]
        synth = row["synthesized_recovery"]
        if (
            r2 is not None
            and r2 >= min_coordinate_r2
            and synth is not None
            and synth >= recovery_target
        ):
            selected_basis = str(row["basis"])
            break
    if selected_basis is None:
        selected_basis = "phi3"
    selection = {
        "selected_basis": selected_basis,
        "min_coordinate_r2": min_coordinate_r2,
        "full_d_close_fraction": full_d_close_fraction,
        "recovery_target": recovery_target,
        f"oracle_rank_{args.rank}_recovery": oracle_rank,
        "oracle_full_d_recovery": oracle.get("full_d"),
    }

    _atomic_write_csv(args.out_root / "basis_results.csv", basis_results)
    _atomic_write_csv(args.out_root / "coordinate_predictions.csv", predictions_rows)
    report = build_report(
        args=args,
        checkpoint_sha=checkpoint_sha,
        basis_results=basis_results,
        selection=selection,
        oracle=oracle,
    )
    (args.out_root / "RESULTS.md").write_text(report, encoding="utf-8")
    _atomic_write_json(
        args.out_root / "summary.json",
        {
            "protocol": PROGRAM_VERSION,
            "model_name": args.model_name,
            "history": args.history,
            "checkpoint_sha256": checkpoint_sha,
            "commitment_block_index": args.commitment_block_index,
            "m_star": args.rank,
            "hidden_size": args.hidden_size,
            "fit_count": len(fit_indices),
            "heldout_count": len(heldout_indices),
            "basis_results": basis_results,
            "oracle_recovery": oracle,
            "selection": selection,
        },
    )
    (args.out_root / "EXPERIMENT_COMPLETE").write_text(
        "complete\n", encoding="utf-8"
    )
    return {
        "selected_basis": selected_basis,
        "basis_results": basis_results,
        "oracle": oracle,
        "out_root": str(args.out_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4 coordinate-difference synthesis (strict bank)."
    )
    parser.add_argument("--model-name", default="frequency_color_circle")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--natural-metrics", type=Path, required=True)
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--ceiling-metrics", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--history", choices=("short", "long"), default="short")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument(
        "--commitment-block-index",
        type=int,
        default=DEFAULT_COMMITMENT_BLOCK_INDEX,
    )
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    args = parser.parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "status": "ok",
                "selected_basis": result["selected_basis"],
                "basis_results": result["basis_results"],
                "oracle": result["oracle"],
                "out_root": str(args.out_root),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
