#!/usr/bin/env python3
"""Target-specific activation PCA at layer-scan-selected Pendulum blocks.

Fit one uncentered PCA basis using only the frozen fit receivers for one
physical-frequency target, then project 32 uniformly spaced held-out boundary
phases and draw PC1--PC2 plus phase fits for PC1--PC4.  PC1--PC3 use the first
harmonic; PC4 uses the second harmonic because it completes two cycles over a
physical phase interval of 0--2pi.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sshv2.experiments.pendulum.mechanism_pca import (
    _build_dense_geometry_manifest,
    _extract_directions,
    _fit_chunked_pca,
    _freeze_manifest,
    _geometry_coordinates,
    _load_runtime,
    _sha256_file,
    _write_csv,
    _write_json,
    manifest_sha256,
    read_jsonl,
    validate_frozen_receiver_manifest,
)


PROGRAM_VERSION = "pendulum_target_specific_phase_pca_v2"
SPECIAL_BLOCKS = {"low": 26, "high": 22}
COORDINATE_SPECS = (
    ("pc1", "a", "pc1_coordinate_a"),
    ("pc2", "b", "pc2_coordinate_b"),
    ("pc3", "c", "pc3_coordinate_c"),
    ("pc4", "d", "pc4_coordinate_d"),
)
HARMONIC_ORDERS = (1, 1, 1, 2)


def _harmonic_fit(
    phases: np.ndarray,
    values: np.ndarray,
    harmonic_order: int,
) -> dict[str, Any]:
    if harmonic_order < 1:
        raise ValueError("harmonic order must be positive")
    angular_phase = harmonic_order * phases
    design = np.column_stack(
        (
            np.ones(len(phases), dtype=np.float64),
            np.cos(angular_phase),
            np.sin(angular_phase),
        )
    )
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    fitted = design @ coefficients
    residual = float(np.sum((values - fitted) ** 2))
    total = float(np.sum((values - values.mean()) ** 2))
    r_squared = 1.0 - residual / total if total > 0.0 else float("nan")
    return {
        "harmonic_order": harmonic_order,
        "intercept": float(coefficients[0]),
        "cosine_coefficient": float(coefficients[1]),
        "sine_coefficient": float(coefficients[2]),
        "amplitude": float(math.hypot(coefficients[1], coefficients[2])),
        "r_squared": r_squared,
    }


def _harmonic_values(
    fit: Mapping[str, Any],
    phases: np.ndarray,
) -> np.ndarray:
    harmonic_order = int(fit["harmonic_order"])
    angular_phase = harmonic_order * phases
    return (
        float(fit["intercept"])
        + float(fit["cosine_coefficient"]) * np.cos(angular_phase)
        + float(fit["sine_coefficient"]) * np.sin(angular_phase)
    )


def _component_harmonic_fits(
    phases: np.ndarray,
    coefficients: np.ndarray,
) -> tuple[dict[str, Any], ...]:
    if coefficients.ndim != 2 or coefficients.shape[1] != len(HARMONIC_ORDERS):
        raise ValueError("expected one PCA-coordinate column per harmonic order")
    return tuple(
        _harmonic_fit(
            phases,
            coefficients[:, component_index],
            harmonic_order,
        )
        for component_index, harmonic_order in enumerate(HARMONIC_ORDERS)
    )


def _target_fit_rows(
    source_root: Path,
    target_label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows = read_jsonl(source_root / "receiver_manifest.jsonl")
    validate_frozen_receiver_manifest(
        all_rows,
        target="frequency",
        expected_fit=24,
        expected_heldout=8,
    )
    selected = [
        dict(row)
        for row in all_rows
        if row["split"] == "fit" and row["target_label"] == target_label
    ]
    if len(selected) != 12:
        raise ValueError(
            f"expected 12 frozen {target_label} fit receivers, got {len(selected)}"
        )
    if {int(row["phase_index"]) for row in selected} != set(range(6)):
        raise ValueError("target fit bank must contain phases 0--5")
    if any(row["split"] != "fit" for row in selected):
        raise AssertionError("held-out receiver leaked into PCA fit")
    return all_rows, selected


def _target_geometry_rows(
    args: argparse.Namespace,
    data_config: Any,
    geometry_root: Path,
) -> list[dict[str, Any]]:
    all_rows = _build_dense_geometry_manifest(args, data_config, geometry_root)
    selected = [
        dict(row)
        for row in all_rows
        if row["target_label"] == args.target_label
    ]
    if len(selected) != args.geometry_phase_count:
        raise ValueError(
            f"expected {args.geometry_phase_count} dense {args.target_label} "
            f"receivers, got {len(selected)}"
        )
    phases = np.asarray(
        [float(row["boundary_phase_rad"]) for row in selected],
        dtype=np.float64,
    )
    expected = (
        2.0
        * math.pi
        * np.arange(args.geometry_phase_count, dtype=np.float64)
        / args.geometry_phase_count
    )
    if not np.allclose(phases, expected, atol=1e-12, rtol=0.0):
        raise ValueError("dense boundary phases are not uniform and exact")
    return selected


def _plot_target_geometry(
    args: argparse.Namespace,
    geometry_root: Path,
    rows: Sequence[Mapping[str, Any]],
    coefficients: np.ndarray,
    fits: Sequence[Mapping[str, Any]],
    display_signs: Sequence[float],
    singular_values: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if coefficients.shape != (len(rows), 4):
        raise ValueError("expected four PCA coordinates for every receiver")
    phases = np.asarray(
        [float(row["boundary_phase_rad"]) for row in rows],
        dtype=np.float64,
    )
    order = np.argsort(phases)
    phases = phases[order]
    coefficients = coefficients[order]
    dense_phase = np.linspace(0.0, 2.0 * math.pi, 721)
    plot_root = geometry_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    model_label = args.model_name.replace("_", "-")
    target_label = f"{args.target_label}-frequency"
    block_label = f"after block {args.block_index}"

    figure, axis = plt.subplots(figsize=(9.6, 8.0))
    scatter = axis.scatter(
        coefficients[:, 0],
        coefficients[:, 1],
        c=phases,
        cmap="twilight",
        vmin=0.0,
        vmax=2.0 * math.pi,
        s=68,
        alpha=0.88,
        linewidths=0.7,
        edgecolors="black",
    )
    axis.axhline(0.0, color="tab:blue", alpha=0.28, linewidth=1.0)
    axis.axvline(0.0, color="tab:blue", alpha=0.28, linewidth=1.0)
    axis.grid(alpha=0.18)
    axis.set_xlabel(r"$a_j = \langle d_j^{\mathrm{canonical}}, v_1\rangle$")
    axis.set_ylabel(r"$b_j = \langle d_j^{\mathrm{canonical}}, v_2\rangle$")
    axis.set_title(
        f"{model_label}: {target_label} directions at {block_label} "
        f"in PC1--PC2 (n={len(rows)})\n"
        r"color = boundary phase $\varphi_j^* = "
        r"\mathrm{atan2}(-\dot{\theta}_j^*/\omega_j,\theta_j^*)\ "
        r"\mathrm{mod}\ 2\pi$"
    )
    colorbar = figure.colorbar(scatter, ax=axis, pad=0.025)
    colorbar.set_label(r"boundary phase $\varphi_j^*$ (rad)")
    colorbar.set_ticks(
        [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi, 2.0 * math.pi],
        labels=["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"],
    )
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(
            plot_root / f"heldout_pc1_pc2_by_phase.{extension}",
            dpi=200 if extension == "png" else None,
        )
    plt.close(figure)

    for component_index, (fit_key, symbol, _column) in enumerate(
        COORDINATE_SPECS
    ):
        fit = fits[component_index]
        figure, axis = plt.subplots(figsize=(10.4, 6.5))
        axis.scatter(
            phases,
            coefficients[:, component_index],
            s=54,
            alpha=0.80,
            color="tab:blue",
            label=f"held-out {target_label} receivers (n={len(rows)})",
        )
        axis.plot(
            dense_phase,
            _harmonic_values(fit, dense_phase),
            linewidth=2.5,
            color="tab:orange",
            label=(
                rf"order-{int(fit['harmonic_order'])} harmonic fit: "
                rf"$R^2={float(fit['r_squared']):.4f}$, "
                rf"$B={float(fit['amplitude']):.3f}$"
            ),
        )
        axis.set_xlim(0.0, 2.0 * math.pi)
        axis.set_xticks(
            [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi, 2.0 * math.pi],
            labels=["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"],
        )
        axis.set_xlabel(r"Boundary physical phase $\varphi_j^*$")
        axis.set_ylabel(rf"${symbol}_j$")
        axis.set_title(
            f"{model_label}: {target_label} {fit_key.upper()} coordinate at "
            f"{block_label} vs. boundary phase"
        )
        axis.grid(alpha=0.2)
        axis.legend(loc="best", frameon=True)
        figure.tight_layout()
        for extension in ("png", "pdf"):
            figure.savefig(
                plot_root / f"{fit_key}_coordinate_vs_phase.{extension}",
                dpi=200 if extension == "png" else None,
            )
        plt.close(figure)

    coordinate_rows = []
    for row_index, manifest_index in enumerate(order):
        row = rows[int(manifest_index)]
        values = coefficients[row_index]
        coordinate_rows.append(
            {
                "receiver_id": row["receiver_id"],
                "target_label": row["target_label"],
                "omega_true": row["omega_true"],
                "phase_index": row["phase_index"],
                "generation_seed": row["generation_seed"],
                "boundary_phase_rad": row["boundary_phase_rad"],
                "pc1_coordinate_a": float(values[0]),
                "pc2_coordinate_b": float(values[1]),
                "pc3_coordinate_c": float(values[2]),
                "pc4_coordinate_d": float(values[3]),
            }
        )
    _write_csv(geometry_root / "coordinates.csv", coordinate_rows)

    energy = np.square(np.asarray(singular_values, dtype=np.float64))
    energy_ratio = energy / energy.sum()
    cumulative = np.cumsum(energy_ratio)
    fit_payload = {
        f"{fit_key}_coordinate_fit": dict(fits[component_index])
        for component_index, (fit_key, _symbol, _column) in enumerate(
            COORDINATE_SPECS
        )
    }
    _write_json(
        geometry_root / "harmonic_fit.json",
        {
            "protocol": PROGRAM_VERSION,
            "model_name": args.model_name,
            "target_label": args.target_label,
            "block_index_zero_based": args.block_index,
            "pca_fit_receivers": args.expected_fit_receivers,
            "geometry_heldout_receivers": len(rows),
            "direction_definition_extracted": "conflict-minus-aligned",
            "direction_definition_plotted": (
                "high-appearance-cue-minus-low-appearance-cue"
            ),
            "pca_component_display_signs": list(display_signs),
            "boundary_phase_definition": (
                "atan2(-angular_velocity_star/omega_true, theta_star) mod 2pi"
            ),
            "harmonic_fit_scope": (
                "independent target-specific fits: first harmonic for PC1--PC3 "
                "and second harmonic for PC4"
            ),
            "pca_coordinate_harmonic_orders": list(HARMONIC_ORDERS),
            "singular_values": [float(value) for value in singular_values],
            "explained_energy_ratio": [float(value) for value in energy_ratio],
            "cumulative_explained_energy_ratio": [
                float(value) for value in cumulative
            ],
            **fit_payload,
        },
    )


def run(args: argparse.Namespace) -> None:
    expected_block = SPECIAL_BLOCKS[args.target_label]
    if args.block_index != expected_block:
        raise ValueError(
            f"the frozen layer-scan result requires {args.target_label} "
            f"after block {expected_block}, not {args.block_index}"
        )
    required = (
        args.source_root / "receiver_manifest.jsonl",
        args.experiment_config,
        args.training_config,
        args.checkpoint,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.out_root.mkdir(parents=True, exist_ok=True)
    all_rows, generated_fit_rows = _target_fit_rows(
        args.source_root, args.target_label
    )
    fit_rows = _freeze_manifest(
        args.out_root / "fit_receiver_manifest.jsonl", generated_fit_rows
    )
    args.expected_fit_receivers = len(fit_rows)
    state = {
        "protocol": PROGRAM_VERSION,
        "model_name": args.model_name,
        "target_label": args.target_label,
        "block_index_zero_based": args.block_index,
        "history": args.history,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256_file(args.checkpoint),
        "source_manifest_sha256": manifest_sha256(all_rows),
        "fit_manifest_sha256": manifest_sha256(fit_rows),
        "fit_receivers": len(fit_rows),
        "geometry_phases": args.geometry_phase_count,
        "steps": args.steps,
        "condition_tokens": args.condition_tokens,
        "hidden_size": args.hidden_size,
        "pca_rank": max(args.ranks),
        "pca_coordinate_harmonic_orders": list(HARMONIC_ORDERS),
        "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    state_path = args.out_root / "run_state.json"
    if state_path.is_file():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        comparable = dict(previous)
        comparable.pop("status", None)
        if comparable != state:
            raise ValueError("existing phase-PCA run state differs from request")
    else:
        _write_json(state_path, {**state, "status": "running"})

    print(
        f"load {args.model_name} target={args.target_label} "
        f"after_block={args.block_index}",
        flush=True,
    )
    pipe, data_config = _load_runtime(args, fit_rows)
    fit_directions = _extract_directions(args, fit_rows, pipe, data_config)
    artifacts = _fit_chunked_pca(args, fit_rows, fit_directions)

    geometry_root = args.out_root / "geometry_dense"
    geometry_rows = _freeze_manifest(
        geometry_root / "receiver_manifest.jsonl",
        _target_geometry_rows(args, data_config, geometry_root),
    )
    geometry_args = argparse.Namespace(**vars(args))
    geometry_args.out_root = geometry_root
    geometry_directions = _extract_directions(
        geometry_args, geometry_rows, pipe, data_config
    )
    (
        coefficients,
        _pooled_first_harmonic_fits,
        display_signs,
    ) = _geometry_coordinates(args, geometry_rows, geometry_directions)
    phases = np.asarray(
        [float(row["boundary_phase_rad"]) for row in geometry_rows],
        dtype=np.float64,
    )
    fits = _component_harmonic_fits(phases, coefficients)
    _plot_target_geometry(
        args,
        geometry_root,
        geometry_rows,
        coefficients,
        fits,
        display_signs,
        artifacts.singular_values,
    )
    _write_json(state_path, {**state, "status": "complete"})
    _write_json(
        args.out_root / "progress.json",
        {
            "status": "complete",
            "fit_receivers": len(fit_rows),
            "geometry_receivers": len(geometry_rows),
            "plots": 5,
        },
    )
    print(f"complete {args.out_root}", flush=True)


def _parse_ranks(value: str) -> tuple[int, ...]:
    try:
        ranks = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("ranks must be comma-separated integers") from error
    if not ranks or any(rank < 1 for rank in ranks):
        raise argparse.ArgumentTypeError("ranks must be positive")
    return ranks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        choices=("frequency_color_circle", "frequency_color_shape"),
        required=True,
    )
    parser.add_argument("--target-label", choices=("low", "high"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--block-index", type=int, required=True)
    parser.add_argument("--history", choices=("short", "long"), default="long")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed-offset", type=int, default=23_000_000)
    parser.add_argument("--low-omega", type=float, default=2.6)
    parser.add_argument("--high-omega", type=float, default=5.8)
    parser.add_argument("--condition-tokens", type=int, default=1088)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--ranks", type=_parse_ranks, default=(1, 2, 3, 4))
    parser.add_argument("--pca-chunk-values", type=int, default=262_144)
    parser.add_argument("--geometry-phase-count", type=int, default=32)
    parser.add_argument("--geometry-noise-repeats", type=int, default=1)
    parser.add_argument("--phase-count", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    positive = (
        "steps",
        "condition_tokens",
        "hidden_size",
        "pca_chunk_values",
        "geometry_phase_count",
    )
    if any(int(getattr(args, name)) < 1 for name in positive):
        raise ValueError("all count and size arguments must be positive")
    if args.geometry_phase_count != 32:
        raise ValueError("the frozen dense phase scan requires 32 phases")
    if args.ranks != (1, 2, 3, 4):
        raise ValueError("the requested protocol requires PCA ranks 1,2,3,4")
    run(args)


if __name__ == "__main__":
    main()
