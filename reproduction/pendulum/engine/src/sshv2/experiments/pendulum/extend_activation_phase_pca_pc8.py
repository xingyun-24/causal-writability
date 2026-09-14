#!/usr/bin/env python3
"""Extend cached target-specific activation PCA directions to PC1--PC8.

The PCA basis is recomputed from the frozen 12-receiver fit bank without model
inference.  The cached 32-receiver dense geometry bank is then projected onto
PC1--PC8.  For every coordinate, a dominant pure harmonic is selected from
orders 1--8 by leave-one-out PRESS R2; weak selections are reported explicitly.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sshv2.experiments.pendulum.mechanism_pca import (
    _write_csv,
    _write_json,
    manifest_sha256,
    read_jsonl,
)


PROGRAM_VERSION = "pendulum_target_specific_phase_pca_pc8_v1"
PCA_RANK = 8
HARMONIC_CANDIDATE_ORDERS = tuple(range(1, 9))
RELIABLE_FULL_R2 = 0.50
RELIABLE_PRESS_R2 = 0.40
COORDINATE_SYMBOLS = tuple("abcdefgh")


def _fit_harmonic(
    phases: np.ndarray,
    values: np.ndarray,
    harmonic_order: int,
) -> dict[str, Any]:
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
    residuals = values - fitted
    residual_sum_squares = float(np.sum(np.square(residuals)))
    total_sum_squares = float(np.sum(np.square(values - values.mean())))
    r_squared = (
        1.0 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 0.0
        else float("nan")
    )

    gram_inverse = np.linalg.pinv(design.T @ design)
    leverage = np.einsum(
        "ij,jk,ik->i",
        design,
        gram_inverse,
        design,
        dtype=np.float64,
    )
    press_residuals = residuals / np.maximum(1.0 - leverage, 1e-12)
    press = float(np.sum(np.square(press_residuals)))
    press_r_squared = (
        1.0 - press / total_sum_squares
        if total_sum_squares > 0.0
        else float("nan")
    )
    return {
        "harmonic_order": harmonic_order,
        "intercept": float(coefficients[0]),
        "cosine_coefficient": float(coefficients[1]),
        "sine_coefficient": float(coefficients[2]),
        "amplitude": float(math.hypot(coefficients[1], coefficients[2])),
        "r_squared": r_squared,
        "press_r_squared": press_r_squared,
        "residual_sum_squares": residual_sum_squares,
        "press": press,
    }


def _harmonic_values(
    fit: Mapping[str, Any],
    phases: np.ndarray,
) -> np.ndarray:
    angular_phase = int(fit["harmonic_order"]) * phases
    return (
        float(fit["intercept"])
        + float(fit["cosine_coefficient"]) * np.cos(angular_phase)
        + float(fit["sine_coefficient"]) * np.sin(angular_phase)
    )


def _select_harmonic(
    phases: np.ndarray,
    values: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = [
        _fit_harmonic(phases, values, harmonic_order)
        for harmonic_order in HARMONIC_CANDIDATE_ORDERS
    ]
    selected = max(
        candidates,
        key=lambda fit: (
            float(fit["press_r_squared"]),
            -int(fit["harmonic_order"]),
        ),
    )
    selected = dict(selected)
    selected["reliable"] = bool(
        float(selected["r_squared"]) >= RELIABLE_FULL_R2
        and float(selected["press_r_squared"]) >= RELIABLE_PRESS_R2
    )
    selected["selection_method"] = (
        "maximum leave-one-out PRESS R2 among pure harmonic orders 1--8"
    )
    return selected, candidates


def _fit_rank8_pca(
    directions: np.ndarray,
    components_path: Path,
    singular_values_path: Path,
    chunk_values: int,
) -> tuple[np.memmap, np.ndarray]:
    if directions.ndim != 4 or directions.shape[0] != 12:
        raise ValueError(
            "expected 12 fit directions with shape [receiver,step,token,hidden]"
        )
    matrix = directions.reshape(directions.shape[0], -1)
    flat_size = matrix.shape[1]
    gram = np.zeros((matrix.shape[0], matrix.shape[0]), dtype=np.float64)
    for start in range(0, flat_size, chunk_values):
        stop = min(flat_size, start + chunk_values)
        chunk = np.asarray(matrix[:, start:stop], dtype=np.float32)
        gram += chunk @ chunk.T

    eigenvalues, left_vectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    left_vectors = left_vectors[:, order]
    singular_values = np.sqrt(eigenvalues)
    positive = singular_values > max(singular_values[0] * 1e-10, 1e-12)
    if int(positive.sum()) < PCA_RANK:
        raise ValueError(
            f"only {int(positive.sum())} non-degenerate components; "
            f"rank {PCA_RANK} requested"
        )

    direction_shape = tuple(int(value) for value in directions.shape[1:])
    components_path.parent.mkdir(parents=True, exist_ok=True)
    components = np.lib.format.open_memmap(
        components_path,
        mode="w+",
        dtype=np.float32,
        shape=(PCA_RANK, *direction_shape),
    ).reshape(PCA_RANK, flat_size)
    for start in range(0, flat_size, chunk_values):
        stop = min(flat_size, start + chunk_values)
        chunk = np.asarray(matrix[:, start:stop], dtype=np.float32)
        components[:, start:stop] = (
            left_vectors[:, :PCA_RANK].T @ chunk
        ) / singular_values[:PCA_RANK, None]
    components.flush()
    np.save(singular_values_path, singular_values)
    return components, singular_values


def _project_geometry(
    directions: np.ndarray,
    components: np.ndarray,
    chunk_values: int,
) -> np.ndarray:
    if directions.ndim != 4 or directions.shape[0] != 32:
        raise ValueError(
            "expected 32 geometry directions with shape "
            "[receiver,step,token,hidden]"
        )
    matrix = directions.reshape(directions.shape[0], -1)
    basis = components.reshape(PCA_RANK, -1)
    if matrix.shape[1] != basis.shape[1]:
        raise ValueError("geometry directions and PCA components do not match")
    coefficients = np.zeros((directions.shape[0], PCA_RANK), dtype=np.float64)
    for start in range(0, matrix.shape[1], chunk_values):
        stop = min(matrix.shape[1], start + chunk_values)
        coefficients += np.asarray(
            matrix[:, start:stop], dtype=np.float32
        ) @ np.asarray(basis[:, start:stop], dtype=np.float32).T
    return coefficients


def _canonicalize_and_fit(
    phases: np.ndarray,
    coefficients: np.ndarray,
    fixed_display_signs: Sequence[float],
) -> tuple[
    np.ndarray,
    tuple[float, ...],
    list[dict[str, Any]],
    list[list[dict[str, Any]]],
]:
    display_signs = np.ones(PCA_RANK, dtype=np.float64)
    if len(fixed_display_signs) > PCA_RANK:
        raise ValueError("too many fixed PCA display signs")
    for component_index, value in enumerate(fixed_display_signs):
        sign = float(value)
        if sign not in (-1.0, 1.0):
            raise ValueError("PCA display signs must be -1 or 1")
        coefficients[:, component_index] *= sign
        display_signs[component_index] = sign

    selected_fits: list[dict[str, Any]] = []
    all_candidates: list[list[dict[str, Any]]] = []
    for component_index in range(PCA_RANK):
        selected, candidates = _select_harmonic(
            phases, coefficients[:, component_index]
        )
        if component_index >= len(fixed_display_signs):
            harmonic_terms = np.asarray(
                [
                    selected["cosine_coefficient"],
                    selected["sine_coefficient"],
                ],
                dtype=np.float64,
            )
            dominant = int(np.argmax(np.abs(harmonic_terms)))
            if harmonic_terms[dominant] < 0.0:
                coefficients[:, component_index] *= -1.0
                display_signs[component_index] = -1.0
                selected, candidates = _select_harmonic(
                    phases, coefficients[:, component_index]
                )
        selected_fits.append(selected)
        all_candidates.append(candidates)
    return (
        coefficients,
        tuple(float(value) for value in display_signs),
        selected_fits,
        all_candidates,
    )


def _plot_results(
    out_root: Path,
    model_name: str,
    target_label: str,
    block_index: int,
    phases: np.ndarray,
    coefficients: np.ndarray,
    selected_fits: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_root = out_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    model_label = model_name.replace("_", "-")
    target_frequency_label = f"{target_label}-frequency"
    block_label = f"after block {block_index}"
    order = np.argsort(phases)
    sorted_phases = phases[order]
    sorted_coefficients = coefficients[order]
    dense_phase = np.linspace(0.0, 2.0 * math.pi, 1441)

    figure, axis = plt.subplots(figsize=(9.6, 8.0))
    scatter = axis.scatter(
        sorted_coefficients[:, 0],
        sorted_coefficients[:, 1],
        c=sorted_phases,
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
        f"{model_label}: {target_frequency_label} directions at {block_label} "
        f"in PC1--PC2 (n={len(phases)})\n"
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

    for component_index in range(PCA_RANK):
        pc_number = component_index + 1
        symbol = COORDINATE_SYMBOLS[component_index]
        fit = selected_fits[component_index]
        harmonic_order = int(fit["harmonic_order"])
        reliable = bool(fit["reliable"])
        fit_strength = "selected" if reliable else "weak best"
        line_style = "-" if reliable else "--"

        figure, axis = plt.subplots(figsize=(10.4, 6.5))
        axis.scatter(
            sorted_phases,
            sorted_coefficients[:, component_index],
            s=54,
            alpha=0.80,
            color="tab:blue",
            label=(
                f"held-out {target_frequency_label} receivers "
                f"(n={len(phases)})"
            ),
        )
        axis.plot(
            dense_phase,
            _harmonic_values(fit, dense_phase),
            linewidth=2.5,
            linestyle=line_style,
            color="tab:orange",
            label=(
                f"{fit_strength} order-{harmonic_order} harmonic: "
                rf"$R^2={float(fit['r_squared']):.4f}$, "
                rf"$R^2_{{\mathrm{{PRESS}}}}="
                rf"{float(fit['press_r_squared']):.4f}$"
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
            f"{model_label}: {target_frequency_label} PC{pc_number} "
            f"coordinate at {block_label} vs. boundary phase\n"
            f"dominant harmonic selected from orders 1--8 by PRESS"
        )
        axis.grid(alpha=0.2)
        axis.legend(loc="best", frameon=True)
        figure.tight_layout()
        for extension in ("png", "pdf"):
            figure.savefig(
                plot_root / f"pc{pc_number}_coordinate_vs_phase.{extension}",
                dpi=200 if extension == "png" else None,
            )
        plt.close(figure)


def _coordinate_rows(
    rows: Sequence[Mapping[str, Any]],
    coefficients: np.ndarray,
) -> list[dict[str, Any]]:
    result = []
    for row, coordinate_values in zip(rows, coefficients, strict=True):
        output = {
            "receiver_id": row["receiver_id"],
            "target_label": row["target_label"],
            "omega_true": row["omega_true"],
            "phase_index": row["phase_index"],
            "generation_seed": row["generation_seed"],
            "boundary_phase_rad": row["boundary_phase_rad"],
        }
        for component_index, (symbol, value) in enumerate(
            zip(COORDINATE_SYMBOLS, coordinate_values, strict=True),
            start=1,
        ):
            output[f"pc{component_index}_coordinate_{symbol}"] = float(value)
        result.append(output)
    return result


def run(args: argparse.Namespace) -> None:
    required = (
        args.source_root / "directions.npy",
        args.source_root / "fit_receiver_manifest.jsonl",
        args.source_root / "geometry_dense" / "directions.npy",
        args.source_root / "geometry_dense" / "receiver_manifest.jsonl",
        args.source_root / "geometry_dense" / "harmonic_fit.json",
        args.source_root / "run_state.json",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    source_state = json.loads(
        (args.source_root / "run_state.json").read_text(encoding="utf-8")
    )
    source_fit = json.loads(
        (args.source_root / "geometry_dense" / "harmonic_fit.json").read_text(
            encoding="utf-8"
        )
    )
    fixed_display_signs = tuple(
        float(value) for value in source_fit["pca_component_display_signs"]
    )
    if len(fixed_display_signs) != 4:
        raise ValueError("source run must provide four frozen display signs")
    fit_rows = read_jsonl(args.source_root / "fit_receiver_manifest.jsonl")
    geometry_rows = read_jsonl(
        args.source_root / "geometry_dense" / "receiver_manifest.jsonl"
    )
    if len(fit_rows) != 12 or len(geometry_rows) != 32:
        raise ValueError("expected 12 fit receivers and 32 geometry receivers")
    if {str(row["target_label"]) for row in geometry_rows} != {
        str(source_state["target_label"])
    }:
        raise ValueError("geometry target does not match source run state")

    phases = np.asarray(
        [float(row["boundary_phase_rad"]) for row in geometry_rows],
        dtype=np.float64,
    )
    expected_phases = (
        2.0 * math.pi * np.arange(32, dtype=np.float64) / 32.0
    )
    if not np.allclose(phases, expected_phases, atol=1e-12, rtol=0.0):
        raise ValueError("geometry phases are not the frozen uniform grid")

    args.out_root.mkdir(parents=True, exist_ok=True)
    state = {
        "protocol": PROGRAM_VERSION,
        "status": "running",
        "source_root": str(args.source_root),
        "source_protocol": source_state["protocol"],
        "source_fit_manifest_sha256": manifest_sha256(fit_rows),
        "source_geometry_manifest_sha256": manifest_sha256(geometry_rows),
        "source_pc1_pc4_display_signs": list(fixed_display_signs),
        "model_name": source_state["model_name"],
        "target_label": source_state["target_label"],
        "block_index_zero_based": source_state["block_index_zero_based"],
        "fit_receivers": len(fit_rows),
        "geometry_receivers": len(geometry_rows),
        "pca_rank": PCA_RANK,
        "harmonic_candidate_orders": list(HARMONIC_CANDIDATE_ORDERS),
        "harmonic_selection": (
            "maximum leave-one-out PRESS R2 among pure harmonic orders 1--8"
        ),
        "reliable_full_r_squared_threshold": RELIABLE_FULL_R2,
        "reliable_press_r_squared_threshold": RELIABLE_PRESS_R2,
        "pca_chunk_values": args.pca_chunk_values,
    }
    _write_json(args.out_root / "run_state.json", state)

    fit_directions = np.load(
        args.source_root / "directions.npy", mmap_mode="r"
    )
    components, singular_values = _fit_rank8_pca(
        fit_directions,
        args.out_root / "pca" / "components.npy",
        args.out_root / "pca" / "singular_values.npy",
        args.pca_chunk_values,
    )
    geometry_directions = np.load(
        args.source_root / "geometry_dense" / "directions.npy", mmap_mode="r"
    )
    coefficients = _project_geometry(
        geometry_directions,
        components,
        args.pca_chunk_values,
    )
    canonical_signs = np.asarray(
        [float(row["canonical_direction_sign"]) for row in geometry_rows],
        dtype=np.float64,
    )
    coefficients *= canonical_signs[:, None]
    (
        coefficients,
        display_signs,
        selected_fits,
        all_candidates,
    ) = _canonicalize_and_fit(phases, coefficients, fixed_display_signs)

    np.save(args.out_root / "pca" / "geometry_coefficients.npy", coefficients)
    energy = np.square(np.asarray(singular_values, dtype=np.float64))
    energy_ratio = energy / energy.sum()
    cumulative = np.cumsum(energy_ratio)
    fit_payload = {
        f"pc{component_index + 1}_coordinate_fit": {
            **selected_fits[component_index],
            "candidate_fits": all_candidates[component_index],
        }
        for component_index in range(PCA_RANK)
    }
    _write_json(
        args.out_root / "harmonic_fit.json",
        {
            "protocol": PROGRAM_VERSION,
            "model_name": source_state["model_name"],
            "target_label": source_state["target_label"],
            "block_index_zero_based": source_state["block_index_zero_based"],
            "pca_fit_receivers": len(fit_rows),
            "geometry_heldout_receivers": len(geometry_rows),
            "pca_rank": PCA_RANK,
            "direction_definition_extracted": "conflict-minus-aligned",
            "direction_definition_plotted": (
                "high-appearance-cue-minus-low-appearance-cue"
            ),
            "pca_component_display_signs": list(display_signs),
            "boundary_phase_definition": (
                "atan2(-angular_velocity_star/omega_true, theta_star) mod 2pi"
            ),
            "harmonic_candidate_orders": list(HARMONIC_CANDIDATE_ORDERS),
            "harmonic_selection": state["harmonic_selection"],
            "reliable_full_r_squared_threshold": RELIABLE_FULL_R2,
            "reliable_press_r_squared_threshold": RELIABLE_PRESS_R2,
            "singular_values": [float(value) for value in singular_values],
            "explained_energy_ratio": [float(value) for value in energy_ratio],
            "cumulative_explained_energy_ratio": [
                float(value) for value in cumulative
            ],
            **fit_payload,
        },
    )
    _write_csv(
        args.out_root / "coordinates.csv",
        _coordinate_rows(geometry_rows, coefficients),
    )
    _plot_results(
        args.out_root,
        str(source_state["model_name"]),
        str(source_state["target_label"]),
        int(source_state["block_index_zero_based"]),
        phases,
        coefficients,
        selected_fits,
    )
    _write_json(
        args.out_root / "progress.json",
        {
            "status": "complete",
            "fit_receivers": len(fit_rows),
            "geometry_receivers": len(geometry_rows),
            "pca_rank": PCA_RANK,
            "plots": PCA_RANK + 1,
        },
    )
    _write_json(args.out_root / "run_state.json", {**state, "status": "complete"})
    selected_summary = ",".join(
        f"PC{index + 1}=k{fit['harmonic_order']}"
        f"/R2{fit['r_squared']:.3f}"
        f"/PRESS{fit['press_r_squared']:.3f}"
        for index, fit in enumerate(selected_fits)
    )
    print(f"complete {args.out_root} {selected_summary}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--pca-chunk-values", type=int, default=262_144)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.pca_chunk_values < 1:
        raise ValueError("pca chunk size must be positive")
    run(args)


if __name__ == "__main__":
    main()
