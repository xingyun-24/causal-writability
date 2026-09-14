#!/usr/bin/env python3
"""Refit an existing target-specific PC4 phase plot with two cycles."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from sshv2.experiments.pendulum.activation_phase_pca import (
    HARMONIC_ORDERS,
    PROGRAM_VERSION,
    _harmonic_fit,
    _harmonic_values,
)
from sshv2.experiments.pendulum.mechanism_pca import _write_json


def _read_coordinates(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 32:
        raise ValueError(f"expected 32 held-out coordinates, got {len(rows)}")
    return rows


def refit(result_root: Path) -> dict[str, Any]:
    coordinates_path = result_root / "coordinates.csv"
    fit_path = result_root / "harmonic_fit.json"
    if not coordinates_path.is_file():
        raise FileNotFoundError(coordinates_path)
    if not fit_path.is_file():
        raise FileNotFoundError(fit_path)

    rows = _read_coordinates(coordinates_path)
    payload = json.loads(fit_path.read_text(encoding="utf-8"))
    phases = np.asarray(
        [float(row["boundary_phase_rad"]) for row in rows],
        dtype=np.float64,
    )
    pc4 = np.asarray(
        [float(row["pc4_coordinate_d"]) for row in rows],
        dtype=np.float64,
    )
    order = np.argsort(phases)
    phases = phases[order]
    pc4 = pc4[order]
    harmonic_order = HARMONIC_ORDERS[3]
    fit = _harmonic_fit(phases, pc4, harmonic_order)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_label = str(payload["model_name"]).replace("_", "-")
    target_label = f"{payload['target_label']}-frequency"
    block_label = f"after block {int(payload['block_index_zero_based'])}"
    dense_phase = np.linspace(0.0, 2.0 * math.pi, 721)

    figure, axis = plt.subplots(figsize=(10.4, 6.5))
    axis.scatter(
        phases,
        pc4,
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
            rf"order-2 harmonic fit: $R^2={float(fit['r_squared']):.4f}$, "
            rf"$B_2={float(fit['amplitude']):.3f}$"
        ),
    )
    axis.set_xlim(0.0, 2.0 * math.pi)
    axis.set_xticks(
        [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi, 2.0 * math.pi],
        labels=["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"],
    )
    axis.set_xlabel(r"Boundary physical phase $\varphi_j^*$")
    axis.set_ylabel(r"$d_j$")
    axis.set_title(
        f"{model_label}: {target_label} PC4 coordinate at {block_label} "
        "vs. boundary phase\norder-2 harmonic (two cycles over $0$--$2\\pi$)"
    )
    axis.grid(alpha=0.2)
    axis.legend(loc="best", frameon=True)
    figure.tight_layout()
    plot_root = result_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(
            plot_root / f"pc4_coordinate_vs_phase.{extension}",
            dpi=200 if extension == "png" else None,
        )
    plt.close(figure)

    payload.update(
        {
            "protocol": PROGRAM_VERSION,
            "harmonic_fit_scope": (
                "independent target-specific fits: first harmonic for PC1--PC3 "
                "and second harmonic for PC4"
            ),
            "pca_coordinate_harmonic_orders": list(HARMONIC_ORDERS),
            "pc4_coordinate_fit": fit,
        }
    )
    for component_index in range(1, 4):
        payload[f"pc{component_index}_coordinate_fit"].setdefault(
            "harmonic_order", 1
        )
    _write_json(fit_path, payload)
    run_state_path = result_root / "run_state.json"
    if run_state_path.is_file():
        run_state = json.loads(run_state_path.read_text(encoding="utf-8"))
        run_state["protocol"] = PROGRAM_VERSION
        run_state["pca_coordinate_harmonic_orders"] = list(HARMONIC_ORDERS)
        _write_json(run_state_path, run_state)
    _write_json(
        result_root / "pc4_refit_state.json",
        {
            "status": "complete",
            "protocol": PROGRAM_VERSION,
            "source_coordinates": str(coordinates_path),
            "harmonic_order": harmonic_order,
            "r_squared": float(fit["r_squared"]),
            "amplitude": float(fit["amplitude"]),
        },
    )
    return fit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    fit = refit(args.result_root)
    print(
        f"complete {args.result_root}: order={fit['harmonic_order']} "
        f"R2={fit['r_squared']:.6f} amplitude={fit['amplitude']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
