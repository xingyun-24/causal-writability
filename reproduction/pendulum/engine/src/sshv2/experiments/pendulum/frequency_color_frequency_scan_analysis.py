#!/usr/bin/env python3
"""Aggregate and plot the Pendulum color x physical-frequency scan."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


PROGRAM_VERSION = "frequency_color_frequency_scan_analysis_v2"
HISTORIES = ("short", "long")


# ======================================================================
# COMMON INPUT / ATOMIC OUTPUT UTILITIES
# Keep this block: every independent plot consumes the same frozen table.
# ======================================================================


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _verify_frozen(root: Path) -> dict[str, Any]:
    frozen_path = root / "FROZEN.json"
    frozen = _read_json(frozen_path)
    if frozen.get("status") != "frozen":
        raise ValueError(f"input is not frozen: {frozen_path}")
    artifacts = frozen.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError(f"missing frozen artifacts: {frozen_path}")
    for name, expected in artifacts.items():
        path = root / str(name)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(expected["size_bytes"]):
            raise ValueError(f"frozen size mismatch: {path}")
        if _sha256(path) != str(expected["sha256"]):
            raise ValueError(f"frozen sha256 mismatch: {path}")
    return {
        "frozen_sha256": _sha256(frozen_path),
        "artifact_count": len(artifacts),
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_savefig(fig: plt.Figure, path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    fig.savefig(temporary, **kwargs)
    os.replace(temporary, path)


def _experiment_identity(
    data: Mapping[str, Any],
    scan: Mapping[str, Any],
) -> tuple[str, str, tuple[str, ...]]:
    shapes = tuple(str(value) for value in scan.get("shapes", ()))
    pairing = str(data.get("model_pairing"))
    target = str(data.get("target"))
    if target != "frequency":
        raise ValueError(f"frequency scan cannot use target={target!r}")
    if pairing == "color":
        fixed_shape = str(data.get("fixed_shape"))
        expected_shapes = (fixed_shape,)
        model_name = f"frequency_color_{fixed_shape}"
        binding = (
            f"training binding: red+{fixed_shape} ↔ low frequency; "
            f"blue+{fixed_shape} ↔ high frequency"
        )
    elif pairing == "color_shape":
        expected_shapes = ("circle", "square")
        model_name = "frequency_color_shape"
        binding = (
            "training binding: red+circle ↔ low frequency; "
            "blue+square ↔ high frequency"
        )
    else:
        raise ValueError(f"unsupported scan pairing: {pairing!r}")
    if shapes != expected_shapes:
        raise ValueError(
            f"{model_name} requires scan shapes {expected_shapes}, got {shapes}"
        )
    return model_name, binding, shapes


def _load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("scan config must be a mapping")
    data = payload.get("data")
    scan = payload.get("scan")
    if not isinstance(data, dict) or not isinstance(scan, dict):
        raise ValueError("scan config must contain data and scan mappings")
    _experiment_identity(data, scan)
    return data, scan


def _dataset_metadata(dataset_root: Path) -> dict[str, dict[str, str]]:
    path = dataset_root / "videos" / "eval" / "metadata.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["sample_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"duplicate sample IDs: {path}")
    return by_id


def _finite_number(value: Any) -> float:
    if value is None:
        return float("nan")
    number = float(value)
    return number if math.isfinite(number) else float("nan")


def _checkpoint_context(
    experiment_root: Path,
    shapes: Sequence[str],
) -> dict[str, Any]:
    states: dict[str, dict[str, Any]] = {}
    for shape in shapes:
        for history in HISTORIES:
            key = f"{shape}_{history}"
            states[key] = _read_json(
                experiment_root / shape / history / "prediction" / "run_state.json"
            )
    short_hashes = {
        states[f"{shape}_short"]["checkpoint_sha256"] for shape in shapes
    }
    long_hashes = {
        states[f"{shape}_long"]["checkpoint_sha256"] for shape in shapes
    }
    if len(short_hashes) != 1 or len(long_hashes) != 1:
        raise ValueError("checkpoint differs across shape slices")
    stems = {Path(str(state["checkpoint"])).stem for state in states.values()}
    if stems != {"step-50000"}:
        raise ValueError(f"scan requires the 50k checkpoints, got {stems}")
    return {
        "short_checkpoint_sha256": short_hashes.pop(),
        "long_checkpoint_sha256": long_hashes.pop(),
        "checkpoint_label": "50k",
        "run_states": states,
    }


def _load_history(
    experiment_root: Path,
    dataset_root: Path,
    shape: str,
    history: str,
    render: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics_root = experiment_root / shape / history / "metrics"
    color_root = experiment_root / shape / history / "color_metrics"
    frozen = {
        "frequency": _verify_frozen(metrics_root),
        "color": _verify_frozen(color_root),
    }
    frequency_rows = _read_jsonl(metrics_root / "per_sample.jsonl")
    color_rows = _read_jsonl(color_root / "per_sample.jsonl")
    color_by_id = {str(row["sample_id"]): row for row in color_rows}
    metadata = _dataset_metadata(dataset_root)
    if set(color_by_id) != {str(row["sample_id"]) for row in frequency_rows}:
        raise ValueError(f"frequency/color sample IDs differ for {shape}/{history}")
    if set(metadata) != set(color_by_id):
        raise ValueError(f"dataset/metric sample IDs differ for {shape}/{history}")

    with np.load(metrics_root / "per_frame.npz", allow_pickle=False) as archive:
        frame_ids = archive["sample_id"].astype(str)
        detected = archive["detected"].astype(bool)
        x_px = archive["x_px"].astype(float)
        y_px = archive["y_px"].astype(float)
    frame_index = {sample_id: index for index, sample_id in enumerate(frame_ids)}
    if len(frame_index) != len(frame_ids) or set(frame_index) != set(metadata):
        raise ValueError(f"per-frame sample IDs differ for {shape}/{history}")

    width = int(render["width"])
    height = int(render["height"])
    pivot_x = float(render["pivot_x"])
    pivot_y = float(render["pivot_y"])
    output: list[dict[str, Any]] = []
    for row in frequency_rows:
        sample_id = str(row["sample_id"])
        meta = metadata[sample_id]
        color = color_by_id[sample_id]
        index = frame_index[sample_id]
        valid_frames = (
            detected[index]
            & np.isfinite(x_px[index])
            & np.isfinite(y_px[index])
        )
        if int(valid_frames.sum()) >= 58:
            dx = x_px[index, valid_frames] / float(width - 1) - pivot_x
            dy = y_px[index, valid_frames] / float(height - 1) - pivot_y
            length_hat = float(np.median(np.hypot(dx, dy)))
        else:
            length_hat = float("nan")
        output.append(
            {
                "shape": shape,
                "history": history,
                "sample_id": sample_id,
                "phase_index": int(meta["phase_index"]),
                "diffusion_repeat": int(meta["diffusion_repeat"]),
                "input_alpha": float(row["test_color_alpha_target"]),
                "omega_true": float(row["omega_true"]),
                "omega_hat": _finite_number(row.get("omega_hat")),
                "frequency_valid": bool(row["valid"]),
                "amplitude_hat": _finite_number(row.get("amplitude_hat")),
                "amplitude_true": float(row["amplitude_true"]),
                "length_hat": length_hat,
                "length_true": float(render["length"]),
                "alpha_hat": _finite_number(color.get("alpha_hat")),
                "color_valid": bool(color["color_valid"]),
                "color_line_distance_rgb": _finite_number(
                    color.get("line_distance_rgb")
                ),
            }
        )
    return output, frozen


def load_enriched_rows(
    experiment_root: Path,
    dataset_base: Path,
    config_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data, scan = _load_config(config_path)
    model_name, training_binding, shapes = _experiment_identity(data, scan)
    render = data["render"]
    expected = (
        len(scan["color_alphas"])
        * len(scan["frequencies_rad_s"])
        * int(scan["phase_count"])
        * int(scan["diffusion_repeats"])
    )
    rows: list[dict[str, Any]] = []
    frozen_inputs: dict[str, Any] = {}
    for shape in shapes:
        dataset_root = dataset_base / shape
        for history in HISTORIES:
            part, frozen = _load_history(
                experiment_root, dataset_root, shape, history, render
            )
            if len(part) != expected:
                raise ValueError(
                    f"{shape}/{history}: expected {expected}, got {len(part)}"
                )
            rows.extend(part)
            frozen_inputs[f"{shape}_{history}"] = frozen
    return rows, {
        "data": data,
        "scan": scan,
        "model_name": model_name,
        "training_binding": training_binding,
        "shapes": shapes,
        "checkpoint": _checkpoint_context(experiment_root, shapes),
        "frozen_inputs": frozen_inputs,
        "config_sha256": _sha256(config_path),
    }


# ======================================================================
# COMMON GRID AGGREGATION / PAIRED DIFFERENCES
# Keep this block: all plot blocks read these auditable CSV-ready tables.
# ======================================================================


def _finite(values: Sequence[Any]) -> np.ndarray:
    array = np.asarray([_finite_number(value) for value in values], dtype=float)
    return array[np.isfinite(array)]


def _median(values: Sequence[Any]) -> float:
    array = _finite(values)
    return float(np.median(array)) if array.size else float("nan")


def _quantile(values: Sequence[Any], q: float) -> float:
    array = _finite(values)
    return float(np.quantile(array, q)) if array.size else float("nan")


def build_grid_tables(
    rows: Sequence[Mapping[str, Any]],
    scan: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    alphas = tuple(float(value) for value in scan["color_alphas"])
    omegas = tuple(float(value) for value in scan["frequencies_rad_s"])
    shapes = tuple(str(value) for value in scan["shapes"])
    expected_per_cell = int(scan["phase_count"]) * int(
        scan["diffusion_repeats"]
    )
    low_center, high_center = (
        float(value) for value in scan["color_frequency_centers_rad_s"]
    )
    low_edge = float(scan["low_band_rad_s"][1])
    high_edge = float(scan["high_band_rad_s"][0])
    grouped: dict[tuple[str, str, float, float], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        grouped[
            (
                str(row["shape"]),
                str(row["history"]),
                float(row["input_alpha"]),
                float(row["omega_true"]),
            )
        ].append(row)

    cells: list[dict[str, Any]] = []
    for shape in shapes:
        for history in HISTORIES:
            for omega_true in omegas:
                for alpha in alphas:
                    group = grouped[(shape, history, alpha, omega_true)]
                    if len(group) != expected_per_cell:
                        raise ValueError(
                            f"{shape}/{history}/alpha={alpha}/omega={omega_true}: "
                            f"expected {expected_per_cell}, got {len(group)}"
                        )
                    valid_frequency = [
                        row
                        for row in group
                        if bool(row["frequency_valid"])
                        and math.isfinite(float(row["omega_hat"]))
                    ]
                    omega_hat = _finite(
                        [row["omega_hat"] for row in valid_frequency]
                    )
                    omega_color = low_center + alpha * (
                        high_center - low_center
                    )
                    dominance = [
                        (
                            abs(float(row["omega_hat"]) - omega_true)
                            - abs(float(row["omega_hat"]) - omega_color)
                        )
                        / (
                            abs(float(row["omega_hat"]) - omega_true)
                            + abs(float(row["omega_hat"]) - omega_color)
                            + 1e-12
                        )
                        for row in valid_frequency
                    ]
                    valid_color = [
                        row
                        for row in group
                        if bool(row["color_valid"])
                        and math.isfinite(float(row["alpha_hat"]))
                    ]
                    cells.append(
                        {
                            "shape": shape,
                            "history": history,
                            "input_alpha": alpha,
                            "omega_true": omega_true,
                            "N_total": len(group),
                            "N_frequency_valid": len(valid_frequency),
                            "frequency_validity_rate": len(valid_frequency)
                            / len(group),
                            "median_omega_hat": _median(omega_hat),
                            "q25_omega_hat": _quantile(omega_hat, 0.25),
                            "q75_omega_hat": _quantile(omega_hat, 0.75),
                            "iqr_omega_hat": _quantile(omega_hat, 0.75)
                            - _quantile(omega_hat, 0.25),
                            "median_signed_physical_error": _median(
                                omega_hat - omega_true
                            ),
                            "median_shortcut_dominance": _median(dominance),
                            "p_high_band": (
                                float(np.mean(omega_hat >= high_edge))
                                if omega_hat.size
                                else float("nan")
                            ),
                            "p_between_bands": (
                                float(
                                    np.mean(
                                        (omega_hat > low_edge)
                                        & (omega_hat < high_edge)
                                    )
                                )
                                if omega_hat.size
                                else float("nan")
                            ),
                            "median_amplitude_hat": _median(
                                [row["amplitude_hat"] for row in group]
                            ),
                            "median_length_hat": _median(
                                [row["length_hat"] for row in group]
                            ),
                            "N_color_valid": len(valid_color),
                            "color_validity_rate": len(valid_color) / len(group),
                            "median_alpha_hat": _median(
                                [row["alpha_hat"] for row in valid_color]
                            ),
                            "median_signed_alpha_error": _median(
                                [
                                    float(row["alpha_hat"]) - alpha
                                    for row in valid_color
                                ]
                            ),
                            "median_color_line_distance_rgb": _median(
                                [
                                    row["color_line_distance_rgb"]
                                    for row in valid_color
                                ]
                            ),
                        }
                    )

    keyed = {
        (
            str(row["shape"]),
            str(row["history"]),
            float(row["input_alpha"]),
            float(row["omega_true"]),
            int(row["phase_index"]),
            int(row["diffusion_repeat"]),
        ): row
        for row in rows
    }
    if len(keyed) != len(rows):
        raise ValueError("duplicate paired scan keys")

    long_minus_short: list[dict[str, Any]] = []
    for shape in shapes:
        for omega_true in omegas:
            for alpha in alphas:
                differences: list[float] = []
                for phase in range(int(scan["phase_count"])):
                    for repeat in range(int(scan["diffusion_repeats"])):
                        short = keyed[(shape, "short", alpha, omega_true, phase, repeat)]
                        long = keyed[(shape, "long", alpha, omega_true, phase, repeat)]
                        if bool(short["frequency_valid"]) and bool(
                            long["frequency_valid"]
                        ):
                            differences.append(
                                float(long["omega_hat"])
                                - float(short["omega_hat"])
                            )
                long_minus_short.append(
                    {
                        "shape": shape,
                        "input_alpha": alpha,
                        "omega_true": omega_true,
                        "N_paired_valid": len(differences),
                        "median_long_minus_short_omega_hat": _median(differences),
                    }
                )

    square_minus_circle: list[dict[str, Any]] = []
    if set(shapes) == {"circle", "square"}:
        for history in HISTORIES:
            for omega_true in omegas:
                for alpha in alphas:
                    differences = []
                    for phase in range(int(scan["phase_count"])):
                        for repeat in range(int(scan["diffusion_repeats"])):
                            circle = keyed[
                                ("circle", history, alpha, omega_true, phase, repeat)
                            ]
                            square = keyed[
                                ("square", history, alpha, omega_true, phase, repeat)
                            ]
                            if bool(circle["frequency_valid"]) and bool(
                                square["frequency_valid"]
                            ):
                                differences.append(
                                    float(square["omega_hat"])
                                    - float(circle["omega_hat"])
                                )
                    square_minus_circle.append(
                        {
                            "history": history,
                            "input_alpha": alpha,
                            "omega_true": omega_true,
                            "N_paired_valid": len(differences),
                            "median_square_minus_circle_omega_hat": _median(
                                differences
                            ),
                        }
                    )
    return cells, long_minus_short, square_minus_circle


def save_aggregate_tables(
    output_root: Path,
    enriched_rows: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    long_minus_short: Sequence[Mapping[str, Any]],
    square_minus_circle: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    table_root = output_root / "tables"
    paths: dict[str, Path] = {
        "per_sample": table_root / "per_sample_enriched.csv",
        "cells": table_root / "grid_cells.csv",
        "long_minus_short": table_root / "paired_long_minus_short.csv",
    }
    if square_minus_circle:
        paths["square_minus_circle"] = (
            table_root / "paired_square_minus_circle.csv"
        )
    _atomic_csv(paths["per_sample"], enriched_rows)
    _atomic_csv(paths["cells"], cells)
    _atomic_csv(paths["long_minus_short"], long_minus_short)
    if square_minus_circle:
        _atomic_csv(paths["square_minus_circle"], square_minus_circle)
    payload = {
        "status": "complete",
        "program_version": PROGRAM_VERSION,
        "num_samples": len(enriched_rows),
        "num_grid_cells": len(cells),
        "num_long_short_cells": len(long_minus_short),
        "num_shape_difference_cells": len(square_minus_circle),
        "model_name": context["model_name"],
        "training_binding": context["training_binding"],
        "x_axis": "equal-spaced physical input alpha (0.0, 0.1, ..., 1.0)",
        "y_axis": "physical input angular frequency in rad/s",
        "context": context,
        "artifacts": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
    }
    _atomic_write_text(
        output_root / "tables" / "summary.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    return payload


# ======================================================================
# COMMON HEATMAP RENDERER
# Keep this block while any PLOT block below is enabled.
# ======================================================================


def _axis_edges(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size < 2 or np.any(np.diff(array) <= 0):
        raise ValueError("heatmap coordinates must be strictly increasing")
    middle = (array[:-1] + array[1:]) / 2.0
    return np.concatenate(
        ([array[0] - (array[1] - array[0]) / 2.0], middle,
         [array[-1] + (array[-1] - array[-2]) / 2.0])
    )


def _matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    selectors: Mapping[str, str],
    metric: str,
    alphas: Sequence[float],
    omegas: Sequence[float],
) -> np.ndarray:
    lookup: dict[tuple[float, float], float] = {}
    for row in rows:
        if all(str(row[key]) == value for key, value in selectors.items()):
            lookup[(float(row["omega_true"]), float(row["input_alpha"]))] = (
                _finite_number(row[metric])
            )
    expected = len(alphas) * len(omegas)
    if len(lookup) != expected:
        raise ValueError(
            f"selectors={selectors}, metric={metric}: "
            f"expected {expected} cells, got {len(lookup)}"
        )
    return np.asarray(
        [[lookup[(omega, alpha)] for alpha in alphas] for omega in omegas],
        dtype=float,
    )


def _robust_limits(
    values: Sequence[float],
    *,
    include: float | None = None,
    symmetric: bool = False,
) -> tuple[float, float]:
    finite = _finite(values)
    if not finite.size:
        raise ValueError("cannot derive plot limits from empty values")
    low = float(np.quantile(finite, 0.02))
    high = float(np.quantile(finite, 0.98))
    if include is not None:
        low = min(low, include)
        high = max(high, include)
    if symmetric:
        bound = max(abs(low), abs(high), 1e-9)
        return -bound, bound
    padding = max((high - low) * 0.05, 1e-9)
    return low - padding, high + padding


def _plot_payload(
    plot_root: Path,
    *,
    plot_id: str,
    metric: str,
    data_rows: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    png: Path,
    pdf: Path,
    csv_path: Path,
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "status": "complete",
        "program_version": PROGRAM_VERSION,
        "plot": plot_id,
        "metric": metric,
        "model_name": context["model_name"],
        "training_binding": context["training_binding"],
        "checkpoint": "50k short and 50k long",
        "x_axis": "physical input color alpha, equally spaced 0.0 to 1.0",
        "y_axis": "physical input angular frequency (rad/s)",
        "num_rows": len(data_rows),
        "config_sha256": context["config_sha256"],
        **dict(extra),
        "artifacts": {},
    }
    for path in (png, pdf, csv_path):
        payload["artifacts"][path.name] = {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    summary = plot_root / "summary.json"
    _atomic_write_text(summary, json.dumps(payload, indent=2) + "\n")
    return payload


def _four_facet_heatmap(
    cells: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    output_root: Path,
    *,
    plot_id: str,
    metric: str,
    colorbar_label: str,
    cmap: str,
    vmin: float,
    vmax: float,
    reference: float | None = None,
) -> dict[str, Any]:
    scan = context["scan"]
    shapes = tuple(str(value) for value in context["shapes"])
    alphas = tuple(float(value) for value in scan["color_alphas"])
    omegas = tuple(float(value) for value in scan["frequencies_rad_s"])
    x_edges = _axis_edges(alphas)
    y_edges = _axis_edges(omegas)
    fig, axes = plt.subplots(
        len(shapes),
        len(HISTORIES),
        figsize=(13.2, 4.4 * len(shapes)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    mesh = None
    for row_index, shape in enumerate(shapes):
        for column, history in enumerate(HISTORIES):
            ax = axes[row_index, column]
            values = _matrix(
                cells,
                selectors={"shape": shape, "history": history},
                metric=metric,
                alphas=alphas,
                omegas=omegas,
            )
            mesh = ax.pcolormesh(
                x_edges, y_edges, values, cmap=cmap, vmin=vmin, vmax=vmax,
                shading="flat", rasterized=True,
            )
            ax.axhline(
                float(scan["low_band_rad_s"][1]), color="white",
                linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.9,
            )
            ax.axhline(
                float(scan["high_band_rad_s"][0]), color="white",
                linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.9,
            )
            ax.set_title(f"{shape} · {history} context")
            ax.set_xticks(alphas)
            ax.set_yticks(omegas)
            ax.set_xlabel(r"physical input color $\alpha_{in}$  (red $\to$ blue)")
            ax.set_ylabel(r"physical input frequency $\omega_{in}$ (rad/s)")
    assert mesh is not None
    colorbar = fig.colorbar(
        mesh, ax=axes.ravel().tolist(), shrink=0.88, pad=0.025
    )
    colorbar.set_label(colorbar_label)
    title = (
        f"{colorbar_label}\n{context['training_binding']}  |  "
        f"model checkpoint: 50k\n"
        f"fixed input amplitude: {float(scan['amplitude_rad']):.3f} rad; "
        f"fixed input length: {float(context['data']['render']['length']):.2f}"
    )
    fig.suptitle(title, fontsize=13.5)

    plot_root = output_root / "plots" / plot_id
    png = plot_root / f"{plot_id}.png"
    pdf = plot_root / f"{plot_id}.pdf"
    csv_path = plot_root / f"{plot_id}_data.csv"
    selected_rows = [
        {
            "shape": row["shape"],
            "history": row["history"],
            "input_alpha": row["input_alpha"],
            "omega_true": row["omega_true"],
            metric: row[metric],
        }
        for row in cells
    ]
    _atomic_savefig(fig, png, dpi=220, facecolor="white")
    _atomic_savefig(fig, pdf, facecolor="white")
    plt.close(fig)
    _atomic_csv(csv_path, selected_rows)
    return _plot_payload(
        plot_root,
        plot_id=plot_id,
        metric=metric,
        data_rows=selected_rows,
        context=context,
        png=png,
        pdf=pdf,
        csv_path=csv_path,
        extra={
            "color_scale": {"vmin": vmin, "vmax": vmax, "cmap": cmap},
            "reference": reference,
            "band_boundaries_rad_s": [
                float(scan["low_band_rad_s"][1]),
                float(scan["high_band_rad_s"][0]),
            ],
        },
    )


def _two_facet_difference_heatmap(
    rows: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    output_root: Path,
    *,
    plot_id: str,
    facet_key: str,
    facets: Sequence[str],
    metric: str,
    colorbar_label: str,
) -> dict[str, Any]:
    scan = context["scan"]
    alphas = tuple(float(value) for value in scan["color_alphas"])
    omegas = tuple(float(value) for value in scan["frequencies_rad_s"])
    values = _finite([row[metric] for row in rows])
    vmin, vmax = _robust_limits(values, symmetric=True)
    if not facets:
        raise ValueError(f"{plot_id} requires at least one facet")
    fig, axes_grid = plt.subplots(
        1,
        len(facets),
        figsize=(6.6 * len(facets), 4.9),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes_grid.ravel()
    mesh = None
    for ax, facet in zip(axes, facets):
        matrix = _matrix(
            rows,
            selectors={facet_key: facet},
            metric=metric,
            alphas=alphas,
            omegas=omegas,
        )
        mesh = ax.pcolormesh(
            _axis_edges(alphas), _axis_edges(omegas), matrix,
            cmap="coolwarm", vmin=vmin, vmax=vmax, shading="flat",
            rasterized=True,
        )
        ax.axhline(
            float(scan["low_band_rad_s"][1]), color="black",
            linewidth=0.9, linestyle=(0, (4, 3)), alpha=0.75,
        )
        ax.axhline(
            float(scan["high_band_rad_s"][0]), color="black",
            linewidth=0.9, linestyle=(0, (4, 3)), alpha=0.75,
        )
        ax.set_title(facet)
        ax.set_xticks(alphas)
        ax.set_yticks(omegas)
        ax.set_xlabel(r"physical input color $\alpha_{in}$  (red $\to$ blue)")
        ax.set_ylabel(r"physical input frequency $\omega_{in}$ (rad/s)")
    assert mesh is not None
    colorbar = fig.colorbar(mesh, ax=axes.tolist(), shrink=0.88, pad=0.025)
    colorbar.set_label(colorbar_label)
    fig.suptitle(
        f"{colorbar_label}\n{context['training_binding']}  |  "
        "model checkpoint: 50k",
        fontsize=13.5,
    )
    plot_root = output_root / "plots" / plot_id
    png = plot_root / f"{plot_id}.png"
    pdf = plot_root / f"{plot_id}.pdf"
    csv_path = plot_root / f"{plot_id}_data.csv"
    selected_rows = [
        {
            facet_key: row[facet_key],
            "input_alpha": row["input_alpha"],
            "omega_true": row["omega_true"],
            "N_paired_valid": row["N_paired_valid"],
            metric: row[metric],
        }
        for row in rows
    ]
    _atomic_savefig(fig, png, dpi=220, facecolor="white")
    _atomic_savefig(fig, pdf, facecolor="white")
    plt.close(fig)
    _atomic_csv(csv_path, selected_rows)
    return _plot_payload(
        plot_root,
        plot_id=plot_id,
        metric=metric,
        data_rows=selected_rows,
        context=context,
        png=png,
        pdf=pdf,
        csv_path=csv_path,
        extra={
            "color_scale": {"vmin": vmin, "vmax": vmax, "cmap": "coolwarm"},
            "difference_is_paired_by": (
                "input alpha, physical frequency, phase index, diffusion repeat"
            ),
        },
    )


# ======================================================================
# PLOT 01: MEDIAN GENERATED FREQUENCY
# Delete this whole block and its PLOTTERS entry to remove only this plot.
# ======================================================================


def plot_01(cells, _long_short, _shape_diff, context, output_root):
    scan = context["scan"]
    return _four_facet_heatmap(
        cells, context, output_root,
        plot_id="01_median_generated_frequency",
        metric="median_omega_hat",
        colorbar_label="median fitted generated frequency (rad/s)",
        cmap="viridis",
        vmin=float(scan["low_band_rad_s"][0]),
        vmax=float(scan["high_band_rad_s"][1]),
    )


# ======================================================================
# PLOT 02: SIGNED PHYSICAL FREQUENCY ERROR
# Delete this whole block and its PLOTTERS entry to remove only this plot.
# ======================================================================


def plot_02(cells, _long_short, _shape_diff, context, output_root):
    span = float(context["scan"]["high_band_rad_s"][1]) - float(
        context["scan"]["low_band_rad_s"][0]
    )
    return _four_facet_heatmap(
        cells, context, output_root,
        plot_id="02_signed_physical_frequency_error",
        metric="median_signed_physical_error",
        colorbar_label=r"median $\hat{\omega}-\omega_{in}$ (rad/s)",
        cmap="coolwarm", vmin=-span, vmax=span, reference=0.0,
    )


# ======================================================================
# PLOT 03: COLOR-SHORTCUT DOMINANCE
# +1 = output closer to color-implied frequency; -1 = closer to physics.
# Delete this whole block and its PLOTTERS entry to remove only this plot.
# ======================================================================


def plot_03(cells, _long_short, _shape_diff, context, output_root):
    return _four_facet_heatmap(
        cells, context, output_root,
        plot_id="03_color_shortcut_dominance",
        metric="median_shortcut_dominance",
        colorbar_label="shortcut dominance  (+1 color; −1 physical input)",
        cmap="coolwarm", vmin=-1.0, vmax=1.0, reference=0.0,
    )


# ======================================================================
# PLOT 04: PROBABILITY OF A HIGH-BAND OUTPUT
# Delete this whole block and its PLOTTERS entry to remove only this plot.
# ======================================================================


def plot_04(cells, _long_short, _shape_diff, context, output_root):
    return _four_facet_heatmap(
        cells, context, output_root,
        plot_id="04_high_band_probability",
        metric="p_high_band",
        colorbar_label=r"$P(\hat{\omega}\geq5.2)$",
        cmap="magma", vmin=0.0, vmax=1.0,
    )


# ======================================================================
# PLOT 05: PROBABILITY OF A BETWEEN-BANDS OUTPUT
# Delete this whole block and its PLOTTERS entry to remove only this plot.
# ======================================================================


def plot_05(cells, _long_short, _shape_diff, context, output_root):
    return _four_facet_heatmap(
        cells, context, output_root,
        plot_id="05_between_bands_probability",
        metric="p_between_bands",
        colorbar_label=r"$P(3.0<\hat{\omega}<5.2)$",
        cmap="cividis", vmin=0.0, vmax=1.0,
    )


# ======================================================================
# PLOT 06: WITHIN-CELL GENERATED-FREQUENCY IQR
# Delete this whole block and its PLOTTERS entry to remove only this plot.
# ======================================================================


def plot_06(cells, _long_short, _shape_diff, context, output_root):
    _, upper = _robust_limits([row["iqr_omega_hat"] for row in cells], include=0.0)
    return _four_facet_heatmap(
        cells, context, output_root,
        plot_id="06_generated_frequency_iqr",
        metric="iqr_omega_hat",
        colorbar_label="generated-frequency IQR (rad/s)",
        cmap="magma", vmin=0.0, vmax=upper, reference=0.0,
    )


# ======================================================================
# PLOT 07: PAIRED LONG MINUS SHORT FREQUENCY
# Delete this whole block and its PLOTTERS entry to remove only this plot.
# ======================================================================


def plot_07(_cells, long_short, _shape_diff, context, output_root):
    return _two_facet_difference_heatmap(
        long_short, context, output_root,
        plot_id="07_paired_long_minus_short_frequency",
        facet_key="shape", facets=context["shapes"],
        metric="median_long_minus_short_omega_hat",
        colorbar_label=r"median paired $\hat{\omega}_{long}-\hat{\omega}_{short}$ (rad/s)",
    )


# ======================================================================
# PLOT 08: PAIRED SQUARE MINUS CIRCLE FREQUENCY
# Delete this whole block and its PLOTTERS entry to remove only this plot.
# ======================================================================


def plot_08(_cells, _long_short, shape_diff, context, output_root):
    return _two_facet_difference_heatmap(
        shape_diff, context, output_root,
        plot_id="08_paired_square_minus_circle_frequency",
        facet_key="history", facets=HISTORIES,
        metric="median_square_minus_circle_omega_hat",
        colorbar_label=r"median paired $\hat{\omega}_{square}-\hat{\omega}_{circle}$ (rad/s)",
    )


# ======================================================================
# PLOT 09: MEDIAN GENERATED AMPLITUDE
# Delete this whole block and its PLOTTERS entry to remove only this plot.
# ======================================================================


def plot_09(cells, _long_short, _shape_diff, context, output_root):
    reference = float(context["scan"]["amplitude_rad"])
    vmin, vmax = _robust_limits(
        [row["median_amplitude_hat"] for row in cells], include=reference
    )
    return _four_facet_heatmap(
        cells, context, output_root,
        plot_id="09_median_generated_amplitude",
        metric="median_amplitude_hat",
        colorbar_label="median fitted generated amplitude (rad)",
        cmap="viridis", vmin=vmin, vmax=vmax, reference=reference,
    )


# ======================================================================
# PLOT 10: MEDIAN GENERATED PIVOT-TO-BOB LENGTH
# Delete this whole block and its PLOTTERS entry to remove only this plot.
# ======================================================================


def plot_10(cells, _long_short, _shape_diff, context, output_root):
    reference = float(context["data"]["render"]["length"])
    vmin, vmax = _robust_limits(
        [row["median_length_hat"] for row in cells], include=reference
    )
    return _four_facet_heatmap(
        cells, context, output_root,
        plot_id="10_median_generated_length",
        metric="median_length_hat",
        colorbar_label="median generated pivot–bob length (normalized)",
        cmap="viridis", vmin=vmin, vmax=vmax, reference=reference,
    )


# ======================================================================
# PLOT 11: SIGNED GENERATED-COLOR ERROR
# Delete this whole block and its PLOTTERS entry to remove only this plot.
# ======================================================================


def plot_11(cells, _long_short, _shape_diff, context, output_root):
    return _four_facet_heatmap(
        cells, context, output_root,
        plot_id="11_signed_generated_color_error",
        metric="median_signed_alpha_error",
        colorbar_label=r"median $\hat{\alpha}_{out}-\alpha_{in}$",
        cmap="coolwarm", vmin=-1.0, vmax=1.0, reference=0.0,
    )


# One line per removable plot block.
PLOTTERS: tuple[Callable[..., dict[str, Any]], ...] = (
    plot_01,
    plot_02,
    plot_03,
    plot_04,
    plot_05,
    plot_06,
    plot_07,
    plot_08,
    plot_09,
    plot_10,
    plot_11,
)


# ======================================================================
# RESUMABLE ORCHESTRATION
# ======================================================================


def _input_fingerprint(context: Mapping[str, Any]) -> str:
    material = {
        "program_version": PROGRAM_VERSION,
        "config_sha256": context["config_sha256"],
        "frozen_inputs": context["frozen_inputs"],
        "short_checkpoint_sha256": context["checkpoint"][
            "short_checkpoint_sha256"
        ],
        "long_checkpoint_sha256": context["checkpoint"][
            "long_checkpoint_sha256"
        ],
    }
    encoded = json.dumps(material, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run(
    experiment_root: Path,
    dataset_base: Path,
    config_path: Path,
    output_root: Path,
) -> Path:
    rows, context = load_enriched_rows(
        experiment_root, dataset_base, config_path
    )
    context["input_fingerprint"] = _input_fingerprint(context)
    cells, long_short, shape_diff = build_grid_tables(rows, context["scan"])
    save_aggregate_tables(
        output_root, rows, cells, long_short, shape_diff, context
    )

    active_plotters = tuple(
        plotter
        for plotter in PLOTTERS
        if plotter is not plot_08 or bool(shape_diff)
    )
    completed: list[str] = []
    progress_path = output_root / "progress.json"
    for plotter in active_plotters:
        plot_number = plotter.__name__.removeprefix("plot_")
        matching = sorted((output_root / "plots").glob(f"{plot_number}_*/summary.json"))
        if len(matching) > 1:
            raise ValueError(f"multiple saved summaries for plot {plot_number}")
        if matching:
            saved = _read_json(matching[0])
            if saved.get("status") != "complete":
                raise ValueError(f"incomplete saved plot summary: {matching[0]}")
            if saved.get("input_fingerprint") not in (
                None,
                context["input_fingerprint"],
            ):
                raise ValueError(f"saved plot input mismatch: {matching[0]}")
            completed.append(str(saved["plot"]))
        else:
            payload = plotter(
                cells, long_short, shape_diff, context, output_root
            )
            summary_path = output_root / "plots" / payload["plot"] / "summary.json"
            saved = _read_json(summary_path)
            saved["input_fingerprint"] = context["input_fingerprint"]
            _atomic_write_text(
                summary_path, json.dumps(saved, indent=2) + "\n"
            )
            completed.append(str(payload["plot"]))
        _atomic_write_text(
            progress_path,
            json.dumps(
                {
                    "status": "running",
                    "completed_plots": completed,
                    "completed_count": len(completed),
                    "expected_count": len(active_plotters),
                    "input_fingerprint": context["input_fingerprint"],
                },
                indent=2,
            )
            + "\n",
        )

    final = {
        "status": "complete",
        "program_version": PROGRAM_VERSION,
        "model_name": context["model_name"],
        "training_binding": context["training_binding"],
        "shapes": list(context["shapes"]),
        "checkpoint": "50k short and 50k long",
        "input_fingerprint": context["input_fingerprint"],
        "num_samples": len(rows),
        "num_plots": len(completed),
        "plots": completed,
    }
    _atomic_write_text(
        output_root / "summary.json",
        json.dumps(final, indent=2) + "\n",
    )
    _atomic_write_text(
        progress_path,
        json.dumps({**final, "completed_plots": completed}, indent=2) + "\n",
    )
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--dataset-base", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.experiment_root, args.dataset_base, args.config, args.output_root))


if __name__ == "__main__":
    main()
