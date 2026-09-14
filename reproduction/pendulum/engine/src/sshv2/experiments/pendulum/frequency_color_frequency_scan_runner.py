#!/usr/bin/env python3
"""Reuse validated Pendulum prediction/measurement code for the 2-D scan."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

from sshv2.experiments.pendulum.frequency_color_frequency_scan_data import (
    ScanSpec,
    load_config,
)


def _read_csv(dataset_root: Path) -> list[dict[str, str]]:
    path = dataset_root / "videos" / "eval" / "metadata.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty scan metadata: {path}")
    return rows


def _validate_rows(
    rows: list[dict[str, str]],
    *,
    spec: ScanSpec,
    training_manifest_id: str,
    model_name: str,
) -> str:
    if len(rows) != spec.samples_per_shape:
        raise ValueError(
            f"expected {spec.samples_per_shape} scan rows, got {len(rows)}"
        )
    if {row["subexperiment_id"] for row in rows} != {spec.subexperiment_id}:
        raise ValueError("scan subexperiment mismatch")
    if {row["training_manifest_id"] for row in rows} != {training_manifest_id}:
        raise ValueError("scan training manifest mismatch")
    if {row["model_name"] for row in rows} != {model_name}:
        raise ValueError("scan model name mismatch")
    shapes = {row["shape_label"] for row in rows}
    if len(shapes) != 1 or not shapes.issubset(set(spec.shapes)):
        raise ValueError(f"scan must contain exactly one supported shape: {shapes}")
    if {float(row["omega_true"]) for row in rows} != set(
        spec.frequencies_rad_s
    ):
        raise ValueError("scan frequency values differ from config")
    alpha_counts = Counter(
        float(row["test_color_alpha_target"]) for row in rows
    )
    omega_counts = Counter(float(row["omega_true"]) for row in rows)
    expected_alpha = len(spec.frequencies_rad_s) * spec.repeats_per_cell
    expected_omega = len(spec.color_alphas) * spec.repeats_per_cell
    if alpha_counts != Counter(
        {alpha: expected_alpha for alpha in spec.color_alphas}
    ):
        raise ValueError(f"scan alpha counts differ: {alpha_counts}")
    if omega_counts != Counter(
        {omega: expected_omega for omega in spec.frequencies_rad_s}
    ):
        raise ValueError(f"scan frequency counts differ: {omega_counts}")
    cell_counts = Counter(
        (
            float(row["omega_true"]),
            float(row["test_color_alpha_target"]),
        )
        for row in rows
    )
    if set(cell_counts.values()) != {spec.repeats_per_cell}:
        raise ValueError("not every scan cell has the configured repeats")
    test_ids = {row["test_manifest_id"] for row in rows}
    if len(test_ids) != 1:
        raise ValueError("scan dataset has multiple test manifests")
    return test_ids.pop()


def run_predict(args: argparse.Namespace) -> Path:
    from sshv2.experiments.pendulum import frequency_color_low_predict as base

    config, spec = load_config(args.experiment_config)
    base.SUBEXPERIMENT_ID = spec.subexperiment_id
    base.COLOR_ALPHAS = spec.color_alphas

    def validate(
        rows: list[dict[str, str]],
        *,
        training_manifest_id: str,
        model_name: str,
    ) -> str:
        return _validate_rows(
            rows,
            spec=spec,
            training_manifest_id=training_manifest_id,
            model_name=model_name,
        )

    base._validate_rows = validate
    return base.predict(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        experiment_config=args.experiment_config,
        training_config=args.training_config,
        checkpoint=args.checkpoint,
        history=args.history,
        device=args.device,
        steps=args.steps,
        seed_offset=args.seed_offset,
    )


def run_evaluate(args: argparse.Namespace) -> Path:
    from sshv2.experiments.pendulum import frequency_color_low_evaluate as base

    config, spec = load_config(args.experiment_config)
    base.SUBEXPERIMENT_ID = spec.subexperiment_id
    base.COLOR_ALPHAS = spec.color_alphas

    def read_rows(dataset_root: Path) -> list[dict[str, str]]:
        rows = _read_csv(dataset_root)
        _validate_rows(
            rows,
            spec=spec,
            training_manifest_id=config.training_manifest_id,
            model_name=config.model_name,
        )
        return rows

    base._read_rows = read_rows
    return base.evaluate(
        dataset_root=args.dataset_root,
        prediction_root=args.prediction_root,
        calibration_root=args.calibration_root,
        experiment_config=args.experiment_config,
        output_root=args.output_root,
        history=args.history,
    )


def run_color(args: argparse.Namespace) -> Path:
    from sshv2.experiments.pendulum import (
        frequency_color_low_color_measure as base,
    )

    config, spec = load_config(args.experiment_config)
    base.SUBEXPERIMENT_ID = spec.subexperiment_id
    base.COLOR_ALPHAS = spec.color_alphas

    def read_rows(dataset_root: Path) -> list[dict[str, str]]:
        rows = _read_csv(dataset_root)
        _validate_rows(
            rows,
            spec=spec,
            training_manifest_id=config.training_manifest_id,
            model_name=config.model_name,
        )
        return rows

    base._read_dataset_rows = read_rows
    return base.analyze_history(
        dataset_root=args.dataset_root,
        prediction_root=args.prediction_root,
        frequency_metrics_root=args.frequency_metrics_root,
        experiment_config=args.experiment_config,
        output_root=args.output_root,
        history=args.history,
    )


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--history", choices=("short", "long"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict")
    _common(predict)
    predict.add_argument("--training-config", type=Path, required=True)
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--device", default="cuda")
    predict.add_argument("--steps", type=int, default=20)
    predict.add_argument("--seed-offset", type=int, default=23_000_000)

    evaluate = subparsers.add_parser("evaluate")
    _common(evaluate)
    evaluate.add_argument("--prediction-root", type=Path, required=True)
    evaluate.add_argument("--calibration-root", type=Path, required=True)

    color = subparsers.add_parser("color")
    _common(color)
    color.add_argument("--prediction-root", type=Path, required=True)
    color.add_argument("--frequency-metrics-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runners: dict[str, Any] = {
        "predict": run_predict,
        "evaluate": run_evaluate,
        "color": run_color,
    }
    print(runners[args.command](args))


if __name__ == "__main__":
    main()
