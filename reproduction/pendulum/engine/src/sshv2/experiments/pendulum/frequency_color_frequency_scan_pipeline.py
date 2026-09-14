#!/usr/bin/env python3
"""Resume the Pendulum 2-D scan from frozen stage boundaries."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from sshv2.experiments.pendulum.frequency_color_frequency_scan_data import (
    load_config,
)

HISTORIES = ("short", "long")


def _calibration_roots(model_name: str) -> dict[str, Path]:
    if model_name == "frequency_color_circle":
        return {
            "circle": Path(
                "runs/pendulum/frequency_color_evaluation/"
                "low_frequency_11color_64states/detector_calibration"
            )
        }
    if model_name == "frequency_color_shape":
        return {
            "circle": Path(
                "runs/pendulum/frequency_color_shape_evaluation/"
                "frequency_color_shape_11point_50k_analysis/"
                "low_frequency_circle_11color_64states/detector_calibration"
            ),
            "square": Path(
                "runs/pendulum/frequency_color_shape_evaluation/"
                "frequency_color_shape_11point_50k_analysis/"
                "high_frequency_square_11color_64states/detector_calibration"
            ),
        }
    raise ValueError(f"unsupported scan model: {model_name}")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _complete(path: Path, count_key: str, expected: int) -> bool:
    if not path.is_file():
        return False
    state = _read_json(path)
    return state.get("status") == "complete" and int(state[count_key]) == expected


def _run_parallel(
    jobs: Sequence[tuple[str, list[str], Path]],
    state_path: Path,
    stage: str,
) -> None:
    pending: list[tuple[str, subprocess.Popen[bytes], Path]] = []
    for name, command, log_path in jobs:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("ab")
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        handle.close()
        pending.append((name, process, log_path))
    _atomic_json(
        state_path,
        {
            "status": "running",
            "stage": stage,
            "jobs": [
                {"name": name, "pid": process.pid, "log": str(log)}
                for name, process, log in pending
            ],
        },
    )
    failures = []
    for name, process, log_path in pending:
        return_code = process.wait()
        if return_code != 0:
            failures.append(
                {"name": name, "return_code": return_code, "log": str(log_path)}
            )
    if failures:
        _atomic_json(
            state_path,
            {"status": "failed", "stage": stage, "failures": failures},
        )
        raise RuntimeError(f"{stage} failed: {failures}")


def run(
    run_root: Path,
    dataset_base: Path,
    config: Path,
    analysis_root: Path,
    triptych_root: Path,
    poll_seconds: float,
) -> Path:
    dataset_config, spec = load_config(config)
    shapes = spec.shapes
    calibration = _calibration_roots(dataset_config.model_name)
    logs = run_root / "logs"
    state_path = run_root / "pipeline_progress.json"
    expected = spec.samples_per_shape
    expected_plots = 11 if set(shapes) == {"circle", "square"} else 10
    expected_videos = (
        len(shapes)
        * len(spec.color_alphas)
        * len(spec.frequencies_rad_s)
    )
    while True:
        missing = []
        for shape in shapes:
            for history in HISTORIES:
                progress = run_root / shape / history / "prediction/progress.json"
                if not _complete(progress, "completed_predictions", expected):
                    missing.append(f"{shape}_{history}")
        if not missing:
            break
        _atomic_json(
            state_path,
            {
                "status": "waiting",
                "stage": "predictions",
                "incomplete": missing,
                "poll_seconds": poll_seconds,
            },
        )
        time.sleep(poll_seconds)

    evaluate_jobs = []
    for shape in shapes:
        for history in HISTORIES:
            output = run_root / shape / history / "metrics"
            if _complete(output / "progress.json", "completed_samples", expected):
                continue
            evaluate_jobs.append(
                (
                    f"{shape}_{history}",
                    [
                        sys.executable,
                        "-m",
                        "sshv2.experiments.pendulum.frequency_color_frequency_scan_runner",
                        "evaluate",
                        "--dataset-root",
                        str(dataset_base / shape),
                        "--experiment-config",
                        str(config),
                        "--history",
                        history,
                        "--output-root",
                        str(output),
                        "--prediction-root",
                        str(run_root / shape / history / "prediction"),
                        "--calibration-root",
                        str(calibration[shape]),
                    ],
                    logs / f"{shape}_{history}_evaluate_pipeline.log",
                )
            )
    if evaluate_jobs:
        _run_parallel(evaluate_jobs, state_path, "frequency_metrics")

    color_jobs = []
    for shape in shapes:
        for history in HISTORIES:
            output = run_root / shape / history / "color_metrics"
            if _complete(output / "progress.json", "completed_samples", expected):
                continue
            color_jobs.append(
                (
                    f"{shape}_{history}",
                    [
                        sys.executable,
                        "-m",
                        "sshv2.experiments.pendulum.frequency_color_frequency_scan_runner",
                        "color",
                        "--dataset-root",
                        str(dataset_base / shape),
                        "--experiment-config",
                        str(config),
                        "--history",
                        history,
                        "--output-root",
                        str(output),
                        "--prediction-root",
                        str(run_root / shape / history / "prediction"),
                        "--frequency-metrics-root",
                        str(run_root / shape / history / "metrics"),
                    ],
                    logs / f"{shape}_{history}_color_pipeline.log",
                )
            )
    if color_jobs:
        _run_parallel(color_jobs, state_path, "color_metrics")

    if not _complete(
        analysis_root / "progress.json", "num_plots", expected_plots
    ):
        _run_parallel(
            [
                (
                    "analysis",
                    [
                        sys.executable,
                        "-m",
                        "sshv2.experiments.pendulum.frequency_color_frequency_scan_analysis",
                        "--experiment-root",
                        str(run_root),
                        "--dataset-base",
                        str(dataset_base),
                        "--config",
                        str(config),
                        "--output-root",
                        str(analysis_root),
                    ],
                    logs / "analysis_pipeline.log",
                )
            ],
            state_path,
            "plots",
        )

    if not _complete(
        triptych_root / "progress.json", "completed_videos", expected_videos
    ):
        _run_parallel(
            [
                (
                    "triptychs",
                    [
                        sys.executable,
                        "-m",
                        "sshv2.experiments.pendulum.frequency_color_frequency_scan_triptych",
                        "--experiment-root",
                        str(run_root),
                        "--dataset-base",
                        str(dataset_base),
                        "--config",
                        str(config),
                        "--output-root",
                        str(triptych_root),
                        "--phase-index",
                        "0",
                        "--diffusion-repeat",
                        "0",
                    ],
                    logs / "triptych_pipeline.log",
                )
            ],
            state_path,
            "triptych_videos",
        )

    _atomic_json(
        state_path,
        {
            "status": "complete",
            "stage": "all",
            "analysis_root": str(analysis_root),
            "triptych_root": str(triptych_root),
            "model_name": dataset_config.model_name,
            "shapes": list(shapes),
            "expected_predictions_per_history_shape": expected,
            "expected_plots": expected_plots,
            "expected_videos": expected_videos,
        },
    )
    return run_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset-base", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--triptych-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    print(
        run(
            args.run_root,
            args.dataset_base,
            args.config,
            args.analysis_root,
            args.triptych_root,
            args.poll_seconds,
        )
    )


if __name__ == "__main__":
    main()
