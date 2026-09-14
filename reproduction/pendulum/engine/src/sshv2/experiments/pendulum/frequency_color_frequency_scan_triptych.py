#!/usr/bin/env python3
"""Build resumable GT | short | long videos for the 2-D Pendulum scan."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml


PROGRAM_VERSION = "frequency_color_frequency_scan_triptych_v2"
HISTORIES = ("short", "long")
TITLES = ("Ground truth", "Short", "Long")
PANEL_WIDTH = 128
PANEL_HEIGHT = 128
TOP_HEIGHT = 34
BOTTOM_HEIGHT = 88
GALLERY_ALPHAS = (0.0, 0.5, 1.0)
GALLERY_OMEGAS = (2.6, 4.2, 5.8)


# ======================================================================
# COMMON INPUT / ATOMIC OUTPUT UTILITIES
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


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if _sha256(destination) != _sha256(source):
            raise ValueError(f"existing gallery file differs: {destination}")
        return
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    os.link(source, temporary)
    os.replace(temporary, destination)


def _load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("scan config must be a mapping")
    data, scan = payload.get("data"), payload.get("scan")
    if not isinstance(data, dict) or not isinstance(scan, dict):
        raise ValueError("scan config must contain data and scan mappings")
    if int(scan["future_frames"]) != 64:
        raise ValueError("triptych output requires exactly 64 future frames")
    return data, scan


def _read_metadata(dataset_root: Path) -> list[dict[str, str]]:
    path = dataset_root / "videos" / "eval" / "metadata.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _prediction_index(
    root: Path,
    history: str,
    expected: int,
) -> dict[str, dict[str, Any]]:
    manifest = _read_json(root / "predictions.json")
    if manifest.get("num_predictions") != expected:
        raise ValueError(f"incomplete predictions: {root}")
    attributes = manifest.get("attributes", {})
    if attributes.get("history") != history:
        raise ValueError(f"prediction history mismatch: {root}")
    rows = _read_jsonl(root / str(manifest["predictions_file"]))
    by_id = {str(row["sample_id"]): row for row in rows}
    if len(by_id) != expected:
        raise ValueError(f"prediction row count mismatch: {root}")
    return by_id


def _metric_index(
    root: Path,
    history: str,
    expected: int,
) -> dict[str, dict[str, Any]]:
    frozen = _read_json(root / "FROZEN.json")
    if (
        frozen.get("status") != "frozen"
        or frozen.get("num_samples") != expected
    ):
        raise ValueError(f"incomplete frozen metrics: {root}")
    rows = _read_jsonl(root / "per_sample.jsonl")
    by_id = {str(row["sample_id"]): row for row in rows}
    if len(by_id) != expected or {row["history"] for row in rows} != {history}:
        raise ValueError(f"metric identity mismatch: {root}")
    return by_id


def _read_video(path: Path, expected_frames: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if len(frames) != expected_frames:
        raise ValueError(
            f"{path}: expected {expected_frames} frames, got {len(frames)}"
        )
    array = np.stack(frames)
    if array.shape[1:] != (PANEL_HEIGHT, PANEL_WIDTH, 3):
        raise ValueError(f"unexpected video size: {path}: {array.shape}")
    return array


# ======================================================================
# VIDEO COMPOSITION
# ======================================================================


def _centered_text(
    canvas: np.ndarray,
    text: str,
    panel_index: int,
    baseline_y: int,
    font_scale: float,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (width, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
    left = panel_index * PANEL_WIDTH
    x = left + max(2, (PANEL_WIDTH - width) // 2)
    cv2.putText(
        canvas, text, (x, baseline_y), font, font_scale,
        (242, 242, 242), thickness, cv2.LINE_AA,
    )


def _compose_frame(
    ground_truth: np.ndarray,
    short: np.ndarray,
    long: np.ndarray,
    *,
    frequencies: tuple[float, float, float],
    amplitudes: tuple[float, float, float],
    shape: str,
    alpha: float,
    omega_input: float,
) -> np.ndarray:
    canvas = np.full(
        (TOP_HEIGHT + PANEL_HEIGHT + BOTTOM_HEIGHT, PANEL_WIDTH * 3, 3),
        22,
        dtype=np.uint8,
    )
    for panel, frame in enumerate((ground_truth, short, long)):
        left = panel * PANEL_WIDTH
        canvas[
            TOP_HEIGHT : TOP_HEIGHT + PANEL_HEIGHT,
            left : left + PANEL_WIDTH,
        ] = frame
        cv2.rectangle(
            canvas,
            (left, TOP_HEIGHT),
            (left + PANEL_WIDTH - 1, TOP_HEIGHT + PANEL_HEIGHT - 1),
            (180, 180, 180),
            1,
        )
        _centered_text(canvas, TITLES[panel], panel, 23, 0.43)
        base = TOP_HEIGHT + PANEL_HEIGHT
        _centered_text(
            canvas, f"freq={frequencies[panel]:.3f} rad/s", panel,
            base + 18, 0.34,
        )
        _centered_text(
            canvas, f"amp={amplitudes[panel]:.3f} rad", panel,
            base + 37, 0.34,
        )
        _centered_text(
            canvas, f"input freq={omega_input:.1f}", panel,
            base + 56, 0.31,
        )
        _centered_text(
            canvas, f"{shape}  alpha={alpha:.1f}", panel,
            base + 75, 0.31,
        )
    return canvas


def _write_video_atomic(path: Path, frames: Sequence[np.ndarray], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.partial.mp4")
    if temporary.exists():
        temporary.unlink()
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create video: {temporary}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError(f"empty video: {temporary}")
    os.replace(temporary, path)


def _write_preview_atomic(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.png")
    if not cv2.imwrite(str(temporary), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"cannot create preview: {temporary}")
    os.replace(temporary, path)


def _saved_record(path: Path, video_path: Path) -> dict[str, Any] | None:
    if not path.is_file() or not video_path.is_file():
        return None
    record = _read_json(path)
    if record.get("program_version") != PROGRAM_VERSION:
        return None
    if record.get("output_video_sha256") != _sha256(video_path):
        return None
    if record.get("num_frames") != 64:
        return None
    return record


# ======================================================================
# RESUMABLE FULL GRID + COMPACT GALLERY
# ======================================================================


def build(
    experiment_root: Path,
    dataset_base: Path,
    config_path: Path,
    output_root: Path,
    phase_index: int,
    diffusion_repeat: int,
) -> Path:
    data, scan = _load_config(config_path)
    if not 0 <= phase_index < int(scan["phase_count"]):
        raise ValueError("phase index is outside the scan")
    if not 0 <= diffusion_repeat < int(scan["diffusion_repeats"]):
        raise ValueError("diffusion repeat is outside the scan")
    alphas = tuple(float(value) for value in scan["color_alphas"])
    omegas = tuple(float(value) for value in scan["frequencies_rad_s"])
    shapes = tuple(str(value) for value in scan["shapes"])
    if not shapes:
        raise ValueError("scan must contain at least one shape")
    expected_samples = (
        len(alphas)
        * len(omegas)
        * int(scan["phase_count"])
        * int(scan["diffusion_repeats"])
    )
    expected_conditions = len(shapes) * len(alphas) * len(omegas)
    prediction_start = int(data["prediction_start"])
    total_frames = int(data["render"]["num_frames"])
    future_frames = int(scan["future_frames"])
    fps = float(data["render"]["fps"])
    if total_frames - prediction_start != future_frames:
        raise ValueError("ground-truth future slice is not 64 frames")

    completed: list[dict[str, Any]] = []
    progress_path = output_root / "progress.json"
    for shape in shapes:
        dataset_root = dataset_base / shape
        metadata = [
            row
            for row in _read_metadata(dataset_root)
            if int(row["phase_index"]) == phase_index
            and int(row["diffusion_repeat"]) == diffusion_repeat
        ]
        if len(metadata) != len(alphas) * len(omegas):
            raise ValueError(
                f"{shape}: expected {len(alphas) * len(omegas)} selected rows, "
                f"got {len(metadata)}"
            )
        short_prediction_root = experiment_root / shape / "short" / "prediction"
        long_prediction_root = experiment_root / shape / "long" / "prediction"
        short_predictions = _prediction_index(
            short_prediction_root, "short", expected_samples
        )
        long_predictions = _prediction_index(
            long_prediction_root, "long", expected_samples
        )
        short_metrics = _metric_index(
            experiment_root / shape / "short" / "metrics",
            "short",
            expected_samples,
        )
        long_metrics = _metric_index(
            experiment_root / shape / "long" / "metrics",
            "long",
            expected_samples,
        )

        for row in sorted(
            metadata,
            key=lambda item: (
                float(item["omega_true"]),
                float(item["test_color_alpha_target"]),
            ),
        ):
            sample_id = row["sample_id"]
            alpha = float(row["test_color_alpha_target"])
            omega_input = float(row["omega_true"])
            stem = f"omega_{omega_input:.1f}_alpha_{alpha:.1f}"
            video_path = output_root / "full_grid" / shape / f"{stem}.mp4"
            preview_path = output_root / "previews" / shape / f"{stem}.png"
            record_path = output_root / "records" / shape / f"{stem}.json"
            saved = _saved_record(record_path, video_path)
            if saved is not None:
                completed.append(saved)
                continue

            if sample_id not in short_predictions or sample_id not in long_predictions:
                raise ValueError(f"short/long prediction missing: {sample_id}")
            short_metric = short_metrics[sample_id]
            long_metric = long_metrics[sample_id]
            identity_fields = (
                "sample_id", "physical_state_id", "pair_id", "omega_true",
                "amplitude_true",
            )
            mismatches = {
                field: (short_metric[field], long_metric[field])
                for field in identity_fields
                if short_metric[field] != long_metric[field]
            }
            if mismatches:
                raise ValueError(f"short/long metric mismatch: {mismatches}")

            gt_path = dataset_root / "videos" / "eval" / str(row["video"])
            short_path = short_prediction_root / str(
                short_predictions[sample_id]["prediction"]
            )
            long_path = long_prediction_root / str(
                long_predictions[sample_id]["prediction"]
            )
            gt = _read_video(gt_path, total_frames)[prediction_start:]
            short = _read_video(short_path, future_frames)
            long = _read_video(long_path, future_frames)
            frequencies = (
                omega_input,
                float(short_metric["omega_hat"]),
                float(long_metric["omega_hat"]),
            )
            amplitudes = (
                float(short_metric["amplitude_true"]),
                float(short_metric["amplitude_hat"]),
                float(long_metric["amplitude_hat"]),
            )
            if not all(math.isfinite(value) for value in frequencies + amplitudes):
                raise ValueError(f"non-finite fitted metric: {sample_id}")
            frames = [
                _compose_frame(
                    gt[index], short[index], long[index],
                    frequencies=frequencies, amplitudes=amplitudes,
                    shape=shape, alpha=alpha, omega_input=omega_input,
                )
                for index in range(future_frames)
            ]
            _write_video_atomic(video_path, frames, fps)
            _write_preview_atomic(preview_path, frames[0])
            record = {
                "program_version": PROGRAM_VERSION,
                "status": "complete",
                "shape": shape,
                "input_alpha": alpha,
                "omega_input_rad_s": omega_input,
                "phase_index": phase_index,
                "diffusion_repeat": diffusion_repeat,
                "sample_id": sample_id,
                "ground_truth_frequency_rad_s": frequencies[0],
                "short_frequency_rad_s": frequencies[1],
                "long_frequency_rad_s": frequencies[2],
                "ground_truth_amplitude_rad": amplitudes[0],
                "short_amplitude_rad": amplitudes[1],
                "long_amplitude_rad": amplitudes[2],
                "ground_truth_source": str(gt_path),
                "ground_truth_frame_slice": [prediction_start, total_frames],
                "short_source": str(short_path),
                "long_source": str(long_path),
                "num_frames": future_frames,
                "fps": fps,
                "panel_order": list(TITLES),
                "output_video": str(video_path),
                "output_video_sha256": _sha256(video_path),
                "output_video_size_bytes": video_path.stat().st_size,
                "preview": str(preview_path),
                "preview_sha256": _sha256(preview_path),
            }
            _atomic_json(record_path, record)
            completed.append(record)
            _atomic_json(
                progress_path,
                {
                    "status": "running",
                    "completed_videos": len(completed),
                    "expected_videos": expected_conditions,
                    "last_condition": f"{shape}/{stem}",
                },
            )
            print(
                f"{len(completed)}/{expected_conditions} {shape}/{stem}",
                flush=True,
            )

    if len(completed) != expected_conditions:
        raise ValueError(
            f"expected {expected_conditions} complete videos, got {len(completed)}"
        )
    by_key = {
        (row["shape"], row["omega_input_rad_s"], row["input_alpha"]): row
        for row in completed
    }
    gallery_sources = [
        by_key[(shape, omega, alpha)]
        for shape in shapes
        for omega in GALLERY_OMEGAS
        for alpha in GALLERY_ALPHAS
    ]
    gallery: list[dict[str, Any]] = []
    gallery_name = f"gallery_{len(gallery_sources)}"
    gallery_preview_name = f"{gallery_name}_previews"
    for record in gallery_sources:
        source_video = Path(str(record["output_video"]))
        source_preview = Path(str(record["preview"]))
        destination_video = (
            output_root
            / gallery_name
            / str(record["shape"])
            / source_video.name
        )
        destination_preview = (
            output_root
            / gallery_preview_name
            / str(record["shape"])
            / source_preview.name
        )
        _atomic_hardlink(source_video, destination_video)
        _atomic_hardlink(source_preview, destination_preview)
        gallery.append(
            {
                **record,
                "gallery_video": str(destination_video),
                "gallery_preview": str(destination_preview),
            }
        )
    _atomic_json(
        output_root / f"{gallery_name}_manifest.json",
        {
            "status": "complete",
            "num_videos": len(gallery),
            "selection": {
                "shapes": list(shapes),
                "omegas_rad_s": list(GALLERY_OMEGAS),
                "input_alphas": list(GALLERY_ALPHAS),
            },
            "records": gallery,
        },
    )
    manifest = {
        "status": "complete",
        "program_version": PROGRAM_VERSION,
        "num_videos": len(completed),
        "num_gallery_videos": len(gallery),
        "shapes": list(shapes),
        "phase_index": phase_index,
        "diffusion_repeat": diffusion_repeat,
        "short_long_ground_truth_identity_check": "passed",
        "ground_truth_alignment": {
            "source_frames": total_frames,
            "frame_slice": [prediction_start, total_frames],
            "future_frames": future_frames,
        },
        "panel_order": list(TITLES),
        "records": completed,
    }
    _atomic_json(output_root / "triptych_manifest.json", manifest)
    _atomic_json(
        progress_path,
        {
            "status": "complete",
            "completed_videos": len(completed),
            "expected_videos": expected_conditions,
            "gallery_videos": len(gallery),
            "manifest": "triptych_manifest.json",
        },
    )
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--dataset-base", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase-index", type=int, default=0)
    parser.add_argument("--diffusion-repeat", type=int, default=0)
    args = parser.parse_args()
    print(
        build(
            args.experiment_root,
            args.dataset_base,
            args.config,
            args.output_root,
            args.phase_index,
            args.diffusion_repeat,
        )
    )


if __name__ == "__main__":
    main()
