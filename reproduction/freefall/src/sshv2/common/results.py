"""Minimal prediction and evaluation result directory formats."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from sshv2.common.dataset import (
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


PREDICTION_FORMAT = "sshv2.predictions.v1"
METRICS_FORMAT = "sshv2.metrics.v1"


def finite_json(value: Any) -> Any:
    """Convert paths, array scalars, and non-finite numbers for strict JSON."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): finite_json(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return finite_json(item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _relative(value: str | Path, field: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be root-relative: {value}")
    return path.as_posix()


def write_predictions(
    root: Path,
    *,
    experiment: str,
    dataset: str,
    predictions: Iterable[Mapping[str, Any]],
    checkpoint: str | Path | None = None,
    config: str | Path | None = None,
    extra: Mapping[str, Any] | None = None,
    check_files: bool = True,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in predictions:
        row = dict(source)
        for field in ("prediction_id", "sample_id", "prediction"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(
                    f"Prediction row is missing {field}: {source}"
                )
        if row["prediction_id"] in seen:
            raise ValueError(
                f"Duplicate prediction_id: {row['prediction_id']}"
            )
        seen.add(row["prediction_id"])
        row["prediction"] = _relative(
            row["prediction"],
            "prediction",
        )
        attributes = row.get("attributes", {})
        if not isinstance(attributes, Mapping):
            raise ValueError("prediction attributes must be a mapping")
        row["attributes"] = dict(attributes)
        if check_files and not (root / row["prediction"]).is_file():
            raise FileNotFoundError(root / row["prediction"])
        rows.append(finite_json(row))
    write_jsonl(root / "predictions.jsonl", rows)
    manifest: dict[str, Any] = {
        "format": PREDICTION_FORMAT,
        "experiment": experiment,
        "dataset": dataset,
        "predictions_file": "predictions.jsonl",
        "num_predictions": len(rows),
    }
    if checkpoint is not None:
        manifest["checkpoint"] = str(checkpoint)
    if config is not None:
        manifest["config"] = str(config)
    if extra:
        manifest["attributes"] = finite_json(dict(extra))
    write_json(root / "predictions.json", manifest)
    return manifest


def read_predictions(
    root: Path,
    *,
    check_files: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(root / "predictions.json")
    if manifest.get("format") != PREDICTION_FORMAT:
        raise ValueError(
            f"Unsupported prediction format: {manifest.get('format')}"
        )
    rows = read_jsonl(
        root / manifest.get("predictions_file", "predictions.jsonl")
    )
    if manifest.get("num_predictions") != len(rows):
        raise ValueError(
            "predictions.json num_predictions does not match JSONL"
        )
    if check_files:
        missing = [
            row["prediction"]
            for row in rows
            if not (root / row["prediction"]).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing prediction files: {missing[:5]}"
            )
    return manifest, rows


def find_prediction(
    root: Path,
    *,
    sample_id: str,
    prediction_id: str | None = None,
) -> Path | None:
    """Resolve a prediction through the common manifest when it exists."""
    manifest_path = root / "predictions.json"
    if not manifest_path.is_file():
        return None
    _, rows = read_predictions(root)
    for row in rows:
        if (
            prediction_id is not None
            and row["prediction_id"] == prediction_id
        ) or row["sample_id"] == sample_id:
            return root / row["prediction"]
    return None


def write_metrics(
    root: Path,
    *,
    experiment: str,
    dataset: str,
    summary: Mapping[str, Any],
    samples: Iterable[Mapping[str, Any]],
    records: Mapping[
        str,
        Iterable[Mapping[str, Any]],
    ] | None = None,
    context: Mapping[str, Any] | None = None,
    legacy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the common metrics envelope and JSONL record files."""
    sample_rows = [
        finite_json(dict(row))
        for row in samples
    ]
    write_jsonl(root / "samples.jsonl", sample_rows)
    record_files: dict[str, str] = {"samples": "samples.jsonl"}
    for name, rows in (records or {}).items():
        if name == "samples":
            continue
        filename = f"{name}.jsonl"
        write_jsonl(
            root / filename,
            (finite_json(dict(row)) for row in rows),
        )
        record_files[name] = filename
    manifest: dict[str, Any] = {
        "format": METRICS_FORMAT,
        "experiment": experiment,
        "dataset": dataset,
        "summary": finite_json(dict(summary)),
        "num_samples": len(sample_rows),
        "record_files": record_files,
    }
    if context:
        manifest["context"] = finite_json(dict(context))
    if legacy:
        for key, value in legacy.items():
            if key in manifest:
                raise ValueError(
                    f"Legacy field collides with metrics field: {key}"
                )
            manifest[key] = finite_json(value)
    write_json(root / "metrics.json", manifest)
    return manifest


def read_metrics(
    root: Path,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    manifest = read_json(root / "metrics.json")
    if manifest.get("format") != METRICS_FORMAT:
        raise ValueError(
            f"Unsupported metrics format: {manifest.get('format')}"
        )
    records = {
        name: read_jsonl(root / filename)
        for name, filename in manifest["record_files"].items()
    }
    if manifest.get("num_samples") != len(records["samples"]):
        raise ValueError("metrics.json num_samples does not match JSONL")
    return manifest, records


def scoped_result_dir(
    results_root: Path,
    *,
    kind: str,
    experiment: str,
    dataset: str,
    run: str,
) -> Path:
    if kind not in {"predictions", "metrics"}:
        raise ValueError("kind must be predictions or metrics")
    return results_root / kind / experiment / dataset / run

