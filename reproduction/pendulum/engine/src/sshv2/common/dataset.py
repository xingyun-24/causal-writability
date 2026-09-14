"""Minimal dataset directory format and JSON/JSONL helpers."""
from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


DATASET_FORMAT = "sshv2.dataset.v1"
DATASET_MANIFEST = "dataset.json"
SAMPLES_FILE = "samples.jsonl"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except (TypeError, ValueError):
            pass
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write one UTF-8 JSON object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(
            _jsonable(payload),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> int:
    """Atomically write JSON objects, one per line, and return the count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            if not isinstance(row, Mapping):
                raise TypeError("JSONL rows must be mappings")
            handle.write(
                json.dumps(
                    _jsonable(row),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
            count += 1
    temporary.replace(path)
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected an object at {path}:{line_number}"
                )
            rows.append(value)
    return rows


def _relative_path(value: str | Path, field: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a root-relative path: {value}")
    return path.as_posix()


def _validate_sample(
    row: Mapping[str, Any],
    *,
    check_files_under: Path | None,
) -> dict[str, Any]:
    required = ("sample_id", "split", "video")
    missing = [
        field
        for field in required
        if not isinstance(row.get(field), str) or not row[field]
    ]
    if missing:
        raise ValueError(f"Dataset sample is missing {missing}: {row}")
    normalized = dict(row)
    normalized["video"] = _relative_path(row["video"], "video")
    metadata = row.get("metadata")
    if metadata is not None:
        normalized["metadata"] = _relative_path(
            metadata,
            "metadata",
        )
    attributes = row.get("attributes", {})
    if not isinstance(attributes, Mapping):
        raise ValueError("sample attributes must be a mapping")
    normalized["attributes"] = dict(attributes)
    if check_files_under is not None:
        for field in ("video", "metadata"):
            relative = normalized.get(field)
            if relative and not (check_files_under / relative).is_file():
                raise FileNotFoundError(
                    f"Missing sample {field}: "
                    f"{check_files_under / relative}"
                )
    return normalized


def write_dataset(
    root: Path,
    *,
    experiment: str,
    dataset: str,
    samples: Iterable[Mapping[str, Any]],
    extra: Mapping[str, Any] | None = None,
    check_files: bool = True,
) -> dict[str, Any]:
    """Write the common dataset manifest without changing experiment files."""
    if not experiment or not dataset:
        raise ValueError("experiment and dataset must be non-empty")
    normalized = [
        _validate_sample(
            row,
            check_files_under=root if check_files else None,
        )
        for row in samples
    ]
    identifiers = [row["sample_id"] for row in normalized]
    if len(identifiers) != len(set(identifiers)):
        duplicates = sorted(
            value
            for value, count in Counter(identifiers).items()
            if count > 1
        )
        raise ValueError(f"Duplicate sample_id values: {duplicates[:5]}")
    split_counts = Counter(row["split"] for row in normalized)
    subset_counts = Counter(
        row["subset"]
        for row in normalized
        if row.get("subset")
    )
    write_jsonl(root / SAMPLES_FILE, normalized)
    manifest: dict[str, Any] = {
        "format": DATASET_FORMAT,
        "experiment": experiment,
        "dataset": dataset,
        "samples_file": SAMPLES_FILE,
        "num_samples": len(normalized),
        "splits": dict(sorted(split_counts.items())),
        "subsets": dict(sorted(subset_counts.items())),
        "video_root": "videos",
    }
    if extra:
        manifest["attributes"] = dict(extra)
    write_json(root / DATASET_MANIFEST, manifest)
    return manifest


def read_dataset(
    root: Path,
    *,
    check_files: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(root / DATASET_MANIFEST)
    if manifest.get("format") != DATASET_FORMAT:
        raise ValueError(
            f"Unsupported dataset format: {manifest.get('format')}"
        )
    samples_path = root / manifest.get("samples_file", SAMPLES_FILE)
    samples = [
        _validate_sample(
            row,
            check_files_under=root if check_files else None,
        )
        for row in read_jsonl(samples_path)
    ]
    if manifest.get("num_samples") != len(samples):
        raise ValueError("dataset.json num_samples does not match samples")
    return manifest, samples


def samples_from_metadata_csv(
    dataset_root: Path,
    metadata_csv: Path,
    *,
    split: str,
    subset: str | None = None,
) -> list[dict[str, Any]]:
    """Convert an existing experiment CSV bank to the common outer fields."""
    bank = metadata_csv.parent
    bank_relative = bank.relative_to(dataset_root)
    subset_name = subset or bank_relative.as_posix()
    with metadata_csv.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    samples: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        source_id = row.get("record_id") or row.get("sample_id")
        sample_id = (
            str(source_id)
            if source_id
            else f"{subset_name}:{row.get('video') or f'row-{index:06d}'}"
        )
        video = bank_relative / row["video"]
        metadata_name = row.get("metadata")
        metadata = (
            (bank_relative / metadata_name).as_posix()
            if metadata_name
            else None
        )
        attributes = {
            key: value
            for key, value in row.items()
            if key not in {
                "video",
                "source",
                "metadata",
            }
        }
        sample: dict[str, Any] = {
            "sample_id": sample_id,
            "split": split,
            "subset": subset_name,
            "video": video.as_posix(),
            "attributes": attributes,
        }
        if metadata is not None:
            sample["metadata"] = metadata
        samples.append(sample)
    return samples


def scoped_dataset_dir(
    videos_root: Path,
    experiment: str,
    dataset: str,
) -> Path:
    """Return the recommended global ``videos/<experiment>/<dataset>`` root."""
    return videos_root / experiment / dataset
