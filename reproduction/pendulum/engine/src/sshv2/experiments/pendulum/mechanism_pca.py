#!/usr/bin/env python3
"""Run the Pendulum activation-PCA mechanism experiment.

The ``run`` command is the end-to-end entry point.  It freezes paired
aligned/conflict receivers from the Pendulum frequency scan, loads one Wan
checkpoint, records the selected block's condition-token activations on every
flow-matching step, fits uncentered PCA on fit receivers only, and injects
norm-matched held-out projections back into aligned rollouts.  It writes
baseline/intervention videos, frequency measurements, plots, and provenance.

The legacy ``prepare`` command remains available for fitting small externally
materialized NPZ direction banks.  The end-to-end path uses a chunked Gram
matrix fit and NPY memmaps so the roughly one-billion-value bank need not be
expanded to float64 in memory.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


class FrozenManifestError(ValueError):
    """The receiver manifest is not safe to use for a frozen intervention."""


REQUIRED_MANIFEST_FIELDS = (
    "receiver_id",
    "pair_id",
    "trajectory_id",
    "split",
    "target",
    "aligned_sample_id",
    "conflict_sample_id",
    "omega_true",
    "amplitude_true",
    "phase",
    "generation_seed",
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def manifest_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    """Hash every manifest field while preserving receiver order."""
    return hashlib.sha256(_canonical_json(list(rows))).hexdigest()


def direction_array_sha256(values: np.ndarray) -> str:
    """Hash dtype, shape, receiver order, and raw direction values."""
    array = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(_canonical_json({"dtype": array.dtype.str, "shape": array.shape}))
    for receiver in array:
        contiguous = np.ascontiguousarray(receiver)
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise FrozenManifestError(
                f"line {line_number}: each JSONL value must be an object"
            )
        rows.append(value)
    return rows


def validate_frozen_receiver_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    target: str,
    expected_fit: int | None = None,
    expected_heldout: int | None = None,
) -> dict[str, Any]:
    """Validate Pendulum receiver identity, pairing, physics, and split state."""
    if not rows:
        raise FrozenManifestError("receiver manifest is empty")
    if target not in ("frequency", "amplitude"):
        raise ValueError("target must be frequency or amplitude")

    counts = {"fit": 0, "heldout": 0}
    receiver_ids: set[str] = set()
    receiver_states: set[tuple[str, int]] = set()

    for index, row in enumerate(rows):
        missing = [field for field in REQUIRED_MANIFEST_FIELDS if field not in row]
        if missing:
            raise FrozenManifestError(
                f"row {index}: missing fields {', '.join(missing)}"
            )

        receiver_id = str(row["receiver_id"])
        pair_id = str(row["pair_id"])
        trajectory_id = str(row["trajectory_id"])
        aligned_id = str(row["aligned_sample_id"])
        conflict_id = str(row["conflict_sample_id"])
        if not all((receiver_id, pair_id, trajectory_id, aligned_id, conflict_id)):
            raise FrozenManifestError(f"row {index}: identifier fields cannot be empty")
        if aligned_id == conflict_id:
            raise FrozenManifestError(
                f"{receiver_id}: aligned and conflict sample IDs are identical"
            )
        if receiver_id in receiver_ids:
            raise FrozenManifestError(f"duplicate receiver_id: {receiver_id}")
        receiver_ids.add(receiver_id)

        try:
            generation_seed = int(row["generation_seed"])
            physical_values = np.asarray(
                [row["omega_true"], row["amplitude_true"], row["phase"]],
                dtype=np.float64,
            )
        except (TypeError, ValueError) as error:
            raise FrozenManifestError(
                f"{receiver_id}: invalid seed or physical parameter"
            ) from error
        if generation_seed < 0:
            raise FrozenManifestError(
                f"{receiver_id}: generation_seed must be non-negative"
            )
        if not np.isfinite(physical_values).all():
            raise FrozenManifestError(
                f"{receiver_id}: physical parameters must be finite"
            )
        if physical_values[0] <= 0.0 or physical_values[1] <= 0.0:
            raise FrozenManifestError(
                f"{receiver_id}: omega_true and amplitude_true must be positive"
            )

        receiver_state = (trajectory_id, generation_seed)
        if receiver_state in receiver_states:
            raise FrozenManifestError(
                f"duplicate trajectory/generation receiver: {receiver_state}"
            )
        receiver_states.add(receiver_state)

        split = str(row["split"])
        if split not in counts:
            raise FrozenManifestError(
                f"{receiver_id}: split must be fit or heldout, got {split!r}"
            )
        counts[split] += 1
        if str(row["target"]) != target:
            raise FrozenManifestError(
                f"{receiver_id}: target {row['target']!r} does not match {target!r}"
            )

    if expected_fit is not None and counts["fit"] != expected_fit:
        raise FrozenManifestError(
            f"fit count {counts['fit']} != expected {expected_fit}"
        )
    if expected_heldout is not None and counts["heldout"] != expected_heldout:
        raise FrozenManifestError(
            f"heldout count {counts['heldout']} != expected {expected_heldout}"
        )
    if counts["fit"] < 2:
        raise FrozenManifestError("at least two fit receivers are required")
    if counts["heldout"] < 1:
        raise FrozenManifestError("at least one heldout receiver is required")

    return {
        "rows": len(rows),
        "fit": counts["fit"],
        "heldout": counts["heldout"],
        "target": target,
        "sha256": manifest_sha256(rows),
    }


def _direction_matrix(
    directions: np.ndarray | Sequence[np.ndarray],
) -> tuple[np.ndarray, tuple[int, int, int]]:
    values = np.asarray(directions, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError(
            "directions must have shape "
            "[receiver, FM step, condition token, hidden]"
        )
    if values.shape[0] < 2 or min(values.shape[1:]) < 1:
        raise ValueError("direction dimensions must all be positive")
    if not np.isfinite(values).all():
        raise ValueError("directions contain non-finite values")
    shape = tuple(int(value) for value in values.shape[1:])
    return values.reshape(values.shape[0], -1), shape


@dataclass(frozen=True)
class PCABasis:
    """Uncentered whole-direction PCA basis."""

    components: np.ndarray
    singular_values: np.ndarray
    direction_shape: tuple[int, int, int]

    def project(self, direction: np.ndarray, rank: int) -> np.ndarray:
        if rank < 1 or rank > len(self.components):
            raise ValueError(
                f"rank must be in [1, {len(self.components)}], got {rank}"
            )
        value = np.asarray(direction, dtype=np.float64)
        if tuple(value.shape) != self.direction_shape:
            raise ValueError(
                f"direction shape {value.shape} != {self.direction_shape}"
            )
        flat = value.reshape(-1)
        basis = self.components[:rank].reshape(rank, -1)
        return ((flat @ basis.T) @ basis).reshape(self.direction_shape)


def fit_uncentered_pca(
    directions: np.ndarray | Sequence[np.ndarray],
) -> PCABasis:
    """Fit economy SVD over complete receiver directions without centering."""
    matrix, shape = _direction_matrix(directions)
    _left, singular_values, right = np.linalg.svd(
        matrix,
        full_matrices=False,
    )
    components = right.reshape((right.shape[0], *shape))
    return PCABasis(components, singular_values, shape)


def norm_match(
    projected: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Match a low-rank projection to its receiver's full-direction norm."""
    projected_array = np.asarray(projected, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if projected_array.shape != reference_array.shape:
        raise ValueError("projected and reference directions must have one shape")
    projected_norm = float(np.linalg.norm(projected_array.reshape(-1)))
    reference_norm = float(np.linalg.norm(reference_array.reshape(-1)))
    if projected_norm == 0.0:
        raise ValueError("cannot norm-match a zero projection")
    scale = reference_norm / projected_norm
    retained_energy = (
        projected_norm * projected_norm / (reference_norm * reference_norm)
        if reference_norm > 0.0
        else 0.0
    )
    return projected_array * scale, {
        "projected_norm": projected_norm,
        "reference_norm": reference_norm,
        "scale": scale,
        "retained_energy_before_norm_match": retained_energy,
    }


@dataclass(frozen=True)
class FMConditionTokenAdditivePatch:
    """Apply one ``[FM, condition token, hidden]`` edit to a hidden state."""

    direction: Any
    num_condition_tokens: int
    strength: float = 1.0

    def apply(self, hidden: Any, fm_step: int) -> Any:
        """Add the selected FM edit and leave every future token untouched."""
        import torch

        if not isinstance(hidden, torch.Tensor):
            raise TypeError("hidden must be a torch.Tensor")
        if not isinstance(self.direction, torch.Tensor):
            raise TypeError("direction must be a torch.Tensor")
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [batch, token, hidden]")
        if self.direction.ndim != 3:
            raise ValueError(
                "direction must have shape [FM step, condition token, hidden]"
            )
        if fm_step < 0 or fm_step >= self.direction.shape[0]:
            raise IndexError(f"FM step {fm_step} is outside the direction array")
        if self.num_condition_tokens != self.direction.shape[1]:
            raise ValueError("condition-token count differs from direction shape")
        if hidden.shape[1] < self.num_condition_tokens:
            raise ValueError("hidden state is shorter than the condition prefix")
        if hidden.shape[2] != self.direction.shape[2]:
            raise ValueError("hidden width differs from direction width")

        edit = self.direction[fm_step].to(
            device=hidden.device,
            dtype=hidden.dtype,
        )
        result = hidden.clone()
        result[:, : self.num_condition_tokens] = (
            hidden[:, : self.num_condition_tokens]
            + float(self.strength) * edit.unsqueeze(0)
        )
        return result


def _validate_cli_shape(
    directions: np.ndarray,
    *,
    fm_steps: int,
    condition_tokens: int,
) -> None:
    if directions.ndim != 4:
        raise ValueError(
            "directions must have shape "
            "[receiver, FM step, condition token, hidden]"
        )
    if directions.shape[1] != fm_steps:
        raise ValueError(
            f"direction FM steps {directions.shape[1]} != configured {fm_steps}"
        )
    if directions.shape[2] != condition_tokens:
        raise ValueError(
            "direction condition tokens "
            f"{directions.shape[2]} != configured {condition_tokens}"
        )


def prepare_mechanism_artifacts(args: argparse.Namespace) -> None:
    """Fit the basis and materialize receiver-specific held-out edits."""
    rows = read_jsonl(args.manifest)
    manifest_audit = validate_frozen_receiver_manifest(
        rows,
        target=args.target,
        expected_fit=args.expected_fit,
        expected_heldout=args.expected_heldout,
    )

    with np.load(args.directions, allow_pickle=False) as archive:
        if args.array_key not in archive:
            raise KeyError(
                f"{args.directions} does not contain {args.array_key!r}"
            )
        directions = np.asarray(archive[args.array_key])
    if len(directions) != len(rows):
        raise ValueError(
            f"{len(directions)} directions != {len(rows)} manifest rows"
        )
    _validate_cli_shape(
        directions,
        fm_steps=args.fm_steps,
        condition_tokens=args.condition_tokens,
    )

    fit_indices = np.asarray(
        [index for index, row in enumerate(rows) if row["split"] == "fit"],
        dtype=np.int64,
    )
    heldout_indices = np.asarray(
        [
            index
            for index, row in enumerate(rows)
            if row["split"] == "heldout"
        ],
        dtype=np.int64,
    )
    basis = fit_uncentered_pca(directions[fit_indices])
    if args.rank > len(basis.components):
        raise ValueError(
            f"rank {args.rank} exceeds fitted rank {len(basis.components)}"
        )

    heldout_edits: list[np.ndarray] = []
    projection_audit: list[dict[str, Any]] = []
    for index in heldout_indices:
        reference = directions[int(index)]
        projected = basis.project(reference, args.rank)
        edit, norm_audit = norm_match(projected, reference)
        heldout_edits.append(edit)
        projection_audit.append(
            {
                "manifest_index": int(index),
                "receiver_id": str(rows[int(index)]["receiver_id"]),
                **norm_audit,
            }
        )

    output_payload = {
        "components": basis.components,
        "singular_values": basis.singular_values,
        "fit_indices": fit_indices,
        "heldout_indices": heldout_indices,
        "heldout_receiver_ids": np.asarray(
            [rows[int(index)]["receiver_id"] for index in heldout_indices]
        ),
        "heldout_edits": np.stack(heldout_edits, axis=0),
        "projection_rank": np.asarray(args.rank, dtype=np.int64),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **output_payload)

    audit = {
        "protocol": "pendulum_activation_pca_v1",
        "manifest": manifest_audit,
        "directions_sha256": direction_array_sha256(directions),
        "direction_shape": list(basis.direction_shape),
        "checkpoint_id": args.checkpoint_id,
        "history": args.history,
        "target": args.target,
        "direction_definition": args.direction_definition,
        "block_index_zero_based": args.block_index,
        "fm_steps": args.fm_steps,
        "condition_tokens": args.condition_tokens,
        "hidden_size": int(directions.shape[3]),
        "projection_rank": args.rank,
        "projection_audit": projection_audit,
        "output": str(args.out),
    }
    audit_path = args.audit or args.out.with_suffix(".audit.json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


PROGRAM_VERSION = "pendulum_activation_pca_v3"
DEFAULT_RANKS = (1, 2, 4, 8)
DEFAULT_HELDOUT_PHASES = (6, 7)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(
        _canonical_json({"dtype": value.dtype.str, "shape": value.shape})
    )
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _parse_int_tuple(value: str, *, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{name} must be a comma-separated integer list"
        ) from error
    if not values or any(item < 0 for item in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(
            f"{name} must contain unique non-negative integers"
        )
    return values


def _parse_ranks(value: str) -> tuple[int, ...]:
    ranks = _parse_int_tuple(value, name="ranks")
    if any(rank < 1 for rank in ranks):
        raise argparse.ArgumentTypeError("ranks must be positive")
    return tuple(sorted(ranks))


def _metadata_path(root: Path) -> Path:
    direct = root / "metadata.csv"
    nested = root / "videos" / "eval" / "metadata.csv"
    if direct.is_file():
        return direct
    if nested.is_file():
        return nested
    raise FileNotFoundError(f"metadata.csv not found below {root}")


def _read_metadata(root: Path) -> tuple[Path, list[dict[str, str]]]:
    path = _metadata_path(root)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty metadata: {path}")
    return path.parent, rows


def _one_row(
    rows: Sequence[dict[str, str]],
    *,
    omega: float,
    alpha: float,
    phase_index: int,
    repeat: int,
) -> dict[str, str]:
    selected = [
        row
        for row in rows
        if math.isclose(float(row["omega_true"]), omega, abs_tol=1e-8)
        and math.isclose(
            float(row["test_color_alpha_target"]), alpha, abs_tol=1e-8
        )
        and int(row["phase_index"]) == phase_index
        and int(row["diffusion_repeat"]) == repeat
    ]
    if len(selected) != 1:
        raise FrozenManifestError(
            "expected exactly one row for "
            f"omega={omega}, alpha={alpha}, phase={phase_index}, "
            f"repeat={repeat}; got {len(selected)}"
        )
    return selected[0]


def _assert_pair(
    aligned: Mapping[str, str],
    conflict: Mapping[str, str],
) -> None:
    exact_fields = (
        "physical_state_id",
        "trajectory_id",
        "pair_id",
        "base_seed",
        "omega_true",
        "amplitude_true",
        "phase",
        "phase_index",
        "diffusion_repeat",
    )
    mismatches = {
        field: (aligned.get(field), conflict.get(field))
        for field in exact_fields
        if aligned.get(field) != conflict.get(field)
    }
    if mismatches:
        raise FrozenManifestError(f"aligned/conflict pair mismatch: {mismatches}")
    if aligned["sample_id"] == conflict["sample_id"]:
        raise FrozenManifestError("aligned/conflict sample IDs are identical")


def build_receiver_manifest(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Build the balanced, frozen 24-fit/8-held-out receiver bank."""
    low_bank, low_rows = _read_metadata(args.low_dataset_root)
    high_bank, high_rows = _read_metadata(args.high_dataset_root)
    all_rows = [*low_rows, *high_rows]
    model_names = {row["model_name"] for row in all_rows}
    if model_names != {args.model_name}:
        raise FrozenManifestError(
            f"dataset model names {model_names} != {args.model_name!r}"
        )
    training_ids = {row["training_manifest_id"] for row in all_rows}
    if len(training_ids) != 1:
        raise FrozenManifestError(
            f"dataset training manifests are not unique: {training_ids}"
        )

    receivers: list[dict[str, Any]] = []
    # ``low_dataset_root`` is the appearance route aligned with the low target
    # (red circle for the color+shape model); ``high_dataset_root`` is the
    # route aligned with the high target (blue square).  For the circle-only
    # model both roots are intentionally identical.  Selecting conflict rows
    # from the opposite root therefore flips color only for the circle model,
    # and flips the complete color+shape route for the color+shape model.
    settings = (
        (
            "low",
            args.low_omega,
            0.0,
            1.0,
            low_bank,
            low_rows,
            high_bank,
            high_rows,
        ),
        (
            "high",
            args.high_omega,
            1.0,
            0.0,
            high_bank,
            high_rows,
            low_bank,
            low_rows,
        ),
    )
    for (
        target_label,
        omega,
        aligned_alpha,
        conflict_alpha,
        aligned_bank,
        aligned_rows,
        conflict_bank,
        conflict_rows,
    ) in settings:
        for phase_index in range(args.phase_count):
            for repeat in range(args.repeats):
                aligned = _one_row(
                    aligned_rows,
                    omega=omega,
                    alpha=aligned_alpha,
                    phase_index=phase_index,
                    repeat=repeat,
                )
                conflict = _one_row(
                    conflict_rows,
                    omega=omega,
                    alpha=conflict_alpha,
                    phase_index=phase_index,
                    repeat=repeat,
                )
                _assert_pair(aligned, conflict)
                receiver_id = (
                    f"{target_label}_phase{phase_index:02d}_repeat{repeat:02d}"
                )
                receivers.append(
                    {
                        "receiver_id": receiver_id,
                        "pair_id": aligned["pair_id"],
                        "physical_state_id": aligned["physical_state_id"],
                        "trajectory_id": aligned["trajectory_id"],
                        "split": (
                            "heldout"
                            if phase_index in args.heldout_phases
                            else "fit"
                        ),
                        "target": "frequency",
                        "target_label": target_label,
                        "target_index": int(aligned["target_index"]),
                        "aligned_sample_id": aligned["sample_id"],
                        "conflict_sample_id": conflict["sample_id"],
                        "aligned_color": aligned["color_label"],
                        "conflict_color": conflict["color_label"],
                        "aligned_shape": aligned["shape_label"],
                        "conflict_shape": conflict["shape_label"],
                        "aligned_alpha": float(aligned_alpha),
                        "conflict_alpha": float(conflict_alpha),
                        "omega_true": float(aligned["omega_true"]),
                        "amplitude_true": float(aligned["amplitude_true"]),
                        "phase": float(aligned["phase"]),
                        "phase_index": phase_index,
                        "diffusion_repeat": repeat,
                        "generation_seed": int(aligned["base_seed"])
                        + args.seed_offset,
                        "base_seed": int(aligned["base_seed"]),
                        "theta_star": float(aligned["theta_star"]),
                        "aligned_video": str(
                            aligned_bank / aligned["video"]
                        ),
                        "conflict_video": str(
                            conflict_bank / conflict["video"]
                        ),
                        "training_manifest_id": aligned[
                            "training_manifest_id"
                        ],
                        "test_manifest_id": aligned["test_manifest_id"],
                        "model_name": args.model_name,
                    }
                )

    validate_frozen_receiver_manifest(
        receivers,
        target="frequency",
        expected_fit=(
            (args.phase_count - len(args.heldout_phases)) * args.repeats * 2
        ),
        expected_heldout=len(args.heldout_phases) * args.repeats * 2,
    )
    if {row["target_label"] for row in receivers} != {"low", "high"}:
        raise FrozenManifestError("receiver bank is not target-balanced")
    return receivers


def _freeze_manifest(
    path: Path,
    generated: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if path.is_file():
        existing = read_jsonl(path)
        if existing != list(generated):
            raise FrozenManifestError(
                f"existing frozen manifest differs from regenerated bank: {path}"
            )
        return existing
    _atomic_text(
        path,
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in generated
        ),
    )
    return [dict(row) for row in generated]


class _BlockActivationHook:
    """Capture or patch one block output, indexed by FM call order."""

    def __init__(
        self,
        *,
        condition_tokens: int,
        expected_steps: int,
        edit: Any | None = None,
        strength: float = 1.0,
        capture: bool = False,
    ) -> None:
        self.condition_tokens = condition_tokens
        self.expected_steps = expected_steps
        self.edit = edit
        self.strength = strength
        self.capture = capture
        self.calls = 0
        self.values: list[Any] = []

    def __call__(self, _module: Any, _inputs: Any, output: Any) -> Any:
        import torch

        if not isinstance(output, torch.Tensor) or output.ndim != 3:
            raise RuntimeError(
                "selected Wan block must return [batch, token, hidden]"
            )
        if output.shape[0] != 1:
            raise RuntimeError("mechanism experiment requires batch size one")
        if output.shape[1] < self.condition_tokens:
            raise RuntimeError(
                f"block has {output.shape[1]} tokens, fewer than the "
                f"{self.condition_tokens} condition tokens"
            )
        if self.calls >= self.expected_steps:
            raise RuntimeError("selected block ran more FM calls than expected")
        if self.capture:
            self.values.append(
                output[0, : self.condition_tokens]
                .detach()
                .to(device="cpu", dtype=torch.float16)
            )
        if self.edit is not None:
            if tuple(self.edit.shape) != (
                self.expected_steps,
                self.condition_tokens,
                output.shape[2],
            ):
                raise RuntimeError(
                    f"edit shape {tuple(self.edit.shape)} is incompatible with "
                    f"block output {tuple(output.shape)}"
                )
            result = output.clone()
            result[:, : self.condition_tokens] = (
                output[:, : self.condition_tokens]
                + self.strength * self.edit[self.calls].unsqueeze(0)
            )
            output = result
        self.calls += 1
        return output

    def stacked(self) -> np.ndarray:
        if not self.capture:
            raise RuntimeError("hook was not configured to capture activations")
        if self.calls != self.expected_steps or len(self.values) != self.expected_steps:
            raise RuntimeError(
                f"captured {len(self.values)}/{self.expected_steps} FM activations"
            )
        return np.stack([value.numpy() for value in self.values], axis=0)


def _load_runtime(
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Any, Any]:
    import yaml
    from sshv2.experiments.pendulum.data import config_from_mapping
    from sshv2.wan.config import StandardTrainingConfig
    from sshv2.wan.trainer import WanTrainingModule

    experiment = yaml.safe_load(
        args.experiment_config.read_text(encoding="utf-8")
    )
    if not isinstance(experiment, Mapping) or not isinstance(
        experiment.get("data"), Mapping
    ):
        raise ValueError("experiment config must contain a data mapping")
    data_config = config_from_mapping(experiment["data"])
    dataset_training_ids = {
        str(row["training_manifest_id"]) for row in rows
    }
    if len(dataset_training_ids) != 1:
        raise ValueError(
            "receiver manifest contains multiple training manifest IDs"
        )
    dataset_training_id = dataset_training_ids.pop()
    if data_config.model_name != args.model_name:
        raise ValueError(
            f"experiment model {data_config.model_name!r} != "
            f"{args.model_name!r}"
        )
    if data_config.training_manifest_id != dataset_training_id:
        raise ValueError(
            "experiment config training manifest does not match receiver bank"
        )

    training_document = yaml.safe_load(
        args.training_config.read_text(encoding="utf-8")
    )
    if not isinstance(training_document, Mapping) or not isinstance(
        training_document.get("pendulum"), Mapping
    ):
        raise ValueError("training config lacks a pendulum binding")
    binding = training_document["pendulum"]
    expected_binding = {
        "model_name": args.model_name,
        "history": args.history,
    }
    mismatches = {
        key: (binding.get(key), expected)
        for key, expected in expected_binding.items()
        if binding.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"training binding mismatch: {mismatches}")
    source_training_id = str(
        binding.get(
            "source_training_manifest_id",
            binding.get("training_manifest_id", ""),
        )
    )
    if source_training_id != dataset_training_id:
        raise ValueError(
            "checkpoint source-training manifest does not match receiver bank: "
            f"{source_training_id!r} != {dataset_training_id!r}"
        )

    training = StandardTrainingConfig.from_file(args.training_config)
    condition_latent_frames = (data_config.prediction_start - 1) // 4 + 1
    if training.model.num_condition_frames != condition_latent_frames:
        raise ValueError(
            "training condition latent count does not match Pendulum config"
        )
    if condition_latent_frames * (data_config.render.height // 16) * (
        data_config.render.width // 16
    ) != args.condition_tokens:
        raise ValueError(
            "configured condition token count is inconsistent with the "
            "Pendulum latent grid"
        )
    training.model.dit.ckpt_file = args.checkpoint
    model = WanTrainingModule(
        dit_config=training.model.dit,
        vae_config=training.model.vae,
        no_encoding=False,
        num_condition_frames=condition_latent_frames,
        num_inference_steps=args.steps,
        pipeline_type=training.model.pipe,
        pipeline_kwargs=training.model.pipe_kwargs,
    )
    pipe = model.pipe
    pipe.to(args.device)
    pipe.load_models_to_device(("dit", "vae"))
    pipe.pre_encoded_(False)
    if args.block_index >= len(pipe.dit.blocks):
        raise ValueError(
            f"block index {args.block_index} outside {len(pipe.dit.blocks)} blocks"
        )
    if int(pipe.dit.dim) != args.hidden_size:
        raise ValueError(
            f"configured hidden size {args.hidden_size} != model {pipe.dit.dim}"
        )
    return pipe, data_config


def _condition_latents(
    pipe: Any,
    data_config: Any,
    video_path: Path,
    *,
    history: str,
) -> Any:
    import torch
    from sshv2.experiments.pendulum.data import (
        apply_short_history_mask,
        load_video,
    )

    raw = load_video(
        video_path,
        expected_frames=data_config.render.num_frames,
    )
    source = (
        apply_short_history_mask(raw, data_config)
        if history == "short"
        else raw
    )
    condition = (
        torch.from_numpy(source[: data_config.prediction_start].copy())
        .permute(3, 0, 1, 2)
        .float()
        .div(127.5)
        .sub(1.0)
        .unsqueeze(0)
        .to(device=pipe.device, dtype=pipe.torch_dtype)
    )
    with torch.inference_mode():
        encoded = pipe.vae.encode(
            condition,
            device=pipe.device,
            tiled=False,
            tile_size=(30, 52),
            tile_stride=(15, 26),
        )
    expected = (data_config.prediction_start - 1) // 4 + 1
    encoded = encoded[:, :, :expected].to(
        device=pipe.device,
        dtype=pipe.torch_dtype,
    )
    if encoded.shape[2] != expected:
        raise RuntimeError(
            f"condition encoding has {encoded.shape[2]} frames, expected {expected}"
        )
    return encoded


def _denoise(
    pipe: Any,
    data_config: Any,
    condition: Any,
    *,
    seed: int,
    steps: int,
    block_index: int,
    condition_tokens: int,
    capture: bool = False,
    edit: Any | None = None,
    strength: float = 1.0,
) -> tuple[Any, np.ndarray | None]:
    import torch

    pipe.scheduler.set_timesteps(
        steps,
        denoising_strength=1.0,
        shift=5.0,
    )
    latent_frames = (data_config.render.num_frames - 1) // 4 + 1
    shape = (
        1,
        pipe.vae.z_dim,
        latent_frames,
        data_config.render.height // pipe.vae.upsampling_factor,
        data_config.render.width // pipe.vae.upsampling_factor,
    )
    latents = pipe.generate_noise(
        shape,
        seed=seed,
        rand_device=pipe.device,
    ).to(device=pipe.device, dtype=pipe.torch_dtype)
    hook = _BlockActivationHook(
        condition_tokens=condition_tokens,
        expected_steps=steps,
        edit=edit,
        strength=strength,
        capture=capture,
    )
    handle = pipe.dit.blocks[block_index].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            for step_index, timestep in enumerate(pipe.scheduler.timesteps):
                timestep_input = timestep.unsqueeze(0).to(
                    device=pipe.device,
                    dtype=pipe.torch_dtype,
                )
                latents[:, :, : condition.shape[2]] = condition
                noise_pred = pipe.model_fn(
                    dit=pipe.dit,
                    latents=latents,
                    timestep=timestep_input,
                )
                latents = pipe.scheduler.step(
                    noise_pred,
                    pipe.scheduler.timesteps[step_index],
                    latents,
                )
                latents[:, :, : condition.shape[2]] = condition
    finally:
        handle.remove()
    if hook.calls != steps:
        raise RuntimeError(f"selected block ran {hook.calls}/{steps} times")
    activations = hook.stacked() if capture else None
    return latents.detach(), activations


def _write_future_video(
    pipe: Any,
    data_config: Any,
    latents: Any,
    destination: Path,
) -> None:
    import torch
    from sshv2.experiments.pendulum.data import load_video, write_video

    with torch.inference_mode():
        decoded = pipe.vae.decode(
            latents,
            device=pipe.device,
            tiled=False,
            tile_size=(30, 52),
            tile_stride=(15, 26),
        )[0]
    frames = (
        decoded.detach()
        .float()
        .cpu()
        .permute(1, 2, 3, 0)
        .add(1.0)
        .mul(127.5)
        .clamp(0, 255)
        .byte()
        .numpy()
    )
    future = frames[data_config.prediction_start :]
    if future.shape[0] != data_config.future_frames:
        raise RuntimeError(
            f"decoded {future.shape[0]} future frames, expected "
            f"{data_config.future_frames}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_video(destination, future, data_config.render.fps)
    load_video(destination, expected_frames=data_config.future_frames)


def _direction_bank(
    path: Path,
    *,
    receivers: int,
    steps: int,
    condition_tokens: int,
    hidden_size: int,
) -> np.memmap:
    shape = (receivers, steps, condition_tokens, hidden_size)
    if path.is_file():
        bank = np.load(path, mmap_mode="r+")
        if bank.shape != shape or bank.dtype != np.float16:
            raise ValueError(
                f"existing direction bank has {bank.shape}/{bank.dtype}, "
                f"expected {shape}/float16"
            )
        return bank
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float16,
        shape=shape,
    )


def _baseline_path(out_root: Path, receiver_id: str, kind: str) -> Path:
    return out_root / "videos" / receiver_id / f"{kind}.mp4"


def _extract_directions(
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    pipe: Any,
    data_config: Any,
) -> np.memmap:
    import torch

    bank_path = args.out_root / "directions.npy"
    bank = _direction_bank(
        bank_path,
        receivers=len(rows),
        steps=args.steps,
        condition_tokens=args.condition_tokens,
        hidden_size=args.hidden_size,
    )
    records_root = args.out_root / "extraction_records"
    records_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for index, row in enumerate(rows):
        receiver_id = str(row["receiver_id"])
        record_path = records_root / f"{receiver_id}.json"
        aligned_path = _baseline_path(args.out_root, receiver_id, "aligned")
        conflict_path = _baseline_path(args.out_root, receiver_id, "conflict")
        if record_path.is_file():
            record = _read_json(record_path)
            if record.get("direction_sha256") != _sha256_array(bank[index]):
                raise ValueError(
                    f"saved direction checksum mismatch for {receiver_id}"
                )
            if row["split"] == "heldout" and not (
                aligned_path.is_file() and conflict_path.is_file()
            ):
                raise FileNotFoundError(
                    f"held-out baseline videos are incomplete for {receiver_id}"
                )
            print(
                f"extract {index + 1}/{len(rows)} {receiver_id} resume",
                flush=True,
            )
            continue

        condition_aligned = _condition_latents(
            pipe,
            data_config,
            Path(str(row["aligned_video"])),
            history=args.history,
        )
        aligned_latents, aligned_activation = _denoise(
            pipe,
            data_config,
            condition_aligned,
            seed=int(row["generation_seed"]),
            steps=args.steps,
            block_index=args.block_index,
            condition_tokens=args.condition_tokens,
            capture=True,
        )
        if row["split"] == "heldout":
            _write_future_video(
                pipe,
                data_config,
                aligned_latents,
                aligned_path,
            )
        del aligned_latents, condition_aligned

        condition_conflict = _condition_latents(
            pipe,
            data_config,
            Path(str(row["conflict_video"])),
            history=args.history,
        )
        conflict_latents, conflict_activation = _denoise(
            pipe,
            data_config,
            condition_conflict,
            seed=int(row["generation_seed"]),
            steps=args.steps,
            block_index=args.block_index,
            condition_tokens=args.condition_tokens,
            capture=True,
        )
        if row["split"] == "heldout":
            _write_future_video(
                pipe,
                data_config,
                conflict_latents,
                conflict_path,
            )
        del conflict_latents, condition_conflict
        if aligned_activation is None or conflict_activation is None:
            raise AssertionError("activation capture unexpectedly returned None")
        direction = (
            conflict_activation.astype(np.float32)
            - aligned_activation.astype(np.float32)
        ).astype(np.float16)
        bank[index] = direction
        bank.flush()
        record = {
            "receiver_id": receiver_id,
            "manifest_index": index,
            "direction_definition": "conflict-minus-aligned",
            "direction_shape": list(direction.shape),
            "direction_dtype": direction.dtype.str,
            "direction_sha256": _sha256_array(direction),
            "aligned_sample_id": row["aligned_sample_id"],
            "conflict_sample_id": row["conflict_sample_id"],
            "generation_seed": row["generation_seed"],
            "split": row["split"],
        }
        _write_json(record_path, record)
        elapsed = max(time.monotonic() - started, 1e-9)
        _write_json(
            args.out_root / "progress.json",
            {
                "status": "extracting",
                "completed_receivers": index + 1,
                "expected_receivers": len(rows),
                "last_receiver_id": receiver_id,
                "elapsed_seconds": elapsed,
            },
        )
        print(
            f"extract {index + 1}/{len(rows)} {receiver_id} "
            f"elapsed={elapsed / 60.0:.1f}m",
            flush=True,
        )
        del aligned_activation, conflict_activation, direction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return bank


@dataclass(frozen=True)
class ChunkedPCAArtifacts:
    components_path: Path
    singular_values: np.ndarray
    heldout_indices: np.ndarray
    coefficients: np.ndarray
    reference_norms: np.ndarray
    direction_shape: tuple[int, int, int]


def _fit_chunked_pca(
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    directions: np.memmap,
) -> ChunkedPCAArtifacts:
    pca_root = args.out_root / "pca"
    pca_root.mkdir(parents=True, exist_ok=True)
    components_path = pca_root / "components.npy"
    singular_path = pca_root / "singular_values.npy"
    coefficients_path = pca_root / "heldout_coefficients.npy"
    norms_path = pca_root / "heldout_reference_norms.npy"
    audit_path = pca_root / "audit.json"
    direction_shape = tuple(int(value) for value in directions.shape[1:])
    fit_indices = np.asarray(
        [index for index, row in enumerate(rows) if row["split"] == "fit"],
        dtype=np.int64,
    )
    heldout_indices = np.asarray(
        [
            index
            for index, row in enumerate(rows)
            if row["split"] == "heldout"
        ],
        dtype=np.int64,
    )
    maximum_rank = max(args.ranks)

    if audit_path.is_file():
        audit = _read_json(audit_path)
        expected = {
            "manifest_sha256": manifest_sha256(rows),
            "maximum_rank": maximum_rank,
            "direction_shape": list(direction_shape),
        }
        if any(audit.get(key) != value for key, value in expected.items()):
            raise ValueError("existing PCA audit does not match this run")
        components = np.load(components_path, mmap_mode="r")
        singular_values = np.load(singular_path)
        coefficients = np.load(coefficients_path)
        reference_norms = np.load(norms_path)
        if components.shape != (maximum_rank, *direction_shape):
            raise ValueError("existing PCA component shape mismatch")
        print("pca resume", flush=True)
        return ChunkedPCAArtifacts(
            components_path,
            singular_values,
            heldout_indices,
            coefficients,
            reference_norms,
            direction_shape,
        )

    matrix = directions.reshape(len(rows), -1)
    flat_size = matrix.shape[1]
    gram = np.zeros((len(fit_indices), len(fit_indices)), dtype=np.float64)
    reference_sq = np.zeros(len(heldout_indices), dtype=np.float64)
    for start in range(0, flat_size, args.pca_chunk_values):
        stop = min(flat_size, start + args.pca_chunk_values)
        fit_chunk = np.asarray(
            matrix[fit_indices, start:stop],
            dtype=np.float32,
        )
        heldout_chunk = np.asarray(
            matrix[heldout_indices, start:stop],
            dtype=np.float32,
        )
        gram += fit_chunk @ fit_chunk.T
        reference_sq += np.einsum(
            "ij,ij->i", heldout_chunk, heldout_chunk, dtype=np.float64
        )
    eigenvalues, left = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    left = left[:, order]
    singular_values = np.sqrt(eigenvalues)
    positive = singular_values > max(singular_values[0] * 1e-10, 1e-12)
    if int(positive.sum()) < maximum_rank:
        raise ValueError(
            f"only {int(positive.sum())} non-degenerate PCA components; "
            f"rank {maximum_rank} requested"
        )

    components = np.lib.format.open_memmap(
        components_path,
        mode="w+",
        dtype=np.float32,
        shape=(maximum_rank, *direction_shape),
    ).reshape(maximum_rank, flat_size)
    for start in range(0, flat_size, args.pca_chunk_values):
        stop = min(flat_size, start + args.pca_chunk_values)
        fit_chunk = np.asarray(
            matrix[fit_indices, start:stop],
            dtype=np.float32,
        )
        components[:, start:stop] = (
            left[:, :maximum_rank].T @ fit_chunk
        ) / singular_values[:maximum_rank, None]
    components.flush()

    coefficients = np.zeros(
        (len(heldout_indices), maximum_rank), dtype=np.float64
    )
    for start in range(0, flat_size, args.pca_chunk_values):
        stop = min(flat_size, start + args.pca_chunk_values)
        heldout_chunk = np.asarray(
            matrix[heldout_indices, start:stop],
            dtype=np.float32,
        )
        coefficients += heldout_chunk @ np.asarray(
            components[:, start:stop], dtype=np.float32
        ).T
    reference_norms = np.sqrt(reference_sq)
    np.save(singular_path, singular_values)
    np.save(coefficients_path, coefficients)
    np.save(norms_path, reference_norms)
    projection_audit: list[dict[str, Any]] = []
    for heldout_position, manifest_index in enumerate(heldout_indices):
        for rank in args.ranks:
            projected_norm = float(
                np.linalg.norm(coefficients[heldout_position, :rank])
            )
            reference_norm = float(reference_norms[heldout_position])
            if projected_norm <= 0.0:
                raise ValueError("held-out PCA projection has zero norm")
            projection_audit.append(
                {
                    "receiver_id": rows[int(manifest_index)]["receiver_id"],
                    "rank": rank,
                    "reference_norm": reference_norm,
                    "projected_norm": projected_norm,
                    "norm_match_scale": reference_norm / projected_norm,
                    "retained_energy_before_norm_match": (
                        projected_norm**2 / reference_norm**2
                    ),
                }
            )
    _write_json(
        audit_path,
        {
            "protocol": PROGRAM_VERSION,
            "manifest_sha256": manifest_sha256(rows),
            "fit_indices": fit_indices.tolist(),
            "heldout_indices": heldout_indices.tolist(),
            "maximum_rank": maximum_rank,
            "direction_shape": list(direction_shape),
            "direction_dtype": directions.dtype.str,
            "fit_method": "uncentered PCA via chunked receiver Gram matrix",
            "projection_audit": projection_audit,
        },
    )
    print(
        "pca fitted singular_values="
        + ",".join(f"{value:.4g}" for value in singular_values[:maximum_rank]),
        flush=True,
    )
    return ChunkedPCAArtifacts(
        components_path,
        singular_values,
        heldout_indices,
        coefficients,
        reference_norms,
        direction_shape,
    )


def _projected_edit(
    artifacts: ChunkedPCAArtifacts,
    heldout_position: int,
    rank: int,
) -> np.ndarray:
    components = np.load(artifacts.components_path, mmap_mode="r").reshape(
        max(rank, artifacts.coefficients.shape[1]), -1
    )
    coefficients = artifacts.coefficients[heldout_position, :rank]
    projected_norm = float(np.linalg.norm(coefficients))
    scale = artifacts.reference_norms[heldout_position] / projected_norm
    flat_size = components.shape[1]
    result = np.empty(flat_size, dtype=np.float32)
    chunk = 262_144
    for start in range(0, flat_size, chunk):
        stop = min(flat_size, start + chunk)
        result[start:stop] = (
            coefficients @ np.asarray(components[:rank, start:stop])
        ) * scale
    return result.reshape(artifacts.direction_shape)


def _build_geometry_manifest(
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Build 64 independent phase-geometry receivers for PCA plotting."""
    low_bank, low_rows = _read_metadata(args.low_dataset_root)
    high_bank, high_rows = _read_metadata(args.high_dataset_root)
    settings = (
        (
            "low",
            args.low_omega,
            0.0,
            1.0,
            1.0,
            low_bank,
            low_rows,
            high_bank,
            high_rows,
        ),
        (
            "high",
            args.high_omega,
            1.0,
            0.0,
            -1.0,
            high_bank,
            high_rows,
            low_bank,
            low_rows,
        ),
    )
    receivers: list[dict[str, Any]] = []
    for (
        target_label,
        omega,
        aligned_alpha,
        conflict_alpha,
        canonical_sign,
        aligned_bank,
        aligned_rows,
        conflict_bank,
        conflict_rows,
    ) in settings:
        for phase_index in range(args.phase_count):
            for data_repeat in range(args.repeats):
                aligned = _one_row(
                    aligned_rows,
                    omega=omega,
                    alpha=aligned_alpha,
                    phase_index=phase_index,
                    repeat=data_repeat,
                )
                conflict = _one_row(
                    conflict_rows,
                    omega=omega,
                    alpha=conflict_alpha,
                    phase_index=phase_index,
                    repeat=data_repeat,
                )
                _assert_pair(aligned, conflict)
                theta_star = float(aligned["theta_star"])
                angular_velocity_star = float(
                    aligned["angular_velocity_star"]
                )
                boundary_phase = math.atan2(
                    -angular_velocity_star / float(aligned["omega_true"]),
                    theta_star,
                ) % (2.0 * math.pi)
                formula_phase = (
                    float(aligned["omega_true"])
                    * (int(aligned["prediction_start"]) - 1)
                    / int(aligned["fps"])
                    + float(aligned["phase"])
                ) % (2.0 * math.pi)
                if not math.isclose(
                    boundary_phase,
                    formula_phase,
                    abs_tol=1e-8,
                ):
                    raise FrozenManifestError(
                        "boundary phase disagrees with the analytic trajectory"
                    )
                for noise_repeat in range(args.geometry_noise_repeats):
                    generation_seed = (
                        int(aligned["base_seed"])
                        + args.seed_offset
                        + (noise_repeat + 1) * 1_000_000
                    )
                    receiver_id = (
                        f"{target_label}_phase{phase_index:02d}_"
                        f"data{data_repeat:02d}_noise{noise_repeat:02d}"
                    )
                    receivers.append(
                        {
                            "receiver_id": receiver_id,
                            "pair_id": aligned["pair_id"],
                            "physical_state_id": aligned[
                                "physical_state_id"
                            ],
                            "trajectory_id": aligned["trajectory_id"],
                            "split": "geometry_heldout",
                            "target": "frequency",
                            "target_label": target_label,
                            "aligned_sample_id": aligned["sample_id"],
                            "conflict_sample_id": conflict["sample_id"],
                            "aligned_color": aligned["color_label"],
                            "conflict_color": conflict["color_label"],
                            "aligned_shape": aligned["shape_label"],
                            "conflict_shape": conflict["shape_label"],
                            "omega_true": float(aligned["omega_true"]),
                            "amplitude_true": float(
                                aligned["amplitude_true"]
                            ),
                            "phase": float(aligned["phase"]),
                            "phase_index": phase_index,
                            "data_repeat": data_repeat,
                            "noise_repeat": noise_repeat,
                            "generation_seed": generation_seed,
                            "theta_star": theta_star,
                            "angular_velocity_star": (
                                angular_velocity_star
                            ),
                            "boundary_phase_rad": boundary_phase,
                            "canonical_direction_sign": canonical_sign,
                            "canonical_direction_definition": (
                                "high-appearance-cue-minus-low-appearance-cue"
                            ),
                            "aligned_video": str(
                                aligned_bank / aligned["video"]
                            ),
                            "conflict_video": str(
                                conflict_bank / conflict["video"]
                            ),
                            "training_manifest_id": aligned[
                                "training_manifest_id"
                            ],
                            "test_manifest_id": aligned[
                                "test_manifest_id"
                            ],
                            "model_name": args.model_name,
                        }
                    )
    expected = (
        2
        * args.phase_count
        * args.repeats
        * args.geometry_noise_repeats
    )
    if len(receivers) != expected:
        raise AssertionError(
            f"geometry receiver count {len(receivers)} != {expected}"
        )
    receiver_ids = {str(row["receiver_id"]) for row in receivers}
    receiver_states = {
        (str(row["trajectory_id"]), int(row["generation_seed"]))
        for row in receivers
    }
    if len(receiver_ids) != expected or len(receiver_states) != expected:
        raise FrozenManifestError(
            "geometry receivers are not unique by ID and trajectory/seed"
        )
    return receivers


def _build_dense_geometry_manifest(
    args: argparse.Namespace,
    data_config: Any,
    geometry_root: Path,
) -> list[dict[str, Any]]:
    """Render 64 receivers with uniformly spaced boundary physical phases."""
    from sshv2.experiments.pendulum.data import (
        PendulumParameters,
        load_video,
        pendulum_trajectory,
        render_video,
        training_appearance,
        write_video,
    )

    receivers: list[dict[str, Any]] = []
    inputs_root = geometry_root / "inputs"
    boundary_time = (
        data_config.prediction_start - 1
    ) / data_config.render.fps
    for target_index, (target_label, omega, canonical_sign) in enumerate(
        (
            ("low", args.low_omega, 1.0),
            ("high", args.high_omega, -1.0),
        )
    ):
        aligned_appearance = training_appearance(data_config, target_index)
        conflict_appearance = training_appearance(
            data_config,
            1 - target_index,
        )
        for phase_index in range(args.geometry_phase_count):
            boundary_phase = (
                2.0 * math.pi * phase_index / args.geometry_phase_count
            )
            initial_phase = (
                boundary_phase - omega * boundary_time
            ) % (2.0 * math.pi)
            parameters = PendulumParameters(
                omega=omega,
                amplitude=0.135,
                phase=initial_phase,
            )
            theta, angular_velocity = pendulum_trajectory(
                parameters,
                fps=data_config.render.fps,
                num_frames=data_config.render.num_frames,
            )
            boundary_index = data_config.prediction_start - 1
            theta_star = float(theta[boundary_index])
            angular_velocity_star = float(
                angular_velocity[boundary_index]
            )
            recovered_phase = math.atan2(
                -angular_velocity_star / omega,
                theta_star,
            ) % (2.0 * math.pi)
            phase_error = math.atan2(
                math.sin(recovered_phase - boundary_phase),
                math.cos(recovered_phase - boundary_phase),
            )
            if abs(phase_error) > 1e-10:
                raise FrozenManifestError(
                    "rendered dense boundary phase is not exact"
                )
            receiver_id = f"{target_label}_phase{phase_index:03d}"
            receiver_root = inputs_root / receiver_id
            aligned_video = receiver_root / "aligned.mp4"
            conflict_video = receiver_root / "conflict.mp4"
            for destination, appearance in (
                (aligned_video, aligned_appearance),
                (conflict_video, conflict_appearance),
            ):
                if not destination.is_file():
                    frames = render_video(
                        theta,
                        appearance,
                        data_config.render,
                    )
                    write_video(
                        destination,
                        frames,
                        data_config.render.fps,
                    )
                load_video(
                    destination,
                    expected_frames=data_config.render.num_frames,
                )
            generation_seed = (
                91_000_000
                + target_index * 100_000
                + phase_index
                + args.seed_offset
            )
            physical_state_id = (
                f"dense_{target_label}_phase{phase_index:03d}"
            )
            receivers.append(
                {
                    "receiver_id": receiver_id,
                    "pair_id": physical_state_id,
                    "physical_state_id": physical_state_id,
                    "trajectory_id": physical_state_id,
                    "split": "geometry_heldout",
                    "target": "frequency",
                    "target_label": target_label,
                    "aligned_sample_id": f"{receiver_id}_aligned",
                    "conflict_sample_id": f"{receiver_id}_conflict",
                    "aligned_color": aligned_appearance.color,
                    "conflict_color": conflict_appearance.color,
                    "aligned_shape": aligned_appearance.shape,
                    "conflict_shape": conflict_appearance.shape,
                    "omega_true": omega,
                    "amplitude_true": parameters.amplitude,
                    "phase": initial_phase,
                    "phase_index": phase_index,
                    "data_repeat": 0,
                    "noise_repeat": 0,
                    "generation_seed": generation_seed,
                    "theta_star": theta_star,
                    "angular_velocity_star": angular_velocity_star,
                    "boundary_phase_rad": boundary_phase,
                    "canonical_direction_sign": canonical_sign,
                    "canonical_direction_definition": (
                        "high-appearance-cue-minus-low-appearance-cue"
                    ),
                    "aligned_video": str(aligned_video),
                    "conflict_video": str(conflict_video),
                    "training_manifest_id": (
                        data_config.training_manifest_id
                    ),
                    "test_manifest_id": "dense_geometry_v2",
                    "model_name": args.model_name,
                }
            )
    expected = 2 * args.geometry_phase_count
    if len(receivers) != expected:
        raise AssertionError(
            f"dense geometry count {len(receivers)} != {expected}"
        )
    return receivers


def _first_harmonic_fit(
    phases: np.ndarray,
    values: np.ndarray,
) -> dict[str, Any]:
    design = np.column_stack(
        (
            np.ones(len(phases), dtype=np.float64),
            np.cos(phases),
            np.sin(phases),
        )
    )
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    fitted = design @ coefficients
    residual = float(np.sum((values - fitted) ** 2))
    total = float(np.sum((values - values.mean()) ** 2))
    r_squared = 1.0 - residual / total if total > 0.0 else float("nan")
    return {
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
    return (
        float(fit["intercept"])
        + float(fit["cosine_coefficient"]) * np.cos(phases)
        + float(fit["sine_coefficient"]) * np.sin(phases)
    )


def _geometry_coordinates(
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    directions: np.memmap,
) -> tuple[
    np.ndarray,
    tuple[dict[str, Any], ...],
    tuple[float, ...],
]:
    components_path = args.out_root / "pca" / "components.npy"
    if not components_path.is_file():
        raise FileNotFoundError(components_path)
    components = np.load(components_path, mmap_mode="r")
    if components.shape[0] < 4 or tuple(components.shape[1:]) != tuple(
        directions.shape[1:]
    ):
        raise ValueError("PCA components do not match geometry directions")
    matrix = directions.reshape(len(rows), -1)
    basis = components[:4].reshape(4, -1)
    coefficients = np.zeros((len(rows), 4), dtype=np.float64)
    flat_size = matrix.shape[1]
    for start in range(0, flat_size, args.pca_chunk_values):
        stop = min(flat_size, start + args.pca_chunk_values)
        coefficients += np.asarray(
            matrix[:, start:stop], dtype=np.float32
        ) @ np.asarray(basis[:, start:stop], dtype=np.float32).T

    canonical_signs = np.asarray(
        [float(row["canonical_direction_sign"]) for row in rows],
        dtype=np.float64,
    )
    coefficients *= canonical_signs[:, None]
    display_signs = [1.0, 1.0, 1.0, 1.0]
    if float(np.mean(coefficients[:, 0])) < 0.0:
        display_signs[0] = -1.0
        coefficients[:, 0] *= -1.0
    phases = np.asarray(
        [float(row["boundary_phase_rad"]) for row in rows],
        dtype=np.float64,
    )
    # PCA component signs are arbitrary.  Keep PC1's shared direction positive,
    # then orient PC2--PC4 so the dominant pooled first-harmonic coefficient is
    # positive.  This changes only display orientation, not fit quality or
    # explained energy.
    for component_index in range(1, 4):
        preliminary = _first_harmonic_fit(
            phases,
            coefficients[:, component_index],
        )
        harmonic_terms = np.asarray(
            [
                preliminary["cosine_coefficient"],
                preliminary["sine_coefficient"],
            ]
        )
        dominant = int(np.argmax(np.abs(harmonic_terms)))
        if harmonic_terms[dominant] < 0.0:
            display_signs[component_index] = -1.0
            coefficients[:, component_index] *= -1.0
    pooled_fits = tuple(
        _first_harmonic_fit(phases, coefficients[:, component_index])
        for component_index in range(4)
    )
    return coefficients, pooled_fits, tuple(display_signs)


def _plot_geometry(
    args: argparse.Namespace,
    geometry_root: Path,
    rows: Sequence[Mapping[str, Any]],
    coefficients: np.ndarray,
    pooled_fits: Sequence[Mapping[str, Any]],
    display_signs: tuple[float, ...],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_root = geometry_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    phases = np.asarray(
        [float(row["boundary_phase_rad"]) for row in rows],
        dtype=np.float64,
    )
    targets = np.asarray([str(row["target_label"]) for row in rows])
    dense_phase = np.linspace(0.0, 2.0 * math.pi, 721)
    model_label = args.model_name.replace("_", "-")
    target_styles = {
        "low": {"marker": "o", "color": "tab:blue"},
        "high": {"marker": "s", "color": "tab:orange"},
    }
    coordinate_specs = (
        ("pc1", "a", "pc1_coordinate_a"),
        ("pc2", "b", "pc2_coordinate_b"),
        ("pc3", "c", "pc3_coordinate_c"),
        ("pc4", "d", "pc4_coordinate_d"),
    )
    fits_by_target = {
        target: {
            fit_key: _first_harmonic_fit(
                phases[targets == target],
                coefficients[targets == target, component_index],
            )
            for component_index, (fit_key, _symbol, _column) in enumerate(
                coordinate_specs
            )
        }
        for target in target_styles
    }
    common = {
        "c": phases,
        "cmap": "twilight",
        "vmin": 0.0,
        "vmax": 2.0 * math.pi,
        "s": 62,
        "alpha": 0.84,
        "linewidths": 0.7,
        "edgecolors": "black",
    }

    figure, axis = plt.subplots(figsize=(9.6, 8.0))
    scatter = None
    for target, style in target_styles.items():
        selected = targets == target
        scatter = axis.scatter(
            coefficients[selected, 0],
            coefficients[selected, 1],
            marker=style["marker"],
            label=f"{target} physical frequency",
            **{
                key: (value[selected] if key == "c" else value)
                for key, value in common.items()
            },
        )
    axis.axhline(0.0, color="tab:blue", alpha=0.28, linewidth=1.0)
    axis.axvline(0.0, color="tab:blue", alpha=0.28, linewidth=1.0)
    axis.grid(alpha=0.18)
    axis.set_xlabel(r"$a_j = \langle d_j^{\mathrm{canonical}}, v_1\rangle$")
    axis.set_ylabel(r"$b_j = \langle d_j^{\mathrm{canonical}}, v_2\rangle$")
    axis.set_title(
        f"{model_label}: held-out Block-13 directions in the first two PCA coordinates "
        f"(n={len(rows)})\n"
        r"color = boundary phase $\varphi_j^* = "
        r"\mathrm{atan2}(-\dot{\theta}_j^*/\omega_j,\theta_j^*)\ \mathrm{mod}\ 2\pi$"
    )
    axis.legend(loc="best", frameon=True)
    if scatter is None:
        raise AssertionError("geometry scatter has no points")
    colorbar = figure.colorbar(scatter, ax=axis, pad=0.025)
    colorbar.set_label(r"boundary phase $\varphi_j^*$ (rad)")
    colorbar.set_ticks(
        [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi, 2.0 * math.pi],
        labels=["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"],
    )
    figure.tight_layout()
    figure.savefig(plot_root / "heldout_pc1_pc2_by_phase.png", dpi=200)
    plt.close(figure)

    def harmonic_plot(
        values: np.ndarray,
        fit_key: str,
        coordinate: str,
        destination: str,
    ) -> None:
        figure, axis = plt.subplots(figsize=(10.4, 6.5))
        for target, style in target_styles.items():
            selected = targets == target
            fit = fits_by_target[target][fit_key]
            axis.scatter(
                phases[selected],
                values[selected],
                s=52,
                alpha=0.78,
                marker=style["marker"],
                color=style["color"],
                label=f"{target} held-out receivers (n={int(selected.sum())})",
            )
            axis.plot(
                dense_phase,
                _harmonic_values(fit, dense_phase),
                linewidth=2.5,
                color=style["color"],
                label=(
                    rf"{target} fit: $R^2={float(fit['r_squared']):.4f}$, "
                    rf"$B={float(fit['amplitude']):.3f}$"
                ),
            )
        axis.set_xlim(0.0, 2.0 * math.pi)
        axis.set_xticks(
            [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi, 2.0 * math.pi],
            labels=["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"],
        )
        axis.set_xlabel(r"Boundary physical phase $\varphi_j^*$")
        axis.set_ylabel(rf"${coordinate}_j$")
        axis.set_title(
            f"{model_label}: first-harmonic least-squares fit: "
            rf"${coordinate}_j$ vs. $\varphi_j^*$"
        )
        axis.grid(alpha=0.2)
        axis.legend(loc="best", frameon=True)
        figure.tight_layout()
        figure.savefig(plot_root / destination, dpi=200)
        plt.close(figure)

    for component_index, (fit_key, symbol, _column) in enumerate(
        coordinate_specs
    ):
        harmonic_plot(
            coefficients[:, component_index],
            fit_key,
            symbol,
            f"{fit_key}_coordinate_vs_phase.png",
        )

    coordinate_rows = []
    for row, values in zip(rows, coefficients):
        coordinate_rows.append(
            {
                "receiver_id": row["receiver_id"],
                "target_label": row["target_label"],
                "omega_true": row["omega_true"],
                "phase_index": row["phase_index"],
                "data_repeat": row["data_repeat"],
                "noise_repeat": row["noise_repeat"],
                "generation_seed": row["generation_seed"],
                "boundary_phase_rad": row["boundary_phase_rad"],
                "pc1_coordinate_a": float(values[0]),
                "pc2_coordinate_b": float(values[1]),
                "pc3_coordinate_c": float(values[2]),
                "pc4_coordinate_d": float(values[3]),
            }
        )
    fit_payload = {
        f"{fit_key}_coordinate_fit_by_target": {
            target: dict(fits_by_target[target][fit_key])
            for target in target_styles
        }
        for fit_key, _symbol, _column in coordinate_specs
    }
    _write_csv(geometry_root / "coordinates.csv", coordinate_rows)
    _write_json(
        geometry_root / "harmonic_fit.json",
        {
            "protocol": "pendulum_activation_pca_dense_geometry_v2",
            "model_name": args.model_name,
            "pca_fit_receivers": 24,
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
                "separate first-harmonic fit for each physical-frequency "
                "receiver group"
            ),
            **fit_payload,
            "pooled_fit_diagnostics_not_plotted": {
                fit_key: dict(pooled_fits[component_index])
                for component_index, (fit_key, _symbol, _column) in enumerate(
                    coordinate_specs
                )
            },
        },
    )


def _run_geometry_experiment(
    args: argparse.Namespace,
    pca_rows: Sequence[Mapping[str, Any]],
) -> None:
    geometry_root = args.out_root / "geometry_dense_v2"
    pipe, data_config = _load_runtime(args, pca_rows)
    geometry_rows = _freeze_manifest(
        geometry_root / "receiver_manifest.jsonl",
        _build_dense_geometry_manifest(
            args,
            data_config,
            geometry_root,
        ),
    )
    expected = 2 * args.geometry_phase_count
    if len(geometry_rows) != expected:
        raise FrozenManifestError(
            f"geometry manifest has {len(geometry_rows)} rows, expected {expected}"
        )
    main_direction_path = args.out_root / "directions.npy"
    components_path = args.out_root / "pca" / "components.npy"
    if not main_direction_path.is_file() or not components_path.is_file():
        raise FileNotFoundError(
            "complete PCA directions/components are required before geometry"
        )
    geometry_args = argparse.Namespace(**vars(args))
    geometry_args.out_root = geometry_root
    directions = _extract_directions(
        geometry_args,
        geometry_rows,
        pipe,
        data_config,
    )
    coefficients, pooled_fits, display_signs = _geometry_coordinates(
        args,
        geometry_rows,
        directions,
    )
    _plot_geometry(
        args,
        geometry_root,
        geometry_rows,
        coefficients,
        pooled_fits,
        display_signs,
    )
    _write_json(
        geometry_root / "progress.json",
        {
            "status": "complete",
            "completed_receivers": len(geometry_rows),
            "expected_receivers": len(geometry_rows),
        },
    )
    print(f"geometry complete {geometry_root}", flush=True)


def _run_interventions(
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    directions: np.memmap,
    artifacts: ChunkedPCAArtifacts,
    pipe: Any,
    data_config: Any,
) -> None:
    import torch

    for heldout_position, manifest_index in enumerate(
        artifacts.heldout_indices
    ):
        row = rows[int(manifest_index)]
        receiver_id = str(row["receiver_id"])
        condition = _condition_latents(
            pipe,
            data_config,
            Path(str(row["aligned_video"])),
            history=args.history,
        )
        interventions: list[tuple[str, np.ndarray]] = [
            ("aligned_plus_full", np.asarray(directions[int(manifest_index)])),
            *[
                (
                    f"aligned_plus_pc{rank}",
                    _projected_edit(artifacts, heldout_position, rank),
                )
                for rank in args.ranks
            ],
        ]
        for label, edit_array in interventions:
            destination = _baseline_path(args.out_root, receiver_id, label)
            if destination.is_file():
                print(f"intervene {receiver_id} {label} resume", flush=True)
                continue
            edit = torch.from_numpy(
                np.array(edit_array, dtype=np.float32, copy=True)
            ).to(device=pipe.device, dtype=pipe.torch_dtype)
            latents, _captured = _denoise(
                pipe,
                data_config,
                condition,
                seed=int(row["generation_seed"]),
                steps=args.steps,
                block_index=args.block_index,
                condition_tokens=args.condition_tokens,
                edit=edit,
                strength=args.intervention_strength,
            )
            _write_future_video(
                pipe,
                data_config,
                latents,
                destination,
            )
            print(f"intervene {receiver_id} {label} complete", flush=True)
            del edit, latents
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del condition, interventions


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _measure_one(
    path: Path,
    row: Mapping[str, Any],
    data_config: Any,
    *,
    expected_colors: Sequence[str],
) -> dict[str, Any]:
    from sshv2.experiments.pendulum.evaluation import (
        centers_to_theta,
        detect_bob_track,
        fit_oscillation,
        track_validity,
    )
    from sshv2.experiments.pendulum.data import load_video

    frames = load_video(path, expected_frames=data_config.future_frames)
    colors = tuple(dict.fromkeys(str(color) for color in expected_colors))
    if not colors:
        raise ValueError("at least one expected bob color is required")
    candidates = [
        (
            color,
            detect_bob_track(
                frames,
                data_config.render,
                expected_color=color,
            ),
        )
        for color in colors
    ]
    detected_color, track = max(
        candidates,
        key=lambda item: float(item[1].detected.mean()),
    )
    validity = track_validity(
        track,
        data_config.render,
        theta_star=float(row["theta_star"]),
    )
    theta = centers_to_theta(track, data_config.render)
    fit = fit_oscillation(
        theta,
        track.detected,
        fps=data_config.render.fps,
        omega_low=max(0.2, data_config.low_frequency.low - 1.0),
        omega_high=data_config.high_frequency.high + 1.0,
    )
    fit_valid = bool(
        math.isfinite(fit.rmse)
        and fit.rmse <= 0.08
        and math.isfinite(fit.omega)
        and math.isfinite(fit.amplitude)
    )
    circle_area = math.pi * data_config.render.bob_radius_px**2
    square_area = (2 * data_config.render.bob_radius_px + 1) ** 2
    predicted_shape = (
        "circle"
        if float(validity["median_area_px"])
        < 0.5 * (circle_area + square_area)
        else "square"
    )
    return {
        **validity,
        "valid": bool(validity["valid"] and fit_valid),
        "fit_valid": fit_valid,
        "detected_color": detected_color,
        "detected_shape": predicted_shape,
        "omega_hat": fit.omega,
        "amplitude_hat": fit.amplitude,
        "fit_center": fit.center,
        "fit_rmse": fit.rmse,
    }


def _measure_interventions(
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    artifacts: ChunkedPCAArtifacts,
    data_config: Any,
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    labels: list[tuple[str, int | str, str]] = [
        ("aligned", 0, "aligned_color"),
        ("conflict", 0, "conflict_color"),
        ("aligned_plus_full", "full", "aligned_color"),
        *[
            (f"aligned_plus_pc{rank}", rank, "aligned_color")
            for rank in args.ranks
        ],
    ]
    for manifest_index in artifacts.heldout_indices:
        row = rows[int(manifest_index)]
        receiver_id = str(row["receiver_id"])
        receiver_metrics: list[dict[str, Any]] = []
        for condition, rank, color_field in labels:
            path = _baseline_path(args.out_root, receiver_id, condition)
            measured = _measure_one(
                path,
                row,
                data_config,
                expected_colors=(
                    str(row[color_field]),
                    str(
                        row[
                            "conflict_color"
                            if color_field == "aligned_color"
                            else "aligned_color"
                        ]
                    ),
                ),
            )
            record = {
                "receiver_id": receiver_id,
                "manifest_index": int(manifest_index),
                "target_label": row["target_label"],
                "phase_index": row["phase_index"],
                "diffusion_repeat": row["diffusion_repeat"],
                "omega_true": row["omega_true"],
                "aligned_color": row["aligned_color"],
                "conflict_color": row["conflict_color"],
                "aligned_shape": row["aligned_shape"],
                "condition": condition,
                "rank": rank,
                "intervention_strength": (
                    0.0
                    if condition in ("aligned", "conflict")
                    else args.intervention_strength
                ),
                "video": str(path),
                **measured,
            }
            receiver_metrics.append(record)

        by_condition = {item["condition"]: item for item in receiver_metrics}
        aligned_omega = float(by_condition["aligned"]["omega_hat"])
        conflict_omega = float(by_condition["conflict"]["omega_hat"])
        denominator = conflict_omega - aligned_omega
        for item in receiver_metrics:
            omega_hat = float(item["omega_hat"])
            item["omega_shift_from_aligned"] = omega_hat - aligned_omega
            item["conflict_recovery_fraction"] = (
                (omega_hat - aligned_omega) / denominator
                if math.isfinite(omega_hat)
                and math.isfinite(denominator)
                and abs(denominator) > 1e-8
                else float("nan")
            )
        metrics.extend(receiver_metrics)
    _write_csv(args.out_root / "metrics.csv", metrics)
    return metrics


def _mean_finite(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _summarize_and_plot(
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    artifacts: ChunkedPCAArtifacts,
    metrics: Sequence[Mapping[str, Any]],
) -> None:
    conditions = [
        "aligned",
        "conflict",
        "aligned_plus_full",
        *[f"aligned_plus_pc{rank}" for rank in args.ranks],
    ]
    summary_rows: list[dict[str, Any]] = []
    for condition in conditions:
        selected = [row for row in metrics if row["condition"] == condition]
        valid_selected = [row for row in selected if bool(row["valid"])]
        summary_rows.append(
            {
                "condition": condition,
                "samples": len(selected),
                "valid_samples": len(valid_selected),
                "valid_fraction": float(
                    np.mean([bool(row["valid"]) for row in selected])
                ),
                "mean_omega_hat": _mean_finite(
                    float(row["omega_hat"]) for row in valid_selected
                ),
                "mean_signed_shift_from_aligned": _mean_finite(
                    (
                        float(row["omega_shift_from_aligned"])
                        * (1.0 if row["target_label"] == "low" else -1.0)
                    )
                    for row in valid_selected
                ),
                "mean_conflict_recovery_fraction": _mean_finite(
                    float(row["conflict_recovery_fraction"])
                    for row in valid_selected
                ),
            }
        )
    _write_csv(args.out_root / "summary.csv", summary_rows)
    summary = {
        "protocol": PROGRAM_VERSION,
        "model_name": args.model_name,
        "history": args.history,
        "checkpoint": str(args.checkpoint),
        "block_index_zero_based": args.block_index,
        "direction_definition": "conflict-minus-aligned",
        "receivers": len(rows),
        "fit_receivers": sum(row["split"] == "fit" for row in rows),
        "heldout_receivers": sum(row["split"] == "heldout" for row in rows),
        "ranks": list(args.ranks),
        "singular_values": artifacts.singular_values.tolist(),
        "condition_summary": [
            {
                key: (_finite_or_none(value) if isinstance(value, float) else value)
                for key, value in row.items()
            }
            for row in summary_rows
        ],
    }
    _write_json(args.out_root / "summary.json", summary)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = args.out_root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    singular = artifacts.singular_values
    energy = singular**2
    cumulative = np.cumsum(energy) / energy.sum()
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(np.arange(1, len(singular) + 1), singular, marker="o")
    axes[0].set(xlabel="component", ylabel="singular value", title="Uncentered PCA")
    axes[1].plot(np.arange(1, len(cumulative) + 1), cumulative, marker="o")
    axes[1].axhline(0.9, color="gray", linestyle="--", linewidth=1)
    axes[1].set(
        xlabel="rank",
        ylabel="cumulative fit energy",
        ylim=(0.0, 1.02),
        title="Cumulative energy",
    )
    figure.tight_layout()
    figure.savefig(plots / "pca_spectrum.png", dpi=180)
    plt.close(figure)

    plot_conditions = [
        "aligned",
        "conflict",
        *[f"aligned_plus_pc{rank}" for rank in args.ranks],
        "aligned_plus_full",
    ]
    labels = [
        "aligned",
        "conflict",
        *[f"PC-{rank}" for rank in args.ranks],
        "full",
    ]
    figure, axis = plt.subplots(figsize=(10, 5))
    for x, condition in enumerate(plot_conditions):
        values = [
            float(row["conflict_recovery_fraction"])
            for row in metrics
            if row["condition"] == condition
            and bool(row["valid"])
            and math.isfinite(float(row["conflict_recovery_fraction"]))
        ]
        if values:
            jitter = np.linspace(-0.08, 0.08, len(values))
            axis.scatter(
                np.full(len(values), x) + jitter,
                values,
                alpha=0.65,
                s=24,
            )
            axis.plot(x, np.mean(values), marker="D", color="black")
    axis.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axis.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    axis.set(
        xticks=range(len(labels)),
        xticklabels=labels,
        ylabel="conflict recovery fraction",
        title=f"{args.model_name}: held-out causal activation edits",
    )
    figure.tight_layout()
    figure.savefig(plots / "heldout_conflict_recovery.png", dpi=180)
    plt.close(figure)


def run_mechanism_experiment(args: argparse.Namespace) -> None:
    required_files = (
        args.experiment_config,
        args.training_config,
        args.checkpoint,
    )
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.out_root.mkdir(parents=True, exist_ok=True)
    generated = build_receiver_manifest(args)
    rows = _freeze_manifest(args.out_root / "receiver_manifest.jsonl", generated)
    manifest_audit = validate_frozen_receiver_manifest(
        rows,
        target="frequency",
        expected_fit=24,
        expected_heldout=8,
    )
    if args.manifest_only:
        print(args.out_root / "receiver_manifest.jsonl")
        return
    if args.measure_only:
        import yaml
        from sshv2.experiments.pendulum.data import config_from_mapping

        experiment = yaml.safe_load(
            args.experiment_config.read_text(encoding="utf-8")
        )
        if not isinstance(experiment, Mapping) or not isinstance(
            experiment.get("data"), Mapping
        ):
            raise ValueError("experiment config must contain a data mapping")
        data_config = config_from_mapping(experiment["data"])
        direction_path = args.out_root / "directions.npy"
        if not direction_path.is_file():
            raise FileNotFoundError(direction_path)
        directions = np.load(direction_path, mmap_mode="r")
        expected_shape = (
            len(rows),
            args.steps,
            args.condition_tokens,
            args.hidden_size,
        )
        if directions.shape != expected_shape or directions.dtype != np.float16:
            raise ValueError(
                f"direction bank has {directions.shape}/{directions.dtype}, "
                f"expected {expected_shape}/float16"
            )
        artifacts = _fit_chunked_pca(args, rows, directions)
        metrics = _measure_interventions(
            args,
            rows,
            artifacts,
            data_config,
        )
        _summarize_and_plot(args, rows, artifacts, metrics)
        print(f"remeasured {args.out_root}", flush=True)
        return
    if args.geometry_only:
        _run_geometry_experiment(args, rows)
        return

    checkpoint_stat = args.checkpoint.stat()
    run_state = {
        "protocol": PROGRAM_VERSION,
        "model_name": args.model_name,
        "history": args.history,
        "checkpoint": str(args.checkpoint),
        "checkpoint_size_bytes": checkpoint_stat.st_size,
        "checkpoint_sha256": _sha256_file(args.checkpoint),
        "training_config": str(args.training_config),
        "training_config_sha256": _sha256_file(args.training_config),
        "experiment_config": str(args.experiment_config),
        "experiment_config_sha256": _sha256_file(args.experiment_config),
        "manifest": manifest_audit,
        "low_omega": args.low_omega,
        "high_omega": args.high_omega,
        "direction_definition": "conflict-minus-aligned",
        "block_index_zero_based": args.block_index,
        "steps": args.steps,
        "condition_tokens": args.condition_tokens,
        "hidden_size": args.hidden_size,
        "ranks": list(args.ranks),
        "intervention_strength": args.intervention_strength,
        "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    state_path = args.out_root / "run_state.json"
    if state_path.is_file():
        existing = _read_json(state_path)
        comparable = dict(existing)
        comparable.pop("status", None)
        if comparable != run_state:
            raise ValueError("existing run state differs from requested experiment")
    else:
        _write_json(state_path, {**run_state, "status": "running"})

    print(
        f"load model={args.model_name} checkpoint={args.checkpoint} "
        f"device={args.device}",
        flush=True,
    )
    pipe, data_config = _load_runtime(args, rows)
    directions = _extract_directions(args, rows, pipe, data_config)
    artifacts = _fit_chunked_pca(args, rows, directions)
    if not args.skip_interventions:
        _run_interventions(
            args,
            rows,
            directions,
            artifacts,
            pipe,
            data_config,
        )
        metrics = _measure_interventions(args, rows, artifacts, data_config)
        _summarize_and_plot(args, rows, artifacts, metrics)
    _write_json(state_path, {**run_state, "status": "complete"})
    _write_json(
        args.out_root / "progress.json",
        {
            "status": "complete",
            "completed_receivers": len(rows),
            "expected_receivers": len(rows),
        },
    )
    print(f"complete {args.out_root}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare",
        help="fit a small externally materialized NPZ direction bank",
    )
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--directions", type=Path, required=True)
    prepare.add_argument("--array-key", default="directions")
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--audit", type=Path)
    prepare.add_argument("--checkpoint-id", required=True)
    prepare.add_argument(
        "--history", choices=("short", "long"), required=True
    )
    prepare.add_argument(
        "--target",
        choices=("frequency", "amplitude"),
        required=True,
    )
    prepare.add_argument(
        "--direction-definition",
        choices=("conflict-minus-aligned", "aligned-minus-conflict"),
        required=True,
    )
    prepare.add_argument("--block-index", type=int, required=True)
    prepare.add_argument("--fm-steps", type=int, required=True)
    prepare.add_argument("--condition-tokens", type=int, required=True)
    prepare.add_argument("--rank", type=int, required=True)
    prepare.add_argument("--expected-fit", type=int)
    prepare.add_argument("--expected-heldout", type=int)

    run = commands.add_parser(
        "run",
        help="freeze receivers, extract Wan activations, fit PCA, and intervene",
    )
    run.add_argument(
        "--model-name",
        choices=("frequency_color_circle", "frequency_color_shape"),
        required=True,
    )
    run.add_argument("--low-dataset-root", type=Path, required=True)
    run.add_argument("--high-dataset-root", type=Path, required=True)
    run.add_argument("--experiment-config", type=Path, required=True)
    run.add_argument("--training-config", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--out-root", type=Path, required=True)
    run.add_argument(
        "--history", choices=("short", "long"), default="long"
    )
    run.add_argument("--device", default="cuda")
    run.add_argument("--steps", type=int, default=20)
    run.add_argument("--seed-offset", type=int, default=23_000_000)
    run.add_argument("--low-omega", type=float, default=2.6)
    run.add_argument("--high-omega", type=float, default=5.8)
    run.add_argument("--phase-count", type=int, default=8)
    run.add_argument("--repeats", type=int, default=2)
    run.add_argument(
        "--heldout-phases",
        type=lambda value: _parse_int_tuple(value, name="heldout phases"),
        default=DEFAULT_HELDOUT_PHASES,
    )
    run.add_argument("--block-index", type=int, default=13)
    run.add_argument("--condition-tokens", type=int, default=1088)
    run.add_argument("--hidden-size", type=int, default=768)
    run.add_argument("--ranks", type=_parse_ranks, default=DEFAULT_RANKS)
    run.add_argument("--intervention-strength", type=float, default=1.0)
    run.add_argument("--pca-chunk-values", type=int, default=262_144)
    run.add_argument("--geometry-noise-repeats", type=int, default=2)
    run.add_argument("--geometry-phase-count", type=int, default=32)
    run.add_argument("--manifest-only", action="store_true")
    run.add_argument("--measure-only", action="store_true")
    run.add_argument("--geometry-only", action="store_true")
    run.add_argument("--skip-interventions", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        for name in ("block_index", "fm_steps", "condition_tokens", "rank"):
            value = int(getattr(args, name))
            minimum = 0 if name == "block_index" else 1
            if value < minimum:
                raise ValueError(
                    f"--{name.replace('_', '-')} must be >= {minimum}"
                )
        prepare_mechanism_artifacts(args)
        return

    positive = (
        "steps",
        "condition_tokens",
        "hidden_size",
        "phase_count",
        "repeats",
        "pca_chunk_values",
        "geometry_noise_repeats",
        "geometry_phase_count",
    )
    for name in positive:
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.block_index < 0:
        raise ValueError("--block-index must be non-negative")
    if args.phase_count != 8 or args.repeats != 2:
        raise ValueError("the frozen protocol requires 8 phases and 2 repeats")
    if args.geometry_phase_count != 32:
        raise ValueError(
            "the dense geometry protocol requires 32 phases per frequency"
        )
    if tuple(args.heldout_phases) != DEFAULT_HELDOUT_PHASES:
        raise ValueError("the frozen protocol requires held-out phases 6,7")
    if max(args.ranks) > 24:
        raise ValueError("PCA rank cannot exceed the 24 fit receivers")
    if not math.isfinite(args.intervention_strength):
        raise ValueError("--intervention-strength must be finite")
    stage_only = (
        bool(args.manifest_only),
        bool(args.measure_only),
        bool(args.geometry_only),
    )
    if sum(stage_only) > 1:
        raise ValueError(
            "--manifest-only, --measure-only, and --geometry-only are exclusive"
        )
    run_mechanism_experiment(args)


if __name__ == "__main__":
    main()
