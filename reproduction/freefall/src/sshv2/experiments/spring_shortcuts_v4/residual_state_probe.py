"""Out-of-sample tests for physical variables encoded in Spring residual deltas.

The input is one paired residual difference per physical trajectory,
``h_aligned - h_conflict``.  Keeping the two colour-conflict directions
separate is essential: fast trajectories change red -> blue, whereas slow
trajectories change blue -> red.

The three primary candidates all have two degrees of freedom:

* trigonometric values: ``(cos(theta*), sin(theta*))``;
* state: ``(x*, v*)``;
* derivative state: ``(v*, a*)`` where ``a* = -omega**2 * x*``.

They are compared with the same linear ridge readout and trajectory-level
cross-validation.  A five-variable combined model is available as an
exploratory upper bound, but must not be compared to the two-variable models
as an equal-capacity winner.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np


RepresentationName = Literal[
    "trigonometric",
    "position_velocity",
    "velocity_acceleration",
    "combined",
]

PRIMARY_REPRESENTATIONS: tuple[RepresentationName, ...] = (
    "trigonometric",
    "position_velocity",
    "velocity_acceleration",
)
ALL_REPRESENTATIONS: tuple[RepresentationName, ...] = (*PRIMARY_REPRESENTATIONS, "combined")


def _number(row: Mapping[str, Any], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Residual-state manifest requires finite {key!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Residual-state manifest requires finite {key!r}")
    return value


def boundary_state_features(row: Mapping[str, Any], representation: RepresentationName) -> np.ndarray:
    """Return physical features at the final observed frame.

    ``x_star`` and ``v_star`` are preferred because they are the physical
    state at the prediction boundary.  The phase pair is reconstructed from
    that state rather than read from the initial-phase metadata, avoiding an
    accidental frame-0/frame-64 mismatch.
    """
    omega = _number(row, "omega_true")
    amplitude = _number(row, "amplitude")
    x_star = _number(row, "x_star")
    v_star = _number(row, "v_star")
    if omega <= 0.0 or amplitude <= 0.0:
        raise ValueError("omega_true and amplitude must be positive")

    cos_theta = x_star / amplitude
    sin_theta = -v_star / (amplitude * omega)
    # Pixel/rendering round-off can move this very slightly away from the
    # unit circle.  Normalising keeps the trigonometric candidate exactly two
    # trigonometric coordinates without changing its angle.
    phase_norm = math.hypot(cos_theta, sin_theta)
    if phase_norm <= 0.0:
        raise ValueError("Boundary phase is undefined for a zero state")
    cos_theta /= phase_norm
    sin_theta /= phase_norm
    acceleration = -(omega**2) * x_star

    values: dict[RepresentationName, tuple[float, ...]] = {
        "trigonometric": (cos_theta, sin_theta),
        "position_velocity": (x_star, v_star),
        "velocity_acceleration": (v_star, acceleration),
        "combined": (cos_theta, sin_theta, x_star, v_star, acceleration),
    }
    return np.asarray(values[representation], dtype=np.float64)


def feature_matrix(rows: Sequence[Mapping[str, Any]], representation: RepresentationName) -> np.ndarray:
    if not rows:
        raise ValueError("Cannot fit a state representation to zero rows")
    return np.stack([boundary_state_features(row, representation) for row in rows])


def deterministic_fold(key: str, folds: int) -> int:
    if folds < 2:
        raise ValueError("folds must be at least two")
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % folds


def _standardise(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def _with_intercept(features: np.ndarray) -> np.ndarray:
    return np.concatenate([np.ones((features.shape[0], 1), dtype=features.dtype), features], axis=1)


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q10": float(np.quantile(array, 0.10)),
        "q90": float(np.quantile(array, 0.90)),
    }


def _flattened_values(residual_deltas: np.ndarray) -> np.ndarray:
    values = np.asarray(residual_deltas)
    if values.ndim < 2:
        raise ValueError("residual_deltas must have at least one residual dimension")
    return values.reshape(values.shape[0], -1)


def _uncentered_pca(
    rows: Sequence[Mapping[str, Any]],
    residual_deltas: np.ndarray,
    *,
    row_indices: np.ndarray,
    components: int,
    chunk_features: int = 262_144,
) -> dict[str, Any]:
    """Fit the document's uncentred PCA without materialising the full matrix.

    For an ``N x D`` residual matrix, the eigendecomposition of its ``N x N``
    Gram matrix yields the same non-zero singular values and sample
    coordinates as a direct SVD.  This matters here because D is the complete
    sampling-step/token/hidden residual width.
    """
    if components <= 0:
        raise ValueError("components must be positive")
    flattened = _flattened_values(residual_deltas)
    indices = np.asarray(row_indices, dtype=np.int64)
    if indices.ndim != 1 or len(rows) != indices.size:
        raise ValueError("PCA rows and row_indices must align")
    sample_count = indices.size
    if sample_count < 2:
        raise ValueError("PCA needs at least two residual deltas")
    gram = np.zeros((sample_count, sample_count), dtype=np.float64)
    for start in range(0, flattened.shape[1], chunk_features):
        stop = min(start + chunk_features, flattened.shape[1])
        chunk = np.asarray(flattened[indices, start:stop], dtype=np.float64)
        gram += chunk @ chunk.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    rank = int(np.count_nonzero(eigenvalues > max(1e-10, eigenvalues[0] * 1e-12)))
    kept = min(components, rank)
    if kept == 0:
        raise ValueError("Residual deltas have zero numerical rank")
    eigenvalues = eigenvalues[:kept]
    eigenvectors = eigenvectors[:, :kept]
    # Fix the otherwise arbitrary PC signs, making coefficient reports stable.
    for column in range(kept):
        pivot = int(np.argmax(np.abs(eigenvectors[:, column])))
        if eigenvectors[pivot, column] < 0.0:
            eigenvectors[:, column] *= -1.0
    scores = eigenvectors * np.sqrt(eigenvalues)[None, :]
    total_energy = float(np.trace(gram))
    per_sample = [
        {
            "trajectory_id": str(row.get("trajectory_id") or row.get("pair_id") or row.get("sample_id") or index),
            "true_band": str(row.get("true_band", "unknown")),
            **{f"pc{column + 1}": float(scores[index, column]) for column in range(kept)},
        }
        for index, row in enumerate(rows)
    ]
    return {
        "kind": "uncentered_gram_pca",
        "num_samples": sample_count,
        "numerical_rank": rank,
        "components": kept,
        "singular_values": [float(math.sqrt(value)) for value in eigenvalues],
        "energy_fraction": [float(value / total_energy) if total_energy > 0.0 else float("nan") for value in eigenvalues],
        "cumulative_energy_fraction": [
            float(eigenvalues[: column + 1].sum() / total_energy) if total_energy > 0.0 else float("nan")
            for column in range(kept)
        ],
        "scores": scores,
        "left_vectors": eigenvectors,
        "singular_values_array": np.sqrt(eigenvalues),
        "per_sample": per_sample,
    }


def _fit_pc_coordinates(
    rows: Sequence[Mapping[str, Any]],
    scores: np.ndarray,
    representation: RepresentationName,
) -> list[dict[str, Any]]:
    """Descriptively fit each PC coordinate; this is not a held-out score."""
    features = feature_matrix(rows, representation)
    design = _with_intercept(features)
    coefficient, *_ = np.linalg.lstsq(design, scores, rcond=None)
    predicted = design @ coefficient
    result: list[dict[str, Any]] = []
    for column in range(scores.shape[1]):
        target = scores[:, column]
        residual = target - predicted[:, column]
        total = float(np.sum((target - target.mean()) ** 2))
        fit: dict[str, Any] = {
            "component": column + 1,
            "r_squared_in_sample": float(1.0 - np.sum(residual**2) / total) if total > 0.0 else float("nan"),
            "intercept": float(coefficient[0, column]),
            "coefficients": [float(value) for value in coefficient[1:, column]],
        }
        if representation == "trigonometric":
            beta_cos, beta_sin = coefficient[1:, column]
            fit["sinusoid_amplitude"] = float(math.hypot(beta_cos, beta_sin))
            fit["trigonometric_peak_angle_radians"] = float(math.atan2(beta_sin, beta_cos) % (2.0 * math.pi))
        result.append(fit)
    return result


def _fit_heldout_indices(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically make the original experiment's fit/held-out split."""
    if len(rows) < 8:
        raise ValueError("Each colour direction needs at least eight paired trajectories")
    ordered = sorted(
        range(len(rows)),
        key=lambda index: hashlib.sha256(
            str(rows[index].get("trajectory_id") or rows[index].get("pair_id") or index).encode("utf-8")
        ).digest(),
    )
    cut = len(ordered) // 2
    return np.asarray(ordered[:cut]), np.asarray(ordered[cut:])


def _project_heldout_onto_fit_pca(
    residual_deltas: np.ndarray,
    *,
    fit_indices: np.ndarray,
    heldout_indices: np.ndarray,
    fit_pca: Mapping[str, Any],
    chunk_features: int = 262_144,
) -> np.ndarray:
    """Project held-out deltas into an uncentred PCA basis learned on fit rows."""
    flattened = _flattened_values(residual_deltas)
    cross_gram = np.zeros((heldout_indices.size, fit_indices.size), dtype=np.float64)
    for start in range(0, flattened.shape[1], chunk_features):
        stop = min(start + chunk_features, flattened.shape[1])
        heldout = np.asarray(flattened[heldout_indices, start:stop], dtype=np.float64)
        fit = np.asarray(flattened[fit_indices, start:stop], dtype=np.float64)
        cross_gram += heldout @ fit.T
    left_vectors = np.asarray(fit_pca["left_vectors"])
    singular_values = np.asarray(fit_pca["singular_values_array"])
    return (cross_gram @ left_vectors) / singular_values[None, :]


def evaluate_representation(
    rows: Sequence[Mapping[str, Any]],
    residual_deltas: np.ndarray,
    representation: RepresentationName,
    *,
    row_indices: np.ndarray | None = None,
    folds: int = 4,
    ridge: float = 1e-3,
    chunk_features: int = 262_144,
) -> dict[str, Any]:
    """Cross-validate a physical readout of paired residual differences.

    ``residual_deltas`` may be a memory-mapped ``.npy`` array.  Its first
    axis is samples and all remaining axes are flattened in bounded chunks,
    so the analysis works on the full ``[step, token, hidden]`` residual
    tensor without materialising a second copy in RAM.
    """
    values = np.asarray(residual_deltas)
    if values.ndim < 2:
        raise ValueError("residual_deltas must have at least one residual dimension")
    source_indices = (
        np.arange(len(rows), dtype=np.int64)
        if row_indices is None
        else np.asarray(row_indices, dtype=np.int64)
    )
    if source_indices.ndim != 1 or source_indices.size != len(rows) or np.any(source_indices < 0) or np.any(source_indices >= values.shape[0]):
        raise ValueError(
            "row_indices must select exactly one valid residual_deltas row per manifest row"
        )
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    if chunk_features <= 0:
        raise ValueError("chunk_features must be positive")

    raw_features = feature_matrix(rows, representation)
    keys = [str(row.get("trajectory_id") or row.get("pair_id") or row.get("sample_id") or index) for index, row in enumerate(rows)]
    fold_ids = np.asarray([deterministic_fold(key, folds) for key in keys], dtype=np.int64)
    observed_folds = sorted(set(int(value) for value in fold_ids))
    if len(observed_folds) < 2:
        raise ValueError("Need at least two populated folds; provide more trajectories or fewer folds")

    flattened = values.reshape(values.shape[0], -1)
    sample_count, width = len(rows), flattened.shape[1]
    dot = np.zeros(sample_count, dtype=np.float64)
    actual_sq = np.zeros(sample_count, dtype=np.float64)
    predicted_sq = np.zeros(sample_count, dtype=np.float64)
    error_sq = np.zeros(sample_count, dtype=np.float64)

    for fold in observed_folds:
        test_mask = fold_ids == fold
        train_mask = ~test_mask
        if train_mask.sum() <= raw_features.shape[1] or not np.any(test_mask):
            raise ValueError(f"Fold {fold} has too few training or test trajectories")
        train_features, test_features = _standardise(raw_features[train_mask], raw_features[test_mask])
        design_train = _with_intercept(train_features)
        design_test = _with_intercept(test_features)
        penalty = np.eye(design_train.shape[1], dtype=np.float64) * ridge
        penalty[0, 0] = 0.0
        inverse_gram = np.linalg.inv(design_train.T @ design_train + penalty)
        projector = inverse_gram @ design_train.T

        for start in range(0, width, chunk_features):
            stop = min(start + chunk_features, width)
            # Indexing one bounded feature chunk at a time preserves the
            # memory-map contract even when fast and slow rows are interleaved.
            train_values = np.asarray(flattened[source_indices[train_mask], start:stop], dtype=np.float64)
            test_values = np.asarray(flattened[source_indices[test_mask], start:stop], dtype=np.float64)
            coefficients = projector @ train_values
            predicted = design_test @ coefficients
            indices = np.flatnonzero(test_mask)
            dot[indices] += np.einsum("ij,ij->i", predicted, test_values)
            actual_sq[indices] += np.einsum("ij,ij->i", test_values, test_values)
            predicted_sq[indices] += np.einsum("ij,ij->i", predicted, predicted)
            error = predicted - test_values
            error_sq[indices] += np.einsum("ij,ij->i", error, error)

    denominator = np.sqrt(actual_sq * predicted_sq)
    cosine = np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator > 0.0)
    relative_projection = np.divide(dot, actual_sq, out=np.zeros_like(dot), where=actual_sq > 0.0)
    explained = 1.0 - np.divide(error_sq, actual_sq, out=np.full_like(error_sq, np.nan), where=actual_sq > 0.0)
    per_sample = [
        {
            "trajectory_id": keys[index],
            "fold": int(fold_ids[index]),
            "true_band": str(rows[index].get("true_band", "unknown")),
            "direction_capture_cosine": float(cosine[index]),
            "relative_projection": float(relative_projection[index]),
            "delta_explained_fraction": float(explained[index]),
        }
        for index in range(sample_count)
    ]
    return {
        "representation": representation,
        "feature_dimension": int(raw_features.shape[1]),
        "folds": int(folds),
        "ridge": float(ridge),
        "num_samples": sample_count,
        "metrics": {
            "direction_capture_cosine": _summary(cosine),
            "relative_projection": _summary(relative_projection),
            "delta_explained_fraction": _summary(explained),
        },
        "per_sample": per_sample,
    }


def evaluate_state_representations(
    rows: Sequence[Mapping[str, Any]],
    residual_deltas: np.ndarray,
    *,
    pca_components: int = 4,
) -> dict[str, Any]:
    """Replicate the document's PCA -> held-out PC-coordinate fit workflow."""
    if residual_deltas.shape[0] != len(rows):
        raise ValueError("residual_deltas and rows have different sample counts")
    groups = {
        "fast_red_conflict_to_blue_aligned": [index for index, row in enumerate(rows) if row.get("true_band") == "fast"],
        "slow_blue_conflict_to_red_aligned": [index for index, row in enumerate(rows) if row.get("true_band") == "slow"],
    }
    if any(not indices for indices in groups.values()):
        raise ValueError("Manifest must include both fast and slow trajectories")

    report_groups: dict[str, Any] = {}
    for group_name, indices in groups.items():
        group_rows = [rows[index] for index in indices]
        index_array = np.asarray(indices)
        fit_local, heldout_local = _fit_heldout_indices(group_rows)
        fit_global = index_array[fit_local]
        heldout_global = index_array[heldout_local]
        fit_rows = [group_rows[index] for index in fit_local]
        heldout_rows = [group_rows[index] for index in heldout_local]
        pca = _uncentered_pca(
            fit_rows,
            residual_deltas,
            row_indices=fit_global,
            components=pca_components,
        )
        heldout_scores = _project_heldout_onto_fit_pca(
            residual_deltas,
            fit_indices=fit_global,
            heldout_indices=heldout_global,
            fit_pca=pca,
        )
        heldout_coordinates = [
            {
                "trajectory_id": str(row.get("trajectory_id") or row.get("pair_id") or row.get("sample_id") or index),
                "true_band": str(row.get("true_band", "unknown")),
                **{f"pc{column + 1}": float(heldout_scores[index, column]) for column in range(heldout_scores.shape[1])},
            }
            for index, row in enumerate(heldout_rows)
        ]
        report_groups[group_name] = {
            "uncentered_pca": {
                **{
                    key: value
                    for key, value in pca.items()
                    if key not in {"scores", "per_sample", "left_vectors", "singular_values_array"}
                },
                "fit_trajectory_ids": [str(row.get("trajectory_id") or row.get("pair_id")) for row in fit_rows],
                "heldout_trajectory_ids": [str(row.get("trajectory_id") or row.get("pair_id")) for row in heldout_rows],
                "heldout_coordinates": heldout_coordinates,
                "coordinate_fits_on_heldout": {
                    representation: _fit_pc_coordinates(heldout_rows, heldout_scores, representation)
                    for representation in PRIMARY_REPRESENTATIONS
                },
            },
        }
    return {
        "design": {
            "target": "h_aligned - h_conflict at one fixed residual site",
            "stratification": "fast and slow physical bands are analysed separately",
            "pca": "uncentred PCA fit only on the deterministic fit split",
            "coordinate_regressions": list(PRIMARY_REPRESENTATIONS),
            "coordinate_regression_rows": "held-out trajectories projected into the fit PCA basis",
        },
        "groups": report_groups,
    }
