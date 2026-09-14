#!/usr/bin/env python3
"""Stage 3: low-rank matched causal route at l* = after block 15.

The main Stage 2 experiment fixed the analysis block to

    l* = after block 15  (DiT block index 15, zero based).

Stage 3 builds the frozen 128 matched-pair bank (64 fit + 64 heldout) from
the in-support frequency_color_circle scan states, captures the aligned and
conflict condition-token activations at l* on all 20 flow-matching steps,
and extracts

    d_i = h_{A,i} - h_{C,i}

for every pair.  An uncentered SVD is fit on the 64 fit directions only
(ranks m in {1,2,3,4,8}), the heldout oracle geometry E_m is recorded, and
the heldout causal ceiling is measured by injecting

    h'_{C,j} = h_{C,j} + d_j^(m),   d_j^(m) = V_m V_m^T d_j,

at l*.  The selected rank is the smallest m whose causal recovery is
basically as good as the full-direction injection.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from sshv2.experiments.pendulum.mechanism_pca import (
    _condition_latents,
    _denoise,
    _fit_chunked_pca,
    _load_runtime,
    _measure_one,
    _sha256_file,
    _write_csv,
    _write_future_video,
    _write_json,
    manifest_sha256,
    read_jsonl,
)


PROGRAM_VERSION = "pendulum_stage3_low_rank_causal_route_v2_hidden_size_parameterized"
DEFAULT_COMMITMENT_BLOCK_INDEX = 15
CONDITION_TOKENS = 1088
DEFAULT_HIDDEN_SIZE = 768
STEPS = 20
RANKS = (1, 2, 3, 4, 8)
FIT_PHASES = (0, 1, 2, 3)
HELDOUT_PHASES = (4, 5, 6, 7)
RECOVERY_THRESHOLD = 0.5
FULL_D_CLOSE_FRACTION = 0.95


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _freeze_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    materialized = [dict(row) for row in rows]
    if path.is_file():
        existing = read_jsonl(path)
        if existing != materialized:
            raise ValueError(f"frozen file differs from current selection: {path}")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in materialized
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return materialized


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _finite(values: Sequence[Any]) -> np.ndarray:
    result = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    return result


def _median(values: Sequence[Any]) -> float | None:
    finite = _finite(values)
    return float(np.median(finite)) if finite.size else None


def _direction_path(out_root: Path, receiver_id: str) -> Path:
    return out_root / "directions" / f"{receiver_id}.npz"


def _video_path(out_root: Path, kind: str, receiver_id: str) -> Path:
    return out_root / "videos" / receiver_id / f"{kind}.mp4"


def build_manifest(args: argparse.Namespace) -> list[dict[str, Any]]:
    """128 matched pairs: in-support frequencies x 8 phases x 2 repeats."""
    rows = _read_rows(args.dataset_root / "videos" / "eval" / "metadata.csv")
    frequencies: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["test_frequency_support"] in ("low_band_id", "high_band_id"):
            frequencies[str(row["omega_true"])].append(row)
    manifest: list[dict[str, Any]] = []
    for omega in sorted(frequencies, key=float):
        band_rows = frequencies[omega]
        is_low = band_rows[0]["test_frequency_support"] == "low_band_id"
        aligned_alpha = 0.0 if is_low else 1.0
        conflict_alpha = 1.0 if is_low else 0.0
        by_phase_repeat: dict[tuple[int, int], dict[str, dict[str, str]]] = {}
        for row in band_rows:
            alpha = float(row["test_color_alpha_target"])
            key = (int(row["phase_index"]), int(row["diffusion_repeat"]))
            condition = "aligned" if abs(alpha - aligned_alpha) < 1e-9 else (
                "conflict" if abs(alpha - conflict_alpha) < 1e-9 else None
            )
            if condition is not None:
                by_phase_repeat.setdefault(key, {})[condition] = row
        for (phase_index, repeat), pair in sorted(by_phase_repeat.items()):
            aligned = pair.get("aligned")
            conflict = pair.get("conflict")
            if aligned is None or conflict is None:
                raise ValueError(
                    f"missing endpoint pair for omega={omega} "
                    f"phase={phase_index} repeat={repeat}"
                )
            for field in (
                "physical_state_id",
                "trajectory_id",
                "pair_id",
                "base_seed",
                "omega_true",
                "amplitude_true",
                "phase",
            ):
                if aligned[field] != conflict[field]:
                    raise ValueError(
                        f"aligned/conflict mismatch on {field}: "
                        f"{omega} phase={phase_index} repeat={repeat}"
                    )
            split = (
                "fit"
                if phase_index in FIT_PHASES
                else "heldout"
                if phase_index in HELDOUT_PHASES
                else None
            )
            if split is None:
                raise ValueError(f"phase {phase_index} not in any split")
            target_label = "low" if is_low else "high"
            receiver_id = (
                f"w{omega.replace('.', 'p')}_phase{phase_index:02d}_"
                f"rep{repeat:02d}"
            )
            manifest.append(
                {
                    "receiver_id": receiver_id,
                    "pair_id": aligned["pair_id"],
                    "physical_state_id": aligned["physical_state_id"],
                    "trajectory_id": aligned["trajectory_id"],
                    "split": split,
                    "target": "frequency",
                    "target_label": target_label,
                    "omega_true": float(aligned["omega_true"]),
                    "amplitude_true": float(aligned["amplitude_true"]),
                    "phase": float(aligned["phase"]),
                    "phase_index": phase_index,
                    "diffusion_repeat": repeat,
                    "generation_seed": int(aligned["base_seed"])
                    + args.seed_offset,
                    "theta_star": float(aligned["theta_star"]),
                    "aligned_sample_id": aligned["sample_id"],
                    "conflict_sample_id": conflict["sample_id"],
                    "aligned_color": aligned["color_label"],
                    "conflict_color": conflict["color_label"],
                    "aligned_shape": aligned["shape_label"],
                    "conflict_shape": conflict["shape_label"],
                    "aligned_video": str(
                        args.dataset_root / "videos" / "eval" / aligned["video"]
                    ),
                    "conflict_video": str(
                        args.dataset_root / "videos" / "eval" / conflict["video"]
                    ),
                    "training_manifest_id": aligned["training_manifest_id"],
                    "test_manifest_id": aligned["test_manifest_id"],
                    "model_name": args.model_name,
                }
            )
    fit = [row for row in manifest if row["split"] == "fit"]
    heldout = [row for row in manifest if row["split"] == "heldout"]
    if len(fit) != 64 or len(heldout) != 64:
        raise ValueError(
            f"expected 64 fit / 64 heldout, got {len(fit)} / {len(heldout)}"
        )
    for name, group in (("fit", fit), ("heldout", heldout)):
        low = sum(1 for row in group if row["target_label"] == "low")
        high = sum(1 for row in group if row["target_label"] == "high")
        print(
            f"manifest {name}: {len(group)} pairs "
            f"(low={low}, high={high})",
            flush=True,
        )
    return _freeze_jsonl(args.out_root / "receiver_manifest.jsonl", manifest)


def natural_baselines(
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    pipe: Any,
    data_config: Any,
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        receiver_id = str(row["receiver_id"])
        for condition, video_key in (
            ("aligned", "aligned_video"),
            ("conflict", "conflict_video"),
        ):
            destination = _video_path(args.out_root, condition, receiver_id)
            if not destination.is_file():
                condition_tensor = _condition_latents(
                    pipe,
                    data_config,
                    Path(str(row[video_key])),
                    history=args.history,
                )
                latents, _ = _denoise(
                    pipe,
                    data_config,
                    condition_tensor,
                    seed=int(row["generation_seed"]),
                    steps=STEPS,
                    block_index=args.commitment_block_index,
                    condition_tokens=CONDITION_TOKENS,
                    capture=False,
                )
                _write_future_video(pipe, data_config, latents, destination)
                del condition_tensor, latents
                if args.device.startswith("cuda"):
                    import torch

                    torch.cuda.empty_cache()
            measured = _measure_one(
                destination,
                row,
                data_config,
                expected_colors=(
                    str(row["aligned_color"]),
                    str(row["conflict_color"]),
                ),
            )
            result[receiver_id][condition] = {
                "omega_hat": measured["omega_hat"],
                "valid": measured["valid"],
                "detected_color": measured["detected_color"],
                "omega_true": float(row["omega_true"]),
            }
        print(
            f"baseline {receiver_id} "
            f"aligned={result[receiver_id]['aligned']['omega_hat']} "
            f"conflict={result[receiver_id]['conflict']['omega_hat']}",
            flush=True,
        )
    rows_out = [
        {
            "receiver_id": receiver_id,
            "condition": condition,
            **values,
        }
        for receiver_id, conditions in sorted(result.items())
        for condition, values in conditions.items()
    ]
    _write_csv(args.out_root / "natural_metrics.csv", rows_out)
    return dict(result)


def capture_directions(
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    pipe: Any,
    data_config: Any,
) -> np.ndarray:
    import torch

    shape = (
        len(rows),
        STEPS,
        CONDITION_TOKENS,
        args.hidden_size,
    )
    directions_path = args.out_root / "directions.npy"
    if directions_path.is_file():
        directions = np.load(directions_path, mmap_mode="r+")
        if tuple(directions.shape) != shape:
            raise ValueError("existing directions.npy shape mismatch")
    else:
        directions = np.lib.format.open_memmap(
            directions_path,
            mode="w+",
            dtype=np.float32,
            shape=shape,
        )
    progress_path = args.out_root / "capture_progress.json"
    if progress_path.is_file():
        completed = set(_read_json(progress_path).get("completed", []))
    else:
        completed = set()
    started = time.monotonic()
    for index, row in enumerate(rows):
        receiver_id = str(row["receiver_id"])
        if receiver_id in completed:
            continue
        aligned_tensor = _condition_latents(
            pipe,
            data_config,
            Path(str(row["aligned_video"])),
            history=args.history,
        )
        _unused, aligned_activation = _denoise(
            pipe,
            data_config,
            aligned_tensor,
            seed=int(row["generation_seed"]),
            steps=STEPS,
            block_index=args.commitment_block_index,
            condition_tokens=CONDITION_TOKENS,
            capture=True,
        )
        del aligned_tensor
        conflict_tensor = _condition_latents(
            pipe,
            data_config,
            Path(str(row["conflict_video"])),
            history=args.history,
        )
        _unused, conflict_activation = _denoise(
            pipe,
            data_config,
            conflict_tensor,
            seed=int(row["generation_seed"]),
            steps=STEPS,
            block_index=args.commitment_block_index,
            condition_tokens=CONDITION_TOKENS,
            capture=True,
        )
        del conflict_tensor
        if aligned_activation is None or conflict_activation is None:
            raise AssertionError(f"activation capture failed for {receiver_id}")
        direction = (
            aligned_activation.astype(np.float32)
            - conflict_activation.astype(np.float32)
        )
        directions[index] = direction
        directions.flush()
        completed.add(receiver_id)
        _atomic_write_json(
            progress_path,
            {
                "completed": sorted(completed),
                "num_completed": len(completed),
                "expected": len(rows),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        print(
            f"capture {len(completed)}/{len(rows)} {receiver_id} "
            f"d_norm={float(np.linalg.norm(direction)):.4g}",
            flush=True,
        )
        del direction, aligned_activation, conflict_activation, _unused
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return directions


def fit_pca(
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    directions: np.ndarray,
) -> Any:
    pca_args = SimpleNamespace(
        out_root=args.out_root,
        ranks=RANKS,
        pca_chunk_values=args.pca_chunk_values,
    )
    return _fit_chunked_pca(pca_args, rows, directions)


def _projection_edit(
    artifacts: Any,
    heldout_position: int,
    rank: int,
    direction_shape: tuple[int, int, int],
) -> np.ndarray:
    """Raw projection d_j^(m) = V_m V_m^T d_j (no norm matching)."""
    components = np.load(
        artifacts.components_path, mmap_mode="r"
    ).reshape(max(rank, artifacts.coefficients.shape[1]), -1)
    coefficients = artifacts.coefficients[heldout_position, :rank]
    flat_size = components.shape[1]
    result = np.empty(flat_size, dtype=np.float32)
    chunk = 262_144
    for start in range(0, flat_size, chunk):
        stop = min(flat_size, start + chunk)
        result[start:stop] = (
            coefficients @ np.asarray(components[:rank, start:stop])
        )
    return result.reshape(direction_shape)


def causal_ceiling(
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    directions: np.ndarray,
    artifacts: Any,
    pipe: Any,
    data_config: Any,
) -> list[dict[str, Any]]:
    import torch

    natural = _natural_map(args.out_root)
    metrics: list[dict[str, Any]] = []
    labels: list[tuple[str, int | str]] = [
        (f"rank_{rank}", rank) for rank in RANKS
    ] + [("full_d", "full")]
    for heldout_position, manifest_index in enumerate(
        artifacts.heldout_indices
    ):
        row = rows[int(manifest_index)]
        receiver_id = str(row["receiver_id"])
        baselines = natural[receiver_id]
        omega_aligned = float(baselines["aligned"]["omega_hat"])
        omega_conflict = float(baselines["conflict"]["omega_hat"])
        denominator = omega_aligned - omega_conflict
        if abs(denominator) <= 1e-8:
            raise ValueError(f"zero recovery space for {receiver_id}")
        conflict_tensor = _condition_latents(
            pipe,
            data_config,
            Path(str(row["conflict_video"])),
            history=args.history,
        )
        for label, rank in labels:
            destination = _video_path(
                args.out_root, f"edit_{label}", receiver_id
            )
            if destination.is_file():
                print(f"ceiling {receiver_id} {label} resume", flush=True)
            else:
                if rank == "full":
                    edit_array = np.asarray(
                        directions[int(manifest_index)], dtype=np.float32
                    )
                else:
                    edit_array = _projection_edit(
                        artifacts,
                        heldout_position,
                        int(rank),
                        tuple(int(value) for value in directions.shape[1:]),
                    )
                edit = torch.from_numpy(
                    np.array(edit_array, dtype=np.float32, copy=True)
                ).to(device=pipe.device, dtype=pipe.torch_dtype)
                latents, _ = _denoise(
                    pipe,
                    data_config,
                    conflict_tensor,
                    seed=int(row["generation_seed"]),
                    steps=STEPS,
                    block_index=args.commitment_block_index,
                    condition_tokens=CONDITION_TOKENS,
                    edit=edit,
                    strength=1.0,
                )
                _write_future_video(pipe, data_config, latents, destination)
                print(f"ceiling {receiver_id} {label} complete", flush=True)
                del edit, latents, edit_array
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            measured = _measure_one(
                destination,
                row,
                data_config,
                expected_colors=(
                    str(row["aligned_color"]),
                    str(row["conflict_color"]),
                ),
            )
            omega_hat = float(measured["omega_hat"])
            metrics.append(
                {
                    "receiver_id": receiver_id,
                    "target_label": row["target_label"],
                    "split": row["split"],
                    "edit": label,
                    "rank": (int(rank) if rank != "full" else None),
                    "omega_true": float(row["omega_true"]),
                    "omega_aligned": omega_aligned,
                    "omega_conflict": omega_conflict,
                    "omega_hat": omega_hat,
                    "frequency_recovery": (
                        (omega_hat - omega_conflict) / denominator
                        if math.isfinite(omega_hat)
                        else float("nan")
                    ),
                    "valid": measured["valid"],
                    "detected_color": measured["detected_color"],
                    "fit_rmse": measured["fit_rmse"],
                }
            )
        del conflict_tensor
    _write_csv(args.out_root / "ceiling_metrics.csv", metrics)
    return metrics


def _natural_map(out_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    def _safe_float(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in _read_csv(out_root / "natural_metrics.csv"):
        result[row["receiver_id"]][row["condition"]] = {
            "omega_hat": _safe_float(row["omega_hat"]),
            "valid": row["valid"] == "True",
        }
    return result


def oracle_geometry(args: argparse.Namespace) -> list[dict[str, Any]]:
    audit = _read_json(args.out_root / "pca" / "audit.json")
    geometry: list[dict[str, Any]] = []
    by_receiver: dict[str, dict[int, float]] = defaultdict(dict)
    for record in audit.get("projection_audit", []):
        by_receiver[record["receiver_id"]][int(record["rank"])] = float(
            record["retained_energy_before_norm_match"]
        )
    for rank in RANKS:
        energies = [
            energies[rank]
            for energies in by_receiver.values()
            if rank in energies
        ]
        geometry.append(
            {
                "rank": rank,
                "num_heldout": len(energies),
                "median_retained_energy_E_m": _median(energies),
                "q25_retained_energy_E_m": (
                    float(np.quantile(energies, 0.25)) if energies else None
                ),
                "q75_retained_energy_E_m": (
                    float(np.quantile(energies, 0.75)) if energies else None
                ),
            }
        )
    return geometry


def select_mstar(
    metrics: Sequence[Mapping[str, Any]],
    geometry: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    valid = [row for row in metrics if bool(row["valid"])]
    full = [
        float(row["frequency_recovery"])
        for row in valid
        if row["edit"] == "full_d"
        and math.isfinite(float(row["frequency_recovery"]))
    ]
    full_median = float(np.median(full)) if full else None
    by_rank: dict[int, float | None] = {}
    for rank in RANKS:
        values = [
            float(row["frequency_recovery"])
            for row in valid
            if row.get("rank") == rank
            and math.isfinite(float(row["frequency_recovery"]))
        ]
        by_rank[rank] = float(np.median(values)) if values else None
    threshold = (
        max(RECOVERY_THRESHOLD, FULL_D_CLOSE_FRACTION * full_median)
        if full_median is not None
        else None
    )
    mstar = next(
        (
            rank
            for rank in RANKS
            if by_rank.get(rank) is not None
            and threshold is not None
            and by_rank[rank] >= threshold
        ),
        None,
    )
    return {
        "recovery_threshold": RECOVERY_THRESHOLD,
        "full_d_close_fraction": FULL_D_CLOSE_FRACTION,
        "full_d_median_recovery": full_median,
        "median_recovery_by_rank": by_rank,
        "selection_threshold": threshold,
        "m_star": mstar,
        "median_retained_energy_by_rank": {
            int(row["rank"]): row["median_retained_energy_E_m"]
            for row in geometry
        },
    }


def plot_results(
    args: argparse.Namespace,
    geometry: Sequence[Mapping[str, Any]],
    mstar: Mapping[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = args.out_root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    ranks = [int(row["rank"]) for row in geometry]
    energies = [row["median_retained_energy_E_m"] for row in geometry]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(
        ranks, energies, marker="o", color="#1b6ca8", lw=1.6
    )
    axes[0].set_xlabel("rank m")
    axes[0].set_ylabel("median heldout retained energy E_m")
    axes[0].set_title("Oracle geometry: E_m = ||V_mV_m^T d||² / ||d||²")
    axes[0].grid(alpha=0.2)

    recovery = mstar["median_recovery_by_rank"]
    rec_ranks = [rank for rank in RANKS if recovery.get(rank) is not None]
    rec_values = [recovery[rank] for rank in rec_ranks]
    axes[1].plot(
        rec_ranks, rec_values, marker="o", color="#d95f02", lw=1.6
    )
    full = mstar["full_d_median_recovery"]
    if full is not None:
        axes[1].axhline(
            full,
            color="black",
            ls="--",
            lw=1.0,
            label=f"full d ({full:.3f})",
        )
    axes[1].axhline(
        RECOVERY_THRESHOLD, color="green", ls=":", lw=1.0
    )
    if mstar.get("m_star") is not None:
        axes[1].axvline(
            mstar["m_star"], color="green", ls="--", lw=1.4
        )
    axes[1].set_xlabel("rank m")
    axes[1].set_ylabel("median heldout causal recovery")
    axes[1].set_title(f"Stage 3 causal ceiling — m* = {mstar.get('m_star')}")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    path = plots / "stage3_rank_selection.png"
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    figure.savefig(temporary, dpi=170, bbox_inches="tight")
    os.replace(temporary, path)
    plt.close(figure)


def build_report(
    *,
    args: argparse.Namespace,
    checkpoint_sha: str,
    rows: Sequence[Mapping[str, Any]],
    geometry: Sequence[Mapping[str, Any]],
    mstar: Mapping[str, Any],
    ceiling: Sequence[Mapping[str, Any]],
) -> str:
    def _fmt(value: Any, digits: int = 3) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "n/a"
        return "n/a" if not math.isfinite(number) else f"{number:.{digits}f}"

    lines = [
        f"# Pendulum Stage 3：Low-Rank Matched Causal Route（ℓ\* = after block {args.commitment_block_index}, 50k-short）",
        "",
        "Status: complete. 协议版本: " + PROGRAM_VERSION.replace("pendulum_", ""),
        "",
        "## 实验身份",
        "",
        f"- Model: `{args.model_name}`，50k-step short checkpoint",
        f"- Checkpoint SHA256: `{checkpoint_sha}`",
        f"- DiT residual width: `{args.hidden_size}`",
        f"- ℓ\* = after block {args.commitment_block_index}（DiT block index {args.commitment_block_index}，来自 Stage 2 commitment scan）",
        f"- Matched pair bank: {len(rows)}（64 fit + 64 heldout；每 split 含 low {sum(1 for r in rows if r['split']=='fit' and r['target_label']=='low')}/high {sum(1 for r in rows if r['split']=='fit' and r['target_label']=='high')}，比例一致）",
        f"- d_i = h_{{A,i}} − h_{{C,i}}，uncentered SVD 只在 fit64 上拟合；rank scan m ∈ {list(RANKS)}",
        "",
        "## Heldout oracle geometry（E_m）",
        "",
        "| rank | median E_m | q25 | q75 |",
        "|---:|---:|---:|---:|",
    ]
    for row in geometry:
        lines.append(
            f"| {row['rank']} | {_fmt(row['median_retained_energy_E_m'])} | "
            f"{_fmt(row['q25_retained_energy_E_m'])} | "
            f"{_fmt(row['q75_retained_energy_E_m'])} |"
        )
    lines += [
        "",
        "## Heldout causal ceiling（注入 h_C + d^(m) at ℓ\*）",
        "",
        "| edit | median recovery |",
        "|---|---:|",
    ]
    valid = [row for row in ceiling if bool(row["valid"])]
    for label in [f"rank_{rank}" for rank in RANKS] + ["full_d"]:
        values = [
            float(row["frequency_recovery"])
            for row in valid
            if row["edit"] == label
            and math.isfinite(float(row["frequency_recovery"]))
        ]
        lines.append(
            f"| {label} | {_fmt(_median(values))} |"
        )
    lines += [
        "",
        "## m\* 选择",
        "",
        f"- full-d median recovery = {_fmt(mstar['full_d_median_recovery'])}",
        f"- 选择规则：最小的 m 使 median recovery ≥ max(0.5, 0.95 × full-d)，"
        f"当前阈值 = {_fmt(mstar['selection_threshold'])}",
        f"- **m\* = {mstar['m_star']}**",
        "- 后续所有 coordinate regression 固定使用 m\*，不再根据 heldout 反复换 rank。",
        "",
        "## Artifacts",
        "",
        "- `receiver_manifest.jsonl`: 128 matched pair 清单",
        "- `natural_metrics.csv`: aligned/conflict 自然基线",
        "- `directions.npy`: 128 × [20, 1088, 768] 的 d = h_A − h_C",
        "- `pca/components.npy`、`pca/audit.json`: fit64 uncentered SVD + E_m",
        "- `ceiling_metrics.csv`: heldout 逐 edit 的因果 recovery",
        "- `plots/stage3_rank_selection.png`: E_m 与 recovery 曲线",
    ]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.out_root = Path(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha = _sha256_file(checkpoint)
    if args.manifest:
        rows = read_jsonl(args.manifest)
        for split in ("fit", "heldout"):
            group = [row for row in rows if row["split"] == split]
            low = sum(1 for row in group if row["target_label"] == "low")
            high = sum(1 for row in group if row["target_label"] == "high")
            if len(group) != 64 or low != 32 or high != 32:
                raise ValueError(
                    f"strict manifest split {split}: {len(group)} rows "
                    f"(low={low}, high={high}); need 64 (32/32)"
                )
    else:
        rows = build_manifest(args)
    pipe, data_config = _load_runtime(
        SimpleNamespace(
            experiment_config=args.experiment_config,
            training_config=args.training_config,
            model_name=args.model_name,
            history=args.history,
            checkpoint=args.checkpoint,
            device=args.device,
            steps=STEPS,
            block_index=args.commitment_block_index,
            condition_tokens=CONDITION_TOKENS,
            hidden_size=args.hidden_size,
        ),
        rows,
    )
    if args.commitment_block_index >= len(pipe.dit.blocks):
        raise ValueError("commitment block index out of range")

    if args.natural_metrics:
        import shutil

        shutil.copyfile(args.natural_metrics, args.out_root / "natural_metrics.csv")
        natural = _natural_map(args.out_root)
    else:
        natural = natural_baselines(args, rows, pipe, data_config)
    directions = capture_directions(args, rows, pipe, data_config)
    artifacts = fit_pca(args, rows, directions)
    geometry = oracle_geometry(args)
    ceiling = causal_ceiling(
        args, rows, directions, artifacts, pipe, data_config
    )
    mstar = select_mstar(ceiling, geometry)
    plot_results(args, geometry, mstar)
    report = build_report(
        args=args,
        checkpoint_sha=checkpoint_sha,
        rows=rows,
        geometry=geometry,
        mstar=mstar,
        ceiling=ceiling,
    )
    (args.out_root / "RESULTS.md").write_text(report, encoding="utf-8")
    _atomic_write_json(
        args.out_root / "summary.json",
        {
            "protocol": PROGRAM_VERSION,
            "model_name": args.model_name,
            "history": args.history,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "hidden_size": args.hidden_size,
            "commitment_block_index": args.commitment_block_index,
            "receiver_manifest_sha256": manifest_sha256(rows),
            "fit_count": sum(1 for row in rows if row["split"] == "fit"),
            "heldout_count": sum(
                1 for row in rows if row["split"] == "heldout"
            ),
            "ranks": list(RANKS),
            "oracle_geometry": geometry,
            "m_star": mstar,
            "natural_valid_rate": (
                float(
                    np.mean(
                        [
                            bool(natural[rid][c]["valid"])
                            for rid in natural
                            for c in ("aligned", "conflict")
                        ]
                    )
                )
                if natural
                else None
            ),
        },
    )
    (args.out_root / "EXPERIMENT_COMPLETE").write_text(
        "complete\n", encoding="utf-8"
    )
    return {
        "receiver_count": len(rows),
        "m_star": mstar.get("m_star"),
        "median_recovery_by_rank": mstar["median_recovery_by_rank"],
        "out_root": str(args.out_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3 low-rank matched causal route at l*."
    )
    parser.add_argument("--model-name", default="frequency_color_circle")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--history", choices=("short", "long"), default="short")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--commitment-block-index",
        type=int,
        default=DEFAULT_COMMITMENT_BLOCK_INDEX,
        help="Zero-based block index selected by the matching Stage 2 scan.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=DEFAULT_HIDDEN_SIZE,
        help="DiT residual width (768 for medium, 1152 for large).",
    )
    parser.add_argument("--seed-offset", type=int, default=23_000_000)
    parser.add_argument("--pca-chunk-values", type=int, default=262_144)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="frozen strict bank manifest (128 rows, pre-fixed fit64/heldout64)",
    )
    parser.add_argument(
        "--natural-metrics",
        type=Path,
        default=None,
        help="pre-measured natural metrics CSV",
    )
    args = parser.parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "status": "ok",
                "receiver_count": result["receiver_count"],
                "m_star": result["m_star"],
                "median_recovery_by_rank": result[
                    "median_recovery_by_rank"
                ],
                "out_root": result["out_root"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
