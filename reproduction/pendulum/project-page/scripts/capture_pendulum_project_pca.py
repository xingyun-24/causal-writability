#!/usr/bin/env python3
"""Capture held-out Pendulum activations and project them onto a frozen PCA basis.

The basis must be the uncentered SVD basis fitted by Stage 3 on the 64 fit
matched differences only.  This exporter never fits or recenters on held-out
data.  Each activation contains the condition tokens from all 20 flow-matching
calls, concatenated in call order before projection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from sshv2.experiments.pendulum.mechanism_pca import (
    _condition_latents,
    _denoise,
    _load_runtime,
    manifest_sha256,
    read_jsonl,
)


PROGRAM_VERSION = "pendulum_project_page_pca_v1"
STEPS = 20
CONDITION_TOKENS = 1088
HIDDEN_SIZE = 1152
BLOCK_INDEX = 12
PLOT_RANK = 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_natural_metrics(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (str(row["receiver_id"]), str(row["condition"]))
            omega = float(row["omega_hat"])
            if not math.isfinite(omega):
                raise ValueError(f"non-finite omega_hat for {key}")
            result[key] = {
                "omega_hat": omega,
                "valid": str(row.get("valid", "")).lower() == "true",
                "detected_color": row.get("detected_color"),
            }
    return result


def _basis_components(path: Path) -> np.ndarray:
    values = np.load(path, mmap_mode="r")
    if values.shape[0] < PLOT_RANK:
        raise ValueError(f"basis has only {values.shape[0]} components")
    expected = STEPS * CONDITION_TOKENS * HIDDEN_SIZE
    if int(np.prod(values.shape[1:])) != expected:
        raise ValueError(
            f"basis vector length {int(np.prod(values.shape[1:]))} != {expected}"
        )
    return values[:PLOT_RANK].reshape(PLOT_RANK, expected)


def _project(activation: np.ndarray, components: np.ndarray) -> list[float]:
    flat = np.asarray(activation, dtype=np.float32).reshape(-1)
    if flat.size != components.shape[1]:
        raise ValueError(f"activation length {flat.size} != {components.shape[1]}")
    score = np.zeros(PLOT_RANK, dtype=np.float64)
    chunk = 262_144
    for start in range(0, flat.size, chunk):
        stop = min(flat.size, start + chunk)
        score += (
            np.asarray(components[:, start:stop], dtype=np.float32)
            @ flat[start:stop]
        )
    if not np.isfinite(score).all():
        raise ValueError("non-finite PCA score")
    return [float(value) for value in score]


def _validate_basis_audit(
    audit_path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    fit_indices = [int(value) for value in audit.get("fit_indices", [])]
    heldout_indices = [int(value) for value in audit.get("heldout_indices", [])]
    if len(fit_indices) != 64 or len(heldout_indices) != 64:
        raise ValueError("PCA audit must record exactly 64 fit and 64 held-out rows")
    expected_fit = [index for index, row in enumerate(rows) if row["split"] == "fit"]
    expected_heldout = [
        index for index, row in enumerate(rows) if row["split"] == "heldout"
    ]
    if fit_indices != expected_fit or heldout_indices != expected_heldout:
        raise ValueError("PCA audit split indices do not match the frozen manifest")
    expected_manifest_sha = manifest_sha256(rows)
    recorded_manifest_sha = audit.get("manifest_sha256")
    if recorded_manifest_sha and recorded_manifest_sha != expected_manifest_sha:
        raise ValueError("PCA audit manifest SHA does not match the frozen manifest")
    return {
        "manifest_sha256_canonical": expected_manifest_sha,
        "fit_indices": fit_indices,
        "heldout_indices": heldout_indices,
        "audit_program_version": audit.get("protocol", audit.get("program_version")),
        "component_sha256_recorded": audit.get("component_sha256"),
    }


def _validate_strict_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    required = {
        "receiver_id", "split", "target_label", "direction", "generation_seed",
        "aligned_video", "conflict_video", "omega_true", "phase", "theta_star",
        "training_manifest_id",
    }
    if len(rows) != 128:
        raise ValueError(f"strict manifest must contain 128 rows, got {len(rows)}")
    if any(required - set(row) for row in rows):
        raise ValueError("strict manifest is missing required fields")
    receiver_ids = [str(row["receiver_id"]) for row in rows]
    if len(set(receiver_ids)) != len(receiver_ids):
        raise ValueError("strict manifest receiver IDs are not unique")
    counts = {
        f"{split}_{target}": sum(
            row["split"] == split and row["target_label"] == target for row in rows
        )
        for split in ("fit", "heldout")
        for target in ("low", "high")
    }
    if any(value != 32 for value in counts.values()):
        raise ValueError(f"strict manifest is not balanced: {counts}")
    return {
        "rows": 128,
        "fit": 64,
        "heldout": 64,
        "target": "frequency",
        "balance": counts,
        "sha256": manifest_sha256(rows),
    }


def capture(args: argparse.Namespace) -> None:
    import torch

    rows = read_jsonl(args.manifest)
    validation = _validate_strict_manifest(rows)
    audit = _validate_basis_audit(args.pca_audit, rows)
    heldout = [row for row in rows if row["split"] == "heldout"]
    shard_rows = [
        row
        for index, row in enumerate(heldout)
        if index % args.num_shards == args.shard_index
    ]
    if not shard_rows:
        raise ValueError("selected shard is empty")
    natural = _read_natural_metrics(args.natural_metrics)
    components = _basis_components(args.components)
    pipe, data_config = _load_runtime(
        SimpleNamespace(
            experiment_config=args.experiment_config,
            training_config=args.training_config,
            model_name=args.model_name,
            history="short",
            checkpoint=args.checkpoint,
            device=args.device,
            steps=STEPS,
            block_index=BLOCK_INDEX,
            condition_tokens=CONDITION_TOKENS,
            hidden_size=HIDDEN_SIZE,
        ),
        rows,
    )

    destination = args.out_dir / f"shard-{args.shard_index:02d}.json"
    endpoints: list[dict[str, Any]] = []
    if destination.is_file():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        endpoints = list(previous.get("endpoints", []))
    completed = {
        (str(item["receiver_id"]), str(item["condition"])) for item in endpoints
    }
    for pair_index, row in enumerate(shard_rows, start=1):
        receiver_id = str(row["receiver_id"])
        for condition in ("aligned", "conflict"):
            if (receiver_id, condition) in completed:
                continue
            condition_latents = _condition_latents(
                pipe,
                data_config,
                Path(str(row[f"{condition}_video"])),
                history="short",
            )
            unused_latents, activation = _denoise(
                pipe,
                data_config,
                condition_latents,
                seed=int(row["generation_seed"]),
                steps=STEPS,
                block_index=BLOCK_INDEX,
                condition_tokens=CONDITION_TOKENS,
                capture=True,
            )
            if activation is None:
                raise RuntimeError(f"activation capture failed: {receiver_id}/{condition}")
            metric = natural[(receiver_id, condition)]
            score = _project(activation, components)
            endpoints.append(
                {
                    "receiver_id": receiver_id,
                    "condition": condition,
                    "marker_symbol": "circle" if condition == "aligned" else "diamond",
                    "target_label": str(row["target_label"]),
                    "direction": str(row.get("direction", "")),
                    "generation_seed": int(row["generation_seed"]),
                    "input_video": str(row[f"{condition}_video"]),
                    "omega_true": float(row["omega_true"]),
                    "omega_hat": float(metric["omega_hat"]),
                    "natural_valid": bool(metric["valid"]),
                    "phase": float(row["phase"]),
                    "theta_star": float(row["theta_star"]),
                    "pc1": score[0],
                    "pc2": score[1],
                    "pc3": score[2],
                }
            )
            completed.add((receiver_id, condition))
            _atomic_json(
                destination,
                {
                    "protocol": PROGRAM_VERSION,
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                    "basis_validation": audit,
                    "manifest_validation": validation,
                    "endpoints": endpoints,
                },
            )
            del condition_latents, unused_latents, activation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(
            f"shard {args.shard_index}: {pair_index}/{len(shard_rows)} {receiver_id}",
            flush=True,
        )


def merge(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.manifest)
    validation = _validate_strict_manifest(rows)
    audit = _validate_basis_audit(args.pca_audit, rows)
    endpoints: list[dict[str, Any]] = []
    for shard_index in range(args.num_shards):
        shard_path = args.out_dir / f"shard-{shard_index:02d}.json"
        shard = json.loads(shard_path.read_text(encoding="utf-8"))
        endpoints.extend(shard["endpoints"])
    endpoints.sort(key=lambda row: (row["receiver_id"], row["condition"]))
    if len(endpoints) != 128:
        raise ValueError(f"expected 128 endpoint points, got {len(endpoints)}")
    by_receiver: dict[str, dict[str, Mapping[str, Any]]] = {}
    for endpoint in endpoints:
        by_receiver.setdefault(str(endpoint["receiver_id"]), {})[
            str(endpoint["condition"])
        ] = endpoint
    if len(by_receiver) != 64 or any(set(pair) != {"aligned", "conflict"} for pair in by_receiver.values()):
        raise ValueError("expected 64 complete aligned/conflict held-out pairs")
    differences: list[dict[str, Any]] = []
    for receiver_id, pair in sorted(by_receiver.items()):
        aligned, conflict = pair["aligned"], pair["conflict"]
        differences.append(
            {
                "receiver_id": receiver_id,
                "target_label": aligned["target_label"],
                "direction": aligned["direction"],
                "generation_seed": aligned["generation_seed"],
                "omega_true": aligned["omega_true"],
                "omega_aligned": aligned["omega_hat"],
                "omega_conflict": conflict["omega_hat"],
                "generated_frequency_gap": abs(aligned["omega_hat"] - conflict["omega_hat"]),
                "pc1": aligned["pc1"] - conflict["pc1"],
                "pc2": aligned["pc2"] - conflict["pc2"],
                "pc3": aligned["pc3"] - conflict["pc3"],
            }
        )
    payload = {
        "schema_version": 1,
        "protocol": PROGRAM_VERSION,
        "title": "Pendulum held-out residual-state geometry",
        "model": "frequency_color_circle Large-Short, seed 3407, step 50000",
        "checkpoint_sha256": args.checkpoint_sha256,
        "basis": {
            "fit": "uncentered SVD of 64 fit matched differences",
            "projection": "raw held-out aligned/conflict activations; no held-out fit or recentering",
            "block_index_zero_based": BLOCK_INDEX,
            "block_label": "after DiT block 12",
            "fm_calls_concatenated": STEPS,
            "condition_tokens_per_call": CONDITION_TOKENS,
            "hidden_size": HIDDEN_SIZE,
            "rank_displayed": PLOT_RANK,
            "components_sha256": _sha256(args.components),
            **audit,
        },
        "manifest": {
            **validation,
            "file_sha256": _sha256(args.manifest),
        },
        "counts": {
            "fit_pairs_used_for_basis": 64,
            "heldout_pairs_projected": 64,
            "endpoint_points": len(endpoints),
            "difference_points": len(differences),
            "endpoint_groups": 4,
        },
        "endpoint_color": "natural generated frequency omega_hat (Viridis)",
        "difference_color": "absolute natural generated-frequency gap (Viridis)",
        "endpoints": endpoints,
        "differences": differences,
    }
    _atomic_json(args.output_json, payload)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--manifest", type=Path, required=True)
    common.add_argument("--pca-audit", type=Path, required=True)
    common.add_argument("--components", type=Path, required=True)
    common.add_argument("--out-dir", type=Path, required=True)
    common.add_argument("--num-shards", type=int, default=4)

    capture_parser = subparsers.add_parser("capture", parents=[common])
    capture_parser.add_argument("--shard-index", type=int, required=True)
    capture_parser.add_argument("--natural-metrics", type=Path, required=True)
    capture_parser.add_argument("--checkpoint", type=Path, required=True)
    capture_parser.add_argument("--training-config", type=Path, required=True)
    capture_parser.add_argument("--experiment-config", type=Path, required=True)
    capture_parser.add_argument("--model-name", default="frequency_color_circle")
    capture_parser.add_argument("--device", default="cuda")

    merge_parser = subparsers.add_parser("merge", parents=[common])
    merge_parser.add_argument("--output-json", type=Path, required=True)
    merge_parser.add_argument("--checkpoint-sha256", required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "capture":
        if not 0 <= args.shard_index < args.num_shards:
            raise ValueError("shard index is out of range")
        capture(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
