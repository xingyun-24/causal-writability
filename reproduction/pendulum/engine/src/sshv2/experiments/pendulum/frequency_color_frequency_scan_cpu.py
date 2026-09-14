#!/usr/bin/env python3
"""Run and merge resumable CPU shards for the Pendulum 2-D scan.

The regular predictor validates the complete scan before inference.  This
module keeps that validation, then partitions whole ``pair_id`` groups so
every physical state retains all eleven color interventions in one shard.
Each shard uses the existing validated prediction implementation and writes
an ordinary SSHV2 prediction directory.  ``merge`` hard-links the disjoint
outputs into the canonical prediction directory and creates the same common
prediction manifest consumed by the evaluation pipeline.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from sshv2.common.dataset import read_json, write_json
from sshv2.common.results import read_predictions
from sshv2.experiments.pendulum.frequency_color_frequency_scan_data import (
    load_config,
)
from sshv2.experiments.pendulum.frequency_color_frequency_scan_runner import (
    _validate_rows,
)


def _read_rows(dataset_root: Path) -> list[dict[str, str]]:
    metadata = dataset_root / "videos" / "eval" / "metadata.csv"
    with metadata.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty scan metadata: {metadata}")
    return rows


def _validated_inputs(
    dataset_root: Path,
    experiment_config: Path,
) -> tuple[Any, Any, list[dict[str, str]], str]:
    config, spec = load_config(experiment_config)
    rows = _read_rows(dataset_root)
    test_manifest_id = _validate_rows(
        rows,
        spec=spec,
        training_manifest_id=config.training_manifest_id,
        model_name=config.model_name,
    )
    return config, spec, rows, test_manifest_id


def _pair_groups(
    rows: list[dict[str, str]],
    color_alphas: tuple[float, ...],
) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["pair_id"], []).append(row)
    expected_alphas = set(color_alphas)
    for pair_id, pair_rows in groups.items():
        if len(pair_rows) != len(color_alphas):
            raise ValueError(
                f"pair {pair_id} has {len(pair_rows)} rows; "
                f"expected {len(color_alphas)}"
            )
        actual_alphas = {
            float(row["test_color_alpha_target"]) for row in pair_rows
        }
        if actual_alphas != expected_alphas:
            raise ValueError(f"pair {pair_id} does not cover every color")
        if len({row["physical_state_id"] for row in pair_rows}) != 1:
            raise ValueError(f"pair {pair_id} mixes physical states")
    return groups


def _select_shard(
    rows: list[dict[str, str]],
    *,
    color_alphas: tuple[float, ...],
    num_shards: int,
    shard_index: int,
) -> tuple[list[dict[str, str]], list[str]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    groups = _pair_groups(rows, color_alphas)
    pair_ids = sorted(groups)
    selected_ids = pair_ids[shard_index::num_shards]
    selected_set = set(selected_ids)
    selected = [row for row in rows if row["pair_id"] in selected_set]
    if not selected:
        raise ValueError(f"empty shard {shard_index}/{num_shards}")
    return selected, selected_ids


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def predict_shard(args: argparse.Namespace) -> Path:
    from sshv2.experiments.pendulum import frequency_color_low_predict as base

    config, spec, rows, test_manifest_id = _validated_inputs(
        args.dataset_root,
        args.experiment_config,
    )
    selected, pair_ids = _select_shard(
        rows,
        color_alphas=spec.color_alphas,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    selected_ids = {row["sample_id"] for row in selected}
    original_read_rows = base._read_rows
    original_validate_rows = base._validate_rows
    original_subexperiment = base.SUBEXPERIMENT_ID
    original_alphas = base.COLOR_ALPHAS

    def read_selected(dataset_root: Path) -> list[dict[str, str]]:
        if dataset_root != args.dataset_root:
            raise ValueError(f"unexpected dataset root: {dataset_root}")
        return selected

    def validate_selected(
        candidate: list[dict[str, str]],
        *,
        training_manifest_id: str,
        model_name: str,
    ) -> str:
        if training_manifest_id != config.training_manifest_id:
            raise ValueError("training manifest changed after scan validation")
        if model_name != config.model_name:
            raise ValueError("model name changed after scan validation")
        if {row["sample_id"] for row in candidate} != selected_ids:
            raise ValueError("CPU shard rows changed after selection")
        return test_manifest_id

    args.output_root.mkdir(parents=True, exist_ok=True)
    assignment = {
        "status": "validated",
        "history": args.history,
        "device": args.device,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "pair_ids": pair_ids,
        "num_pairs": len(pair_ids),
        "num_predictions": len(selected),
        "sample_ids_sha256": hashlib.sha256(
            "\n".join(sorted(selected_ids)).encode("utf-8")
        ).hexdigest(),
        "test_manifest_id": test_manifest_id,
    }
    assignment_path = args.output_root / "shard_assignment.json"
    if assignment_path.is_file():
        if read_json(assignment_path) != assignment:
            raise ValueError(f"existing shard assignment differs: {assignment_path}")
    else:
        write_json(assignment_path, assignment)

    try:
        base._read_rows = read_selected
        base._validate_rows = validate_selected
        base.SUBEXPERIMENT_ID = spec.subexperiment_id
        base.COLOR_ALPHAS = spec.color_alphas
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
    finally:
        base._read_rows = original_read_rows
        base._validate_rows = original_validate_rows
        base.SUBEXPERIMENT_ID = original_subexperiment
        base.COLOR_ALPHAS = original_alphas


def _link_or_verify(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.stat().st_size != destination.stat().st_size:
            raise ValueError(f"existing merged file differs: {destination}")
        if _sha256(source) != _sha256(destination):
            raise ValueError(f"existing merged file differs: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        import shutil

        shutil.copy2(source, destination)


def merge(args: argparse.Namespace) -> Path:
    from sshv2.experiments.pendulum import frequency_color_low_predict as base

    config, spec, rows, test_manifest_id = _validated_inputs(
        args.dataset_root,
        args.experiment_config,
    )
    expected_sample_ids = {row["sample_id"] for row in rows}
    collected: dict[str, tuple[dict[str, Any], Path, Path]] = {}
    shard_summaries = []
    for shard_index in range(args.num_shards):
        shard_root = args.shards_root / f"shard_{shard_index:02d}"
        assignment = read_json(shard_root / "shard_assignment.json")
        progress = read_json(shard_root / "progress.json")
        if assignment.get("history") != args.history:
            raise ValueError(f"history mismatch in {shard_root}")
        if assignment.get("num_shards") != args.num_shards:
            raise ValueError(f"shard count mismatch in {shard_root}")
        if assignment.get("shard_index") != shard_index:
            raise ValueError(f"shard index mismatch in {shard_root}")
        expected = int(assignment["num_predictions"])
        if progress.get("status") != "complete" or int(
            progress.get("completed_predictions", -1)
        ) != expected:
            raise ValueError(f"incomplete CPU shard: {shard_root}")
        _, prediction_rows = read_predictions(shard_root, check_files=True)
        if len(prediction_rows) != expected:
            raise ValueError(f"manifest count mismatch in {shard_root}")
        for record in prediction_rows:
            sample_id = str(record["sample_id"])
            if sample_id in collected:
                raise ValueError(f"duplicate sharded sample: {sample_id}")
            prediction = shard_root / str(record["prediction"])
            record_path = shard_root / "records" / f"{sample_id}.json"
            if not record_path.is_file():
                raise FileNotFoundError(record_path)
            collected[sample_id] = (record, prediction, record_path)
        shard_summaries.append(
            {
                "shard_index": shard_index,
                "num_pairs": int(assignment["num_pairs"]),
                "num_predictions": expected,
                "output_root": str(shard_root),
            }
        )

    if set(collected) != expected_sample_ids:
        missing = sorted(expected_sample_ids - set(collected))
        extra = sorted(set(collected) - expected_sample_ids)
        raise ValueError(
            f"CPU shard coverage mismatch; missing={missing[:5]} extra={extra[:5]}"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    merge_progress = args.output_root / "cpu_merge_progress.json"
    ordered_records: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        sample_id = row["sample_id"]
        record, prediction, record_path = collected[sample_id]
        _link_or_verify(
            prediction,
            args.output_root / "predictions" / f"{sample_id}.mp4",
        )
        _link_or_verify(
            record_path,
            args.output_root / "records" / f"{sample_id}.json",
        )
        ordered_records.append(record)
        if index % 64 == 0 or index == len(rows):
            write_json(
                merge_progress,
                {
                    "status": "running",
                    "history": args.history,
                    "completed_links": index,
                    "expected_links": len(rows),
                },
            )

    checkpoint_sha256 = base._sha256(args.checkpoint)
    checkpoint_training_manifest_id = base._validate_training_binding(
        args.training_config,
        history=args.history,
        model_name=config.model_name,
        training_manifest_id=config.training_manifest_id,
    )
    data_config = base._load_data_config(args.experiment_config)
    run_state = {
        "subexperiment_id": spec.subexperiment_id,
        "history": args.history,
        "dataset_root": str(args.dataset_root),
        "dataset_sha256": base._sha256(args.dataset_root / "dataset.json"),
        "experiment_config": str(args.experiment_config),
        "training_config": str(args.training_config),
        "training_config_sha256": base._sha256(args.training_config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "training_manifest_id": config.training_manifest_id,
        "checkpoint_training_manifest_id": checkpoint_training_manifest_id,
        "pair_filter": None,
        "test_manifest_id": test_manifest_id,
        "device": "cpu_sharded",
        "cuda_visible_devices": "",
        "steps": args.steps,
        "seed_offset": args.seed_offset,
        "expected_predictions": len(rows),
        "future_frames": data_config.future_frames,
        "cpu_shards": args.num_shards,
        "cpu_shard_summaries": shard_summaries,
    }
    state_path = args.output_root / "run_state.json"
    if state_path.is_file() and read_json(state_path) != run_state:
        raise ValueError(f"existing merged run state differs: {state_path}")
    write_json(state_path, run_state)
    base.SUBEXPERIMENT_ID = spec.subexperiment_id
    base._finalize_predictions(
        args.output_root,
        ordered_records,
        checkpoint=args.checkpoint,
        training_config=args.training_config,
        history=args.history,
        test_manifest_id=test_manifest_id,
        training_manifest_id=config.training_manifest_id,
        checkpoint_training_manifest_id=checkpoint_training_manifest_id,
        steps=args.steps,
        seed_offset=args.seed_offset,
        checkpoint_sha256=checkpoint_sha256,
    )
    write_json(
        args.output_root / "progress.json",
        {
            "status": "complete",
            "history": args.history,
            "completed_predictions": len(rows),
            "expected_predictions": len(rows),
            "checkpoint_sha256": checkpoint_sha256,
            "device": "cpu_sharded",
            "cpu_shards": args.num_shards,
        },
    )
    write_json(
        merge_progress,
        {
            "status": "complete",
            "history": args.history,
            "completed_links": len(rows),
            "expected_links": len(rows),
            "num_shards": args.num_shards,
        },
    )
    return args.output_root


def promote(args: argparse.Namespace) -> Path:
    """Atomically promote a complete isolated GPU run to canonical output."""

    from sshv2.experiments.pendulum import frequency_color_low_predict as base

    config, spec, rows, test_manifest_id = _validated_inputs(
        args.dataset_root,
        args.experiment_config,
    )
    progress = read_json(args.source_root / "progress.json")
    if progress.get("status") != "complete" or int(
        progress.get("completed_predictions", -1)
    ) != len(rows):
        raise ValueError(f"GPU prediction is incomplete: {args.source_root}")
    _, prediction_rows = read_predictions(args.source_root, check_files=True)
    expected_ids = [row["sample_id"] for row in rows]
    actual_ids = [str(row["sample_id"]) for row in prediction_rows]
    if actual_ids != expected_ids:
        raise ValueError("GPU prediction rows differ from dataset order")
    run_state = read_json(args.source_root / "run_state.json")
    checkpoint_sha256 = base._sha256(args.checkpoint)
    expected_state = {
        "subexperiment_id": spec.subexperiment_id,
        "history": args.history,
        "test_manifest_id": test_manifest_id,
        "training_manifest_id": config.training_manifest_id,
        "checkpoint_sha256": checkpoint_sha256,
        "device": "cuda",
        "expected_predictions": len(rows),
        "steps": args.steps,
        "seed_offset": args.seed_offset,
    }
    mismatches = {
        key: (run_state.get(key), value)
        for key, value in expected_state.items()
        if run_state.get(key) != value
    }
    if mismatches:
        raise ValueError(f"GPU promotion identity mismatch: {mismatches}")

    if args.output_root.exists():
        existing_progress = args.output_root / "progress.json"
        if existing_progress.is_file():
            state = read_json(existing_progress)
            if state.get("status") == "complete" and int(
                state.get("completed_predictions", -1)
            ) == len(rows):
                existing_run = read_json(args.output_root / "run_state.json")
                if existing_run.get("checkpoint_sha256") == checkpoint_sha256:
                    read_predictions(args.output_root, check_files=True)
                    return args.output_root
        raise ValueError(
            f"refusing to replace incomplete canonical output: {args.output_root}"
        )

    staging = args.output_root.parent / (
        f".{args.output_root.name}.gpu-promotion-{os.getpid()}"
    )
    if staging.exists():
        raise FileExistsError(staging)

    def hardlink(source: str, destination: str) -> str:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        return destination

    shutil.copytree(args.source_root, staging, copy_function=hardlink)
    write_json(
        staging / "promotion.json",
        {
            "status": "complete",
            "source_root": str(args.source_root),
            "output_root": str(args.output_root),
            "history": args.history,
            "device": "cuda",
            "checkpoint_sha256": checkpoint_sha256,
            "test_manifest_id": test_manifest_id,
            "num_predictions": len(rows),
        },
    )
    read_predictions(staging, check_files=True)
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    os.rename(staging, args.output_root)
    return args.output_root


def audit(args: argparse.Namespace) -> None:
    _, spec, rows, test_manifest_id = _validated_inputs(
        args.dataset_root,
        args.experiment_config,
    )
    counts = Counter()
    seen: set[str] = set()
    for shard_index in range(args.num_shards):
        selected, pair_ids = _select_shard(
            rows,
            color_alphas=spec.color_alphas,
            num_shards=args.num_shards,
            shard_index=shard_index,
        )
        sample_ids = {row["sample_id"] for row in selected}
        if seen & sample_ids:
            raise ValueError(f"overlap in shard {shard_index}")
        seen |= sample_ids
        counts[len(selected)] += 1
        print(
            f"shard={shard_index:02d} pairs={len(pair_ids)} "
            f"predictions={len(selected)}"
        )
    if len(seen) != len(rows):
        raise ValueError(f"only {len(seen)}/{len(rows)} samples assigned")
    print(
        f"complete samples={len(seen)} test_manifest_id={test_manifest_id} "
        f"shard_size_counts={dict(counts)}"
    )


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    audit_parser = commands.add_parser("audit")
    _common(audit_parser)

    predict_parser = commands.add_parser("predict-shard")
    _common(predict_parser)
    predict_parser.add_argument("--shard-index", type=int, required=True)
    predict_parser.add_argument("--training-config", type=Path, required=True)
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--history", choices=("short", "long"), required=True)
    predict_parser.add_argument("--output-root", type=Path, required=True)
    predict_parser.add_argument("--device", default="cpu")
    predict_parser.add_argument("--steps", type=int, default=20)
    predict_parser.add_argument("--seed-offset", type=int, default=23_000_000)

    merge_parser = commands.add_parser("merge")
    _common(merge_parser)
    merge_parser.add_argument("--shards-root", type=Path, required=True)
    merge_parser.add_argument("--training-config", type=Path, required=True)
    merge_parser.add_argument("--checkpoint", type=Path, required=True)
    merge_parser.add_argument("--history", choices=("short", "long"), required=True)
    merge_parser.add_argument("--output-root", type=Path, required=True)
    merge_parser.add_argument("--steps", type=int, default=20)
    merge_parser.add_argument("--seed-offset", type=int, default=23_000_000)

    promote_parser = commands.add_parser("promote")
    promote_parser.add_argument("--dataset-root", type=Path, required=True)
    promote_parser.add_argument("--experiment-config", type=Path, required=True)
    promote_parser.add_argument("--source-root", type=Path, required=True)
    promote_parser.add_argument("--output-root", type=Path, required=True)
    promote_parser.add_argument("--checkpoint", type=Path, required=True)
    promote_parser.add_argument("--history", choices=("short", "long"), required=True)
    promote_parser.add_argument("--steps", type=int, default=20)
    promote_parser.add_argument("--seed-offset", type=int, default=23_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "audit":
        audit(args)
    elif args.command == "predict-shard":
        print(predict_shard(args))
    elif args.command == "merge":
        print(merge(args))
    elif args.command == "promote":
        print(promote(args))
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
