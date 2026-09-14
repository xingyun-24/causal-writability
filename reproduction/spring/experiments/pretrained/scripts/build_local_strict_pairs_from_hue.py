#!/usr/bin/env python3
"""Freeze checkpoint-local strict endpoint failures from a completed hue sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


FIELDS = (
    "pair_index", "split", "target_direction", "trajectory_id", "pair_id",
    "base_seed", "generation_seed", "true_band", "omega_true", "amplitude",
    "phase", "x_star", "v_star", "aligned_sample_id", "aligned_video",
    "aligned_metadata", "conflict_sample_id", "conflict_video", "conflict_metadata",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--seed-offset", type=int, default=17_000_000)
    parser.add_argument("--min-gap", type=float, default=2.0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    with (args.dataset / "metadata.csv").open(newline="", encoding="utf-8") as handle:
        metadata = list(csv.DictReader(handle))
    metrics = {row["sample_id"]: row for row in read_jsonl(args.metrics)}
    if len(metadata) != 704 or len(metrics) != 704:
        raise AssertionError(f"Expected 704 matched hue rows, got metadata={len(metadata)} metrics={len(metrics)}")

    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for meta in metadata:
        if meta["color_label"] in {"mix00", "mix10"}:
            grouped[meta["trajectory_id"]][meta["color_label"]] = meta

    selected = []
    rejection = Counter()
    for trajectory_id in sorted(grouped):
        endpoints = grouped[trajectory_id]
        if set(endpoints) != {"mix00", "mix10"}:
            rejection["missing_endpoint"] += 1
            continue
        exemplar = endpoints["mix00"]
        true_band = exemplar["true_band"]
        aligned_label = "mix00" if true_band == "slow" else "mix10"
        conflict_label = "mix10" if true_band == "slow" else "mix00"
        aligned_meta = endpoints[aligned_label]
        conflict_meta = endpoints[conflict_label]
        aligned = metrics[aligned_meta["sample_id"]]
        conflict = metrics[conflict_meta["sample_id"]]
        if not aligned.get("valid") or not conflict.get("valid"):
            rejection["invalid"] += 1
            continue
        if aligned.get("route_label") != "physics_frequency":
            rejection["aligned_not_physics"] += 1
            continue
        if conflict.get("route_label") != "opposite_band_frequency":
            rejection["conflict_not_shortcut"] += 1
            continue
        aligned_omega = float(aligned.get("omega_hat_free", float("nan")))
        conflict_omega = float(conflict.get("omega_hat_free", float("nan")))
        if not math.isfinite(aligned_omega) or not math.isfinite(conflict_omega):
            rejection["nonfinite_frequency"] += 1
            continue
        if abs(aligned_omega - conflict_omega) < args.min_gap:
            rejection["gap_below_threshold"] += 1
            continue
        selected.append({
            "pair_index": len(selected),
            "split": "checkpoint_local_strict",
            "target_direction": "fast-target" if true_band == "fast" else "slow-target",
            "trajectory_id": trajectory_id,
            "pair_id": exemplar["pair_id"],
            "base_seed": int(exemplar["base_seed"]),
            "generation_seed": int(exemplar["base_seed"]) + args.seed_offset,
            "true_band": true_band,
            "omega_true": exemplar["omega_true"],
            "amplitude": exemplar["amplitude"],
            "phase": exemplar["phase"],
            "x_star": exemplar["x_star"],
            "v_star": exemplar["v_star"],
            "aligned_sample_id": aligned_meta["sample_id"],
            "aligned_video": aligned_meta["video"],
            "aligned_metadata": aligned_meta["metadata"],
            "conflict_sample_id": conflict_meta["sample_id"],
            "conflict_video": conflict_meta["video"],
            "conflict_metadata": conflict_meta["metadata"],
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(selected)
    summary = {
        "status": "PASS" if selected else "EMPTY",
        "selection": "checkpoint-local strict matched failures",
        "n_histories": len(grouped),
        "n_selected": len(selected),
        "by_target_direction": dict(Counter(row["target_direction"] for row in selected)),
        "rejection_counts": dict(rejection),
        "min_frequency_gap": args.min_gap,
        "generation_seed_offset": args.seed_offset,
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not selected:
        raise SystemExit("No strict matched failures were selected")


if __name__ == "__main__":
    main()
