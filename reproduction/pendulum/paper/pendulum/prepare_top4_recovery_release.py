#!/usr/bin/env python3
"""Prepare the same-rank Top-4 recovery release from the frozen remote audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
CONDITIONS = (
    "natural_conflict",
    "full_matched",
    "top4_oracle",
    "fit_only_predicted",
)


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(group: list[dict[str, str]]) -> dict[str, object]:
    valid = [row for row in group if parse_bool(row["valid"])]
    successful = [
        row
        for row in valid
        if row["normalized_recovery"]
        and 0.75 < float(row["normalized_recovery"]) < 1.25
    ]
    recoveries = [
        float(row["normalized_recovery"])
        for row in valid
        if row["normalized_recovery"]
    ]
    reasons = Counter(
        reason
        for row in group
        for reason in row["invalid_reasons"].split(";")
        if reason
    )
    outcomes = Counter(row["outcome"] for row in group)
    return {
        "n": len(group),
        "valid_n": len(valid),
        "valid_rate": len(valid) / len(group),
        "near_full_success_n": len(successful),
        "near_full_success_rate_all": len(successful) / len(group),
        "conditional_median_recovery": statistics.median(recoveries),
        "invalid_reason_counts": dict(sorted(reasons.items())),
        "outcome_counts": dict(sorted(outcomes.items())),
        "physics_follow_rate_all": outcomes["physics"] / len(group),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-audit", type=Path, required=True)
    args = parser.parse_args()

    source_rows = read_rows(args.source_audit)
    top4_rows = [
        dict(row, condition="top4_oracle")
        for row in source_rows
        if row["condition"] == "low_rank_oracle"
        and "/edit_rank_4.mp4" in row["output_video_path"]
    ]
    if len(top4_rows) != 64 or len({row["receiver_id"] for row in top4_rows}) != 64:
        raise ValueError("source must contain 64 unique Top-4 oracle receiver rows")
    if any("edit_rank_1.mp4" in row["output_video_path"] for row in top4_rows):
        raise ValueError("Rank-1 output detected in Top-4 release")

    current_path = HERE / "decoded_recovery.csv"
    current_rows = read_rows(current_path)
    fields = list(current_rows[0])
    receiver_ids = {row["receiver_id"] for row in current_rows}
    if {row["receiver_id"] for row in top4_rows} != receiver_ids:
        raise ValueError("Top-4 receiver cohort differs from the released held-out cohort")

    top4_rows.sort(key=lambda row: row["receiver_id"])
    write_rows(HERE / "top4_oracle_audit.csv", top4_rows, fields)

    retained = [row for row in current_rows if row["condition"] != "low_rank_oracle"]
    merged = retained + top4_rows
    order = {condition: index for index, condition in enumerate(CONDITIONS)}
    merged.sort(key=lambda row: (row["receiver_id"], order[row["condition"]]))
    if len(merged) != 256:
        raise ValueError("release must contain 64 receivers by four conditions")
    write_rows(current_path, merged, fields)

    prior = json.loads((HERE / "decoded_recovery_summary.json").read_text())
    condition_summary = {
        condition: summarize([row for row in merged if row["condition"] == condition])
        for condition in CONDITIONS
    }
    summary = {
        "evaluator": "geometry_annulus_v1",
        "thresholds_frozen": prior["thresholds_frozen"],
        "conditions": condition_summary,
        "selected_example": prior["selected_example"],
        "notes": [
            "The visible oracle and fit-only controller use the same frozen top-four coordinate space.",
            "Top-4 oracle rows come from edit_rank_4.mp4 outputs and are not relabeled Rank-1 rows.",
            "All four conditions use the same frozen evaluator gates and 64-receiver cohort.",
            "Rank-1 remains available only in pendulum_invalid_audit.csv as a secondary fragility control.",
            "Fit-only scale 1.0 was selected globally on the fit split before held-out generation.",
        ],
    }
    (HERE / "decoded_recovery_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    top4_release = {
        "source_audit_sha256": sha256(args.source_audit),
        "source_selection": "condition=low_rank_oracle AND output_video_path contains edit_rank_4.mp4",
        "released_condition": "top4_oracle",
        "receiver_level_rows": 64,
        "condition_summary": condition_summary["top4_oracle"],
    }
    (HERE / "top4_oracle_summary.json").write_text(
        json.dumps(top4_release, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
