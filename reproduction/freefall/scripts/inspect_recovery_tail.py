#!/usr/bin/env python3
"""Inspect whether late-layer recovery hits are genuine or baseline overlap."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def summarize(values: list[dict]) -> dict:
    hats = sorted(row["gravity_hat"] for row in values if row.get("gravity_hat") is not None)
    errors = [row["gravity_error"] for row in values if row.get("gravity_error") is not None]
    labels = ("valid", "true_band_correct", "physics_side", "shortcut_band", "middle_gap", "outside", "invalid")
    counts = {label: sum(bool(row.get(label)) if label in {"valid", "true_band_correct"} else row.get("route_label") == label for row in values) for label in labels}
    return {
        "pairs": len(values),
        **counts,
        "mean_gravity_error": sum(errors) / len(errors) if errors else None,
        "gravity_hat_min": hats[0] if hats else None,
        "gravity_hat_median": hats[len(hats) // 2] if hats else None,
        "gravity_hat_max": hats[-1] if hats else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict] = []
    for path in glob.glob(str(args.root / "shard*" / "outcomes.json")):
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8")))
    for label, rows_for_label in [("conflict_baseline", [r for r in rows if r["condition"] == "conflict_baseline"]), *[(f"block_{block}", [r for r in rows if r["condition"] == "replacement" and r["block"] == block]) for block in (0, 8, 9, 29)]]:
        print(label, json.dumps(summarize(rows_for_label), sort_keys=True))
        for band in ("low", "high"):
            print(f"  {band}", json.dumps(summarize([r for r in rows_for_label if r["true_band"] == band]), sort_keys=True))
    baseline = {row["pair_id"]: bool(row["true_band_correct"]) for row in rows if row["condition"] == "conflict_baseline"}
    failed = {pair_id for pair_id, correct in baseline.items() if not correct}
    correct = set(baseline) - failed
    print(f"baseline_failed={len(failed)} baseline_correct={len(correct)}")
    for block in range(30):
        changed = {row["pair_id"]: bool(row["true_band_correct"]) for row in rows if row["condition"] == "replacement" and row["block"] == block}
        recovered = sum(changed[pair_id] for pair_id in failed)
        retained = sum(changed[pair_id] for pair_id in correct)
        print(f"GAIN block={block} recovered={recovered}/{len(failed)} retained={retained}/{len(correct)} total={sum(changed.values())}/{len(changed)}")


if __name__ == "__main__":
    main()
