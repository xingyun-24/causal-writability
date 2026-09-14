#!/usr/bin/env python3
"""Split a balanced recovery manifest evenly across independent GPU workers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()
    if args.shards < 1:
        parser.error("--shards must be positive")
    source = json.loads(args.manifest.read_text(encoding="utf-8"))
    pairs = source["pairs"]
    by_interval = {interval: [item for item in pairs if item["gravity_interval"] == interval] for interval in ("low", "high")}
    shards = [[] for _ in range(args.shards)]
    for interval in ("low", "high"):
        for index, pair in enumerate(by_interval[interval]):
            shards[index % args.shards].append(pair)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for index, items in enumerate(shards):
        payload = {**source, "shard_index": index, "shard_count": args.shards, "pairs": items}
        (args.out_dir / f"recovery_pairs_shard{index}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"shard={index} pairs={len(items)}")


if __name__ == "__main__":
    main()
