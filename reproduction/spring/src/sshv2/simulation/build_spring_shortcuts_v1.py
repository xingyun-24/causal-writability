"""Build the complete raw-video splits for spring_shortcuts_v4."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

from sshv2.simulation.spring_shortcuts_v1 import (
    VERSION,
    config_to_dict,
    dataclass_config_from_dict,
    generate_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=("all", "sanity", "train", "eval"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite non-empty directory: {path}")
        shutil.rmtree(path)


def main() -> None:
    args = parse_args()
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg = dataclass_config_from_dict(payload)
    args.root.mkdir(parents=True, exist_ok=True)
    requested = ("sanity", "train", "eval") if args.split == "all" else (args.split,)
    counts = {
        "sanity": cfg.sanity_base_seeds,
        "train": cfg.train_base_seeds,
        "eval": cfg.eval_base_seeds,
    }
    summary = {"benchmark_version": VERSION, "splits": {}}
    for split in requested:
        out = args.root / "videos" / split
        prepare_dir(out, args.overwrite)
        rows = generate_split(
            out,
            cfg,
            split=split,
            num_base_seeds=counts[split],
            include_conflicts=(split != "train"),
            overwrite=False,
        )
        summary["splits"][split] = {
            "directory": str(out),
            "rows": len(rows),
            "physical_trajectories": counts[split] * 2,
        }

    metadata_dir = args.root / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    resolved = config_to_dict(cfg)
    config_text = yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True)
    (metadata_dir / "generation_config_resolved.yaml").write_text(config_text, encoding="utf-8")
    (args.root / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
