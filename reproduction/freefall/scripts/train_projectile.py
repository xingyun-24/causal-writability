#!/usr/bin/env python3
"""Train Projectile Gravity with the standard Wan trainer, without a Spring alias."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resolved-dir", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    from sshv2.wan.trainer import train_standard

    source = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.resolved_dir.mkdir(parents=True, exist_ok=True)
    (args.resolved_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump({
            "experiment": source.get("log", {}).get("project", "projectile-gravity"),
            "config_file": str(args.config),
            "parameters": {
                "checkpoint_dir": str(args.ckpt_dir),
                "output_dir": str(args.out_dir),
                "resume": str(args.resume) if args.resume else None,
                "steps": args.steps,
                "seed": args.seed,
                "with_wandb": not args.no_wandb,
            },
            "source_config": source,
        }, sort_keys=False),
        encoding="utf-8",
    )
    train_standard(
        args.config,
        resume=args.resume,
        steps=args.steps,
        ckpt_dir=args.ckpt_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        with_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
