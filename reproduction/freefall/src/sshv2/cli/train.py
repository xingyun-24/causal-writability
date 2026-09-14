#!/usr/bin/env python3
"""Train one experiment through the shared Wan training entry point."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

try:
    from ._common import (
        bootstrap_repository,
        load_yaml,
        training_module,
        write_resolved_config,
    )
except ImportError:  # Direct execution from this source directory.
    from _common import (
        bootstrap_repository,
        load_yaml,
        training_module,
        write_resolved_config,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resolved-dir", type=Path, required=True)
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument(
        "-r",
        "--resume",
        nargs="?",
        type=Path,
        const=True,
    )
    resume.add_argument(
        "--resume-top",
        nargs="?",
        type=Path,
        const=True,
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--val-every", type=int)
    parser.add_argument("--ckpt-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args(argv)

    bootstrap_repository()
    module = training_module(args.experiment)
    source_config = load_yaml(args.config)
    requested = {
        "resume": args.resume,
        "resume_top": args.resume_top,
        "steps": args.steps,
        "val_every": args.val_every,
        "ckpt_dir": args.ckpt_dir,
        "out_dir": args.out_dir,
        "seed": args.seed,
        "with_wandb": False if args.no_wandb else None,
    }
    parameters = {
        "config_file": args.config,
        **{
            key: value
            for key, value in requested.items()
            if value is not None
        },
    }
    write_resolved_config(
        args.resolved_dir,
        stage="train",
        experiment=args.experiment,
        config_file=args.config,
        profile=None,
        config=parameters,
        source_config=source_config,
        parameters=parameters,
    )
    signature = inspect.signature(module.train)
    unsupported = [
        key
        for key, value in requested.items()
        if value is not None and key not in signature.parameters
    ]
    if unsupported:
        parser.error(
            f"{args.experiment} does not support: "
            f"{', '.join(sorted(unsupported))}"
        )
    module.train(
        args.config,
        **{
            key: value
            for key, value in requested.items()
            if value is not None and key in signature.parameters
        },
    )


if __name__ == "__main__":
    main()
