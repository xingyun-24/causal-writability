#!/usr/bin/env python3
"""Generate predictions through an experiment-owned predict function."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from ._common import (
        bootstrap_repository,
        experiment_module,
        resolve_profile,
        write_resolved_config,
    )
except ImportError:  # Direct execution from this source directory.
    from _common import (
        bootstrap_repository,
        experiment_module,
        resolve_profile,
        write_resolved_config,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile")
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed-offset", type=int)
    parser.add_argument("--save-videos-limit", type=int)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    args = parser.parse_args(argv)

    bootstrap_repository()
    config, source_config = resolve_profile(
        args.config,
        "prediction_profiles",
        args.profile,
        args.overrides,
    )
    runtime = {
        "training_config": args.training_config,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "steps": args.steps,
        "limit": args.limit,
        "seed_offset": args.seed_offset,
        "save_videos_limit": args.save_videos_limit,
    }
    config.update(
        {
            key: value
            for key, value in runtime.items()
            if value is not None
        }
    )
    write_resolved_config(
        args.prediction_dir,
        stage="predict",
        experiment=args.experiment,
        config_file=args.config,
        profile=args.profile,
        config=config,
        source_config=source_config,
        parameters={
            "dataset_dir": args.dataset_dir,
            "prediction_dir": args.prediction_dir,
            **runtime,
        },
    )
    module = experiment_module(args.experiment, "evaluation")
    if not hasattr(module, "predict"):
        raise AttributeError(
            f"{module.__name__} does not expose predict()"
        )
    module.predict(args.dataset_dir, args.prediction_dir, config)


if __name__ == "__main__":
    main()
