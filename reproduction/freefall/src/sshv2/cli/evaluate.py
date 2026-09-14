#!/usr/bin/env python3
"""Evaluate predictions through an experiment-owned evaluator."""

from __future__ import annotations

import argparse
import inspect
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
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
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
        "evaluation_profiles",
        args.profile,
        args.overrides,
    )
    if args.limit is not None:
        config["limit"] = args.limit
    write_resolved_config(
        args.output_dir,
        stage="evaluate",
        experiment=args.experiment,
        config_file=args.config,
        profile=args.profile,
        config=config,
        source_config=source_config,
        parameters={
            "dataset_dir": args.dataset_dir,
            "prediction_dir": args.prediction_dir,
            "output_dir": args.output_dir,
            "limit": args.limit,
        },
    )
    module = experiment_module(args.experiment, "evaluation")
    result = module.evaluate(
        args.dataset_dir,
        args.prediction_dir,
        config,
    )
    write_parameters = {
        "context": {
            "config": str(args.config),
            "profile": args.profile,
            "prediction_dir": str(args.prediction_dir),
        },
    }
    signature = inspect.signature(module.write_evaluation)
    if "dataset" in signature.parameters and config.get("dataset_id"):
        write_parameters["dataset"] = config["dataset_id"]
    module.write_evaluation(result, args.output_dir, **write_parameters)


if __name__ == "__main__":
    main()
