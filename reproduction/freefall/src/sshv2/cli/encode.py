#!/usr/bin/env python3
"""Encode one video dataset with the canonical Wan VAE."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from ._common import (
        bootstrap_repository,
        load_yaml,
        write_resolved_config,
    )
except ImportError:  # Direct execution from this source directory.
    from _common import (
        bootstrap_repository,
        load_yaml,
        write_resolved_config,
    )


def _vae_from_config(config: dict) -> Path | None:
    value = config.get("model", {}).get("vae", {}).get("ckpt_file")
    return Path(value) if value else None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--latent-dir", type=Path, required=True)
    parser.add_argument("--vae", type=Path)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args(argv)

    bootstrap_repository()
    from sshv2.wan.vae import (
        EncodeDatasetConfig,
        encode_dataset,
    )

    source_config = load_yaml(args.config)
    vae = args.vae or _vae_from_config(source_config)
    if vae is None:
        parser.error(
            "--vae is required when model.vae.ckpt_file is absent"
        )
    stage_config = EncodeDatasetConfig(
        vae=vae,
        batch_size=args.batch,
        device=args.device,
        cpu_threads=args.cpu_threads,
        start=args.start,
        stop=args.stop,
        finalize_only=args.finalize_only,
    )
    parameters = {
        "dataset_dir": args.dataset_dir,
        "latent_dir": args.latent_dir,
        **stage_config.as_dict(),
    }
    write_resolved_config(
        args.latent_dir,
        stage="encode",
        experiment=args.experiment,
        config_file=args.config,
        profile=None,
        config=parameters,
        source_config=source_config,
        parameters=parameters,
    )
    encode_dataset(args.dataset_dir, args.latent_dir, stage_config)


if __name__ == "__main__":
    main()
