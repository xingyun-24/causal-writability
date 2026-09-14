#!/usr/bin/env python3
"""Build isolated short/long training configs for one Pendulum model."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from sshv2.experiments.pendulum.data import config_from_mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT_CONFIG = (
    REPO_ROOT / "src/sshv2/experiments/pendulum/config.yaml"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "configs/pendulum"


def _save_steps(total: int) -> list[int]:
    if total <= 0:
        raise ValueError("training steps must be positive")
    values = list(range(1000, total + 1, 1000))
    if not values or values[-1] != total:
        values.append(total)
    return values


def build_training_config(
    experiment_config: Path,
    *,
    history: str,
    steps: int = 10_000,
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[str, dict[str, Any]]:
    if history not in ("short", "long"):
        raise ValueError("history must be short or long")
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("invalid loader settings")
    document = yaml.safe_load(
        experiment_config.read_text(encoding="utf-8")
    )
    if not isinstance(document, dict) or not isinstance(
        document.get("data"),
        dict,
    ):
        raise ValueError("experiment config must contain a data mapping")
    data = config_from_mapping(document["data"])
    training_id = data.training_manifest_id
    try:
        config_reference = experiment_config.resolve().relative_to(
            REPO_ROOT.resolve()
        )
    except ValueError:
        config_reference = experiment_config
    latent_dir = (
        Path("data")
        / "pendulum"
        / training_id
        / "latents"
        / f"train_{history}"
    )
    run_dir = Path("runs") / "pendulum" / training_id / history
    filename = f"Train-{data.model_name}-{history}.yaml"
    payload: dict[str, Any] = {
        "seed": data.seed,
        "model": {
            "dit": {
                "dim": 768,
                "in_dim": 16,
                "ffn_dim": 3072,
                "out_dim": 16,
                "text_dim": 4096,
                "freq_dim": 256,
                "eps": 1.0e-6,
                "patch_size": [1, 2, 2],
                "num_heads": 6,
                "num_layers": 30,
                "has_text_input": False,
                "has_image_input": False,
                "num_inference_steps": 20,
                "type": "bidirectional",
            },
            "vae": {
                "z_dim": 16,
                "queued": False,
                "ckpt_file": "models/Wan2.1_VAE.pth",
            },
            "type": "default",
            "num_condition_frames": 17,
        },
        "data": {
            "dataset": latent_dir.as_posix(),
            "labels": (latent_dir / "metadata.csv").as_posix(),
            "size": [128, 128],
            "num_frames": 33,
            "load_as": "tensor",
            "encoded": True,
        },
        "loader": {
            "num_training_steps": steps,
            "batch_size": batch_size,
            "num_workers": num_workers,
        },
        "optimizer": {
            "learning_rate": 2.0e-4,
            "weight_decay": 1.0e-2,
            "gradient_accumulation_steps": 1,
            "with_ema": False,
        },
        "log": {
            "project": "pendulum",
            "name": f"{data.model_name}-{history}",
            "log_every": 10,
            "save_at": _save_steps(steps),
            "save_last_every": min(1000, steps),
            "ckpt_dir": (run_dir / "ckpt").as_posix(),
            "out_dir": (run_dir / "output").as_posix(),
            "fps": data.render.fps,
        },
        "pendulum": {
            "model_name": data.model_name,
            "training_manifest_id": training_id,
            "history": history,
            "experiment_config": config_reference.as_posix(),
        },
    }
    return filename, payload


def write_training_configs(
    experiment_config: Path,
    output_dir: Path,
    *,
    steps: int = 10_000,
    batch_size: int = 32,
    num_workers: int = 4,
    overwrite: bool = False,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    planned: list[tuple[Path, dict[str, Any]]] = []
    for history in ("short", "long"):
        filename, payload = build_training_config(
            experiment_config,
            history=history,
            steps=steps,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        planned.append((output_dir / filename, payload))
    existing = [
        destination
        for destination, _ in planned
        if destination.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite training configs: "
            + ", ".join(str(path) for path in existing)
        )

    written: list[Path] = []
    for destination, payload in planned:
        destination.write_text(
            "# Generated from the Pendulum experiment config.\n"
            + yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
        written.append(destination)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=DEFAULT_EXPERIMENT_CONFIG,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = write_training_configs(
        args.experiment_config,
        args.output_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
