#!/usr/bin/env python3
"""Encode one Pendulum training split into long and masked-short latents."""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm

from sshv2.common.dataset import write_json
from sshv2.experiments.pendulum.data import (
    PendulumDatasetConfig,
    apply_short_history_mask_tensor,
    config_from_mapping,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT_CONFIG = (
    REPO_ROOT / "src/sshv2/experiments/pendulum/config.yaml"
)
DEFAULT_VAE = REPO_ROOT / "models/Wan2.1_VAE.pth"


def load_experiment_config(path: Path) -> PendulumDatasetConfig:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(
        document.get("data"),
        dict,
    ):
        raise ValueError("experiment config must contain a data mapping")
    config = config_from_mapping(document["data"])
    if not config.generate_train:
        raise ValueError("experiment config has generate_train=false")
    if not (
        config.latent_frames == 33
        and config.condition_latents == 17
        and config.target_latents == 16
    ):
        raise AssertionError("Pendulum latent geometry changed")
    return config


def resolve_training_bank(source: Path) -> Path:
    if (source / "metadata.csv").is_file():
        return source
    nested = source / "videos" / "train"
    if (nested / "metadata.csv").is_file():
        return nested
    raise FileNotFoundError(
        f"training metadata.csv not found under {source}"
    )


def default_output_root(config: PendulumDatasetConfig) -> Path:
    return (
        REPO_ROOT
        / "data"
        / "pendulum"
        / config.training_manifest_id
        / "latents"
    )


def read_training_rows(
    source: Path,
    config: PendulumDatasetConfig,
    *,
    limit: int = 0,
) -> list[dict[str, str]]:
    with (source / "metadata.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    if limit:
        if limit < 2 or limit % 2:
            raise ValueError("--limit must be an even integer >= 2")
        rows = rows[:limit]
    if not rows:
        raise ValueError("no training rows selected")
    for row in rows:
        if row["variant"] != "aligned":
            raise ValueError("training latent input must be aligned")
        if row["model_name"] != config.model_name:
            raise ValueError("training row model_name does not match config")
        if (
            row["training_manifest_id"]
            != config.training_manifest_id
        ):
            raise ValueError(
                "training row manifest ID does not match config"
            )
    target_counts = {
        target: sum(int(row["target_index"]) == target for row in rows)
        for target in (0, 1)
    }
    if target_counts[0] != target_counts[1]:
        raise ValueError("selected training rows are not target-balanced")
    return rows


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        payload = value.view(torch.uint16).numpy().tobytes()
    else:
        payload = value.numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty metadata: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_output(
    root: Path,
    *,
    overwrite: bool,
) -> tuple[Path, Path]:
    if root.is_symlink():
        raise ValueError("latent output root must not be a symbolic link")
    if root.exists() and any(root.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"latent output root is not empty: {root}"
            )
        allowed = {
            "train_short",
            "train_long",
            "encoding_audit.json",
        }
        unexpected = sorted(
            path.name
            for path in root.iterdir()
            if path.name not in allowed
        )
        if unexpected:
            raise ValueError(
                "refusing to overwrite unknown latent files: "
                + ", ".join(unexpected)
            )
        resolved = root.resolve()
        if resolved == Path.cwd().resolve() or len(resolved.parts) < 3:
            raise ValueError(f"refusing to replace broad path: {resolved}")
        shutil.rmtree(root)
    short_dir = root / "train_short"
    long_dir = root / "train_long"
    short_dir.mkdir(parents=True, exist_ok=True)
    long_dir.mkdir(parents=True, exist_ok=True)
    return short_dir, long_dir


def load_vae(path: Path, device: str):
    from diffsynth.models.model_manager import ModelManager
    from sshv2.wan.config import WanVAEConfig

    manager = ModelManager(
        device=device,
        torch_dtype=torch.bfloat16,
    )
    vae_config = WanVAEConfig(
        z_dim=16,
        queued=False,
        ckpt_file=path,
    )
    vae_config.get_model_config(
        torch_dtype=torch.bfloat16
    ).load_model(manager)
    vae = manager.fetch_model("wan_video_vae").eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    return vae


def make_video_loader(config: PendulumDatasetConfig):
    from diffsynth.trainers.unified_dataset import ImageCropAndResize
    from sshv2.wan.trainer import LoadVideoAsTensor

    return LoadVideoAsTensor(
        config.render.num_frames,
        4,
        1,
        frame_processor=ImageCropAndResize(
            config.render.height,
            config.render.width,
            config.render.height * config.render.width,
            16,
            16,
        ),
    )


def encode_histories(
    source: Path,
    output_root: Path,
    config: PendulumDatasetConfig,
    *,
    vae_path: Path,
    batch_size: int,
    device: str,
    limit: int = 0,
    overwrite: bool = False,
) -> Path:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    source = resolve_training_bank(source)
    rows = read_training_rows(source, config, limit=limit)
    short_dir, long_dir = prepare_output(
        output_root,
        overwrite=overwrite,
    )
    vae = load_vae(vae_path, device)
    loader = make_video_loader(config)
    short_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    audit_samples: list[dict[str, Any]] = []

    for start in tqdm(
        range(0, len(rows), batch_size),
        desc="pendulum VAE encode",
    ):
        batch_rows = rows[start : start + batch_size]
        long_pixels = torch.cat(
            [
                loader(str(source / row["video"]))
                for row in batch_rows
            ],
            dim=0,
        ).to(device=device, dtype=torch.bfloat16)
        short_pixels = apply_short_history_mask_tensor(
            long_pixels,
            config,
        )
        if not torch.equal(
            short_pixels[:, :, config.short_prefix_start :],
            long_pixels[:, :, config.short_prefix_start :],
        ):
            raise AssertionError(
                "short masking changed real frames 57..128"
            )

        with torch.inference_mode():
            long_batch = (
                vae.encode(
                    long_pixels,
                    device=device,
                    tiled=False,
                )
                .to(torch.bfloat16)
                .cpu()
            )
            short_batch = (
                vae.encode(
                    short_pixels,
                    device=device,
                    tiled=False,
                )
                .to(torch.bfloat16)
                .cpu()
            )
        expected_shape = (16, 33, 16, 16)
        if (
            long_batch.ndim != 5
            or tuple(long_batch.shape[1:]) != expected_shape
        ):
            raise AssertionError(
                f"unexpected long latent shape {tuple(long_batch.shape)}"
            )
        if (
            short_batch.ndim != 5
            or tuple(short_batch.shape[1:]) != expected_shape
        ):
            raise AssertionError(
                f"unexpected short latent shape {tuple(short_batch.shape)}"
            )

        for row, short_view, long_view in zip(
            batch_rows,
            short_batch,
            long_batch,
            strict=True,
        ):
            short_view = short_view.unsqueeze(0).contiguous()
            long_view = long_view.unsqueeze(0).contiguous()
            short_target = short_view[
                :, :, config.condition_latents :
            ].contiguous()
            long_target = long_view[
                :, :, config.condition_latents :
            ].contiguous()
            expected_target = (1, 16, 16, 16, 16)
            if tuple(short_target.shape) != expected_target:
                raise AssertionError("unexpected short target shape")
            if tuple(long_target.shape) != expected_target:
                raise AssertionError("unexpected long target shape")

            filename = Path(row["video"]).with_suffix(".pt").name
            torch.save(short_view, short_dir / filename)
            torch.save(long_view, long_dir / filename)
            common = dict(row)
            common.update(
                {
                    "source_video": row["video"],
                    "video": filename,
                    "latent_frames": config.latent_frames,
                    "condition_latents": config.condition_latents,
                    "target_latents": config.target_latents,
                    "pixel_future_slice": (
                        f"{config.prediction_start}:"
                        f"{config.render.num_frames}"
                    ),
                }
            )
            short_row = {
                **common,
                "history": "short",
                "pixel_condition_policy": (
                    "mask_0_56_background_then_real_57_64"
                ),
                "latent_hash": tensor_hash(short_view),
                "target_hash": tensor_hash(short_target),
            }
            long_row = {
                **common,
                "history": "long",
                "pixel_condition_policy": "real_0_64",
                "latent_hash": tensor_hash(long_view),
                "target_hash": tensor_hash(long_target),
            }
            short_rows.append(short_row)
            long_rows.append(long_row)
            difference = (
                short_target.float() - long_target.float()
            ).abs()
            audit_samples.append(
                {
                    "sample_id": row["sample_id"],
                    "short_shape": list(short_view.shape),
                    "long_shape": list(long_view.shape),
                    "short_target_hash": short_row["target_hash"],
                    "long_target_hash": long_row["target_hash"],
                    "pixel_future_is_identical": True,
                    "target_latents_elementwise_equal": bool(
                        torch.equal(short_target, long_target)
                    ),
                    "target_mean_absolute_difference": float(
                        difference.mean()
                    ),
                    "target_max_absolute_difference": float(
                        difference.max()
                    ),
                }
            )

    write_rows(short_dir / "metadata.csv", short_rows)
    write_rows(long_dir / "metadata.csv", long_rows)
    write_json(
        output_root / "encoding_audit.json",
        {
            "model_name": config.model_name,
            "training_manifest_id": config.training_manifest_id,
            "source": str(source),
            "vae": str(vae_path),
            "num_samples": len(rows),
            "latent_shape": [1, 16, 33, 16, 16],
            "condition_latents_both": config.condition_latents,
            "target_latents_both": config.target_latents,
            "pixel_future_slice_both": [
                config.prediction_start,
                config.render.num_frames,
            ],
            "short_condition": (
                "mask frames 0..56; real frames 57..64"
            ),
            "long_condition": "real frames 0..64",
            "samples": audit_samples,
        },
    )
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=DEFAULT_EXPERIMENT_CONFIG,
    )
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--vae", type=Path, default=DEFAULT_VAE)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.experiment_config)
    output_root = args.out_root or default_output_root(config)
    result = encode_histories(
        args.source,
        output_root,
        config,
        vae_path=args.vae,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(result)


if __name__ == "__main__":
    main()
