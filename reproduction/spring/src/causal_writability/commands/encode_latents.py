"""Encode natural long and masked-short 129-frame V4 latent videos.

V4 keeps the complete 129-frame timeline for both models.

Long input:
* real pixel frames 0..128;
* condition latent frames 0:17;
* target latent frames 17:33.

Short input:
* pixel frames 0..56 are replaced, before VAE encoding, by one fixed
  background-only RGB mask that is identical for every sample;
* real pixel frames 57..128 are unchanged;
* condition latent frames 0:17 therefore contain a fixed missing-history
  prefix plus the real eight-frame observation 57..64;
* target latent frames 17:33 correspond to the same pixel future 65..128.

The short and long targets are natural outputs of two complete causal VAE
encodings.  They are not expected to be elementwise equal; pixel-space future
identity, equal latent length, equal condition count, and equal temporal
positions are the controlled invariants.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm

from diffsynth.models.model_manager import ModelManager
from diffsynth.trainers.unified_dataset import ImageCropAndResize
from sshv2.diffsynth.configs.wan_config import WanVAEConfig
from sshv2.diffsynth.trainers.dataset import LoadVideoAsTensor
from sshv2.simulation.spring_shortcuts_v1 import (
    apply_short_history_mask_tensor,
    dataclass_config_from_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Raw training split containing metadata.csv")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Debug-only cap; 0 encodes all rows")
    return parser.parse_args()


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        payload = value.view(torch.uint16).numpy().tobytes()
    else:
        payload = value.numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty metadata: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def balanced_prefix(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count > len(rows):
        raise ValueError(f"Requested subset of {count} from only {len(rows)} rows")
    selected = rows[:count]
    bands = [r["true_band"] for r in selected]
    if bands.count("slow") != bands.count("fast"):
        raise AssertionError(f"First {count} encoded rows are not slow/fast balanced")
    return selected


def prepare_output(out_root: Path, overwrite: bool) -> tuple[Path, Path]:
    short_dir = out_root / "train_short"
    long_dir = out_root / "train_long"
    if out_root.exists() and any(out_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite non-empty output root: {out_root}")
        shutil.rmtree(out_root)
    for path in (short_dir, long_dir):
        path.mkdir(parents=True, exist_ok=True)
    return short_dir, long_dir


def load_vae(path: Path, device: str):
    manager = ModelManager(device=device, torch_dtype=torch.bfloat16)
    config = WanVAEConfig(z_dim=16, queued=False, ckpt_file=path)
    config.get_model_config(torch_dtype=torch.bfloat16).load_model(manager)
    vae = manager.fetch_model("wan_video_vae").eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    return vae


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    cfg = dataclass_config_from_dict(yaml.safe_load(args.data_config.read_text(encoding="utf-8")))
    if not (
        cfg.long_latent_frames == cfg.short_latent_frames == 33
        and cfg.long_condition_latents == cfg.short_condition_latents == 17
        and cfg.target_latents == 16
    ):
        raise AssertionError("Frozen V4 latent geometry changed unexpectedly")

    source_rows = read_rows(args.source / "metadata.csv")
    if args.limit:
        if args.limit < 2 or args.limit % 2:
            raise ValueError("--limit must be an even integer >=2 so slow/fast subsets stay balanced")
        source_rows = source_rows[: args.limit]
    if not source_rows:
        raise ValueError("No source rows found")
    for row in source_rows:
        if row["variant"] != "aligned":
            raise ValueError("Training encoding may only consume aligned colour-frequency rows")

    short_dir, long_dir = prepare_output(args.out_root, args.overwrite)
    vae = load_vae(args.vae, args.device)
    loader = LoadVideoAsTensor(
        cfg.render.num_frames,
        4,
        1,
        frame_processor=ImageCropAndResize(
            cfg.render.height,
            cfg.render.width,
            cfg.render.height * cfg.render.width,
            16,
            16,
        ),
    )

    encoded_short: list[dict[str, Any]] = []
    encoded_long: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(source_rows), args.batch_size), desc="spring VAE encode"):
        batch_rows = source_rows[start : start + args.batch_size]
        long_pixels = torch.cat([loader(str(args.source / row["video"])) for row in batch_rows], dim=0)
        long_pixels = long_pixels.to(device=args.device, dtype=torch.bfloat16)
        short_pixels = apply_short_history_mask_tensor(long_pixels, cfg)

        # The real observed and target pixels must remain exactly unchanged.
        if not torch.equal(
            short_pixels[:, :, cfg.short_prefix_start :],
            long_pixels[:, :, cfg.short_prefix_start :],
        ):
            raise AssertionError("Short masking modified real frames 57..128")
        with torch.inference_mode():
            long_batch = vae.encode(long_pixels, device=args.device, tiled=False).to(torch.bfloat16).cpu()
            short_batch = vae.encode(short_pixels, device=args.device, tiled=False).to(torch.bfloat16).cpu()

        expected = (16, 33, 16, 16)
        if long_batch.ndim != 5 or tuple(long_batch.shape[1:]) != expected:
            raise AssertionError(f"Unexpected long VAE output shape {tuple(long_batch.shape)}")
        if short_batch.ndim != 5 or tuple(short_batch.shape[1:]) != expected:
            raise AssertionError(f"Unexpected short VAE output shape {tuple(short_batch.shape)}")

        for row, short_view, long_view in zip(batch_rows, short_batch, long_batch, strict=True):
            short_view = short_view.unsqueeze(0).contiguous()
            long_view = long_view.unsqueeze(0).contiguous()
            short_target = short_view[:, :, cfg.short_condition_latents :].contiguous()
            long_target = long_view[:, :, cfg.long_condition_latents :].contiguous()
            if tuple(short_target.shape) != (1, 16, 16, 16, 16):
                raise AssertionError(f"Unexpected short target shape {tuple(short_target.shape)}")
            if tuple(long_target.shape) != (1, 16, 16, 16, 16):
                raise AssertionError(f"Unexpected long target shape {tuple(long_target.shape)}")
            target_diff = (short_target.float() - long_target.float()).abs()

            file_name = Path(row["video"]).with_suffix(".pt").name
            torch.save(short_view, short_dir / file_name)
            torch.save(long_view, long_dir / file_name)

            common = dict(row)
            common["source_video"] = row["video"]
            common["video"] = file_name
            common["latent_frames"] = cfg.long_latent_frames
            common["condition_latents"] = cfg.long_condition_latents
            common["target_latents"] = cfg.target_latents
            common["pixel_future_slice"] = f"{cfg.prediction_start}:{cfg.render.num_frames}"

            short_row = dict(common)
            short_row.update(
                {
                    "history": "short",
                    "pixel_condition_policy": "mask_0_56_background_then_real_57_64",
                    "mask_rgb": json.dumps(list(cfg.render.background_rgb)),
                    "latent_hash": tensor_hash(short_view),
                    "target_hash": tensor_hash(short_target),
                }
            )
            long_row = dict(common)
            long_row.update(
                {
                    "history": "long",
                    "pixel_condition_policy": "real_0_64",
                    "mask_rgb": "",
                    "latent_hash": tensor_hash(long_view),
                    "target_hash": tensor_hash(long_target),
                }
            )
            encoded_short.append(short_row)
            encoded_long.append(long_row)
            audit_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "short_shape": list(short_view.shape),
                    "long_shape": list(long_view.shape),
                    "short_target_shape": list(short_target.shape),
                    "long_target_shape": list(long_target.shape),
                    "short_target_hash": short_row["target_hash"],
                    "long_target_hash": long_row["target_hash"],
                    "targets_elementwise_equal": bool(torch.equal(short_target, long_target)),
                    "target_mean_abs_diff": float(target_diff.mean()),
                    "target_max_abs_diff": float(target_diff.max()),
                    "pixel_future_is_identical_by_construction": True,
                }
            )

    write_rows(short_dir / "metadata.csv", encoded_short)
    write_rows(long_dir / "metadata.csv", encoded_long)
    write_rows(short_dir / "metadata_tiny.csv", balanced_prefix(encoded_short, min(cfg.tiny_train_videos, len(encoded_short))))
    write_rows(long_dir / "metadata_tiny.csv", balanced_prefix(encoded_long, min(cfg.tiny_train_videos, len(encoded_long))))
    write_rows(short_dir / "metadata_pilot.csv", balanced_prefix(encoded_short, min(cfg.pilot_train_videos, len(encoded_short))))
    write_rows(long_dir / "metadata_pilot.csv", balanced_prefix(encoded_long, min(cfg.pilot_train_videos, len(encoded_long))))

    summary = {
        "source": str(args.source),
        "vae": str(args.vae),
        "num_samples": len(source_rows),
        "short_shape": [1, 16, 33, 16, 16],
        "long_shape": [1, 16, 33, 16, 16],
        "short_pixel_construction": "background-mask frames 0..56; real frames 57..128",
        "long_pixel_construction": "real frames 0..128",
        "short_mask_rgb": list(cfg.render.background_rgb),
        "condition_latents_both": 17,
        "target_latents_both": 16,
        "target_latent_slice_both": [17, 33],
        "pixel_future_slice_both": [65, 129],
        "targets_elementwise_equal_required": False,
        "pixel_futures_identical_required": True,
        "samples": audit_rows,
    }
    (args.out_root / "encoding_audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
