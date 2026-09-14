"""Generate spring_shortcuts_v4 futures from a trained short- or long-history checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml

from sshv2.diffsynth.trainers.wan_video_train import WanTrainingModule
from sshv2.simulation.spring_shortcuts_v1 import (
    apply_short_history_mask_tensor,
    dataclass_config_from_dict,
    load_video,
    write_video,
)
from sshv2.utils.spring_configs import SpringTrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="Raw eval split")
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--history", choices=("short", "long"), required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--variant", choices=("all", "aligned", "conflict"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-offset", type=int, default=17_000_000)
    parser.add_argument("--save-full", action="store_true")
    parser.add_argument("--save-comparison", action="store_true")
    parser.add_argument("--save-latents", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing future MP4s in --out/predictions and regenerate the complete manifest.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_video_tensor(frames: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(frames.copy()).permute(3, 0, 1, 2).float().div(127.5).sub(1.0)


def tensor_to_uint8(video: torch.Tensor) -> np.ndarray:
    # [C,T,H,W] -> [T,H,W,C]
    return (
        video.detach().float().cpu().permute(1, 2, 3, 0).add(1.0).mul(127.5).clamp(0, 255).byte().numpy()
    )


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    data_cfg = dataclass_config_from_dict(yaml.safe_load(args.data_config.read_text(encoding="utf-8")))
    train_cfg = SpringTrainingConfig.from_file(args.config)
    expected_conditions = data_cfg.long_condition_latents
    if data_cfg.short_condition_latents != data_cfg.long_condition_latents:
        raise AssertionError("V4 requires equal short/long condition-latent counts")
    if train_cfg.model.num_condition_frames != expected_conditions:
        raise ValueError(
            f"Config has {train_cfg.model.num_condition_frames} condition latents, expected {expected_conditions} for {args.history}"
        )
    train_cfg.model.dit.ckpt_file = args.checkpoint
    model = WanTrainingModule(
        dit_config=train_cfg.model.dit,
        vae_config=train_cfg.model.vae,
        no_encoding=False,
        num_condition_frames=train_cfg.model.num_condition_frames,
        num_inference_steps=args.steps,
        pipeline_type=train_cfg.model.pipe,
        pipeline_kwargs=train_cfg.model.pipe_kwargs,
    )
    pipe = model.pipe
    pipe.to(args.device)
    pipe.load_models_to_device(("dit", "vae"))
    # Both histories use a 65-frame pixel prefix. The short prefix is masked
    # in pixel space before VAE encoding, so true frames 0..56 cannot leak into
    # its condition latents while temporal length and RoPE positions stay equal.
    pipe.pre_encoded_(False)

    rows = read_rows(args.dataset / "metadata.csv")
    if args.variant != "all":
        rows = [row for row in rows if row["variant"] == args.variant]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No evaluation rows selected")

    prediction_dir = args.out / "predictions"
    full_dir = args.out / "full"
    comparison_dir = args.out / "comparisons"
    latent_dir = args.out / "latents"
    directories = [prediction_dir, full_dir, comparison_dir]
    if args.save_latents:
        directories.append(latent_dir)
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        prediction_path = prediction_dir / f"{row['sample_id']}.mp4"
        latent_path = latent_dir / f"{row['sample_id']}.pt" if args.save_latents else None
        generation_seed = int(row["base_seed"]) + args.seed_offset

        if args.resume and prediction_path.is_file() and (not args.save_latents or latent_path.is_file()):
            manifest.append(
                {
                    "sample_id": row["sample_id"],
                    "pair_id": row["pair_id"],
                    "variant": row["variant"],
                    "true_band": row["true_band"],
                    "color_label": row["color_label"],
                    "history": args.history,
                    "checkpoint": str(args.checkpoint),
                    "generation_seed": generation_seed,
                    "num_inference_steps": args.steps,
                    "condition_source": "mask_0_56_background_then_real_57_64" if args.history == "short" else "real_0_64",
                    "condition_pixel_slice": [0, data_cfg.prediction_start],
                    "real_condition_pixel_slice": [data_cfg.short_prefix_start, data_cfg.prediction_start]
                    if args.history == "short" else [0, data_cfg.prediction_start],
                    "short_mask_rgb": list(data_cfg.render.background_rgb) if args.history == "short" else None,
                    "prediction": str(prediction_path.relative_to(args.out)),
                    "latent": str(latent_path.relative_to(args.out)) if latent_path is not None else None,
                }
            )
            print(f"[{index + 1}/{len(rows)}] {row['sample_id']} (resume: existing prediction)")
            continue

        raw = load_video(args.dataset / row["video"], expected_frames=data_cfg.render.num_frames)
        raw_tensor = to_video_tensor(raw).unsqueeze(0).to(device=args.device, dtype=pipe.torch_dtype)

        condition_pixels = raw_tensor[:, :, : data_cfg.prediction_start].contiguous()
        if args.history == "short":
            condition_pixels = apply_short_history_mask_tensor(condition_pixels, data_cfg)
            condition_policy = "mask_0_56_background_then_real_57_64"
            real_condition_pixel_slice = [data_cfg.short_prefix_start, data_cfg.prediction_start]
        else:
            condition_policy = "real_0_64"
            real_condition_pixel_slice = [0, data_cfg.prediction_start]
        generated_num_frames = data_cfg.render.num_frames
        generated_future_start = data_cfg.prediction_start
        condition_pixel_slice = [0, data_cfg.prediction_start]
        expected_pixel_conditions = data_cfg.long_prefix_frames
        if condition_pixels.shape[2] != expected_pixel_conditions:
            raise AssertionError(f"Unexpected condition pixel shape {tuple(condition_pixels.shape)}")

        with torch.inference_mode():
            generated_output = pipe(
                prompt="",
                negative_prompt="",
                cfg_scale=1.0,
                height=data_cfg.render.height,
                width=data_cfg.render.width,
                num_frames=generated_num_frames,
                num_condition_frames=expected_conditions,
                condition_frames=condition_pixels,
                num_inference_steps=args.steps,
                tiled=False,
                num_samples=1,
                return_as_tensor=True,
                return_latents=args.save_latents,
                progress_bar_cmd=lambda values: values,
                seed=generation_seed,
            )
        if args.save_latents:
            generated_batch, final_latents = generated_output
            if final_latents is None:
                raise AssertionError("Pipeline did not return final latents")
            generated = generated_batch[0]
            expected_latent_shape = (1, 16, data_cfg.long_latent_frames, 16, 16)
            if tuple(final_latents.shape) != expected_latent_shape:
                raise AssertionError(
                    f"Generated latent shape {tuple(final_latents.shape)}, "
                    f"expected {expected_latent_shape}"
                )
            assert latent_path is not None
            torch.save(final_latents.detach().float().cpu(), latent_path)
        else:
            generated = generated_output[0]
            latent_path = None
        full_frames = tensor_to_uint8(generated)
        if full_frames.shape[0] != generated_num_frames:
            raise AssertionError(f"Generated {full_frames.shape[0]} frames, expected {generated_num_frames}")
        future = full_frames[generated_future_start:]
        if future.shape[0] != data_cfg.future_frames:
            raise AssertionError(f"Future has {future.shape[0]} frames, expected {data_cfg.future_frames}")

        write_video(prediction_path, future, data_cfg.render.fps)
        if args.save_full:
            write_video(full_dir / f"{row['sample_id']}.mp4", full_frames, data_cfg.render.fps)
        if args.save_comparison:
            condition_display = raw[: data_cfg.prediction_start].copy()
            gt_future = raw[data_cfg.prediction_start :]
            # Repeat the last visible condition frame so all three panels have 64 frames.
            cond_panel = np.repeat(condition_display[-1:], data_cfg.future_frames, axis=0)
            comparison = np.concatenate([cond_panel, gt_future, future], axis=2)
            write_video(comparison_dir / f"{row['sample_id']}.mp4", comparison, data_cfg.render.fps)

        manifest.append(
            {
                "sample_id": row["sample_id"],
                "pair_id": row["pair_id"],
                "variant": row["variant"],
                "true_band": row["true_band"],
                "color_label": row["color_label"],
                "history": args.history,
                "checkpoint": str(args.checkpoint),
                "generation_seed": generation_seed,
                "num_inference_steps": args.steps,
                "condition_source": condition_policy,
                "condition_pixel_slice": condition_pixel_slice,
                "real_condition_pixel_slice": real_condition_pixel_slice,
                "short_mask_rgb": list(data_cfg.render.background_rgb) if args.history == "short" else None,
                "prediction": str(prediction_path.relative_to(args.out)),
                "latent": str(latent_path.relative_to(args.out)) if latent_path is not None else None,
            }
        )
        print(f"[{index + 1}/{len(rows)}] {row['sample_id']}")

    with (args.out / "generation_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for item in manifest:
            handle.write(json.dumps(item) + "\n")
    (args.out / "generation_summary.json").write_text(
        json.dumps(
            {
                "history": args.history,
                "config": str(args.config),
                "checkpoint": str(args.checkpoint),
                "dataset": str(args.dataset),
                "num_predictions": len(manifest),
                "future_frames_per_prediction": data_cfg.future_frames,
                "latents_saved": bool(args.save_latents),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
