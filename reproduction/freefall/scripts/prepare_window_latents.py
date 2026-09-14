#!/usr/bin/env python3
"""Encode complete-timeline inputs with a fixed-size observed window visible."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml


def load_video(path: Path) -> torch.Tensor:
    reader = imageio.get_reader(path)
    try: frames = np.stack([frame[..., :3] for frame in reader])
    finally: reader.close()
    return torch.from_numpy(frames.copy()).permute(3, 0, 1, 2).float().div(127.5).sub(1).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--source", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True); parser.add_argument("--window", type=int, required=True); parser.add_argument("--device", default="cuda"); parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prediction-starts", type=int, nargs="+", help="deterministic per-pair prediction starts; defaults to the configured start")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--allow-window-override", action="store_true", help="allow a fixed-start training window other than the config default")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--no-metadata", action="store_true")
    args = parser.parse_args(); cfg = yaml.safe_load(args.data_config.read_text(encoding="utf-8")); render, history = cfg["render"], cfg["history"]; frames, default_start = int(render["num_frames"]), int(history["prediction_start"])
    if frames != 129 or default_start != 65: raise ValueError("window must use the 129-frame profile with prediction start 65")
    if args.window != int(history["short_observation_window_pixels"]) and not args.allow_window_override: raise ValueError("window differs from the configured default; pass --allow-window-override for an explicit fixed-start experiment")
    starts = tuple(args.prediction_starts or (default_start,))
    if args.batch_size < 1: raise ValueError("batch size must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards: raise ValueError("invalid shard index")
    if any(start < args.window or start > frames - args.window for start in starts): raise ValueError("each prediction start must retain the configured history and at least one observation window of future")
    background = torch.tensor(render["background_rgb"], dtype=torch.float32).div(127.5).sub(1).view(1, 3, 1, 1, 1)
    with (args.source / "metadata.csv").open(newline="", encoding="utf-8") as f: rows = list(csv.DictReader(f))
    args.out.mkdir(parents=True, exist_ok=True)
    def prediction_start(row: dict[str, str]) -> int:
        return starts[int(row["base_seed"]) % len(starts)]

    def write_metadata() -> None:
        output_rows = [dict(row) for row in rows]
        fields = list(output_rows[0]) + ["observation_window_pixels", "masked_prefix_pixels", "pixel_condition_policy"]
        for row in output_rows:
            start = prediction_start(row); boundary = start - 1; original_boundary = int(row["boundary_frame"]); dt = (boundary - original_boundary) / int(row["simulation_fps"])
            gravity, x, y, vx, vy = float(row["gravity"]), float(row["x_boundary"]), float(row["y_boundary"]), float(row["vx_boundary"]), float(row["vy_boundary"])
            x += vx * dt; y += vy * dt - .5 * gravity * dt ** 2; vy -= gravity * dt
            mask_until = start - args.window
            row["video"] = f"{Path(row['video']).stem}.pt"; row["boundary_frame"] = boundary; row["prediction_start"] = start
            row["x_boundary"] = x; row["y_boundary"] = y; row["vx_boundary"] = vx; row["vy_boundary"] = vy
            row["position_at_boundary"] = json.dumps([x, y]); row["velocity_at_boundary"] = json.dumps([vx, vy])
            row["observation_window_pixels"] = args.window; row["masked_prefix_pixels"] = mask_until; row["pixel_condition_policy"] = f"background_0_{mask_until - 1}_real_{mask_until}_{start - 1}_target_{start}_128"
        with (args.out / "metadata.csv").open("w", newline="", encoding="utf-8") as f: writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(output_rows)

    if args.metadata_only:
        write_metadata()
        return

    from diffsynth import ModelManager
    manager = ModelManager(torch_dtype=torch.bfloat16, device=args.device); manager.load_models(["models/Wan2.1_VAE.pth"]); vae = manager.fetch_model("wan_video_vae").eval()

    pending: list[tuple[Path, torch.Tensor]] = []
    encoded = sum((args.out / f"{Path(row['video']).stem}.pt").exists() for row in rows) if not args.overwrite else 0

    def encode_pending() -> None:
        nonlocal encoded
        if not pending: return
        pixels = torch.cat([item[1] for item in pending])
        with torch.inference_mode(): latent = vae.encode(pixels.to(args.device, torch.bfloat16), device=args.device, tiled=False)
        expected = (len(pending), 16, 33, 16, 16)
        if tuple(latent.shape) != expected: raise ValueError(f"unexpected latent {tuple(latent.shape)}, expected {expected}")
        for item, sample_latent in zip(pending, latent, strict=True): torch.save(sample_latent.unsqueeze(0).cpu(), item[0])
        encoded += len(pending); pending.clear()
        if encoded % 32 == 0 or encoded == len(rows): print(f"window={args.window} encoded={encoded}/{len(rows)}", flush=True)

    for row in rows:
        if int(row["base_seed"]) % args.num_shards != args.shard_index: continue
        target = args.out / f"{Path(row['video']).stem}.pt"
        if target.exists() and not args.overwrite: continue
        mask_until = prediction_start(row) - args.window
        pixels = load_video(args.source / "videos" / row["video"]); pixels[:, :, :mask_until] = background
        pending.append((target, pixels))
        if len(pending) == args.batch_size: encode_pending()
    encode_pending()
    if not args.no_metadata: write_metadata()


if __name__ == "__main__": main()
