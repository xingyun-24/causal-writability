#!/usr/bin/env python3
"""Encode 129-frame continuous-gravity videos as 33-frame Wan VAE latents."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import imageio.v2 as imageio
import torch
import yaml


def read_video(path: Path) -> torch.Tensor:
    reader = imageio.get_reader(path)
    try: frames = [frame[..., :3] for frame in reader]
    finally: reader.close()
    tensor = torch.from_numpy(__import__("numpy").stack(frames).copy()).permute(3, 0, 1, 2).float().div(127.5).sub(1).unsqueeze(0)
    if tensor.shape[2] != 129: raise ValueError(f"expected 129 frames: {path}")
    return tensor


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--source", type=Path, required=True); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--overwrite", action="store_true"); args = parser.parse_args()
    cfg = yaml.safe_load(args.data_config.read_text(encoding="utf-8"))
    if (int(cfg["render"]["num_frames"]), int(cfg["history"]["prediction_start"])) != (129, 65): raise ValueError("this profile requires 129 frames and a 65-frame prefix")
    from diffsynth import ModelManager
    manager = ModelManager(torch_dtype=torch.bfloat16, device=args.device); manager.load_models(["models/Wan2.1_VAE.pth"]); vae = manager.fetch_model("wan_video_vae").eval()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.source / "metadata.csv").open(newline="", encoding="utf-8") as f: rows = list(csv.DictReader(f))
    for index, row in enumerate(rows):
        target = args.out / f"{Path(row['video']).stem}.pt"
        if not args.overwrite and target.exists(): continue
        video = read_video(args.source / "videos" / row["video"]).to(args.device, dtype=torch.bfloat16)
        with torch.inference_mode(): latent = vae.encode(video, device=args.device, tiled=False)
        if tuple(latent.shape) != (1, 16, 33, 16, 16): raise ValueError(f"unexpected latent shape {tuple(latent.shape)}")
        # The encoded training loader collates one stored batch per file.
        # Keep this singleton batch axis: [1, C, T, H, W].
        torch.save(latent.cpu(), target)
        if (index + 1) % 32 == 0 or index + 1 == len(rows): print(f"encoded={index + 1}/{len(rows)}", flush=True)
    # Training loader uses the same metadata but replaces video filenames with .pt basenames.
    fields = list(rows[0]);
    for row in rows: row["video"] = f"{Path(row['video']).stem}.pt"
    with (args.out / "metadata.csv").open("w", newline="", encoding="utf-8") as f: writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__": main()
