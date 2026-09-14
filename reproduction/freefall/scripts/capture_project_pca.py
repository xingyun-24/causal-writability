"""Capture B1 fit differences and held-out raw endpoints for the project page.

Uses the frozen 64/64 split and original renderer, video codec and sampler.
No training or controller intervention is performed.
"""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from audit_v1_estimator import audit_track
from build_dataset_v2 import read_config, render, trajectory, write_video
from layer_residual_replacement import condition_token_count, generate, load_condition, measure
from projectile_block_pca_fit import cfg_from
from sshv2.wan.config import StandardTrainingConfig
from sshv2.wan.trainer import WanTrainingModule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("fit", "heldout"), required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    assert 0 <= args.shard < args.shards
    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "videos").mkdir(exist_ok=True)
    packed = np.load(root / "redo/results-grouped/train_only_pca.npz", allow_pickle=False)
    indices = np.flatnonzero(packed["train_mask" if args.split == "fit" else "heldout_mask"])
    assert len(indices) == 64
    indices = indices[args.shard::args.shards]
    if args.limit:
        indices = indices[:args.limit]
    with (root / "redo/delivery_inputs/metadata.csv").open(newline="") as handle:
        metadata = {(r["pair_id"], r["variant"]): r for r in csv.DictReader(handle)}
    config = root / "config/data-v3-fixedpos-freefall.yaml"
    raw = yaml.safe_load(config.read_text())
    renderer, _ = read_config(config)
    cfg = cfg_from(config)
    cfg.radius = int(raw["render"]["ball_radius_px"])
    cfg.gravity_bands = {b: tuple(raw["physics"][b + "_gravity_range"]) for b in ("low", "high")}
    cfg.short_masked_prefix = cfg.prediction_start - 32
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    training = StandardTrainingConfig.from_file(root / "config/Train-short-v3-large-fixedpos-freefall-hist32.yaml")
    training.model.dit.ckpt_file = root / "checkpoints/short-v3-large-fixedpos-freefall-hist32/hist32/step-100000.safetensors"
    training.model.vae.ckpt_file = root / "models/Wan2.1_VAE.pth"
    module = WanTrainingModule(dit_config=training.model.dit, vae_config=training.model.vae,
        no_encoding=False, num_condition_frames=training.model.num_condition_frames,
        num_inference_steps=20, pipeline_type=training.model.pipe, pipeline_kwargs=training.model.pipe_kwargs)
    pipe = module.pipe
    pipe.to(args.device)
    pipe.load_models_to_device(("dit", "vae"))
    pipe.pre_encoded_(False)
    cfg.condition_latents = int(training.model.num_condition_frames)
    reference_basis = np.load(root / "paper/pca_components.npy", mmap_mode="r").reshape(2, -1)
    started = time.monotonic()
    for pos, i in enumerate(indices, 1):
        pair = str(packed["pair_ids"][i])
        record_path = out / f"{pair}.json"
        if record_path.exists():
            print(f"cached {pair}", flush=True)
            continue
        endpoints, measured = {}, {}
        for variant in ("aligned", "conflict"):
            row = metadata[(pair, variant)]
            video = out / "videos" / row["video"]
            if not video.exists():
                state = {key: float(row[key]) for key in ("x_boundary", "y_boundary", "vx_boundary", "vy_boundary")}
                frames = render(trajectory(state, float(row["gravity"]), renderer), renderer.colors[row["color_label"]], renderer)
                write_video(video, frames, renderer.fps)
            condition = load_condition(row, out, cfg, pipe.torch_dtype, args.device, True)
            captures = []
            def hook(_module, _inputs, hidden):
                count = condition_token_count(int(hidden.shape[1]), cfg)
                assert count == 1088
                captures.append(hidden[0, :count].detach().cpu().clone())
            handle = pipe.dit.blocks[1].register_forward_hook(hook)
            try:
                future = generate(pipe, condition, cfg, 20, int(row["base_seed"]) + 17000000)
            finally:
                handle.remove()
            assert len(captures) == 20
            endpoints[variant] = torch.stack(captures).float().numpy()
            measured[variant] = {"g_E3": audit_track(future, row, cfg)["g_E3"],
                                 "valid": bool(measure(future, row, cfg)["valid"]),
                                 "input_colour": row["color_label"]}
            if args.split == "heldout":
                np.save(out / f"{pair}_{variant}.npy", endpoints[variant])
        delta = endpoints["aligned"] - endpoints["conflict"]
        assert delta.shape == (20, 1088, 1536)
        if args.split == "fit":
            np.save(out / f"{pair}_delta.npy", delta)
        scores = np.array([np.sum(delta.reshape(-1).astype(np.float64) * v, dtype=np.float64) for v in reference_basis])
        expected = packed["scores"][i, :2]
        record = {"pair_id": pair, "split": args.split, "base_seed": int(row["base_seed"]),
                  "generation_seed": int(row["base_seed"]) + 17000000,
                  "target_band": row["gravity_interval"], "target_g": float(row["gravity"]),
                  "block": 1, "observed_frames": 32, "fm_calls": 20,
                  "measurements": measured, "archived_pc12": expected.tolist(),
                  "recaptured_pc12": scores.tolist(),
                  "pc12_relative_error": float(np.linalg.norm(scores - expected) / np.linalg.norm(expected))}
        record_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        print(f"{args.split} shard={args.shard} {pos}/{len(indices)} {pair} error={record['pc12_relative_error']:.6g} elapsed={time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
