#!/usr/bin/env python3
"""Evaluate norm-scaled rank-k PCA reconstruction on full condition-token residuals."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from layer_residual_replacement import condition_token_count, load_condition, measure
from projectile_block_pca_fit import capture_residual, cfg_from, generate


def load_matrix(root: Path, device: torch.device) -> tuple[torch.Tensor, dict[str, int], torch.Tensor, torch.Tensor, tuple[int, ...]]:
    shards = sorted(path for path in root.glob("shard*") if path.is_dir())
    arrays = [np.load(path / "deltas.npy", mmap_mode="r") for path in shards]
    metadata = sum((json.loads((path / "metadata.json").read_text()) for path in shards), [])
    locations = [(array_index, index) for array_index, array in enumerate(arrays) for index in range(len(array))]
    order = np.argsort([item["pair_id"] for item in metadata])
    packed = np.load(root / "decomposition" / "sample_pca.npz")
    pair_ids = [str(item) for item in packed["pair_ids"]]
    if pair_ids != [metadata[int(index)]["pair_id"] for index in order]:
        raise ValueError("PCA pair ordering does not match captured deltas")
    shape = tuple(int(value) for value in packed["delta_shape"])
    matrix_dtype = torch.float32 if arrays[0].dtype == np.float32 else torch.float16
    matrix = torch.empty((len(order), int(np.prod(shape))), dtype=matrix_dtype, device=device)
    for target, source in enumerate(order):
        array_index, index = locations[int(source)]
        matrix[target].copy_(torch.from_numpy(np.array(arrays[array_index][index], copy=True)).flatten().to(device))
    vectors = torch.from_numpy(packed["vectors"].astype(np.float32)).to(device)
    eigenvalues = torch.from_numpy(packed["eigenvalues"].astype(np.float64)).to(device)
    return matrix, {pair_id: index for index, pair_id in enumerate(pair_ids)}, vectors, eigenvalues, shape


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("dataset-dir", "data-config", "training-config", "checkpoint", "pair-manifest", "pca-root", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--components", type=int, required=True)
    parser.add_argument("--block", type=int, default=2)
    parser.add_argument("--observed-window", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    matrix, index_by_pair, vectors, eigenvalues, shape = load_matrix(args.pca_root, device)
    if not 1 <= args.components <= vectors.shape[1]:
        raise ValueError(f"components must be in [1, {vectors.shape[1]}]")
    data = yaml.safe_load(args.data_config.read_text())
    cfg = cfg_from(args.data_config)
    cfg.gravity_bands = {"low": tuple(data["physics"]["low_gravity_range"]), "high": tuple(data["physics"]["high_gravity_range"])}
    cfg.radius = int(data["render"]["ball_radius_px"]); cfg.gravity_accuracy_threshold = float(data["evaluation"]["E3_threshold"])
    from sshv2.wan.config import StandardTrainingConfig
    from sshv2.wan.trainer import WanTrainingModule
    training = StandardTrainingConfig.from_file(args.training_config); training.model.dit.ckpt_file = args.checkpoint
    module = WanTrainingModule(dit_config=training.model.dit, vae_config=training.model.vae, no_encoding=False, num_condition_frames=training.model.num_condition_frames, num_inference_steps=args.steps, pipeline_type=training.model.pipe, pipeline_kwargs=training.model.pipe_kwargs)
    pipe = module.pipe; pipe.to(args.device); pipe.load_models_to_device(("dit", "vae")); pipe.pre_encoded_(False)
    cfg.condition_latents = int(training.model.num_condition_frames)
    rows = {(row["pair_id"], row["variant"]): row for row in csv.DictReader((args.dataset_dir / "metadata.csv").open(newline=""))}
    pairs = json.loads(args.pair_manifest.read_text())["pairs"]
    results = []
    components = vectors[:, :args.components]
    retained_energy = eigenvalues[:args.components].sum()
    energy_scale = (
        torch.ones((), dtype=matrix.dtype, device=device)
        if args.components == matrix.shape[0]
        else torch.sqrt(eigenvalues.sum() / retained_energy).to(dtype=matrix.dtype)
    )
    for count, item in enumerate(pairs, 1):
        pair_id = item["pair_id"]; source = index_by_pair[pair_id]; conflict = rows[(pair_id, "conflict")]
        if args.components == matrix.shape[0]:
            reconstructed = matrix[source]
        else:
            weights = components @ components[source]
            reconstructed = weights.to(matrix.dtype) @ matrix
        delta = (reconstructed * energy_scale).reshape(shape)
        old_prefix = cfg.short_masked_prefix; cfg.short_masked_prefix = cfg.prediction_start - args.observed_window
        condition = load_condition(conflict, args.dataset_dir, cfg, pipe.torch_dtype, args.device, True); cfg.short_masked_prefix = old_prefix
        conflict_residual = torch.from_numpy(
            capture_residual(pipe, condition, cfg, args.steps, int(conflict["base_seed"]) + 17_000_000, args.block)
        ).to(device=device)
        call = 0; block = pipe.dit.blocks[args.block]
        def hook(_module, _inputs, output):
            nonlocal call
            tokens = condition_token_count(int(output.shape[1]), cfg)
            edited = output.clone()
            edited[:, :tokens] = (conflict_residual[call].unsqueeze(0) + delta[call].unsqueeze(0)).to(dtype=output.dtype)
            call += 1
            return edited
        handle = block.register_forward_hook(hook)
        try: frames = generate(pipe, condition, cfg, args.steps, int(conflict["base_seed"]) + 17_000_000)
        finally: handle.remove()
        results.append({"pair_id": pair_id, "components": args.components, "block": args.block, **measure(frames, conflict, cfg)})
        print(f"pairs={count}/{len(pairs)}", flush=True)
    payload = {"method": "uncentered PCA of aligned-minus-conflict full condition-token residuals; captured conflict residual plus energy-scaled rank-k reconstruction", "components": args.components, "energy_scale": float(energy_scale), "retained_energy_fraction": float(retained_energy / eigenvalues.sum()), "pairs": len(results), "E3_accuracy": float(np.mean([row["E3_correct"] for row in results])), "E0_accuracy": float(np.mean([row["E0_correct"] for row in results])), "results": results}
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n"); (args.out / "outcomes.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("components", "energy_scale", "retained_energy_fraction", "pairs", "E3_accuracy", "E0_accuracy")}, indent=2))


if __name__ == "__main__":
    main()
