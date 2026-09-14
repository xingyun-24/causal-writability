#!/usr/bin/env python3
"""PCA and continuous-gravity fit for a Projectile DiT block residual."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from layer_residual_replacement import condition_token_count, generate, load_condition


def finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def cfg_from(path: Path) -> Any:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return SimpleNamespace(
        width=int(data["render"]["width"]), height=int(data["render"]["height"]),
        frames=int(data["render"]["num_frames"]),
        prediction_start=int(data["history"]["prediction_start"]),
        short_masked_prefix=int(data["history"]["short_masked_prefix_pixels"]),
        background=data["render"]["background_rgb"], simulation_fps=int(data["physics"]["simulation_fps"]),
        latent_frames=33, condition_latents=17,
    )


def capture_residual(pipe: Any, condition: torch.Tensor, cfg: Any, steps: int, seed: int, block_index: int) -> np.ndarray:
    captures: list[torch.Tensor] = []
    block = pipe.dit.blocks[block_index]

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        if not isinstance(output, torch.Tensor) or output.ndim != 3:
            raise TypeError("Expected [batch, tokens, hidden] block output")
        count = condition_token_count(int(output.shape[1]), cfg)
        captures.append(output[:, :count].detach().cpu().clone()[0])

    handle = block.register_forward_hook(hook)
    try:
        generate(pipe, condition, cfg, steps, seed)
    finally:
        handle.remove()
    if len(captures) != steps:
        raise RuntimeError(f"capture calls={len(captures)}, expected={steps}")
    # NumPy cannot directly represent torch.bfloat16 on the H200 path.
    return torch.stack(captures).float().numpy()


def capture(args: argparse.Namespace) -> None:
    out = args.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    cfg = cfg_from(args.data_config)
    manifest = json.loads(args.pair_manifest.read_text(encoding="utf-8"))
    pairs = manifest["pairs"]
    from sshv2.wan.config import StandardTrainingConfig
    from sshv2.wan.trainer import WanTrainingModule

    training = StandardTrainingConfig.from_file(args.training_config)
    training.model.dit.ckpt_file = args.checkpoint
    module = WanTrainingModule(
        dit_config=training.model.dit, vae_config=training.model.vae, no_encoding=False,
        num_condition_frames=training.model.num_condition_frames,
        num_inference_steps=args.steps, pipeline_type=training.model.pipe,
        pipeline_kwargs=training.model.pipe_kwargs,
    )
    pipe = module.pipe
    pipe.to(args.device)
    pipe.load_models_to_device(("dit", "vae"))
    pipe.pre_encoded_(False)
    cfg.condition_latents = int(training.model.num_condition_frames)
    deltas: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    with (args.dataset_dir / "metadata.csv").open(newline="", encoding="utf-8") as handle:
        metadata_rows = list(csv.DictReader(handle))
    metadata_by_key = {(row["pair_id"], row["variant"]): row for row in metadata_rows}
    for index, item in enumerate(pairs, 1):
        pair_id = item["pair_id"]
        base_seed = int(item["base_seed"])
        aligned = metadata_by_key[(pair_id, "aligned")]
        conflict = metadata_by_key[(pair_id, "conflict")]
        old_prefix = cfg.short_masked_prefix
        cfg.short_masked_prefix = cfg.prediction_start - args.observed_window
        try:
            donor = load_condition(aligned, args.dataset_dir, cfg, pipe.torch_dtype, args.device, True)
            receiver = load_condition(conflict, args.dataset_dir, cfg, pipe.torch_dtype, args.device, True)
        finally:
            cfg.short_masked_prefix = old_prefix
        seed = base_seed + args.seed_offset
        aligned_residual = capture_residual(pipe, donor, cfg, args.steps, seed, args.block)
        conflict_residual = capture_residual(pipe, receiver, cfg, args.steps, seed, args.block)
        deltas.append(aligned_residual - conflict_residual)
        metadata.append({
            "pair_id": pair_id, "gravity_true": float(item["gravity_true"]),
            "gravity_interval": item["gravity_interval"], "base_seed": base_seed,
            "vx_boundary": float(json.loads(aligned["velocity_at_boundary"])[0]),
            "vy_boundary": float(json.loads(aligned["velocity_at_boundary"])[1]),
            "block": args.block, "observed_window": args.observed_window,
            "steps": args.steps, "feature_reduction": "full condition-token residual",
        })
        print(f"pairs={index}/{len(pairs)}", flush=True)
    np.save(out / "deltas.npy", np.stack(deltas).astype(args.delta_dtype))
    (out / "metadata.json").write_text(json.dumps(finite(metadata), indent=2) + "\n", encoding="utf-8")


def ridge_fit(design: np.ndarray, target: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + ridge * penalty, design.T @ target)


def basis_design(vx: np.ndarray, vy: np.ndarray, gravity: np.ndarray) -> np.ndarray:
    """Intercept plus the three requested physical basis functions."""
    features = np.column_stack((vx, vy, gravity)).astype(np.float64)
    features -= features.mean(axis=0, keepdims=True)
    return np.column_stack((np.ones(len(features)), features))


def analyze(args: argparse.Namespace) -> None:
    shards = sorted(args.shards)
    deltas = np.concatenate([np.load(path / "deltas.npy") for path in shards], axis=0)
    metadata = sum((json.loads((path / "metadata.json").read_text(encoding="utf-8")) for path in shards), [])
    order = np.argsort([item["pair_id"] for item in metadata])
    deltas = deltas[order]
    metadata = [metadata[int(index)] for index in order]
    flat = deltas.reshape(len(deltas), -1).astype(np.float64)
    gram = flat @ flat.T
    eigenvalues, vectors = np.linalg.eigh(gram)
    order_pc = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order_pc], 0.0)
    vectors = vectors[:, order_pc]
    singular = np.sqrt(np.maximum(eigenvalues, 1e-30))
    scores = vectors * singular[None, :]
    total_energy = float(eigenvalues.sum())
    gravity = np.asarray([item["gravity_true"] for item in metadata], dtype=np.float64)
    vx = np.asarray([item["vx_boundary"] for item in metadata], dtype=np.float64)
    vy = np.asarray([item["vy_boundary"] for item in metadata], dtype=np.float64)
    intervals = np.asarray([item["gravity_interval"] for item in metadata])
    fit_mask = np.zeros(len(metadata), dtype=bool)
    for interval in ("low", "high"):
        indices = np.flatnonzero(intervals == interval)
        fit_mask[indices[::2]] = True
    holdout = ~fit_mask
    design = basis_design(vx, vy, gravity)
    component_results = []
    for components in (1, 2, 4, 8, 16):
        m = min(components, scores.shape[1])
        coef = ridge_fit(design[fit_mask], scores[fit_mask, :m])
        predicted = design[holdout] @ coef
        actual = scores[holdout, :m]
        mse = float(np.mean((predicted - actual) ** 2))
        baseline = float(np.mean((actual - actual.mean(axis=0)) ** 2))
        component_results.append({
            "components": m,
            "cumulative_pca_energy": float(eigenvalues[:m].sum() / total_energy) if total_energy else None,
            "heldout_score_rmse": float(np.sqrt(mse)), "heldout_score_r2": 1.0 - mse / baseline if baseline > 0 else None,
        })
    basis_results = [{"basis": "vx, vy, g", "features": ["vx_boundary", "vy_boundary", "gravity_true"], "results": component_results}]
    selected_components, selected_rmse = min(
        ((result["components"], result["heldout_score_rmse"]) for result in component_results), key=lambda item: item[1]
    )
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "block": int(metadata[0]["block"]), "pairs": len(metadata), "fit_pairs": int(fit_mask.sum()),
        "heldout_pairs": int(holdout.sum()), "fit_split": "every other pair within each gravity interval",
        "delta_shape": list(deltas.shape), "feature_reduction": metadata[0]["feature_reduction"],
        "recovery_metric": {
            "name": "endpoint_normalized_recovery_R",
            "formula": "R = (g_edit - g_conflict) / (g_aligned - g_conflict)",
            "state_source": "E3 gravity_hat from generated videos",
            "success_interval": [0.9, 1.1],
        },
        "pca": {"total_energy": total_energy, "eigenvalues": eigenvalues[:16].tolist(),
                "energy_fraction": (eigenvalues[:16] / total_energy).tolist() if total_energy else []},
        "fit": {"target": "PCA scores of aligned-minus-conflict block residual", "basis": "intercept + centered vx_boundary + centered vy_boundary + centered gravity_true", "physical_basis_functions": ["vx_boundary", "vy_boundary", "gravity_true"], "selected_components": selected_components, "selected_heldout_score_rmse": selected_rmse, "ridge": 1e-6, "results": basis_results},
    }
    (args.out / "pca_fit.json").write_text(json.dumps(finite(payload), indent=2) + "\n", encoding="utf-8")
    x = np.arange(1, len(eigenvalues[:16]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=160)
    axes[0].plot(x, np.cumsum(eigenvalues[:16]) / total_energy * 100, marker="o")
    axes[0].set(xlabel="PCA components", ylabel="Cumulative energy (%)", title="Block 6 residual PCA")
    axes[0].set_title(f"Block {metadata[0]['block']} residual PCA")
    axes[0].set_ylim(0, 105); axes[0].grid(axis="y", alpha=.25)
    axes[1].plot([r["components"] for r in component_results], [r["heldout_score_rmse"] for r in component_results], marker="o", label="vx + vy + g")
    axes[1].set(xlabel="PCA components used", ylabel="Held-out PC-score RMSE", title="Physical-basis fit")
    axes[1].legend(frameon=False, fontsize=7)
    axes[1].grid(axis="y", alpha=.25)
    fig.suptitle(f"Projectile V3 hist32 100k | selected block {metadata[0]['block']}")
    fig.tight_layout(); fig.savefig(args.out / "pca_fit_summary.png", bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=160)
    for interval, color in (("low", "#2f7d32"), ("high", "#b23b3b")):
        mask = intervals == interval
        ax.scatter(gravity[mask], scores[mask, 0], s=18, alpha=.8, label=interval, color=color)
    ax.set(xlabel="True continuous g", ylabel="PC1 score", title=f"Block {metadata[0]['block']} PC1 vs true gravity")
    ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(args.out / "pc1_vs_gravity.png", bbox_inches="tight"); plt.close(fig)
    print(json.dumps(payload["fit"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    capture_parser = sub.add_parser("capture")
    for name in ("dataset-dir", "data-config", "training-config", "checkpoint", "pair-manifest", "out"):
        capture_parser.add_argument(f"--{name}", type=Path, required=True)
    capture_parser.add_argument("--block", type=int, default=6)
    capture_parser.add_argument("--steps", type=int, default=20)
    capture_parser.add_argument("--observed-window", type=int, default=16)
    capture_parser.add_argument("--seed-offset", type=int, default=17_000_000)
    capture_parser.add_argument("--delta-dtype", choices=("float16", "float32"), default="float16")
    capture_parser.add_argument("--device", default="cuda")
    capture_parser.set_defaults(func=capture)
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--shards", type=Path, nargs="+", required=True)
    analyze_parser.add_argument("--out", type=Path, required=True)
    analyze_parser.set_defaults(func=analyze)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
