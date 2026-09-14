#!/usr/bin/env python3
"""Layerwise aligned-to-conflict condition-residual replacement for Projectile V1.

For each paired evaluation sample, the script captures an aligned run's output
at every DiT block, then generates the colour-conflict input while replacing
only that block's condition-token output.  It never changes the generated
tokens directly.  Dynamics are reported as a continuous fitted gravity.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import yaml


def read_pairs(dataset_dir: Path, limit: int) -> list[tuple[dict[str, str], dict[str, str]]]:
    with (dataset_dir / "metadata.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        if row.get("variant") in {"aligned", "conflict"}:
            grouped.setdefault(row["pair_id"], {})[row["variant"]] = row
    pairs = [
        (value["aligned"], value["conflict"])
        for _, value in sorted(grouped.items())
        if {"aligned", "conflict"}.issubset(value)
    ]
    if not pairs:
        raise ValueError("No aligned/conflict pairs in evaluation metadata")
    return pairs[:limit] if limit else pairs


def load_condition(row: dict[str, str], dataset_dir: Path, cfg: Any, dtype: torch.dtype, device: str, short: bool) -> torch.Tensor:
    raw = imageio.mimread(dataset_dir / "videos" / row["video"])
    frames = np.asarray(raw, dtype=np.uint8)
    if frames.shape[0] != cfg.frames:
        raise ValueError(f"Expected {cfg.frames} frames, got {frames.shape}")
    condition = torch.from_numpy(frames.copy()).permute(3, 0, 1, 2).float().div(127.5).sub(1.0)[None]
    condition = condition[:, :, :cfg.prediction_start].to(device=device, dtype=dtype)
    if short:
        # Frozen V1: source pixels 0..56 are exact background, not history.
        background = torch.tensor(cfg.background, device=device, dtype=dtype).view(1, 3, 1, 1, 1).div(127.5).sub(1.0)
        condition[:, :, :cfg.short_masked_prefix] = background
    return condition.contiguous()


def frames_from_tensor(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().permute(1, 2, 3, 0).add(1.0).mul(127.5).clamp(0, 255).byte().numpy()


def condition_token_count(tokens: int, cfg: Any) -> int:
    if tokens % cfg.latent_frames:
        raise ValueError(f"Unexpected DiT token count {tokens}")
    return cfg.condition_latents * (tokens // cfg.latent_frames)


def generate(pipe: Any, condition: torch.Tensor, cfg: Any, steps: int, seed: int) -> np.ndarray:
    with torch.inference_mode():
        output = pipe(
            prompt="", negative_prompt="", cfg_scale=1.0,
            height=cfg.height, width=cfg.width, num_frames=cfg.frames,
            num_condition_frames=cfg.condition_latents, condition_frames=condition,
            num_inference_steps=steps, tiled=False, num_samples=1,
            return_as_tensor=True, progress_bar_cmd=lambda values: values, seed=seed,
        )[0]
    return frames_from_tensor(output)[cfg.prediction_start:]


def capture_all_blocks(pipe: Any, condition: torch.Tensor, cfg: Any, steps: int, seed: int) -> dict[int, list[torch.Tensor]]:
    captures: dict[int, list[torch.Tensor]] = {index: [] for index in range(len(pipe.dit.blocks))}
    handles = []
    for index, block in enumerate(pipe.dit.blocks):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any, *, block_index: int = index) -> None:
            if not isinstance(output, torch.Tensor) or output.ndim != 3:
                raise TypeError("Expected [batch, tokens, hidden] block output")
            count = condition_token_count(int(output.shape[1]), cfg)
            captures[block_index].append(output[:, :count].detach().cpu().clone())
        handles.append(block.register_forward_hook(hook))
    try:
        generate(pipe, condition, cfg, steps, seed)
    finally:
        for handle in handles:
            handle.remove()
    if any(len(items) != steps for items in captures.values()):
        raise RuntimeError("Did not capture every block at every denoising step")
    return captures


def generate_with_replacement(pipe: Any, block_index: int, donor: list[torch.Tensor], condition: torch.Tensor, cfg: Any, steps: int, seed: int) -> np.ndarray:
    block = pipe.dit.blocks[block_index]
    call_index = 0
    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> torch.Tensor:
        nonlocal call_index
        if call_index >= len(donor):
            raise RuntimeError("Hook invoked more times than donor capture")
        count = condition_token_count(int(output.shape[1]), cfg)
        replacement = donor[call_index].to(device=output.device, dtype=output.dtype)
        if tuple(replacement.shape) != tuple(output[:, :count].shape):
            raise ValueError("Donor residual shape does not match receiver")
        edited = output.clone()
        edited[:, :count] = replacement
        call_index += 1
        return edited
    handle = block.register_forward_hook(hook)
    try:
        return generate(pipe, condition, cfg, steps, seed)
    finally:
        handle.remove()
        if call_index != len(donor):
            raise RuntimeError(f"Replacement hook calls={call_index}, expected={len(donor)}")


def detect_ball(frames: np.ndarray, cfg: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys = np.full(len(frames), np.nan, dtype=np.float64)
    rgbs = np.full((len(frames), 3), np.nan, dtype=np.float64)
    prior: tuple[float, float] | None = None
    for index, frame in enumerate(frames):
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask = ((hsv[..., 1] >= 45) & (hsv[..., 2] >= 55)).astype(np.uint8)
        count, labels, stats, centres = cv2.connectedComponentsWithStats(mask, connectivity=8)
        candidates: list[tuple[float, int]] = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if not 28 <= area <= 420:
                continue
            x, y = centres[label]
            temporal = 0.0 if prior is None else .2 * math.hypot(x - prior[0], y - prior[1])
            candidates.append((abs(area - math.pi * cfg.radius ** 2) * .02 + temporal, label))
        if not candidates:
            continue
        _, label = min(candidates)
        x, y = centres[label]
        component = labels == label
        ys[index] = 1.0 - y / (cfg.height - 1)
        rgbs[index] = frame[component].mean(axis=0)
        prior = (float(x), float(y))
    return ys, rgbs, np.isfinite(ys)


def colour_name(rgbs: np.ndarray) -> str:
    valid = np.isfinite(rgbs).all(axis=1)
    if not valid.any():
        return "unknown"
    rgb = np.median(rgbs[valid], axis=0)
    return "red" if rgb[0] > rgb[2] else "blue"


def distance_to_band(value: float, band: tuple[float, float]) -> float:
    return max(band[0] - value, 0.0, value - band[1])


def measure(frames: np.ndarray, row: dict[str, str], cfg: Any) -> dict[str, Any]:
    y, rgb, valid = detect_ball(frames, cfg)
    detected = int(valid.sum())
    result: dict[str, Any] = {
        "valid": bool(detected >= 58),
        "detected_frames": detected,
        "detected_colour_majority": colour_name(rgb),
        "gravity_true": float(row["gravity"]),
        "true_band": row["gravity_interval"],
        "input_colour": row["color_label"],
    }
    if detected < 6:
        result.update({"gravity_hat": None, "gravity_error": None, "E3_correct": False, "gravity_hat_e0": None, "gravity_error_e0": None, "E0_correct": False, "shortcut_band_distance": None, "route_label": "invalid"})
        return result
    time = (np.arange(len(y), dtype=np.float64) + 1.0) / cfg.simulation_fps
    boundary_y = float(json.loads(row["position_at_boundary"])[1])
    boundary_vy = float(json.loads(row["velocity_at_boundary"])[1])
    basis = -.5 * time ** 2
    target = y - boundary_y - boundary_vy * time
    gravity_hat_e0 = float(np.dot(basis[valid], target[valid]) / np.dot(basis[valid], basis[valid]))
    design_e3 = np.column_stack((np.ones_like(time), time, basis))
    beta_e3, *_ = np.linalg.lstsq(design_e3[valid], target[valid], rcond=None)
    gravity_hat = float(beta_e3[-1])
    true_band = cfg.gravity_bands[row["gravity_interval"]]
    shortcut_band = cfg.gravity_bands["low"] if row["color_label"] == "red" else cfg.gravity_bands["high"]
    gravity_error = abs(gravity_hat - float(row["gravity"]))
    gravity_error_e0 = abs(gravity_hat_e0 - float(row["gravity"]))
    shortcut_distance = distance_to_band(gravity_hat, shortcut_band)
    error_threshold = float(getattr(cfg, "gravity_accuracy_threshold", 0.002))
    e3_correct = bool(gravity_error < error_threshold)
    e0_correct = bool(gravity_error_e0 < error_threshold)
    if not result["valid"]:
        route = "invalid"
    elif cfg.gravity_bands["low"][1] < gravity_hat < cfg.gravity_bands["high"][0]:
        route = "middle_gap"
    elif gravity_hat < cfg.gravity_bands["low"][0] or gravity_hat > cfg.gravity_bands["high"][1]:
        route = "outside"
    elif shortcut_distance <= gravity_error:
        route = "shortcut_band"
    else:
        route = "physics_side"
    result.update({"gravity_hat": gravity_hat, "gravity_error": gravity_error, "E3_correct": e3_correct, "gravity_hat_e0": gravity_hat_e0, "gravity_error_e0": gravity_error_e0, "E0_correct": e0_correct, "shortcut_band_distance": shortcut_distance,
                   "route_label": route})
    return result


def finite(value: Any) -> Any:
    if isinstance(value, dict): return {key: finite(item) for key, item in value.items()}
    if isinstance(value, list): return [finite(item) for item in value]
    if isinstance(value, float): return value if math.isfinite(value) else None
    return value


def write_outcomes(path: Path, results: list[dict[str, Any]]) -> None:
    """Persist complete pairs atomically so interrupted jobs can resume."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(finite(results), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed-offset", type=int, default=17_000_000)
    parser.add_argument("--short-history", action="store_true")
    parser.add_argument("--observed-window", type=int, help="real frames before prediction_start; overrides --short-history")
    parser.add_argument("--pair-manifest", type=Path, help="recovery pair manifest from select_recovery_pairs.py")
    parser.add_argument("--max-block", type=int, help="evaluate only blocks 0..max-block inclusive")
    parser.add_argument("--baseline-only", action="store_true", help="Generate aligned/conflict baselines without layer replacements.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.observed_window is not None and not 1 <= args.observed_window <= 64:
        parser.error("--observed-window must be in [1, 64]")
    if args.max_block is not None and args.max_block < 0:
        parser.error("--max-block must be non-negative")
    # Worker outcomes are committed once per complete pair.  Permit an
    # interrupted shard to resume from its existing outcomes.json.
    if args.out.exists():
        unexpected = [path.name for path in args.out.iterdir() if path.name not in {"outcomes.json", "summary.json"}]
        if unexpected:
            raise FileExistsError(f"Refusing output with unexpected files: {args.out}: {unexpected}")
    args.out.mkdir(parents=True, exist_ok=True)
    data = yaml.safe_load(args.data_config.read_text())
    cfg = SimpleNamespace(
        width=int(data["render"]["width"]), height=int(data["render"]["height"]), frames=int(data["render"]["num_frames"]),
        radius=int(data["render"]["ball_radius_px"]), background=data["render"]["background_rgb"],
        simulation_fps=int(data["physics"]["simulation_fps"]), prediction_start=int(data["history"]["prediction_start"]),
        short_masked_prefix=int(data["history"]["short_masked_prefix_pixels"]), latent_frames=33,
        condition_latents=17, gravity_bands={"low": tuple(data["physics"]["low_gravity_range"]), "high": tuple(data["physics"]["high_gravity_range"])},
        gravity_accuracy_threshold=float(data.get("evaluation", {}).get("E3_threshold", 0.002)),
    )
    from sshv2.wan.config import StandardTrainingConfig
    from sshv2.wan.trainer import WanTrainingModule
    training = StandardTrainingConfig.from_file(args.training_config)
    training.model.dit.ckpt_file = args.checkpoint
    module = WanTrainingModule(dit_config=training.model.dit, vae_config=training.model.vae, no_encoding=False,
        num_condition_frames=training.model.num_condition_frames, num_inference_steps=args.steps,
        pipeline_type=training.model.pipe, pipeline_kwargs=training.model.pipe_kwargs)
    pipe = module.pipe
    pipe.to(args.device); pipe.load_models_to_device(("dit", "vae")); pipe.pre_encoded_(False)
    cfg.condition_latents = int(training.model.num_condition_frames)
    pairs = read_pairs(args.dataset_dir, 0 if args.pair_manifest is not None else args.limit)
    if args.pair_manifest is not None:
        requested = {item["pair_id"] for item in json.loads(args.pair_manifest.read_text(encoding="utf-8"))["pairs"]}
        pairs = [pair for pair in pairs if pair[0]["pair_id"] in requested]
        if len(pairs) != len(requested):
            raise ValueError(f"Requested {len(requested)} pairs, found {len(pairs)} in dataset")
    args.out.mkdir(parents=True, exist_ok=True)
    outcomes_path = args.out / "outcomes.json"
    if outcomes_path.exists():
        loaded = json.loads(outcomes_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(f"Expected a list in {outcomes_path}")
        results: list[dict[str, Any]] = loaded
    else:
        results = []
    completed_pair_ids = {str(row["pair_id"]) for row in results}
    for pair_index, (aligned, conflict) in enumerate(pairs, 1):
        if aligned["pair_id"] in completed_pair_ids:
            print(f"pairs={pair_index}/{len(pairs)} already_complete", flush=True)
            continue
        seed = int(aligned["base_seed"]) + args.seed_offset
        if args.observed_window is not None:
            original_prefix = cfg.short_masked_prefix
            cfg.short_masked_prefix = cfg.prediction_start - args.observed_window
            donor_condition = load_condition(aligned, args.dataset_dir, cfg, pipe.torch_dtype, args.device, True)
            receiver_condition = load_condition(conflict, args.dataset_dir, cfg, pipe.torch_dtype, args.device, True)
            cfg.short_masked_prefix = original_prefix
        else:
            donor_condition = load_condition(aligned, args.dataset_dir, cfg, pipe.torch_dtype, args.device, args.short_history)
            receiver_condition = load_condition(conflict, args.dataset_dir, cfg, pipe.torch_dtype, args.device, args.short_history)
        if args.baseline_only:
            aligned_frames = generate(pipe, donor_condition, cfg, args.steps, seed)
            conflict_frames = generate(pipe, receiver_condition, cfg, args.steps, seed)
            results.append({"pair_id": aligned["pair_id"], "block": None, "condition": "aligned_baseline", **measure(aligned_frames, aligned, cfg)})
            results.append({"pair_id": aligned["pair_id"], "block": None, "condition": "conflict_baseline", **measure(conflict_frames, conflict, cfg)})
            write_outcomes(outcomes_path, results)
            completed_pair_ids.add(aligned["pair_id"])
            print(f"pairs={pair_index}/{len(pairs)}", flush=True)
            continue
        donor = capture_all_blocks(pipe, donor_condition, cfg, args.steps, seed)
        conflict_frames = generate(pipe, receiver_condition, cfg, args.steps, seed)
        for block, residual in donor.items():
            if args.max_block is not None and block > args.max_block:
                continue
            edited = generate_with_replacement(pipe, block, residual, receiver_condition, cfg, args.steps, seed)
            results.append({"pair_id": aligned["pair_id"], "block": block, "condition": "replacement", **measure(edited, conflict, cfg)})
        results.append({"pair_id": aligned["pair_id"], "block": None, "condition": "aligned_baseline", **measure(generate(pipe, donor_condition, cfg, args.steps, seed), aligned, cfg)})
        results.append({"pair_id": aligned["pair_id"], "block": None, "condition": "conflict_baseline", **measure(conflict_frames, conflict, cfg)})
        del donor
        write_outcomes(outcomes_path, results)
        completed_pair_ids.add(aligned["pair_id"])
        print(f"pairs={pair_index}/{len(pairs)}", flush=True)
    summaries: dict[str, Any] = {}
    block_indices = [] if args.baseline_only else range(len(pipe.dit.blocks))
    if args.max_block is not None:
        block_indices = range(min(args.max_block + 1, len(pipe.dit.blocks)))
    for block in block_indices:
        rows = [row for row in results if row["block"] == block]
        valid_error = [row["gravity_error"] for row in rows if row["gravity_error"] is not None]
        valid_shortcut = [row["shortcut_band_distance"] for row in rows if row["shortcut_band_distance"] is not None]
        summaries[str(block)] = {"pairs": len(rows), "route_counts": dict(Counter(row["route_label"] for row in rows)),
            "E3_accuracy": float(np.mean([bool(row.get("E3_correct", False)) for row in rows])) if rows else None,
            "mean_gravity_error": float(np.mean(valid_error)) if valid_error else None,
            "mean_shortcut_band_distance": float(np.mean(valid_shortcut)) if valid_shortcut else None}
    write_outcomes(outcomes_path, results)
    baseline_rows = {label: [row for row in results if row["condition"] == label] for label in ("aligned_baseline", "conflict_baseline")}
    baseline_summary = {label: {"route_counts": dict(Counter(row["route_label"] for row in values)), "E3_accuracy": float(np.mean([bool(row.get("E3_correct", False)) for row in values])), "mean_gravity_error": float(np.mean([row["gravity_error"] for row in values if row["gravity_error"] is not None]))} for label, values in baseline_rows.items()}
    (args.out / "summary.json").write_text(json.dumps(finite({"model": "short" if args.short_history else "long", "pairs": len(pairs), "steps": args.steps, "baseline_only": args.baseline_only, "replacement": None if args.baseline_only else "aligned donor condition-token residual replaces conflict receiver condition-token residual after one DiT block", "baselines": baseline_summary, "blocks": summaries}), indent=2) + "\n")


if __name__ == "__main__":
    main()
