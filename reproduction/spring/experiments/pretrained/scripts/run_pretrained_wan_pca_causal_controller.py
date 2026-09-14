#!/usr/bin/env python3
"""Evaluate oracle PCA projections and a donor-free phase controller on held-out pairs."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch
import yaml

from sshv2.diffsynth.trainers.wan_video_train import WanTrainingModule
from sshv2.interpretability.condition_residual_patching import ResidualPatchController, sample_final_latents
from sshv2.interpretability.mean_direction_128_runtime import decode_latent, evaluate_future
from sshv2.interpretability.qualified_pair_bank_128_runtime import encode_long_condition, load_frames_npz, load_latent
from sshv2.simulation.spring_shortcuts_v1 import apply_short_history_mask_numpy, dataclass_config_from_dict
from sshv2.utils.spring_configs import SpringTrainingConfig


STEPS = 20
ARMS = ("full", "pc1", "top2", "top4", "phase_predicted_top4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--fast-bank-root", type=Path, required=True)
    parser.add_argument("--slow-bank-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows))
    temporary.replace(path)


def safe_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise AssertionError(f"Path escapes bank root: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def bank_root(args: argparse.Namespace, direction: str) -> Path:
    return args.fast_bank_root if direction == "fast" else args.slow_bank_root


def dot(left: list[torch.Tensor], right: list[torch.Tensor]) -> float:
    return float(sum(torch.sum(a.float() * b.float(), dtype=torch.float64) for a, b in zip(left, right, strict=True)))


def norm(value: list[torch.Tensor]) -> float:
    return math.sqrt(max(dot(value, value), 0.0))


def reconstruct(coefficients: np.ndarray, pcs: list[list[torch.Tensor]]) -> list[torch.Tensor]:
    return [
        sum((float(coefficients[k]) * pcs[k][step].float() for k in range(len(coefficients))),
            torch.zeros_like(pcs[0][step], dtype=torch.float32)).to(torch.bfloat16)
        for step in range(STEPS)
    ]


def oracle_projection(full: list[torch.Tensor], pcs: list[list[torch.Tensor]], rank: int) -> list[torch.Tensor]:
    coefficients = np.asarray([dot(full, pcs[index]) for index in range(rank)], dtype=float)
    projected = reconstruct(coefficients, pcs[:rank])
    projected_norm = norm(projected)
    full_norm = norm(full)
    if projected_norm <= 0 or full_norm <= 0:
        raise AssertionError("Invalid oracle projection norm")
    scale = full_norm / projected_norm
    return [(scale * value.float()).to(torch.bfloat16) for value in projected]


@dataclass
class AdditiveController:
    expected_steps: int
    num_condition_frames: int
    inject_layer: int
    direction_by_step: list[torch.Tensor]
    conflict_reference_by_step: list[torch.Tensor]
    arm: str
    step_index: int = 0
    hits: int = 0
    audits: list[dict[str, float]] = field(default_factory=list)

    def apply(self, x: torch.Tensor, *, layer: int, f: int, h: int, w: int) -> torch.Tensor:
        if layer != self.inject_layer:
            return x
        count = self.num_condition_frames * h * w
        reference = self.conflict_reference_by_step[self.step_index].to(x.device, x.dtype)
        live = x[:, :count]
        replay = float((live.float() - reference.float()).abs().max())
        if self.step_index == 0 and replay != 0.0:
            raise AssertionError(f"Conflict receiver did not exactly replay: {replay}")
        edit = self.direction_by_step[self.step_index].to(x.device, x.dtype)
        if tuple(edit.shape) != tuple(live.shape):
            raise AssertionError("Edit shape differs from condition prefix")
        future = x[:, count:].clone()
        patched = x.clone()
        patched[:, :count] = live + edit
        if not torch.equal(patched[:, count:], future):
            raise AssertionError("Future suffix changed directly")
        self.audits.append({"replay_max_abs": replay, "edit_max_abs": float(edit.float().abs().max())})
        self.hits += 1
        return patched

    def finish_model_call(self) -> None:
        self.step_index += 1

    def finish_run(self) -> None:
        if self.step_index != self.expected_steps or self.hits != self.expected_steps:
            raise AssertionError("Incomplete additive intervention")


def load_model(args: argparse.Namespace):
    cfg = dataclass_config_from_dict(yaml.safe_load(args.data_config.read_text()))
    train = SpringTrainingConfig.from_file(args.config)
    train.model.dit.ckpt_file = args.checkpoint
    module = WanTrainingModule(
        dit_config=train.model.dit, vae_config=train.model.vae, no_encoding=False,
        num_condition_frames=train.model.num_condition_frames, num_inference_steps=STEPS,
        pipeline_type=train.model.pipe, pipeline_kwargs=train.model.pipe_kwargs,
    )
    pipe = module.pipe
    pipe.to(args.device)
    pipe.load_models_to_device(("dit", "vae"))
    pipe.pre_encoded_(False)
    return pipe, cfg, int(train.model.num_condition_frames)


def sample(pipe: Any, condition: torch.Tensor, cfg: Any, ncond: int, seed: int, controller: Any):
    return sample_final_latents(
        pipe=pipe, condition_latents=condition, num_frames=cfg.render.num_frames,
        height=cfg.render.height, width=cfg.render.width, num_condition_frames=ncond,
        num_inference_steps=STEPS, seed=seed, controller=controller,
        sigma_shift=5.0, denoising_strength=1.0,
    )


def metric(pipe: Any, latent: torch.Tensor, cfg: Any, metadata: dict, conflict_color: str) -> dict:
    frames = decode_latent(pipe, latent, device=str(pipe.device), expected_frames=cfg.render.num_frames)
    value = evaluate_future(frames[cfg.prediction_start:], metadata=metadata, color_label_for_route=conflict_color, cfg=cfg)
    return {key: item for key, item in value.items() if not isinstance(item, float) or math.isfinite(item)}


def fit_phase_models(coordinates: list[dict]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    models = {}
    r2 = {}
    for direction in ("fast", "slow"):
        fit = [row for row in coordinates if row["direction"] == direction and row["split"] == "fit"]
        held = [row for row in coordinates if row["direction"] == direction and row["split"] == "heldout"]
        x_fit = np.asarray([[1.0, math.cos(row["theta_boundary"]), math.sin(row["theta_boundary"])] for row in fit])
        y_fit = np.asarray([row["z"] for row in fit], dtype=float)
        beta = np.linalg.lstsq(x_fit, y_fit, rcond=None)[0]
        x_held = np.asarray([[1.0, math.cos(row["theta_boundary"]), math.sin(row["theta_boundary"])] for row in held])
        y_held = np.asarray([row["z"] for row in held], dtype=float)
        prediction = x_held @ beta
        denominator = np.square(y_held - y_held.mean(axis=0, keepdims=True)).sum()
        models[direction] = beta
        r2[direction] = float(1.0 - np.square(y_held - prediction).sum() / denominator)
    return models, r2


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "summary.json"
    if args.resume and summary_path.is_file():
        print(summary_path)
        return
    split = json.loads((args.geometry_root / "split_manifest.json").read_text())["rows"]
    coordinates = [json.loads(line) for line in (args.geometry_root / "coordinates.jsonl").read_text().splitlines() if line]
    phase_models, coordinate_r2 = fit_phase_models(coordinates)
    basis = torch.load(args.geometry_root / "pca_basis.pt", map_location="cpu", weights_only=False)
    pcs = basis["pcs"]
    if len(pcs) != 4:
        raise AssertionError("Expected frozen top-4 basis")
    heldout = [row for row in split if row["split"] == "heldout"]
    coordinate_index = {(row["direction"], row["pair_id"]): row for row in coordinates}
    records_dir = args.out / "records"
    records_dir.mkdir(exist_ok=True)
    pipe, cfg, ncond = load_model(args)
    all_runs = []
    for index, row in enumerate(heldout, 1):
        record_path = records_dir / f"{row['direction']}_{row['pair_id']}.json"
        if args.resume and record_path.is_file():
            all_runs.extend(json.loads(record_path.read_text())["runs"])
            continue
        root = bank_root(args, row["direction"])
        conflict_frames = load_frames_npz(safe_path(root, row["conflict_input_frames_npz"]))
        conflict_frames = apply_short_history_mask_numpy(conflict_frames, cfg)
        condition = encode_long_condition(pipe, conflict_frames, prediction_start=cfg.prediction_start, num_condition_frames=ncond)
        metadata = json.loads(safe_path(root, row["conflict_metadata"]).read_text())
        seed = int(row["generation_seed"])
        reference_controller = ResidualPatchController(
            expected_steps=STEPS, num_condition_frames=ncond,
            record_condition_layers=(args.layer,),
        )
        conflict_latent = sample(pipe, condition, cfg, ncond, seed, reference_controller)
        frozen_conflict = load_latent(safe_path(root, row["conflict_baseline_latent"]))
        if float((conflict_latent.detach().cpu().float() - frozen_conflict.float()).abs().max()) != 0.0:
            raise AssertionError("Conflict baseline replay failed")
        references = reference_controller.bank.condition[args.layer]
        payload = torch.load(
            args.geometry_root / "differences" / f"{row['direction']}_{row['pair_id']}.pt",
            map_location="cpu", weights_only=False,
        )
        full = payload["difference"]
        directions = {
            "full": full,
            "pc1": oracle_projection(full, pcs, 1),
            "top2": oracle_projection(full, pcs, 2),
            "top4": oracle_projection(full, pcs, 4),
        }
        theta = float(row["theta_boundary"])
        feature = np.asarray([1.0, math.cos(theta), math.sin(theta)])
        predicted_z = feature @ phase_models[row["direction"]]
        directions["phase_predicted_top4"] = reconstruct(predicted_z, pcs)
        aligned_metrics = row["aligned_pixel_metrics"]
        conflict_metrics = row["conflict_pixel_metrics"]
        omega_a, omega_c = float(aligned_metrics["omega_pixel"]), float(conflict_metrics["omega_pixel"])
        conflict_color = "red" if row["direction"] == "fast" else "blue"
        runs = []
        for arm in ARMS:
            controller = AdditiveController(
                expected_steps=STEPS, num_condition_frames=ncond, inject_layer=args.layer,
                direction_by_step=directions[arm], conflict_reference_by_step=references,
                arm=arm,
            )
            latent = sample(pipe, condition, cfg, ncond, seed, controller)
            controller.finish_run()
            result = metric(pipe, latent, cfg, metadata, conflict_color)
            omega = result.get("omega_pixel")
            recovery = None if omega is None else (float(omega) - omega_c) / (omega_a - omega_c)
            runs.append({
                "trajectory_id": row["trajectory_id"], "pair_id": row["pair_id"],
                "direction": row["direction"], "split": "heldout", "arm": arm,
                "valid": bool(result.get("valid")), "R": recovery,
                "omega_aligned": omega_a, "omega_conflict": omega_c, **result,
                "future_suffix_directly_unchanged": True,
                "heldout_matched_difference_read": arm != "phase_predicted_top4",
            })
        atomic_json(record_path, {"predicted_z": predicted_z.tolist(), "runs": runs})
        all_runs.extend(runs)
        atomic_jsonl(args.out / "per_run.partial.jsonl", all_runs)
        print(f"[{index}/{len(heldout)}] {row['direction']} {row['trajectory_id']}", flush=True)
    atomic_jsonl(args.out / "per_run.jsonl", all_runs)
    groups = {}
    for arm in ARMS:
        for direction in ("fast", "slow", "pooled"):
            rows = [run for run in all_runs if run["arm"] == arm and (direction == "pooled" or run["direction"] == direction)]
            recoveries = [float(run["R"]) for run in rows if run["valid"] and run["R"] is not None and math.isfinite(float(run["R"]))]
            groups[f"{arm}/{direction}"] = {
                "n": len(rows), "valid_rate": sum(run["valid"] for run in rows) / len(rows),
                "median_R": median(recoveries) if recoveries else None,
                "strong_R_rate": sum(run["valid"] and run["R"] is not None and .75 < float(run["R"]) < 1.25 for run in rows) / len(rows),
                "route_counts": dict(Counter(str(run.get("route_label")) for run in rows if run["valid"])),
            }
    summary = {
        "status": "PASS", "checkpoint": str(args.checkpoint), "layer": args.layer,
        "fit_n": 128, "heldout_n": len(heldout),
        "heldout_coordinate_R2": coordinate_r2,
        "phase_features": "separate direction-specific [1, cos(theta*), sin(theta*)] maps",
        "phase_controller_reads_heldout_donor": False,
        "oracle_projection_uses_heldout_difference_and_norm": True,
        "groups": groups,
    }
    atomic_json(summary_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
