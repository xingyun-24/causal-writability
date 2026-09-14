"""Stage-1 direct donor-to-conflict scan for frozen Short or Long endpoint pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch
import yaml

from sshv2.diffsynth.trainers.wan_video_train import WanTrainingModule
from sshv2.interpretability.condition_residual_patching import ResidualPatchController, sample_final_latents
from sshv2.interpretability.positive_condition_residual_all_layer_scan_128 import AuditedSingleSiteResidualPatchController
from sshv2.interpretability.mean_direction_128_runtime import decode_latent, evaluate_future
from sshv2.simulation.spring_shortcuts_v1 import apply_short_history_mask_tensor, dataclass_config_from_dict, load_video
from sshv2.utils.spring_configs import SpringTrainingConfig

SITES = (11, 12, 13, 14, 15, 16)
STEPS = 20
MIN_DENOMINATOR = 2.0


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-config", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--history", choices=("short", "long"), default="short")
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument(
        "--expected-pairs", type=int, default=128,
        help="Expected input-pair count; use 0 for a checkpoint-local variable-size strict bank.",
    )
    p.add_argument(
        "--sites",
        default=",".join(str(site) for site in SITES),
        help="Comma-separated zero-based post-block sites. Default is the preregistered B11..B16 scan.",
    )
    return p.parse_args()


def finite(x: float | None) -> bool:
    return x is not None and math.isfinite(float(x))


def encode_condition(pipe: Any, frames: np.ndarray, cfg: Any, ncond: int, history: str) -> torch.Tensor:
    x = torch.from_numpy(frames.copy()).permute(3, 0, 1, 2).float().div(127.5).sub(1).unsqueeze(0)
    x = x.to(pipe.device, dtype=pipe.torch_dtype)[:, :, :cfg.prediction_start].contiguous()
    if history == "short":
        x = apply_short_history_mask_tensor(x, cfg)
    encoded = pipe.vae.encode(x, device=pipe.device, tiled=False)[:, :, :ncond]
    if tuple(encoded.shape) != (1, 16, ncond, 16, 16):
        raise AssertionError(f"Unexpected {history} condition shape {tuple(encoded.shape)}")
    return encoded.to(pipe.device, dtype=pipe.torch_dtype)


def sample(pipe: Any, condition: torch.Tensor, cfg: Any, ncond: int, seed: int, controller: Any) -> torch.Tensor:
    return sample_final_latents(pipe=pipe, condition_latents=condition, num_frames=cfg.render.num_frames,
        height=cfg.render.height, width=cfg.render.width, num_condition_frames=ncond,
        num_inference_steps=STEPS, seed=seed, controller=controller, sigma_shift=5.0, denoising_strength=1.0)


def score(pipe: Any, latent: torch.Tensor, cfg: Any, metadata: dict[str, Any], colour: str) -> dict[str, Any]:
    all_frames = decode_latent(pipe, latent, device=str(pipe.device), expected_frames=cfg.render.num_frames)
    result = evaluate_future(all_frames[cfg.prediction_start:], metadata=metadata, color_label_for_route=colour, cfg=cfg)
    return {key: value for key, value in result.items() if not isinstance(value, float) or math.isfinite(value)}


def row_summary(rows: list[dict[str, Any]], site: int) -> dict[str, Any]:
    at_site = [r for r in rows if r["site_index"] == site]
    qualified = [r for r in at_site if r["qualified"]]
    result: dict[str, Any] = {"site_index": site, "site_label": f"after-B{site}", "n_total": len(at_site), "n_qualified": len(qualified)}
    for direction in ("fast-target", "slow-target", "pooled"):
        group = qualified if direction == "pooled" else [r for r in qualified if r["target_direction"] == direction]
        prefix = direction.replace("-", "_")
        values = [float(r["recovery"]) for r in group]
        patched_valid = [bool(r["patched"].get("valid", False)) for r in (at_site if direction == "pooled" else [r for r in at_site if r["target_direction"] == direction])]
        result[f"{prefix}_n_qualified"] = len(group)
        result[f"{prefix}_median_recovery"] = float(median(values)) if values else None
        result[f"{prefix}_strong_rate"] = sum(v > 0.8 for v in values) / len(values) if values else None
        result[f"{prefix}_validity"] = sum(patched_valid) / len(patched_valid) if patched_valid else None
    return result


def main() -> None:
    a = args()
    sites = tuple(int(value) for value in a.sites.split(",") if value.strip())
    if not sites or any(site < -1 or site > 29 for site in sites):
        raise ValueError(f"Invalid sites: {a.sites!r}")
    if a.out.exists() and any(a.out.iterdir()) and not a.resume:
        raise FileExistsError(f"Refusing to overwrite {a.out}")
    a.out.mkdir(parents=True, exist_ok=True)
    for p in (a.config, a.checkpoint, a.data_config, a.pairs, a.dataset / "metadata.csv"):
        if not p.is_file(): raise FileNotFoundError(p)
    with a.pairs.open(newline="", encoding="utf-8") as h: pairs = list(csv.DictReader(h))
    if a.expected_pairs > 0 and len(pairs) != a.expected_pairs:
        raise AssertionError(f"Expected frozen {a.expected_pairs} pairs, got {len(pairs)}")
    if not pairs:
        raise AssertionError("The matched-pair bank is empty")
    if a.limit is not None: pairs = pairs[:a.limit]
    cfg = dataclass_config_from_dict(yaml.safe_load(a.data_config.read_text()))
    train = SpringTrainingConfig.from_file(a.config); train.model.dit.ckpt_file = a.checkpoint
    model = WanTrainingModule(dit_config=train.model.dit, vae_config=train.model.vae, no_encoding=False,
        num_condition_frames=train.model.num_condition_frames, num_inference_steps=STEPS,
        pipeline_type=train.model.pipe, pipeline_kwargs=train.model.pipe_kwargs)
    pipe = model.pipe; pipe.to(a.device); pipe.load_models_to_device(("dit", "vae")); pipe.pre_encoded_(False)
    ncond = train.model.num_condition_frames
    meta = {p.stem: json.loads(p.read_text()) for p in a.dataset.glob("*.json")}
    done_path = a.out / "per_pair.jsonl"
    completed: list[dict[str, Any]] = []
    if a.resume and done_path.is_file(): completed = [json.loads(x) for x in done_path.read_text().splitlines() if x]
    done = {r["trajectory_id"] for r in completed}
    for index, pair in enumerate(pairs, 1):
        if pair["trajectory_id"] in done: continue
        aligned_frames = load_video(a.dataset / pair["aligned_video"], expected_frames=cfg.render.num_frames)
        conflict_frames = load_video(a.dataset / pair["conflict_video"], expected_frames=cfg.render.num_frames)
        aligned_meta, conflict_meta = meta[Path(pair["aligned_metadata"]).stem], meta[Path(pair["conflict_metadata"]).stem]
        aligned_condition = encode_condition(pipe, aligned_frames, cfg, ncond, a.history)
        conflict_condition = encode_condition(pipe, conflict_frames, cfg, ncond, a.history)
        seed = int(pair["generation_seed"])
        donor_ctl = ResidualPatchController(expected_steps=STEPS, num_condition_frames=ncond, record_condition_layers=sites)
        receiver_ctl = ResidualPatchController(expected_steps=STEPS, num_condition_frames=ncond, record_condition_layers=sites)
        donor_latent = sample(pipe, aligned_condition, cfg, ncond, seed, donor_ctl)
        receiver_latent = sample(pipe, conflict_condition, cfg, ncond, seed, receiver_ctl)
        donor = score(pipe, donor_latent, cfg, aligned_meta, "blue" if pair["true_band"] == "fast" else "red")
        receiver = score(pipe, receiver_latent, cfg, conflict_meta, "red" if pair["true_band"] == "fast" else "blue")
        denominator = (float(donor["omega_pixel"]) - float(receiver["omega_pixel"])) if finite(donor.get("omega_pixel")) and finite(receiver.get("omega_pixel")) else None
        base = {"tag": a.tag, "pair_index": int(pair["pair_index"]), "trajectory_id": pair["trajectory_id"], "pair_id": pair["pair_id"], "split": pair["split"], "target_direction": pair["target_direction"], "generation_seed": seed, "true_band": pair["true_band"], "aligned": donor, "conflict": receiver, "denominator": denominator}
        records=[]
        for site in sites:
            ctl = AuditedSingleSiteResidualPatchController(expected_steps=STEPS, num_condition_frames=ncond, inject_bank=donor_ctl.bank, inject_site=site)
            patched_latent = sample(pipe, conflict_condition, cfg, ncond, seed, ctl)
            patched = score(pipe, patched_latent, cfg, conflict_meta, "red" if pair["true_band"] == "fast" else "blue")
            recovery = (float(patched["omega_pixel"]) - float(receiver["omega_pixel"])) / denominator if denominator is not None and finite(patched.get("omega_pixel")) else None
            qualified = bool(donor.get("valid")) and bool(receiver.get("valid")) and finite(denominator) and abs(float(denominator)) >= MIN_DENOMINATOR and finite(patched.get("omega_pixel"))
            records.append({**base, "site_index":site, "patched":patched, "recovery":recovery, "qualified":qualified, "audit":ctl.audit_dict()})
        completed.extend(records)
        tmp=done_path.with_suffix('.tmp'); tmp.write_text(''.join(json.dumps(r, ensure_ascii=False)+"\n" for r in completed)); tmp.replace(done_path)
        print(f"[{index}/{len(pairs)}] {pair['trajectory_id']} done", flush=True)
    if a.limit is None and len(completed) != len(pairs) * len(sites):
        raise AssertionError(f"Incomplete scan: {len(completed)} rows")
    summary=[row_summary(completed, site) for site in sites]
    with (a.out / "layer_scan.json").open("w") as h: json.dump({"tag":a.tag,"history":a.history,"checkpoint":str(a.checkpoint),"checkpoint_sha256":hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),"pairs":str(a.pairs),"pairs_sha256":hashlib.sha256(a.pairs.read_bytes()).hexdigest(),"n_pairs":len(pairs),"steps":STEPS,"sites":list(sites),"min_qualified_denominator":MIN_DENOMINATOR,"summary":summary},h,indent=2)
    with (a.out / "layer_scan.csv").open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    print(a.out / "layer_scan.csv")


if __name__ == "__main__":
    main()
