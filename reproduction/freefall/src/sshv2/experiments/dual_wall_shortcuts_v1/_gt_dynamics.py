"""Private GT-perception dynamics profiles for ``dual_wall_shortcuts_v1``.

This module deliberately has no video/RGB loader.  The predictor consumes an
explicit :class:`DynamicsInput` built from simulator poses only, so that the
appearance firewall is structural rather than a convention in a caller.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

SIDES = ("left", "right")
PREFIX, FUTURE = 5, 44


def wrap(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def angle(v: np.ndarray) -> float:
    return math.degrees(math.atan2(float(v[1]), float(v[0])))


def angle_error(a: float, b: float) -> float:
    return abs(wrap(a - b))


def wall_frame(event: dict) -> np.ndarray:
    """Columns are outward normal and tangent, in global x/y coordinates."""
    normal = np.asarray(event["normal"], dtype=np.float32)
    phi = math.radians(float(event["phi_wall"]))
    tangent = np.asarray((math.cos(phi), math.sin(phi)), dtype=np.float32)
    return np.stack((normal, tangent), axis=1)


@dataclass(frozen=True)
class DynamicsInput:
    """Only legal inputs to the small shared dynamics predictor.

    ``features`` always has 18 values: ten trajectory values, six geometry
    values, and two appearance values (all zero under the firewall).  In
    local mode the geometry values are zero by default; an explicitly enabled
    selector audit may place only ``(cos(phi), sin(phi))`` in its first two
    slots.  There is intentionally no RGB, slot, filename, sample index,
    wall-family, or screen-side field.
    """

    features: torch.Tensor
    mode: str
    appearance: bool
    global_pose: bool

    def __post_init__(self) -> None:
        if self.features.shape != (18,):
            raise ValueError(f"expected exactly 18 dynamics values, got {tuple(self.features.shape)}")


def make_input(event: dict, *, local: bool, appearance: bool, global_pose: bool = False,
               global_pose_phi: float | None = None) -> DynamicsInput:
    if global_pose and not local:
        raise ValueError("global-pose selector audit is defined only for local-coordinate inputs")
    p = np.asarray(event["positions"], dtype=np.float32)
    c = np.asarray(event["contact_point"], dtype=np.float32)
    frame = wall_frame(event)
    if local:
        trajectory = ((p[:PREFIX] - c) @ frame).reshape(-1)
        geometry = np.zeros(6, dtype=np.float32)
        if global_pose:
            # The optional override is an evaluator-only counterfactual cue.
            # It must never alter the actual local coordinate transform.
            phi = math.radians(float(event["phi_wall"] if global_pose_phi is None else global_pose_phi))
            geometry[:2] = (math.cos(phi), math.sin(phi))
    else:
        trajectory = p[:PREFIX].reshape(-1)
        geometry = np.concatenate((c, frame[:, 0], frame[:, 1])).astype(np.float32)
    color = np.zeros(2, dtype=np.float32)
    if appearance:
        color[0 if event["color"] == "red" else 1] = 1.0
    return DynamicsInput(torch.from_numpy(np.concatenate((trajectory, geometry, color))), "local" if local else "global", appearance, global_pose)


def make_target(event: dict, *, local: bool) -> torch.Tensor:
    p = np.asarray(event["positions"], dtype=np.float32)[PREFIX:]
    if local:
        c = np.asarray(event["contact_point"], dtype=np.float32)
        p = (p - c) @ wall_frame(event)
    return torch.from_numpy(p.reshape(-1).copy())


def to_global(event: dict, values: np.ndarray, *, local: bool) -> np.ndarray:
    p = np.asarray(values, dtype=np.float32).reshape(FUTURE, 2)
    if local:
        p = p @ wall_frame(event).T + np.asarray(event["contact_point"], dtype=np.float32)
    prefix = np.asarray(event["positions"], dtype=np.float32)[:PREFIX]
    return np.concatenate((prefix, p), axis=0)


class PoseDataset(Dataset):
    """One role-free interaction per item; rows are deliberately shuffled."""

    def __init__(self, root: Path, *, local: bool, appearance: bool, global_pose: bool = False):
        self.root, self.local, self.appearance, self.global_pose = Path(root), bool(local), bool(appearance), bool(global_pose)
        rows = list(csv.DictReader((self.root / "metadata.csv").open()))
        self.items: list[tuple[dict, str]] = []
        for row in rows:
            meta = json.loads((self.root / row["metadata"]).read_text())
            self.items.extend((meta, side) for side in SIDES)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        meta, side = self.items[index]
        event = meta["interactions"][side]
        x = make_input(event, local=self.local, appearance=self.appearance, global_pose=self.global_pose)
        return {"x": x.features, "y": make_target(event, local=self.local)}


class SharedTemporalPredictor(nn.Module):
    """Same four-layer residual MLP for every quadrant and every interaction."""

    def __init__(self, width: int = 256, depth: int = 3):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(18, width), nn.SiLU())
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
            for _ in range(depth)
        ])
        self.output = nn.Linear(width, FUTURE * 2)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input(x)
        for block in self.blocks:
            h = h + block(h)
        return self.output(h)


def records(root: Path) -> list[tuple[dict, dict]]:
    root = Path(root)
    out = []
    for row in csv.DictReader((root / "metadata.csv").open()):
        out.append((row, json.loads((root / row["metadata"]).read_text())))
    return out


def opposite_events(eval_root: Path) -> dict[str, dict[tuple[str, str], dict]]:
    result: dict[str, dict[tuple[str, str], dict]] = {}
    for _, meta in records(eval_root):
        bucket = result.setdefault(meta["quartet_id"], {})
        for event in meta["interactions"].values():
            bucket[(event["wall_instance"], event["state"])] = event
    return result


@torch.no_grad()
def predict_record(model: nn.Module, meta: dict, *, local: bool, appearance: bool, global_pose: bool = False,
                   device: str, local_noise_px: float = 0.0, noise_seed: int | None = None) -> dict[str, np.ndarray]:
    out = {}
    for side in SIDES:
        event = meta["interactions"][side]
        x = make_input(event, local=local, appearance=appearance, global_pose=global_pose).features[None]
        if local_noise_px:
            # Coordinates are normalized to the 128px image domain.  A
            # quartet/wall keyed generator makes every ID/G/C/GC counterpart
            # receive the same five-frame perturbation for this wall instance.
            seed = int(noise_seed if noise_seed is not None else 0)
            seed += 104729 * (0 if event["wall_instance"] == "g-" else 1)
            generator = torch.Generator().manual_seed(seed)
            x[:,:10] += torch.randn((1, 10), generator=generator) * (float(local_noise_px) / 128.0)
        x = x.to(device)
        out[side] = to_global(event, model(x).float().cpu().numpy()[0], local=local)
    return out


def trajectory_metrics(prediction: dict[str, np.ndarray], meta: dict, opposite: dict[tuple[str, str], dict]) -> list[dict]:
    out = []
    for side in SIDES:
        event = meta["interactions"][side]
        pred = prediction[side]
        target = np.asarray(event["positions"], dtype=np.float32)
        other = np.asarray(opposite[(event["wall_instance"], "B" if event["state"] == "A" else "A")]["positions"], dtype=np.float32)
        pv = np.median(np.diff(pred[8:], axis=0) * 15.0, axis=0)
        gt = np.asarray(event["velocity_out"], dtype=np.float32)
        e = np.linalg.norm(pred[PREFIX:] - target[PREFIX:], axis=1)
        w = np.linalg.norm(pred[PREFIX:] - other[PREFIX:], axis=1)
        d_correct, d_wrong = float(e.mean()), float(w.mean())
        out.append({
            "side": side,
            "wall_instance": event["wall_instance"],
            "state": event["state"],
            "color": event["color"],
            "outgoing_angle_mae": angle_error(angle(pv), angle(gt)),
            "predicted_outgoing_angle": angle(pv),
            "correct_outgoing_angle": angle(gt),
            "opposite_outgoing_angle": angle(np.asarray(opposite[(event["wall_instance"], "B" if event["state"] == "A" else "A")]["velocity_out"], dtype=np.float32)),
            "trajectory_rmse": float(np.sqrt(np.mean(e ** 2))),
            "endpoint_error": float(np.linalg.norm(pred[-1] - target[-1])),
            "d_correct": d_correct,
            "d_wrong": d_wrong,
            "correct_follow": float(d_correct < d_wrong),
            "shortcut_follow": float(d_wrong < d_correct),
        })
    return out


def aggregate(rows: Iterable[dict]) -> dict[str, float]:
    rows = list(rows)
    keys = ("outgoing_angle_mae", "trajectory_rmse", "endpoint_error", "d_correct", "d_wrong", "correct_follow", "shortcut_follow")
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def audit_firewall(local: bool, appearance: bool, global_pose: bool = False) -> dict:
    probe = {
        "positions": [[.1, .2]] * 49,
        "contact_point": [.2, .3], "normal": [0., 1.], "phi_wall": 0., "color": "red",
    }
    inp = make_input(probe, local=local, appearance=appearance, global_pose=global_pose)
    return {
        "input_dim": int(inp.features.numel()),
        "contains_rgb": False,
        "contains_slot_or_filename": False,
        "contains_global_geometry": bool(not local or global_pose),
        "global_pose_encoding": "cos_phi,sin_phi" if global_pose else None,
        "appearance_values_nonzero": bool(inp.features[-2:].abs().sum() > 0),
        "firewall_pass": bool((not appearance) == bool(inp.features[-2:].abs().sum() == 0)),
    }
