"""Private GT-route predictor profiles for ``support_release_v1``.

The target is always the same pixel displacement from prefix frame four.  The
only experimental changes are which causal-state/cue fields are exposed and
whether their *training-split-only* normalization is balanced.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

PREFIX, FUTURE, SCALE, STD_EPS_PX = 5, 44, 128.0, 1e-4


@dataclass(frozen=True)
class InputSpec:
    centered: bool = True
    appearance: bool = False
    global_pos: bool = False
    positions: bool = True
    velocity: bool = False
    # ``balanced_standardization`` is the original composite switch.  The
    # three switches below split it into independently auditable blocks while
    # retaining compatibility with all previously saved checkpoints.
    balanced_standardization: bool = False
    trajectory_standardization: bool = False
    signed_colour: bool = False
    global_pos_standardization: bool = False

    @property
    def standardized_trajectory(self):
        return self.balanced_standardization or self.trajectory_standardization

    @property
    def signed_colour_code(self):
        return self.balanced_standardization or self.signed_colour

    @property
    def standardized_global_pos(self):
        return self.balanced_standardization or self.global_pos_standardization

    @property
    def standardized_velocity(self):
        # Velocity was first introduced only in the balanced controls.  Keep
        # that exact convention rather than silently changing old V runs.
        return self.balanced_standardization

    @property
    def needs_statistics(self):
        return self.standardized_trajectory or self.standardized_velocity or self.standardized_global_pos

    @property
    def dim(self):
        return (10 * int(self.positions) + 2 * int(self.velocity)
                + (int(self.appearance) if self.signed_colour_code else 2 * int(self.appearance))
                + int(self.global_pos))


def app_center(event):
    """Static nominal V-centre, never a moving-support centroid."""
    return np.asarray(event["support_base"], np.float32)


def positions(event):
    return np.asarray(event["positions"], np.float32)


def velocity_frame4(event):
    """Exact simulator velocity at frame 4, expressed in pixel/frame."""
    return np.asarray(event["velocities"], np.float32)[PREFIX - 1] * (SCALE / 15.0)


def metas(root):
    root = Path(root)
    return [json.loads((root / row["metadata"]).read_text()) for row in csv.DictReader((root / "metadata.csv").open())]


def compute_stats(root, eps_px=STD_EPS_PX):
    """Compute and return only train-manifest statistics in pixel units."""
    records = metas(root)
    q = np.stack([(positions(m["event"])[:PREFIX] - app_center(m["event"])).reshape(-1) * SCALE for m in records])
    v = np.stack([velocity_frame4(m["event"]) for m in records])
    g = np.asarray([app_center(m["event"])[0] - .5 for m in records], np.float32)[:, None]

    def values(x):
        std = x.std(axis=0)
        return {"mean": x.mean(axis=0).astype(np.float32).tolist(), "std": std.astype(np.float32).tolist(),
                "std_clipped": np.maximum(std, eps_px).astype(np.float32).tolist(),
                "min": x.min(axis=0).astype(np.float32).tolist(), "max": x.max(axis=0).astype(np.float32).tolist()}

    return {"source": str(Path(root)), "n": len(records), "eps_px": float(eps_px),
            "q_px": values(q), "v_px_per_frame": values(v), "g_pos": values(g)}


def _standardize(x, section, stats):
    mean = np.asarray(stats[section]["mean"], np.float32)
    std = np.asarray(stats[section]["std_clipped"], np.float32)
    return (x - mean) / std


def input_slices(spec: InputSpec):
    """Named first-layer column blocks in their frozen concatenation order."""
    start, out = 0, {}
    for name, width in (("trajectory", 10 * int(spec.positions)), ("velocity", 2 * int(spec.velocity)),
                        ("colour", (1 if spec.signed_colour_code else 2) * int(spec.appearance)),
                        ("global_pos", int(spec.global_pos))):
        if width:
            out[name] = slice(start, start + width)
            start += width
    return out


def features(meta, spec: InputSpec, stats=None, color_override=None, pos_override=None):
    """Return the allowed predictor input, in the frozen block order."""
    event = meta["event"]
    centre = app_center(event)
    pieces = []
    if spec.positions:
        q_px = (positions(event)[:PREFIX] - (centre if spec.centered else 0)).reshape(-1) * SCALE
        pieces.append(_standardize(q_px, "q_px", stats) if spec.standardized_trajectory else q_px / SCALE)
    if spec.velocity:
        v = velocity_frame4(event)
        pieces.append(_standardize(v, "v_px_per_frame", stats) if spec.standardized_velocity else v / SCALE)
    if spec.appearance:
        color = event["color"] if color_override is None else color_override
        if spec.signed_colour_code:
            pieces.append(np.asarray((1.0 if color == "red" else -1.0,), np.float32))
        else:
            pieces.append(np.asarray((1.0, 0.0) if color == "red" else (0.0, 1.0), np.float32))
    if spec.global_pos:
        x = np.asarray((float(centre[0] if pos_override is None else pos_override) - .5,), np.float32)
        pieces.append(_standardize(x, "g_pos", stats) if spec.standardized_global_pos else x)
    return torch.from_numpy(np.concatenate(pieces).astype(np.float32))


def target(meta):
    p = positions(meta["event"])
    return torch.from_numpy(((p[PREFIX:] - p[PREFIX - 1]) * SCALE).reshape(-1))


def decode(meta, y):
    p = positions(meta["event"])
    return np.concatenate((p[:PREFIX], p[PREFIX - 1] + np.asarray(y).reshape(FUTURE, 2) / SCALE))


class DatasetGT(Dataset):
    def __init__(self, root, spec, noise_px=0.0, stats=None):
        self.root, self.spec, self.noise, self.stats = Path(root), spec, float(noise_px), stats
        self.items = metas(root)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        x = features(item, self.spec, self.stats)
        # Noise is specified in pixel units, then converted to the active
        # representation.  This legacy Dataset path is retained for callers;
        # the trainer uses the same distribution directly on GPU.
        if self.noise:
            if not self.spec.positions:
                raise ValueError("position noise requires position inputs")
            if self.spec.standardized_trajectory:
                scale = torch.as_tensor(self.stats["q_px"]["std_clipped"], dtype=x.dtype)
                x[:10] += torch.randn(10) * self.noise / scale
            else:
                x[:10] += torch.randn(10) * self.noise / SCALE
        return {"x": x, "y": target(item)}


class Predictor(nn.Module):
    def __init__(self, dim, width=256, depth=3):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(dim, width), nn.SiLU())
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
                                     for _ in range(depth)])
        self.out = nn.Linear(width, FUTURE * 2)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        h = self.inp(x)
        for block in self.blocks:
            h = h + block(h)
        return self.out(h)


def index_opposite(root):
    return {(m["octet_id"], m["geometry"], m["color"], m["state"]): m for m in metas(root)}
