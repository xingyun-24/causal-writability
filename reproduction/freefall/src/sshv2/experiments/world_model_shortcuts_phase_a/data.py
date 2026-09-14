"""Phase-A raw-world-model shortcut benchmark generators.

This module intentionally lives beside, rather than modifies, the frozen
``dual_wall_shortcuts_v1`` and ``support_release_v1`` generators.  It supplies
three scenes with one manifest schema and deterministic matched evaluation
banks:

* a single analytic ball--wall reflection;
* a non-symmetric support-release apparatus and its horizontal mirror; and
* a ball undergoing occluded uniform motion.

The data layer exposes S/C/G and analytic trajectories in metadata only.  Raw
video training continues to receive just the rendered video, exactly as in the
existing B-768 pipeline.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import imageio.v2 as imageio
import numpy as np
import yaml

from sshv2.common.dataset import (
    samples_from_metadata_csv,
    write_dataset,
)
from sshv2.common.video import (
    as_list as _as_list,
    deterministic_rng as _rng,
    draw_circle as _circle,
    draw_segment as _segment,
    valid_positions,
)


# v1 Support encoded the release side before applying a mirror, which changed
# the *screen-coordinate* state.  v2 keeps the already-generated v1 roots
# immutable and fixes Support's public S semantics.
VERSION = "world_model_shortcuts_phase_a_v2"
SCENES = ("ball_wall", "support_release", "occluded_uniform_motion")
RELIABILITIES = (0.5, 0.7, 0.8, 0.9, 0.95, 1.0)
REGIMES = {
    "ball_wall": ("P", "C", "G", "GC"),
    "support_release": ("P", "C", "G", "GC"),
    "occluded_uniform_motion": ("P", "C"),
}
EVAL_VARIANTS = {
    "ball_wall": ("ID", "C_flip", "G_flip", "GC_flip"),
    "support_release": ("ID", "C_flip", "G_flip", "GC_flip"),
    "occluded_uniform_motion": ("ID", "C_flip"),
}

FPS, FRAMES, PREFIX, RESOLUTION = 15, 49, 5, 128
RADIUS = 0.040
BACKGROUND = (22, 22, 26)
WALL = (224, 224, 224)
SUPPORT = (158, 158, 158)
OCCLUDER = (104, 104, 110)
COLORS = {0: (235, 45, 45), 1: (45, 95, 235)}
COLOR_NAMES = {0: "red", 1: "blue"}


@dataclass(frozen=True)
class BallWallDataConfig:
    """Direct input for the Ball-Wall/Phase-A dataset generator."""

    root: Path
    scenes: tuple[str, ...] = SCENES
    train_count: int = 2048
    eval_count: int = 64
    alphas: tuple[float, ...] = RELIABILITIES
    include_training: bool = True
    include_eval: bool = True


def wrap(degrees: float) -> float:
    return (float(degrees) + 180.0) % 360.0 - 180.0


def unit(angle_degrees: float, scale: float = 1.0) -> np.ndarray:
    a = math.radians(float(angle_degrees))
    return scale * np.asarray((math.cos(a), math.sin(a)), dtype=np.float64)


def normal(phi_degrees: float) -> np.ndarray:
    tangent = unit(phi_degrees)
    return np.asarray((-tangent[1], tangent[0]), dtype=np.float64)


def reflect(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype=np.float64) - 2.0 * float(np.dot(v, n)) * n


def _alpha_tag(alpha: float) -> str:
    return f"a{int(round(float(alpha) * 100)):03d}"


def _json_default(x: Any):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x).__name__)


@dataclass(frozen=True)
class CueAssignment:
    S: int
    C: int
    G: int | None


class ReliabilitySampler:
    """Balanced finite-bank S/C/G assignment with independent cue masks.

    Within each S stratum, exactly ``round(alpha * n/2)`` records agree with
    S.  This makes S, C and G exactly marginally balanced for even n, while
    the realised reliability differs from the requested value by at most
    1/(n/2).  C/G permutations use separate salts, so they are conditionally
    independently shuffled rather than copied from one another.
    """

    def __init__(self, n: int, regime: str, alpha: float, has_geometry: bool, seed: int = 81231):
        if n <= 0 or n % 2:
            raise ValueError("n must be positive and even for exact marginal balance")
        if regime not in ("P", "C", "G", "GC"):
            raise ValueError(regime)
        if not has_geometry and regime in ("G", "GC"):
            raise ValueError(f"{regime} is invalid without G")
        if not any(abs(alpha - x) < 1e-10 for x in RELIABILITIES):
            raise ValueError(f"unsupported alpha={alpha}")
        if regime == "P" and abs(alpha - 0.5) > 1e-10:
            raise ValueError("P is defined only at alpha=0.5")
        self.n, self.regime, self.alpha, self.has_geometry, self.seed = n, regime, float(alpha), has_geometry, int(seed)
        states = np.asarray([0, 1] * (n // 2), dtype=np.int8)
        self.assignments: list[CueAssignment] = [CueAssignment(int(s), 0, 0 if has_geometry else None) for s in states]
        c = self._cue(states, alpha if regime in ("C", "GC") else 0.5, salt=31)
        g = self._cue(states, alpha if regime in ("G", "GC") else 0.5, salt=47) if has_geometry else None
        self.assignments = [CueAssignment(int(states[i]), int(c[i]), None if g is None else int(g[i])) for i in range(n)]

    def _cue(self, states: np.ndarray, alpha: float, salt: int) -> np.ndarray:
        result = np.empty_like(states)
        r = _rng(self.seed, salt)
        group_size = len(states) // 2
        agree = int(round(alpha * group_size))
        for state in (0, 1):
            idx = np.flatnonzero(states == state)
            chosen = idx[r.permutation(group_size)[:agree]]
            result[idx] = 1 - state
            result[chosen] = state
        return result

    def __getitem__(self, index: int) -> CueAssignment:
        return self.assignments[index]

    def summary(self) -> dict[str, float | None]:
        x = self.assignments
        ans: dict[str, float | None] = {
            "requested_alpha": self.alpha,
            "realized_C_agreement": float(np.mean([q.C == q.S for q in x])),
            "C_marginal_one": float(np.mean([q.C for q in x])),
        }
        if self.has_geometry:
            ans.update({
                "realized_G_agreement": float(np.mean([q.G == q.S for q in x])),
                "G_marginal_one": float(np.mean([q.G for q in x])),
                "S_marginal_one": float(np.mean([q.S for q in x])),
                "conditional_CG_agreement_covariance": float(np.mean([
                    np.mean((np.asarray([int(q.C == q.S) for q in x if q.S == s], dtype=float) - self.alpha)
                            * (np.asarray([int(q.G == q.S) for q in x if q.S == s], dtype=float) - self.alpha))
                    for s in (0, 1)
                ])),
            })
        else:
            ans.update({"realized_G_agreement": None, "G_marginal_one": None})
        return ans


def _rect(frame: np.ndarray, lo: tuple[float, float], hi: tuple[float, float], color: tuple[int, int, int]) -> None:
    h, w = frame.shape[:2]
    x0, x1 = sorted((int(round(lo[0] * w)), int(round(hi[0] * w))))
    y0, y1 = sorted((int(round((1.0 - hi[1]) * h)), int(round((1.0 - lo[1]) * h))))
    frame[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = color


def _valid_positions(positions: np.ndarray, radius: float = RADIUS) -> bool:
    return valid_positions(positions, radius=radius)


class BallWallScene:
    """One analytic continuous-time reflection with S=local incident branch."""

    phi_bands = {0: (-32.0, -12.0), 1: (12.0, 32.0)}
    alpha_bands = {0: (-120.0, -105.0), 1: (-75.0, -60.0)}
    wall_half_length, wall_thickness = .18, 4

    @staticmethod
    def nuisance(seed: int) -> dict[str, float]:
        r = _rng(seed, 101)
        return {
            "phi_0": float(r.uniform(*BallWallScene.phi_bands[0])),
            "phi_1": float(r.uniform(*BallWallScene.phi_bands[1])),
            "alpha_0": float(r.uniform(*BallWallScene.alpha_bands[0])),
            "alpha_1": float(r.uniform(*BallWallScene.alpha_bands[1])),
            "speed": float(r.uniform(.050, .068)),
            "collision_time": float(r.uniform(.215, .255)),
            "scene_dx": float(r.uniform(-.055, .055)),
            "scene_dy": float(r.uniform(-.040, .040)),
        }

    @staticmethod
    def event(seed: int, S: int, G: int, C: int) -> dict[str, Any]:
        q = BallWallScene.nuisance(seed)
        phi, alpha = q[f"phi_{G}"], q[f"alpha_{S}"]
        # The contact point is globally translated nuisance, never tied to S/G.
        contact = np.asarray((.50 + q["scene_dx"], .52 + q["scene_dy"]), dtype=np.float64)
        n = normal(phi)
        theta_in = wrap(phi + alpha)
        vin = unit(theta_in, q["speed"])
        if float(vin @ n) >= -1e-10:
            raise RuntimeError("invalid incident direction")
        vout = reflect(vin, n)
        centre_contact = contact + RADIUS * n
        positions = np.asarray([
            centre_contact + (vin if f / FPS <= q["collision_time"] else vout) * (f / FPS - q["collision_time"])
            for f in range(FRAMES)
        ])
        if not _valid_positions(positions):
            raise RuntimeError("out-of-frame ball-wall nuisance")
        return {
            "scene_type": "ball_wall", "S": int(S), "C": int(C), "G": int(G),
            "color": COLOR_NAMES[C], "color_rgb": list(COLORS[C]),
            "phi_wall": phi, "alpha_in": alpha,
            "theta_in": theta_in,
            "theta_out": wrap(math.degrees(math.atan2(vout[1], vout[0]))),
            "alpha_out": wrap(math.degrees(math.atan2(vout[1], vout[0])) - phi),
            "contact_point": _as_list(contact), "normal": _as_list(n),
            "velocity_in": _as_list(vin), "velocity_out": _as_list(vout),
            "collision_time": q["collision_time"], "first_contact_frame": int(math.ceil(q["collision_time"] * FPS)),
            "positions": _as_list(positions), "radius": RADIUS, "speed": q["speed"],
            "mass": 1.0, "restitution": 1.0, "wall_half_length": BallWallScene.wall_half_length, "wall_thickness": BallWallScene.wall_thickness,
            "nuisance": q,
        }

    @staticmethod
    def render(event: dict[str, Any]) -> np.ndarray:
        frames = np.full((FRAMES, RESOLUTION, RESOLUTION, 3), BACKGROUND, dtype=np.uint8)
        c, phi = np.asarray(event["contact_point"]), float(event["phi_wall"])
        t = unit(phi)
        a, b = c - BallWallScene.wall_half_length * t, c + BallWallScene.wall_half_length * t
        for f in range(FRAMES):
            _segment(frames[f], a, b, WALL, BallWallScene.wall_thickness)
            _circle(frames[f], np.asarray(event["positions"])[f], RADIUS, tuple(event["color_rgb"]))
        return frames


class SupportReleaseScene:
    """Asymmetric apparatus with screen-coordinate release state.

    ``S`` always names the side that retracts in the final rendered image:
    S=0 means screen-left first / future-left, S=1 screen-right first /
    future-right.  G=1 mirrors the apparatus, so the canonical side simulated
    before the mirror must be ``S XOR G``.
    """

    # Angles relative to horizontal.  They are fixed by design; only the
    # delay between retractions is continuously sampled.
    beta_left, beta_right = 34.0, 48.0
    ramp_length, ramp_thickness = .160, 4
    gravity, retreat_duration = .160, .060

    @staticmethod
    def nuisance(seed: int) -> dict[str, float]:
        r = _rng(seed, 211)
        return {
            # With 15 FPS these ranges make frame 1 show the first support
            # moving, frame 2 show it absent, frame 3 show the second moving,
            # and frame 4 show both absent.  Delta_t is the only directly
            # sampled physical timing variable controlling release velocity.
            "delta_t": float(r.uniform(.100, .135)),
            "first_retraction_time": float(r.uniform(.035, .050)),
            "scene_dx": float(r.uniform(-.035, .035)),
            "scene_dy": float(r.uniform(-.004, .004)),
            "light": float(r.uniform(.98, 1.02)),
        }

    @staticmethod
    def _mirror(x: np.ndarray, G: int) -> np.ndarray:
        y = np.asarray(x, dtype=np.float64).copy()
        if G:
            y[..., 0] = 1.0 - y[..., 0]
        return y

    @staticmethod
    def event(seed: int, S: int, G: int, C: int) -> dict[str, Any]:
        q = SupportReleaseScene.nuisance(seed)
        bottom_canonical = np.asarray((.50 + q["scene_dx"], .815 + q["scene_dy"]), dtype=np.float64)
        p0_canonical = np.asarray((.50 + q["scene_dx"], .860 + q["scene_dy"]), dtype=np.float64)
        bl, br = math.radians(SupportReleaseScene.beta_left), math.radians(SupportReleaseScene.beta_right)
        left_end = bottom_canonical + np.asarray((-math.cos(bl), math.sin(bl))) * SupportReleaseScene.ramp_length
        right_end = bottom_canonical + np.asarray((math.cos(br), math.sin(br))) * SupportReleaseScene.ramp_length
        if S not in (0, 1) or G not in (0, 1):
            raise ValueError((S, G))
        canonical_state = S ^ G
        canonical_retract_side = "left" if canonical_state == 0 else "right"
        screen_retract_side = "left" if S == 0 else "right"
        # Generate G=0 canonically and mirror all geometry/trajectory for
        # G=1.  Choosing canonical_state=S XOR G preserves the public screen
        # state after that mirror.
        tangent = (np.asarray((-math.cos(br), -math.sin(br)))
                   if canonical_state == 0 else np.asarray((math.cos(bl), -math.sin(bl))))
        a_slide = SupportReleaseScene.gravity * abs(tangent[1]) * tangent
        t_first = q["first_retraction_time"]
        t_slide_start = t_first + SupportReleaseScene.retreat_duration
        t_release = t_first + q["delta_t"] + SupportReleaseScene.retreat_duration
        p_release_canonical = p0_canonical + .5 * a_slide * q["delta_t"] ** 2
        v_release_canonical = a_slide * q["delta_t"]
        gravity = np.asarray((0.0, -SupportReleaseScene.gravity))
        positions, velocities = [], []
        for f in range(FRAMES):
            t = f / FPS
            if t <= t_slide_start:
                p, v = p0_canonical, np.zeros(2)
            elif t <= t_release:
                dt = t - t_slide_start
                p, v = p0_canonical + .5 * a_slide * dt * dt, a_slide * dt
            else:
                dt = t - t_release
                p, v = p_release_canonical + v_release_canonical * dt + .5 * gravity * dt * dt, v_release_canonical + gravity * dt
            positions.append(p); velocities.append(v)
        pos = SupportReleaseScene._mirror(np.asarray(positions), G)
        # Points reflect as x -> 1-x; vectors reflect by flipping x only.
        vel = np.asarray(velocities, dtype=np.float64).copy()
        vrel = np.asarray(v_release_canonical, dtype=np.float64).copy()
        arel = np.asarray(a_slide, dtype=np.float64).copy()
        if G:
            vel[:, 0] *= -1.0; vrel[0] *= -1.0; arel[0] *= -1.0
        bottom, p0, p_release = (SupportReleaseScene._mirror(x, G) for x in (bottom_canonical, p0_canonical, p_release_canonical))
        supports = {
            "canonical_left": {"endpoints": [_as_list(SupportReleaseScene._mirror(left_end, G)), _as_list(bottom)], "first_removed": canonical_retract_side == "left"},
            "canonical_right": {"endpoints": [_as_list(bottom), _as_list(SupportReleaseScene._mirror(right_end, G))], "first_removed": canonical_retract_side == "right"},
        }
        if t_release >= (PREFIX - 1) / FPS:
            raise RuntimeError("support still present at prefix end")
        if not _valid_positions(pos):
            raise RuntimeError("out-of-frame support-release nuisance")
        return {
            "scene_type": "support_release", "S": int(S), "C": int(C), "G": int(G),
            "color": COLOR_NAMES[C], "color_rgb": list(COLORS[C]),
            "canonical_apparatus": {"phi_left_deg": SupportReleaseScene.beta_left, "phi_right_deg": SupportReleaseScene.beta_right,
                                    "first_removed_support": canonical_retract_side},
            "mirror_transform": {"type": "horizontal", "applied": bool(G), "x_map": "x -> 1-x"},
            "geometry_name": "Gamma_1_mirror" if G else "Gamma_0",
            "screen_state": {"first_removed_support": screen_retract_side,
                             "future_branch": "left" if S == 0 else "right",
                             "left_slope_deg": SupportReleaseScene.beta_right if G else SupportReleaseScene.beta_left,
                             "right_slope_deg": SupportReleaseScene.beta_left if G else SupportReleaseScene.beta_right,
                             "canonical_state": canonical_state},
            "positions": _as_list(pos), "velocities": _as_list(vel),
            "velocity_at_release": _as_list(vrel), "slide_acceleration": _as_list(arel), "gravity": _as_list(gravity),
            "initial_position": _as_list(p0), "release_position": _as_list(p_release),
            "support_base": _as_list(bottom), "supports": supports,
            "radius": RADIUS, "mass": 1.0, "friction": 0.0, "support_timing": {"first_begin_time": t_first, "first_absent_time": t_slide_start, "slide_start_time": t_slide_start,
                "second_begin_time": t_first + q["delta_t"], "second_absent_time": t_release, "release_time": t_release},
            "nuisance": q,
        }

    @staticmethod
    def _support_offset(t: float, start: float) -> float | None:
        duration = SupportReleaseScene.retreat_duration
        if t >= start + duration:
            return None
        if t <= start:
            return 0.0
        return -.125 * (t - start) / duration

    @staticmethod
    def render(event: dict[str, Any]) -> np.ndarray:
        frames = np.full((FRAMES, RESOLUTION, RESOLUTION, 3), BACKGROUND, dtype=np.uint8)
        timing, supports = event["support_timing"], event["supports"]
        for f in range(FRAMES):
            t = f / FPS
            for support in supports.values():
                start = timing["first_begin_time"] if support["first_removed"] else timing["second_begin_time"]
                offset = SupportReleaseScene._support_offset(t, start)
                if offset is not None:
                    a, b = np.asarray(support["endpoints"])
                    _segment(frames[f], a + (0.0, offset), b + (0.0, offset), SUPPORT, SupportReleaseScene.ramp_thickness)
            _circle(frames[f], np.asarray(event["positions"])[f], RADIUS, tuple(event["color_rgb"]))
        return frames


class OccludedUniformMotionScene:
    """Strict constant-velocity motion through a neutral vertical occluder."""

    occluder_lo, occluder_hi = (.44, .16), (.64, .84)

    @staticmethod
    def nuisance(seed: int) -> dict[str, float]:
        r = _rng(seed, 307)
        return {
            "vx": float(r.uniform(.120, .150)),
            "abs_vy": float(r.uniform(.025, .040)),
            "enter_time": float(r.uniform(.380, .480)),
            "enter_x": .40,
            "enter_y": float(r.uniform(.455, .545)),
        }

    @staticmethod
    def event(seed: int, S: int, C: int) -> dict[str, Any]:
        q = OccludedUniformMotionScene.nuisance(seed)
        vy = -q["abs_vy"] if S == 0 else q["abs_vy"]
        velocity = np.asarray((q["vx"], vy), dtype=np.float64)
        p_enter = np.asarray((q["enter_x"], q["enter_y"]), dtype=np.float64)
        positions = np.asarray([p_enter + velocity * (f / FPS - q["enter_time"]) for f in range(FRAMES)])
        if not _valid_positions(positions):
            raise RuntimeError("out-of-frame occlusion nuisance")
        return {
            "scene_type": "occluded_uniform_motion", "S": int(S), "C": int(C), "G": None,
            "color": COLOR_NAMES[C], "color_rgb": list(COLORS[C]), "positions": _as_list(positions),
            "velocity": _as_list(velocity), "last_visible_enter_point": _as_list(p_enter), "occluder": {"lo": list(OccludedUniformMotionScene.occluder_lo), "hi": list(OccludedUniformMotionScene.occluder_hi)},
            "radius": RADIUS, "mass": 1.0, "acceleration": [0.0, 0.0], "nuisance": q,
        }

    @staticmethod
    def render(event: dict[str, Any]) -> np.ndarray:
        frames = np.full((FRAMES, RESOLUTION, RESOLUTION, 3), BACKGROUND, dtype=np.uint8)
        for f in range(FRAMES):
            _circle(frames[f], np.asarray(event["positions"])[f], RADIUS, tuple(event["color_rgb"]))
            # Opaque, textureless occluder is drawn after the ball.
            _rect(frames[f], OccludedUniformMotionScene.occluder_lo, OccludedUniformMotionScene.occluder_hi, OCCLUDER)
        return frames


SCENE_CLASSES = {"ball_wall": BallWallScene, "support_release": SupportReleaseScene, "occluded_uniform_motion": OccludedUniformMotionScene}


def make_record(scene: str, seed: int, S: int, C: int, G: int | None, *, train_regime: str, alpha: float | None, eval_variant: str | None, sample_id: str, record_id: str) -> dict[str, Any]:
    cls = SCENE_CLASSES[scene]
    if scene == "occluded_uniform_motion":
        event = cls.event(seed, S, C)
        counterfactual = cls.event(seed, 1 - S, C)
    else:
        assert G is not None
        event = cls.event(seed, S, G, C)
        counterfactual = cls.event(seed, 1 - S, G, C)
    return {
        "benchmark_version": VERSION, "scene_type": scene, "record_id": record_id, "sample_id": sample_id,
        "seed": int(seed), "S": int(S), "C": int(C), "G": None if G is None else int(G),
        "train_regime": train_regime, "alpha": None if alpha is None else float(alpha), "eval_variant": eval_variant,
        "prefix": {"pixel_frames": PREFIX, "future_start_frame": PREFIX, "frames": FRAMES, "fps": FPS, "resolution": RESOLUTION},
        "nuisance": event["nuisance"], "physical_parameters": {k: event[k] for k in event if k not in ("positions", "velocities", "nuisance", "color_rgb")},
        "ground_truth_trajectory": event["positions"], "true_branch_oracle_trajectory": event["positions"],
        "counterfactual_branch_trajectory": counterfactual["positions"],
        "renderer_metadata": {"color": event["color"], "color_rgb": event["color_rgb"], "radius": event["radius"], "event": event},
    }


def _eval_cues(S: int, variant: str, has_geometry: bool) -> tuple[int, int | None]:
    if variant == "ID": return S, S if has_geometry else None
    if variant == "C_flip": return 1 - S, S if has_geometry else None
    if variant == "G_flip" and has_geometry: return S, 1 - S
    if variant == "GC_flip" and has_geometry: return 1 - S, 1 - S
    raise ValueError((variant, has_geometry))


def _save_record(record: dict[str, Any], directory: Path, index: int) -> dict[str, str | int | float | None]:
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"sample_{index:06d}"
    cls = SCENE_CLASSES[record["scene_type"]]
    event = record["renderer_metadata"]["event"]
    video_path, meta_path = directory / f"{stem}.mp4", directory / f"{stem}.json"
    with imageio.get_writer(video_path, fps=FPS, codec="libx264", quality=10, macro_block_size=None) as writer:
        for frame in cls.render(event):
            writer.append_data(frame)
    meta_path.write_text(json.dumps(record, indent=2, default=_json_default) + "\n")
    return {"video": video_path.name, "source": video_path.name, "metadata": meta_path.name,
            "record_id": record["record_id"], "sample_id": record["sample_id"], "seed": record["seed"],
            "scene_type": record["scene_type"], "S": record["S"], "C": record["C"], "G": record["G"],
            "train_regime": record["train_regime"], "alpha": record["alpha"], "eval_variant": record["eval_variant"]}


def _write_csv(directory: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with (directory / "metadata.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _iter_train_conditions(scene: str, alphas: Iterable[float]) -> Iterable[tuple[str, float]]:
    yield "P", .5
    for regime in REGIMES[scene]:
        if regime == "P":
            continue
        for alpha in alphas:
            yield regime, float(alpha)


def _build_training_bank(root: Path, scene: str, regime: str, alpha: float, count: int, seed_start: int) -> dict[str, Any]:
    has_g = scene != "occluded_uniform_motion"
    sampler = ReliabilitySampler(count, regime, alpha, has_g, seed=seed_start + 831)
    directory = root / "videos" / scene / f"train_{regime}_{_alpha_tag(alpha)}"
    rows = []
    for i in range(count):
        cue = sampler[i]
        rec = make_record(scene, seed_start + i, cue.S, cue.C, cue.G, train_regime=regime, alpha=alpha,
                          eval_variant=None, sample_id=f"{scene}:train:{regime}:{_alpha_tag(alpha)}:{i:06d}",
                          record_id=f"{scene}:train:{regime}:{_alpha_tag(alpha)}:{i:06d}")
        rows.append(_save_record(rec, directory, i))
    _write_csv(directory, rows)
    return {"scene": scene, "regime": regime, "alpha": alpha, "path": str(directory.relative_to(root)), "samples": count, **sampler.summary()}


def _build_eval_bank(root: Path, scene: str, count: int, seed_start: int) -> dict[str, Any]:
    directory = root / "videos" / scene / "eval"
    rows: list[dict[str, Any]] = []
    variants, has_g = EVAL_VARIANTS[scene], scene != "occluded_uniform_motion"
    for base_i in range(count):
        seed = seed_start + base_i
        sample_id = f"{scene}:eval:{base_i:06d}"
        for S in (0, 1):
            for variant in variants:
                C, G = _eval_cues(S, variant, has_g)
                rec = make_record(scene, seed, S, C, G, train_regime="eval", alpha=None, eval_variant=variant,
                                  sample_id=sample_id, record_id=f"{sample_id}:S{S}:{variant}")
                rows.append(_save_record(rec, directory, len(rows)))
    _write_csv(directory, rows)
    return {"scene": scene, "path": str(directory.relative_to(root)), "base_samples": count, "records": len(rows), "variants": list(variants)}


def build(root: Path, *, scenes: Iterable[str] = SCENES, train_count: int = 2048, eval_count: int = 64,
          alphas: Iterable[float] = RELIABILITIES, include_training: bool = True, include_eval: bool = True) -> dict[str, Any]:
    scenes = tuple(scenes)
    if train_count % 2:
        raise ValueError("train_count must be even")
    if eval_count <= 0:
        raise ValueError("eval_count must be positive")
    for scene in scenes:
        if scene not in SCENE_CLASSES:
            raise ValueError(scene)
    root.mkdir(parents=True, exist_ok=True)
    alphas = tuple(float(x) for x in alphas)
    report: dict[str, Any] = {"benchmark_version": VERSION, "train": [], "eval": []}
    for scene_i, scene in enumerate(scenes):
        if include_training:
            for condition_i, (regime, alpha) in enumerate(_iter_train_conditions(scene, alphas)):
                report["train"].append(_build_training_bank(root, scene, regime, alpha, train_count,
                                                              3_100_000 + scene_i * 100_000 + condition_i * 10_000))
        if include_eval:
            report["eval"].append(_build_eval_bank(root, scene, eval_count, 4_100_000 + scene_i * 100_000))
    metadata = root / "metadata"; manifests = root / "manifests"
    metadata.mkdir(exist_ok=True); manifests.mkdir(exist_ok=True)
    config = {"benchmark_version": VERSION, "video": {"resolution": RESOLUTION, "frames": FRAMES, "fps": FPS, "prefix_pixel_frames": PREFIX},
              "train_count": train_count, "eval_base_count": eval_count, "reliabilities": list(alphas), "scenes": list(scenes),
              "continuous": {"ball_wall": ["phi_wall", "alpha_in", "speed", "collision_time", "scene_dx", "scene_dy"],
                  "support_release": ["delta_t", "first_retraction_time", "scene_dx", "scene_dy"],
                  "occluded_uniform_motion": ["vx", "abs_vy", "enter_time", "enter_y"]},
              "fixed": {"ball_wall": ["radius", "restitution", "wall_length", "wall_thickness", "camera", "background"],
                  "support_release": ["phi_left", "phi_right", "gravity", "support_size", "retraction_speed", "radius", "camera", "background"],
                  "occluded_uniform_motion": ["radius", "occluder_geometry", "camera", "background"]}}
    config["config_hash"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    (metadata / "generation_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (metadata / "build_report.json").write_text(json.dumps(report, indent=2, default=_json_default) + "\n")
    common_samples: list[dict[str, Any]] = []
    for path in sorted((root / "videos").glob("**/metadata.csv")):
        name = "__".join(path.relative_to(root / "videos").parts[:-1])
        (manifests / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in csv.DictReader(path.open())))
        bank_name = path.parent.name
        split = (
            "eval"
            if bank_name == "eval"
            else "train"
            if bank_name.startswith("train_")
            else bank_name
        )
        common_samples.extend(
            samples_from_metadata_csv(
                root,
                path,
                split=split,
                subset=path.parent.relative_to(
                    root / "videos"
                ).as_posix(),
            )
        )
    write_dataset(
        root,
        experiment="world_model_shortcuts",
        dataset=VERSION,
        samples=common_samples,
        extra={
            "generation_config": "metadata/generation_config.yaml",
            "legacy_manifests": "manifests",
            "scenes": list(scenes),
        },
    )
    return report


def generate_dataset(config: BallWallDataConfig) -> Path:
    """Generate one configured dataset and return its root directory."""
    build(
        config.root,
        scenes=config.scenes,
        train_count=config.train_count,
        eval_count=config.eval_count,
        alphas=config.alphas,
        include_training=config.include_training,
        include_eval=config.include_eval,
    )
    return config.root


def _audit_rows(path: Path) -> list[dict[str, str]]:
    with (path / "metadata.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _audit_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads((path / row["metadata"]).read_text(encoding="utf-8"))
        for row in _audit_rows(path)
    ]


def _audit_trajectory(
    record: dict[str, Any],
    key: str = "ground_truth_trajectory",
) -> np.ndarray:
    return np.asarray(record[key], dtype=np.float64)


def _audit_expected_cues(
    state: int,
    variant: str,
    has_geometry: bool,
) -> tuple[int, int | None]:
    if variant == "ID":
        return state, state if has_geometry else None
    if variant == "C_flip":
        return 1 - state, state if has_geometry else None
    if variant == "G_flip" and has_geometry:
        return state, 1 - state
    if variant == "GC_flip" and has_geometry:
        return 1 - state, 1 - state
    raise ValueError((variant, has_geometry))


def _numeric_nuisance(event: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in event["nuisance"].items()
        if isinstance(value, (int, float))
    }


def _audit_future_right(record: dict[str, Any]) -> bool:
    positions = _audit_trajectory(record)
    return bool(positions[-1, 0] > positions[PREFIX - 1, 0])


def _audit_training_bank(
    path: Path,
    scene: str,
    regime: str,
    alpha: float,
) -> tuple[dict[str, Any], list[list[Any]]]:
    records = _audit_records(path)
    violations: list[list[Any]] = []
    state = np.asarray([record["S"] for record in records], dtype=float)
    color = np.asarray([record["C"] for record in records], dtype=float)
    has_geometry = scene != "occluded_uniform_motion"
    geometry = (
        np.asarray([record["G"] for record in records], dtype=float)
        if has_geometry
        else None
    )
    requested_color = alpha if regime in ("C", "GC") else 0.5
    requested_geometry = alpha if regime in ("G", "GC") else 0.5
    summary: dict[str, Any] = {
        "records": len(records),
        "S_one": float(state.mean()),
        "C_one": float(color.mean()),
        "C_agreement": float(np.mean(color == state)),
        "requested_C_agreement": requested_color,
        "C_error": float(np.mean(color == state) - requested_color),
    }
    if has_geometry:
        assert geometry is not None
        summary.update(
            {
                "G_one": float(geometry.mean()),
                "G_agreement": float(np.mean(geometry == state)),
                "requested_G_agreement": requested_geometry,
                "G_error": float(
                    np.mean(geometry == state) - requested_geometry
                ),
            }
        )
        conditional = []
        for state_value in (0, 1):
            index = state == state_value
            color_agrees = (color[index] == state[index]).astype(float)
            geometry_agrees = (
                geometry[index] == state[index]
            ).astype(float)
            conditional.append(
                float(
                    np.mean(
                        (color_agrees - color_agrees.mean())
                        * (geometry_agrees - geometry_agrees.mean())
                    )
                )
            )
        summary["conditional_CG_agreement_covariance"] = float(
            np.mean(conditional)
        )

    if scene == "support_release":
        assert geometry is not None
        signs = np.asarray(
            [_audit_future_right(record) for record in records],
            dtype=int,
        )
        summary["SCG_contingency"] = {
            f"S{state_value}_C{color_value}_G{geometry_value}": int(
                np.sum(
                    (state == state_value)
                    & (color == color_value)
                    & (geometry == geometry_value)
                )
            )
            for state_value in (0, 1)
            for color_value in (0, 1)
            for geometry_value in (0, 1)
        }
        summary["future_right_by_SG"] = {
            f"S{state_value}_G{geometry_value}": float(
                signs[
                    (state == state_value) & (geometry == geometry_value)
                ].mean()
            )
            for state_value in (0, 1)
            for geometry_value in (0, 1)
        }
        summary["future_right_by_C"] = {
            f"C{value}": float(signs[color == value].mean())
            for value in (0, 1)
        }
        summary["future_right_by_G"] = {
            f"G{value}": float(signs[geometry == value].mean())
            for value in (0, 1)
        }
        for state_value in (0, 1):
            for geometry_value in (0, 1):
                mask = (
                    (state == state_value)
                    & (geometry == geometry_value)
                )
                if (
                    mask.any()
                    and float(signs[mask].mean()) != float(state_value)
                ):
                    violations.append(
                        [
                            scene,
                            regime,
                            alpha,
                            "screen_state_future_sign",
                            state_value,
                            geometry_value,
                        ]
                    )
        if alpha == 1.0 and regime == "C" and (
            summary["future_right_by_C"]["C0"] != 0.0
            or summary["future_right_by_C"]["C1"] != 1.0
        ):
            violations.append(
                [scene, regime, alpha, "C_screen_branch_binding"]
            )
        if alpha == 1.0 and regime == "G" and (
            summary["future_right_by_G"]["G0"] != 0.0
            or summary["future_right_by_G"]["G1"] != 1.0
        ):
            violations.append(
                [scene, regime, alpha, "G_screen_branch_binding"]
            )

    nuisance: dict[str, dict[str, float]] = {}
    events = [
        record["renderer_metadata"]["event"]
        for record in records
    ]
    for key in sorted(
        set().union(*[_numeric_nuisance(event) for event in events])
    ):
        values = np.asarray(
            [_numeric_nuisance(event)[key] for event in events]
        )
        nuisance[key] = {
            "S_mean_gap": float(
                abs(values[state == 0].mean() - values[state == 1].mean())
            ),
            "C_mean_gap": float(
                abs(values[color == 0].mean() - values[color == 1].mean())
            ),
        }
        if has_geometry:
            assert geometry is not None
            nuisance[key]["G_mean_gap"] = float(
                abs(
                    values[geometry == 0].mean()
                    - values[geometry == 1].mean()
                )
            )
    summary["nuisance_mean_gaps"] = nuisance
    tolerance = 1.0 / (len(records) / 2.0) + 1e-12
    if not (
        abs(summary["S_one"] - 0.5) < 1e-12
        and abs(summary["C_one"] - 0.5) < 1e-12
    ):
        violations.append([scene, regime, alpha, "marginal_balance"])
    if abs(summary["C_error"]) > tolerance:
        violations.append([scene, regime, alpha, "C_reliability"])
    if has_geometry and (
        abs(summary["G_one"] - 0.5) > 1e-12
        or abs(summary["G_error"]) > tolerance
    ):
        violations.append([scene, regime, alpha, "G_reliability"])
    return summary, violations


def _write_audit_video_grid(
    root: Path,
    records: list[dict[str, Any]],
    out: Path,
    scene: str,
) -> None:
    grouped: dict[
        tuple[str, int],
        dict[str, dict[str, Any]],
    ] = defaultdict(dict)
    for record in records:
        grouped[(record["sample_id"], record["S"])][
            record["eval_variant"]
        ] = record
    out.mkdir(parents=True, exist_ok=True)
    eval_dir = root / "videos" / scene / "eval"
    rows = _audit_rows(eval_dir)
    for index, ((_, state), group) in enumerate(
        list(grouped.items())[:2]
    ):
        variants = EVAL_VARIANTS[scene]
        if set(group) != set(variants):
            continue
        clips = []
        for variant in variants:
            row_index = next(
                row_index
                for row_index, row in enumerate(rows)
                if row["record_id"] == group[variant]["record_id"]
            )
            reader = imageio.get_reader(
                eval_dir / f"sample_{row_index:06d}.mp4"
            )
            try:
                clips.append(
                    np.stack(
                        [reader.get_data(frame) for frame in range(FRAMES)]
                    )
                )
            finally:
                reader.close()
        output = out / (
            f"{scene}_{index:02d}_S{state}_{'_'.join(variants)}.mp4"
        )
        with imageio.get_writer(
            output,
            fps=FPS,
            codec="libx264",
            quality=10,
            macro_block_size=None,
        ) as writer:
            for frame in np.concatenate(clips, axis=2):
                writer.append_data(frame)


def _audit_evaluation_bank(
    root: Path,
    scene: str,
    out: Path,
) -> tuple[dict[str, Any], list[list[Any]]]:
    directory = root / "videos" / scene / "eval"
    records = _audit_records(directory)
    violations: list[list[Any]] = []
    groups: dict[
        tuple[str, int],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for record in records:
        groups[(record["sample_id"], record["S"])].append(record)
    has_geometry = scene != "occluded_uniform_motion"
    for (sample_id, state), group in groups.items():
        by_variant = {
            record["eval_variant"]: record
            for record in group
        }
        expected = set(EVAL_VARIANTS[scene])
        if set(by_variant) != expected:
            violations.append(
                [scene, sample_id, state, "incomplete_eval_bank"]
            )
            continue
        reference = by_variant[next(iter(expected))]
        for variant, record in by_variant.items():
            color, geometry = _audit_expected_cues(
                state,
                variant,
                has_geometry,
            )
            if record["C"] != color or record["G"] != geometry:
                violations.append(
                    [scene, sample_id, state, variant, "cue_assignment"]
                )
            if (
                record["seed"] != reference["seed"]
                or record["nuisance"] != reference["nuisance"]
            ):
                violations.append(
                    [scene, sample_id, state, variant, "nuisance_changed"]
                )
            if not np.allclose(
                _audit_trajectory(record),
                _audit_trajectory(
                    record,
                    "true_branch_oracle_trajectory",
                ),
            ):
                violations.append(
                    [scene, sample_id, state, variant, "true_oracle_mismatch"]
                )
            if np.allclose(
                _audit_trajectory(record),
                _audit_trajectory(
                    record,
                    "counterfactual_branch_trajectory",
                ),
            ):
                violations.append(
                    [
                        scene,
                        sample_id,
                        state,
                        variant,
                        "counterfactual_not_distinct",
                    ]
                )
        if not np.allclose(
            _audit_trajectory(by_variant["ID"]),
            _audit_trajectory(by_variant["C_flip"]),
        ):
            violations.append(
                [
                    scene,
                    sample_id,
                    state,
                    "C_flip",
                    "appearance_changed_trajectory",
                ]
            )
        if has_geometry and np.allclose(
            _audit_trajectory(by_variant["ID"]),
            _audit_trajectory(by_variant["G_flip"]),
        ):
            violations.append(
                [
                    scene,
                    sample_id,
                    state,
                    "G_flip",
                    "geometry_did_not_change_global_trajectory",
                ]
            )

        if scene == "ball_wall":
            for record in group:
                event = record["renderer_metadata"]["event"]
                positions = np.asarray(event["positions"])
                normal = np.asarray(event["normal"])
                contact = np.asarray(event["contact_point"])
                if (
                    positions @ normal - contact @ normal
                    < RADIUS - 1e-8
                ).any():
                    violations.append(
                        [
                            scene,
                            sample_id,
                            state,
                            record["eval_variant"],
                            "wall_penetration",
                        ]
                    )
                reflection_error = (
                    (
                        event["alpha_in"]
                        + event["alpha_out"]
                        + 180
                    )
                    % 360
                ) - 180
                if abs(reflection_error) > 1e-7:
                    violations.append(
                        [
                            scene,
                            sample_id,
                            state,
                            record["eval_variant"],
                            "reflection_identity",
                        ]
                    )
        elif scene == "support_release":
            geometry_zero = next(
                record for record in group if record["G"] == 0
            )
            geometry_one = next(
                record for record in group if record["G"] == 1
            )
            if (
                _audit_future_right(geometry_zero) != bool(state)
                or _audit_future_right(geometry_one) != bool(state)
            ):
                violations.append(
                    [
                        scene,
                        sample_id,
                        state,
                        "screen_state",
                        "wrong_future_sign",
                    ]
                )
            if np.allclose(
                _audit_trajectory(geometry_zero),
                _audit_trajectory(geometry_one),
            ):
                violations.append(
                    [
                        scene,
                        sample_id,
                        state,
                        "geometry",
                        "did_not_change_trajectory",
                    ]
                )
            for record in group:
                event = record["renderer_metadata"]["event"]
                timing = event["support_timing"]
                expected_side = "left" if state == 0 else "right"
                if (
                    event["screen_state"]["first_removed_support"]
                    != expected_side
                ):
                    violations.append(
                        [
                            scene,
                            sample_id,
                            state,
                            record["eval_variant"],
                            "screen_retract_side",
                        ]
                    )
                if timing["second_absent_time"] >= (PREFIX - 1) / FPS:
                    violations.append(
                        [
                            scene,
                            sample_id,
                            state,
                            record["eval_variant"],
                            "support_present_at_prefix_end",
                        ]
                    )
                if not np.allclose(
                    np.asarray(event["velocity_at_release"]),
                    np.asarray(event["slide_acceleration"])
                    * event["nuisance"]["delta_t"],
                ):
                    violations.append(
                        [
                            scene,
                            sample_id,
                            state,
                            record["eval_variant"],
                            "release_velocity_not_simulated",
                        ]
                    )
        else:
            for record in group:
                event = record["renderer_metadata"]["event"]
                positions = np.asarray(event["positions"])
                velocity = np.asarray(event["velocity"])
                if not np.allclose(
                    np.diff(positions, axis=0),
                    velocity[None] / FPS,
                ):
                    violations.append(
                        [
                            scene,
                            sample_id,
                            state,
                            record["eval_variant"],
                            "non_uniform_motion",
                        ]
                    )
                x = positions[:, 0]
                inside = (
                    (x >= 0.44 - RADIUS)
                    & (x <= 0.64 + RADIUS)
                )
                if not (
                    inside[PREFIX:].any()
                    and (~inside[PREFIX:]).any()
                ):
                    violations.append(
                        [
                            scene,
                            sample_id,
                            state,
                            record["eval_variant"],
                            "occlusion_visibility_horizon",
                        ]
                    )
    _write_audit_video_grid(
        root,
        records,
        out / "matched_grids",
        scene,
    )
    return (
        {
            "records": len(records),
            "matched_state_groups": len(groups),
            "variants": list(EVAL_VARIANTS[scene]),
        },
        violations,
    )


def audit_dataset(root: Path, out: Path) -> dict[str, Any]:
    """Audit generated Phase-A data and persist a human-readable report."""
    report: dict[str, Any] = {
        "benchmark_version": VERSION,
        "train": {},
        "eval": {},
        "violations": [],
    }
    for scene_dir in sorted((root / "videos").iterdir()):
        if not scene_dir.is_dir():
            continue
        scene = scene_dir.name
        for train_dir in sorted(scene_dir.glob("train_*")):
            if train_dir.name.endswith("_enc"):
                continue
            _, regime, alpha_tag = train_dir.name.split("_")
            alpha = int(alpha_tag[1:]) / 100.0
            summary, violations = _audit_training_bank(
                train_dir,
                scene,
                regime,
                alpha,
            )
            report["train"][f"{scene}/{train_dir.name}"] = summary
            report["violations"].extend(violations)
        if (scene_dir / "eval" / "metadata.csv").exists():
            summary, violations = _audit_evaluation_bank(
                root,
                scene,
                out,
            )
            report["eval"][scene] = summary
            report["violations"].extend(violations)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if report["violations"]:
        raise ValueError(
            f"Dataset audit found {len(report['violations'])} violations"
        )
    return report


def generation_cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--scenes", nargs="*", choices=SCENES, default=list(SCENES))
    p.add_argument("--train-count", type=int, default=2048)
    p.add_argument("--eval-count", type=int, default=64)
    p.add_argument("--alphas", nargs="*", type=float, default=list(RELIABILITIES))
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--no-eval", action="store_true")
    a = p.parse_args()
    print(json.dumps(build(a.root, scenes=a.scenes, train_count=a.train_count, eval_count=a.eval_count, alphas=a.alphas,
                           include_training=not a.eval_only, include_eval=not a.no_eval), indent=2, default=_json_default))


if __name__ == "__main__":
    generation_cli()
