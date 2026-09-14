"""Data generation for the frozen v1 dual ball--wall shortcut benchmark.

This is deliberately a new generator.  It does not change the earlier R1/R2
matched-quartet generator: v1 uses two fixed *pose families*, a balanced
left/right permutation, and a factorial evaluation set for geometry and colour
shortcuts.  All collisions are analytic continuous-time reflections.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import yaml

from sshv2.common.dataset import (
    samples_from_metadata_csv,
    write_dataset,
)

FPS, FRAMES, CONTACT_FRAME = 15, 49, 4
RADIUS, SPEED, WALL_THICKNESS, WALL_HALF_LENGTH = .040, .060, 4, .18
BACKGROUND, WALL = (18, 18, 18), (225, 225, 225)
COLORS = {"red": (235, 45, 45), "blue": (45, 95, 235)}
ALPHA = {"A": (-120., -105.), "B": (-75., -60.)}
PHI = {"minus": (-32., -12.), "plus": (12., 32.)}
EVAL_VARIANTS = ("ID", "G", "C", "GC")
VERSION = "dual_wall_shortcuts_v1"


@dataclass(frozen=True)
class DualWallDataConfig:
    root: Path
    train_count: int = 2048
    eval_count: int = 64
    audit_count: int = 32
    audit_only: bool = False


def wrap(deg: float) -> float:
    return (float(deg) + 180.) % 360. - 180.


def vec(deg: float, scale: float = 1.) -> np.ndarray:
    r = math.radians(deg)
    return scale * np.asarray((math.cos(r), math.sin(r)), dtype=np.float64)


def normal(phi: float) -> np.ndarray:
    t = vec(phi)
    return np.asarray((-t[1], t[0]), dtype=np.float64)


def reflect(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    return v - 2. * float(v @ n) * n


def rng_for(seed: int, salt: int) -> np.random.Generator:
    return np.random.default_rng((int(seed) * 6364136223846793005 + salt) & ((1 << 64) - 1))


def event(phi: float, alpha: float, contact: np.ndarray, side: str, wall_id: str, state: str, color: str) -> dict:
    theta_in = wrap(phi + alpha)
    n, vin = normal(phi), vec(theta_in, SPEED)
    if vin @ n >= -1e-10:
        raise ValueError(f"non-incident draw phi={phi}, alpha={alpha}")
    vout, tc = reflect(vin, n), CONTACT_FRAME / FPS
    centre_contact = contact + RADIUS * n
    positions = np.asarray([
        centre_contact + (vin if f / FPS <= tc else vout) * (f / FPS - tc)
        for f in range(FRAMES)
    ])
    return {
        "screen_slot": side, "wall_instance": wall_id, "pose_family": "Phi_minus" if wall_id == "g_minus" else "Phi_plus",
        "state": state, "color": color, "color_rgb": list(COLORS[color]), "phi_wall": float(phi),
        "alpha_in": float(alpha), "alpha_out": float(wrap(math.degrees(math.atan2(vout[1], vout[0])) - phi)),
        "theta_in": float(theta_in), "theta_out": float(wrap(math.degrees(math.atan2(vout[1], vout[0])))),
        "contact_point": contact.tolist(), "normal": n.tolist(), "velocity_in": vin.tolist(), "velocity_out": vout.tolist(),
        "collision_time": tc, "first_contact_frame": CONTACT_FRAME, "positions": positions.tolist(), "radius": RADIUS, "speed": SPEED,
    }


def _valid(events: dict[str, dict]) -> bool:
    for e in events.values():
        p, n, c = np.asarray(e["positions"]), np.asarray(e["normal"]), np.asarray(e["contact_point"])
        if p.min() < RADIUS or p.max() > 1 - RADIUS:
            return False
        if ((p - c) @ n < RADIUS - 1e-8).any():
            return False
        d = np.diff(p, axis=0)
        if not np.allclose(d[:CONTACT_FRAME], np.asarray(e["velocity_in"]) / FPS, atol=1e-8):
            return False
        if not np.allclose(d[CONTACT_FRAME:], np.asarray(e["velocity_out"]) / FPS, atol=1e-8):
            return False
    left, right = events["left"], events["right"]
    if np.linalg.norm(np.asarray(left["positions"]) - np.asarray(right["positions"]), axis=1).min() <= 2 * RADIUS + .04:
        return False
    for a, b in ((left, right), (right, left)):
        p, c, n, t = np.asarray(a["positions"]), np.asarray(b["contact_point"]), np.asarray(b["normal"]), vec(b["phi_wall"])
        along, dist = abs((p - c) @ t), abs((p - c) @ n)
        if ((along <= WALL_HALF_LENGTH + RADIUS) & (dist <= RADIUS + WALL_THICKNESS / 128)).any():
            return False
    return True


def _base(seed: int, quartet_index: int) -> dict:
    """Draw all v1 nuisance once; position permutation is exactly balanced."""
    r = rng_for(seed, 101)
    # A deterministic retry only changes nuisance for geometrically invalid
    # renders; it is shared by all tracks and all factorial variants.
    for _ in range(1000):
        phi_minus = float(r.uniform(*PHI["minus"])); phi_plus = float(r.uniform(*PHI["plus"]))
        a = float(rng_for(seed, 211).uniform(*ALPHA["A"])); b = float(rng_for(seed, 223).uniform(*ALPHA["B"]))
        jl, jr = r.uniform(-.012, .012, 2), r.uniform(-.012, .012, 2)
        base = {
            "phi_minus": phi_minus, "phi_plus": phi_plus, "a": a, "b": b,
            "pos_perm": int(quartet_index % 2),
            "contact_left": np.asarray((.25 + jl[0], .50 + jl[1])),
            "contact_right": np.asarray((.75 + jr[0], .50 + jr[1])),
        }
        # Validate all possible state bindings before freezing the nuisance.
        for left_is_minus in (True, False):
            es = _events(base, "A", "B", "red", "blue", left_is_minus)
            fs = _events(base, "B", "A", "blue", "red", left_is_minus)
            if not (_valid(es) and _valid(fs)):
                break
        else:
            return base
    raise RuntimeError(f"could not draw legal v1 quartet seed={seed}")


def _events(base: dict, state_minus: str, state_plus: str, color_minus: str, color_plus: str, left_is_minus: bool | None = None) -> dict[str, dict]:
    if left_is_minus is None:
        left_is_minus = base["pos_perm"] == 0
    side_minus, side_plus = ("left", "right") if left_is_minus else ("right", "left")
    contact_minus = base["contact_left"] if side_minus == "left" else base["contact_right"]
    contact_plus = base["contact_left"] if side_plus == "left" else base["contact_right"]
    return {
        side_minus: event(base["phi_minus"], base["a"] if state_minus == "A" else base["b"], contact_minus, side_minus, "g_minus", state_minus, color_minus),
        side_plus: event(base["phi_plus"], base["a"] if state_plus == "A" else base["b"], contact_plus, side_plus, "g_plus", state_plus, color_plus),
    }


def _assignment(variant: str) -> tuple[str, str, str, str]:
    # Return (state(g-), state(g+), colour(g-), colour(g+)).
    return {
        "ID": ("A", "B", "red", "blue"),
        "G": ("B", "A", "blue", "red"),
        "C": ("A", "B", "blue", "red"),
        "GC": ("B", "A", "red", "blue"),
    }[variant]


def record(seed: int, quartet_index: int, variant: str, track: str | None = None) -> dict:
    base = _base(seed, quartet_index)
    if track is None:
        sm, sp, cm, cp = _assignment(variant)
    elif track == "G":
        sm, sp = "A", "B"
        cm, cp = ("red", "blue") if (quartet_index // 2) % 2 == 0 else ("blue", "red")
    elif track == "C":
        sm, sp = ("A", "B") if (quartet_index // 2) % 2 == 0 else ("B", "A")
        # Colour is bound to *state*, never to the g-/g+ wall instance.
        cm, cp = (("red", "blue") if sm == "A" else ("blue", "red"))
    elif track == "GC":
        sm, sp, cm, cp = "A", "B", "red", "blue"
    else:
        raise ValueError(track)
    events = _events(base, sm, sp, cm, cp)
    if not _valid(events):
        raise RuntimeError(f"unexpected invalid frozen record seed={seed} {variant} {track}")
    return {
        "benchmark_version": VERSION, "base_seed": int(seed), "seed": int(seed), "quartet_id": f"q{seed:08d}", "track": track or "eval",
        "variant": variant, "pos_perm": int(base["pos_perm"]), "fps": FPS, "frames": FRAMES,
        "background": list(BACKGROUND), "wall_thickness_px": WALL_THICKNESS, "wall_half_length": WALL_HALF_LENGTH,
        "interactions": events,
        "nuisance": {"phi_minus": base["phi_minus"], "phi_plus": base["phi_plus"], "contact_left": base["contact_left"].tolist(),
                     "contact_right": base["contact_right"].tolist(), "radius": RADIUS, "speed": SPEED,
                     "collision_time": CONTACT_FRAME / FPS, "a": base["a"], "b": base["b"], "simulator_nuisance_seed": int(seed)},
        "paired_counterfactual_ids": {"g_minus": f"q{seed:08d}:g_minus:opposite_state", "g_plus": f"q{seed:08d}:g_plus:opposite_state"},
    }


def render(rec: dict, resolution: int = 128) -> np.ndarray:
    yy, xx = np.mgrid[:resolution, :resolution]
    world = np.stack(((xx + .5) / resolution, 1 - (yy + .5) / resolution), -1)
    frames = np.full((FRAMES, resolution, resolution, 3), BACKGROUND, dtype=np.uint8)
    for e in rec["interactions"].values():
        c, n, t = np.asarray(e["contact_point"]), np.asarray(e["normal"]), vec(e["phi_wall"])
        wall = (abs((world - c) @ n) <= WALL_THICKNESS / resolution / 2) & (abs((world - c) @ t) <= WALL_HALF_LENGTH)
        frames[:, wall] = WALL
    for f in range(FRAMES):
        for side in ("left", "right"):
            e = rec["interactions"][side]; p = np.asarray(e["positions"])[f]
            ball = (world[..., 0] - p[0]) ** 2 + (world[..., 1] - p[1]) ** 2 <= RADIUS ** 2
            frames[f, ball] = e["color_rgb"]
    return frames


def _save(rec: dict, root: Path, index: int) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    stem = f"sample_{index:06d}"; video, meta = root / f"{stem}.mp4", root / f"{stem}.json"
    with imageio.get_writer(video, fps=FPS, codec="libx264", quality=10, macro_block_size=None) as w:
        for frame in render(rec): w.append_data(frame)
    meta.write_text(json.dumps(rec, indent=2) + "\n")
    return {"video": video.name, "source": video.name, "metadata": meta.name,
            "sample_id": (
                f"{rec['quartet_id']}:{rec['track']}:{rec['variant']}"
            ),
            "base_seed": rec["base_seed"], "seed": rec["seed"],
            "quartet_id": rec["quartet_id"], "track": rec["track"], "variant": rec["variant"], "pos_perm": rec["pos_perm"]}


def _write_rows(root: Path, rows: list[dict]) -> None:
    with (root / "metadata.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def _write_train(root: Path, track: str, count: int, seed_start: int) -> None:
    rows = [_save(record(seed_start + q, q, "train", track), root, q) for q in range(count)]
    _write_rows(root, rows)


def _write_eval(root: Path, count: int, seed_start: int) -> None:
    rows = []
    for q in range(count):
        for variant in EVAL_VARIANTS:
            rows.append(_save(record(seed_start + q, q, variant), root, len(rows)))
    _write_rows(root, rows)


def config() -> dict:
    return {"benchmark_version": VERSION, "video": {"resolution": 128, "frames": FRAMES, "fps": FPS, "prefix_pixel_frames": 5, "contact_frame": CONTACT_FRAME},
            "physics": {"radius": RADIUS, "speed": SPEED, "reflection": "alpha_out = -alpha_in", "analytic_continuous_time": True},
            "alpha": ALPHA, "phi": {"Phi_minus": PHI["minus"], "Phi_plus": PHI["plus"]}, "train_seeds": "2100000...", "eval_seeds": "2200000...",
            "tracks": {"G": "g-<->A,g+<->B; colour balanced independently", "C": "A<->red,B<->blue; pose/state balanced independently", "GC": "g-<->A<->red,g+<->B<->blue"},
            "eval_mapping": {v: dict(zip(("g-_state", "g+_state", "g-_color", "g+_color"), _assignment(v))) for v in EVAL_VARIANTS},
            "pos_perm": "quartet-index alternating, exactly 50/50 in each even-sized split"}


def build(root: Path, train_count: int = 2048, eval_count: int = 64, audit_count: int = 32, audit_only: bool = False) -> None:
    if train_count % 4 or eval_count % 2 or audit_count % 2:
        raise ValueError("v1 counts must be even; train count must also divide by four for exact balance")
    root.mkdir(parents=True, exist_ok=True)
    if audit_only:
        _write_eval(root / "audit" / "eval", audit_count, 2_200_000)
    else:
        for t in ("G", "C", "GC"):
            _write_train(root / "videos" / f"train_{t}", t, train_count, 2_100_000)
        _write_eval(root / "videos" / "eval", eval_count, 2_200_000)
    (root / "metadata").mkdir(exist_ok=True); (root / "manifests").mkdir(exist_ok=True)
    with (root / "metadata" / "generation_config.yaml").open("w") as f: yaml.safe_dump(config(), f, sort_keys=False)
    for name, rel in (("train_G", "videos/train_G/metadata.csv"), ("train_C", "videos/train_C/metadata.csv"), ("train_GC", "videos/train_GC/metadata.csv"), ("eval_factorial", "videos/eval/metadata.csv")):
        if (root / rel).exists(): (root / "manifests" / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in csv.DictReader((root / rel).open())))
    (root / "metadata" / "dataset_card.md").write_text(
        f"# {VERSION}\n\nFrozen dual interaction ball--wall benchmark.  Train tracks contain 2048 videos; eval is 64 matched quartets × ID/G/C/GC.  "
        "Each interaction is a continuous analytic elastic reflection and both slots are balanced by `pos_perm`.\n")
    common_samples = []
    for metadata_csv in sorted(root.glob("**/metadata.csv")):
        bank_name = metadata_csv.parent.name
        split = (
            "eval"
            if "eval" in bank_name
            else "train"
            if bank_name.startswith("train_")
            else "audit"
        )
        common_samples.extend(
            samples_from_metadata_csv(
                root,
                metadata_csv,
                split=split,
                subset=metadata_csv.parent.relative_to(root).as_posix(),
            )
        )
    write_dataset(
        root,
        experiment="dual_wall",
        dataset=VERSION,
        samples=common_samples,
        extra={"generation_config": "metadata/generation_config.yaml"},
    )


def generate_dataset(config: DualWallDataConfig) -> Path:
    build(
        config.root,
        config.train_count,
        config.eval_count,
        config.audit_count,
        config.audit_only,
    )
    return config.root


def _records(directory: Path) -> list[dict]:
    with (directory / "metadata.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        json.loads((directory / row["metadata"]).read_text())
        for row in rows
    ]


def audit_dataset(root: Path, out: Path) -> dict:
    """Check balance, matched quartets, and analytic collision invariants."""
    violations: list[list[object]] = []
    tracks: dict[str, object] = {}
    for track in ("G", "C", "GC"):
        records = _records(root / "videos" / f"train_{track}")
        interactions = [
            event
            for record in records
            for event in record["interactions"].values()
        ]
        state_pose = Counter(
            (item["state"], item["wall_instance"])
            for item in interactions
        )
        state_color = Counter(
            (item["state"], item["color"])
            for item in interactions
        )
        left = Counter(
            item["screen_slot"] for item in interactions
        )
        if left["left"] != left["right"]:
            violations.append(["balance", track, "screen_slot"])
        if track in ("G", "GC"):
            expected = {
                ("A", "g_minus"): len(interactions) // 2,
                ("B", "g_plus"): len(interactions) // 2,
            }
            if dict(state_pose) != expected:
                violations.append(["binding", track, "state_pose"])
        if track in ("C", "GC"):
            expected = {
                ("A", "red"): len(interactions) // 2,
                ("B", "blue"): len(interactions) // 2,
            }
            if dict(state_color) != expected:
                violations.append(["binding", track, "state_color"])
        tracks[track] = {
            "samples": len(records),
            "state_pose": {
                str(key): value for key, value in state_pose.items()
            },
            "state_color": {
                str(key): value for key, value in state_color.items()
            },
        }

    groups: defaultdict[str, list[dict]] = defaultdict(list)
    reflection_errors = []
    for record in _records(root / "videos" / "eval"):
        groups[record["quartet_id"]].append(record)
        for event in record["interactions"].values():
            reflection_errors.append(
                abs(wrap(event["alpha_out"] + event["alpha_in"]))
            )
            positions = np.asarray(event["positions"])
            normal_vector = np.asarray(event["normal"])
            contact = np.asarray(event["contact_point"])
            if (
                positions.min() < event["radius"] - 1e-8
                or positions.max() > 1.0 - event["radius"] + 1e-8
                or (
                    (positions - contact) @ normal_vector
                    < event["radius"] - 1e-8
                ).any()
            ):
                violations.append([
                    record["quartet_id"],
                    record["variant"],
                    "physics",
                ])
    for group_id, records in groups.items():
        if {record["variant"] for record in records} != set(
            EVAL_VARIANTS
        ):
            violations.append([group_id, "incomplete_quartet"])
        nuisance = records[0]["nuisance"]
        for record in records[1:]:
            if record["nuisance"] != nuisance:
                violations.append([
                    group_id,
                    record["variant"],
                    "nuisance_changed",
                ])
    report = {
        "benchmark_version": VERSION,
        "tracks": tracks,
        "eval": {
            "quartets": len(groups),
            "reflection_error_mae_deg": float(
                np.mean(reflection_errors)
            ),
        },
        "violations": violations,
    }
    out.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2) + "\n"
    (out / "audit.json").write_text(payload)
    (out / "balance_audit.json").write_text(payload)
    if violations:
        raise RuntimeError(json.dumps(report, indent=2))
    return report


def audit_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_dataset(args.root, args.out), indent=2))


def generation_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=2048)
    parser.add_argument("--eval-count", type=int, default=64)
    parser.add_argument("--audit-count", type=int, default=32)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    generate_dataset(
        DualWallDataConfig(
            root=args.root,
            train_count=args.train_count,
            eval_count=args.eval_count,
            audit_count=args.audit_count,
            audit_only=args.audit_only,
        )
    )


if __name__ == "__main__":
    generation_cli()
