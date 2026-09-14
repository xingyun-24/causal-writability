"""Data generation for frozen ``support_release_v1``.

The simulator intentionally has no contact solver in its dataset loop.  The
only contact phase is an analytic point mass sliding down the *remaining*
incline under gravity; at frame four both kinematic supports have left the
scene and the ball is propagated by an analytic ballistic trajectory.  This
makes continuity, release velocity and all matched counterfactuals exact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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

FPS, FRAMES, PREFIX = 15, 49, 5
RADIUS = .040
GRAVITY = .160                 # world units / second^2, y points upward
RAMP_ANGLE_DEG = 42.0
RAMP_LENGTH, RAMP_THICKNESS = .160, 4
CENTRES = {"g_left": .34, "g_right": .66}
BALL_Y, SUPPORT_BASE_Y = .860, .815
BACKGROUND, SUPPORT = (24, 24, 28), (158, 158, 158)
COLORS = {"red": (235, 45, 45), "blue": (45, 95, 235)}
EVAL_VARIANTS = ("ID", "G", "C", "GC")
VERSION = "support_release_v1"


@dataclass(frozen=True)
class SupportReleaseDataConfig:
    root: Path
    train_count: int = 2048
    eval_count: int = 64
    audit_only: bool = False
# Release starts immediately after frame zero; frame one shows the first
# support beginning to retreat.  The 0.255 s supported interval is long enough
# to produce a >15 px A/B endpoint separation while the later ballistic arc
# remains safely within the 49-frame canvas.
T_START, T_RELEASE = .012, 4 / FPS


def _rng(seed: int, salt: int) -> np.random.Generator:
    return np.random.default_rng((int(seed) * 6364136223846793005 + salt) & ((1 << 64) - 1))


def _as_list(x):
    return np.asarray(x, dtype=np.float64).tolist()


def support_offset(frame: int, first: bool) -> float | None:
    """Vertical visual retreat; ``None`` means collision is disabled/offscreen."""
    # State-dependent first support begins leaving at f1 and is absent f2.
    # The other begins at f3 and is absent at the final prefix frame f4.
    begin = 1 if first else 3
    if frame >= begin + 1:
        return None
    return 0.0 if frame < begin else -.125


def base_params(seed: int) -> dict:
    # Both states share this nuisance; only a small common vertical placement
    # jitter is admitted.  It cannot encode state, cue, or geometry.
    r = _rng(seed, 17)
    return {"ball_y": float(BALL_Y + r.uniform(-.004, .004)), "light": float(r.uniform(.98, 1.02)),
            "ramp_angle_deg": RAMP_ANGLE_DEG, "gravity": GRAVITY, "radius": RADIUS,
            "simulator_nuisance_seed": int(seed)}


def state_event(seed: int, state: str, geometry: str, color: str) -> dict:
    """Generate a state A/B mirror pair in one geometry chart."""
    if state not in ("A", "B") or geometry not in CENTRES or color not in COLORS:
        raise ValueError((state, geometry, color))
    q = base_params(seed)
    beta = math.radians(RAMP_ANGLE_DEG)
    # Remaining right ramp sends the ball left/down after left support is
    # removed; the other state is the exact horizontal mirror.
    sign = -1.0 if state == "A" else 1.0
    tangent = np.asarray((sign * math.cos(beta), -math.sin(beta)))
    a_slide = GRAVITY * math.sin(beta) * tangent
    p0 = np.asarray((CENTRES[geometry], q["ball_y"]))
    duration = T_RELEASE - T_START
    p_release = p0 + .5 * a_slide * duration * duration
    v_release = a_slide * duration
    gravity = np.asarray((0.0, -GRAVITY))
    pos = []
    vel = []
    for f in range(FRAMES):
        t = f / FPS
        if t <= T_START:
            p, v = p0, np.zeros(2)
        elif t <= T_RELEASE:
            dt = t - T_START
            p, v = p0 + .5 * a_slide * dt * dt, a_slide * dt
        else:
            dt = t - T_RELEASE
            p, v = p_release + v_release * dt + .5 * gravity * dt * dt, v_release + gravity * dt
        pos.append(p); vel.append(v)
    # V has a bottom just under the ball.  The support which remains is the
    # one whose ascending endpoint is opposite the direction of travel.
    bottom = np.asarray((CENTRES[geometry], SUPPORT_BASE_Y))
    left_end = bottom + np.asarray((-math.cos(beta), math.sin(beta))) * RAMP_LENGTH
    right_end = bottom + np.asarray((math.cos(beta), math.sin(beta))) * RAMP_LENGTH
    first_removed = "left" if state == "A" else "right"
    supports = {
        "left": {"endpoints": [_as_list(left_end), _as_list(bottom)], "first_removed": first_removed == "left"},
        "right": {"endpoints": [_as_list(bottom), _as_list(right_end)], "first_removed": first_removed == "right"},
    }
    return {"state": state, "geometry": geometry, "color": color, "color_rgb": list(COLORS[color]),
            "first_removed_support": first_removed, "support_timing": {"first_begin_frame": 1, "first_absent_frame": 2, "second_begin_frame": 3, "second_absent_frame": 4},
            "positions": _as_list(pos), "velocities": _as_list(vel), "velocity_at_release": _as_list(v_release),
            "gravity": _as_list(gravity), "initial_position": _as_list(p0), "release_position": _as_list(p_release),
            "radius": RADIUS, "ramp_angle_deg": RAMP_ANGLE_DEG, "support_base": _as_list(bottom), "supports": supports,
            "physics": q}


def _valid(e: dict) -> tuple[bool, str]:
    p = np.asarray(e["positions"]); r = float(e["radius"])
    if not (p[:, 0].min() >= r and p[:, 0].max() <= 1-r and p[:, 1].min() >= r and p[:, 1].max() <= 1-r):
        return False, "out_of_frame"
    vx = float(e["velocity_at_release"][0])
    if not ((e["state"] == "A" and vx < -.012) or (e["state"] == "B" and vx > .012)):
        return False, "weak_or_wrong_vx"
    if abs((p[-1, 0] - p[4, 0])) < .055:
        return False, "weak_separation"
    if not np.allclose(np.diff(p[5:], 2, axis=0), np.asarray(e["gravity"])[None] / FPS**2, atol=1e-10):
        return False, "non_ballistic_future"
    return True, ""


def _assignment(state: str, variant: str) -> tuple[str, str]:
    # (geometry, colour) for this true physical state.
    canonical = ("g_left", "red") if state == "A" else ("g_right", "blue")
    opposite = ("g_right", "blue") if state == "A" else ("g_left", "red")
    if variant == "ID": return canonical
    if variant == "G": return (opposite[0], canonical[1])
    if variant == "C": return (canonical[0], opposite[1])
    if variant == "GC": return opposite
    raise ValueError(variant)


def train_assignment(index: int, track: str) -> tuple[str, str, str]:
    state = "A" if index % 2 == 0 else "B"
    # Pairs alternate every two records, giving exact joint balance in a
    # 2048-element split while every track uses identical base seeds.
    flip = (index // 2) % 2
    if track == "G":
        geometry = "g_left" if state == "A" else "g_right"
        color = ("blue" if state == "A" else "red") if flip else ("red" if state == "A" else "blue")
    elif track == "C":
        color = "red" if state == "A" else "blue"
        geometry = (("g_right" if state == "A" else "g_left") if flip else ("g_left" if state == "A" else "g_right"))
    elif track == "GC":
        geometry, color = ("g_left", "red") if state == "A" else ("g_right", "blue")
    else:
        raise ValueError(track)
    return state, geometry, color


def record(seed: int, index: int, state: str, geometry: str, color: str, *, track: str, variant: str) -> dict:
    event = state_event(seed, state, geometry, color)
    ok, why = _valid(event)
    if not ok: raise RuntimeError(f"illegal support record {seed}: {why}")
    return {"benchmark_version": VERSION, "base_seed": int(seed), "seed": int(seed), "octet_id": f"o{seed:08d}",
            "track": track, "variant": variant, "state": state, "geometry": geometry, "color": color,
            "fps": FPS, "frames": FRAMES, "prefix_pixel_frames": PREFIX, "event": event,
            "nuisance": base_params(seed),
            "paired_counterfactual_ids": {"correct": f"o{seed:08d}:{geometry}:{color}:{state}",
                                           "opposite_state": f"o{seed:08d}:{geometry}:{color}:{'B' if state=='A' else 'A'}"}}


def _draw_segment(img: np.ndarray, a: np.ndarray, b: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
    h, w = img.shape[:2]; yy, xx = np.mgrid[:h, :w]
    # Coordinate conversion world y-up -> pixels y-down.
    a = np.asarray((a[0]*w, (1-a[1])*h)); b = np.asarray((b[0]*w, (1-b[1])*h))
    d=b-a; den=max(float(d@d),1e-12); u=np.clip(((xx-a[0])*d[0]+(yy-a[1])*d[1])/den,0,1)
    dist=np.hypot(xx-(a[0]+u*d[0]),yy-(a[1]+u*d[1]))
    img[dist<=thickness/2]=color


def render(rec: dict, resolution: int=128) -> np.ndarray:
    event=rec["event"]; frames=np.full((FRAMES,resolution,resolution,3),BACKGROUND,dtype=np.uint8)
    for f in range(FRAMES):
        for support in event["supports"].values():
            off=support_offset(f,bool(support["first_removed"]))
            if off is not None:
                a,b=np.asarray(support["endpoints"]); shift=np.asarray((0.,off))
                _draw_segment(frames[f],a+shift,b+shift,SUPPORT,RAMP_THICKNESS)
        p=np.asarray(event["positions"])[f]; yy,xx=np.mgrid[:resolution,:resolution]; wx=(xx+.5)/resolution;wy=1-(yy+.5)/resolution
        ball=(wx-p[0])**2+(wy-p[1])**2<=RADIUS**2;frames[f,ball]=event["color_rgb"]
    return frames


def _save(rec:dict,root:Path,index:int)->dict:
    root.mkdir(parents=True,exist_ok=True);stem=f"sample_{index:06d}";video=root/f"{stem}.mp4";meta=root/f"{stem}.json"
    with imageio.get_writer(video,fps=FPS,codec="libx264",quality=10,macro_block_size=None) as w:
        for frame in render(rec):w.append_data(frame)
    meta.write_text(json.dumps(rec,indent=2)+"\n")
    return {
        "video": video.name,
        "source": video.name,
        "metadata": meta.name,
        "sample_id": (
            f"{rec['octet_id']}:{rec['track']}:"
            f"{rec['state']}:{rec['variant']}"
        ),
        "base_seed": rec["base_seed"],
        "seed": rec["seed"],
        "octet_id": rec["octet_id"],
        "track": rec["track"],
        "variant": rec["variant"],
        "state": rec["state"],
        "geometry": rec["geometry"],
        "color": rec["color"],
    }


def _write_csv(root:Path,rows:list[dict])->None:
    with (root/"metadata.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def frozen_config(train_count:int,eval_count:int)->dict:
    d={"benchmark_version":VERSION,"video":{"resolution":128,"frames":FRAMES,"fps":FPS,"prefix_pixel_frames":PREFIX},
       "physics":{"integrator":"analytic ramp gravity then ballistic gravity","gravity":GRAVITY,"radius":RADIUS,"ramp_angle_deg":RAMP_ANGLE_DEG,"release_frames":[1,2,3,4]},
       "cues":{"state":{"A":"left support first -> vx<0","B":"right support first -> vx>0"},"geometry":{"g_left":CENTRES["g_left"],"g_right":CENTRES["g_right"]},"appearance":{"red":"A canonical","blue":"B canonical"}},
       "eval_mapping":{"A":{"ID":["g_left","red"],"G":["g_right","red"],"C":["g_left","blue"],"GC":["g_right","blue"]},"B":{"ID":["g_right","blue"],"G":["g_left","blue"],"C":["g_right","red"],"GC":["g_left","red"]}},
       "counts":{"train_per_track":train_count,"eval_base_seeds":eval_count,"eval_videos":8*eval_count},"seeds":{"train_start":2300000,"eval_start":2400000}}
    d["config_hash"]=hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()[:16];return d


def build(root:Path,train_count:int=2048,eval_count:int=64,audit_only:bool=False)->None:
    if train_count%4 or eval_count%2:raise ValueError("train count must divide by four; eval count must be even")
    root.mkdir(parents=True,exist_ok=True)
    if audit_only:
        eval_count=min(eval_count,32); eval_root=root/"audit"/"eval"
        rows=[]
        for i in range(eval_count):
            seed=2400000+i
            for state in ("A","B"):
                for variant in EVAL_VARIANTS:
                    g,c=_assignment(state,variant);rows.append(_save(record(seed,i,state,g,c,track="eval",variant=variant),eval_root,len(rows)))
        _write_csv(eval_root,rows)
    else:
        for track in ("G","C","GC"):
            rows=[];root_t=root/"videos"/f"train_{track}"
            for i in range(train_count):
                state,g,c=train_assignment(i,track);rows.append(_save(record(2300000+i,i,state,g,c,track=track,variant="train"),root_t,i))
            _write_csv(root_t,rows)
        rows=[];eval_root=root/"videos"/"eval"
        for i in range(eval_count):
            seed=2400000+i
            for state in ("A","B"):
                for variant in EVAL_VARIANTS:
                    g,c=_assignment(state,variant);rows.append(_save(record(seed,i,state,g,c,track="eval",variant=variant),eval_root,len(rows)))
        _write_csv(eval_root,rows)
    (root/"metadata").mkdir(exist_ok=True);(root/"manifests").mkdir(exist_ok=True)
    cfg=frozen_config(train_count,eval_count)
    (root/"metadata"/"generation_config.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False))
    for name,rel in (("train_G","videos/train_G/metadata.csv"),("train_C","videos/train_C/metadata.csv"),("train_GC","videos/train_GC/metadata.csv"),("eval_octets","videos/eval/metadata.csv")):
        p=root/rel
        if p.exists():(root/"manifests"/f"{name}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in csv.DictReader(p.open())))
    (root/"metadata"/"dataset_card.md").write_text(f"# {VERSION}\n\nFrozen single-ball support-release benchmark.  Each eval base seed has two true states × ID/G/C/GC = eight matched videos.  Config hash: `{cfg['config_hash']}`.\n")
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
        experiment="support_release",
        dataset=VERSION,
        samples=common_samples,
        extra={"generation_config": "metadata/generation_config.yaml"},
    )


def generate_dataset(config: SupportReleaseDataConfig) -> Path:
    build(
        config.root,
        config.train_count,
        config.eval_count,
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


def audit_dataset(
    root: Path,
    out: Path,
    *,
    audit_only: bool = False,
) -> dict:
    """Check balance, matched octets, and ballistic invariants."""
    violations: list[list[object]] = []
    tracks: dict[str, object] = {}
    for track in (() if audit_only else ("G", "C", "GC")):
        records = _records(root / "videos" / f"train_{track}")
        state = Counter(record["state"] for record in records)
        state_geometry = Counter(
            (record["state"], record["geometry"])
            for record in records
        )
        state_color = Counter(
            (record["state"], record["color"])
            for record in records
        )
        if state["A"] != state["B"]:
            violations.append(["balance", track, "state"])
        if track in ("G", "GC") and dict(state_geometry) != {
            ("A", "g_left"): len(records) // 2,
            ("B", "g_right"): len(records) // 2,
        }:
            violations.append(["binding", track, "state_geometry"])
        if track in ("C", "GC") and dict(state_color) != {
            ("A", "red"): len(records) // 2,
            ("B", "blue"): len(records) // 2,
        }:
            violations.append(["binding", track, "state_color"])
        tracks[track] = {
            "samples": len(records),
            "state": dict(state),
            "state_geometry": {
                str(key): value
                for key, value in state_geometry.items()
            },
            "state_color": {
                str(key): value
                for key, value in state_color.items()
            },
        }

    groups: defaultdict[str, list[dict]] = defaultdict(list)
    eval_directory = (
        root / "audit" / "eval"
        if audit_only
        else root / "videos" / "eval"
    )
    for record in _records(eval_directory):
        groups[record["octet_id"]].append(record)
        event = record["event"]
        positions = np.asarray(event["positions"])
        if (
            np.any(positions < event["radius"] - 1e-10)
            or np.any(positions > 1.0 - event["radius"] + 1e-10)
        ):
            violations.append([
                record["octet_id"],
                record["state"],
                record["variant"],
                "out_of_frame",
            ])
        if not np.allclose(
            np.diff(positions[5:], 2, axis=0),
            np.asarray(event["gravity"])[None] / record["fps"] ** 2,
            atol=1e-10,
        ):
            violations.append([
                record["octet_id"],
                record["state"],
                record["variant"],
                "future_not_ballistic",
            ])
    expected = {
        (state, variant)
        for state in ("A", "B")
        for variant in EVAL_VARIANTS
    }
    for group_id, records in groups.items():
        if {
            (record["state"], record["variant"])
            for record in records
        } != expected:
            violations.append([group_id, "incomplete_octet"])
        nuisance = records[0]["nuisance"]
        if any(record["nuisance"] != nuisance for record in records):
            violations.append([group_id, "nuisance_changed"])
    report = {
        "benchmark_version": VERSION,
        "tracks": tracks,
        "eval": {"octets": len(groups)},
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
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            audit_dataset(
                args.root,
                args.out,
                audit_only=args.audit_only,
            ),
            indent=2,
        )
    )


def generation_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=2048)
    parser.add_argument("--eval-count", type=int, default=64)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    generate_dataset(
        SupportReleaseDataConfig(
            root=args.root,
            train_count=args.train_count,
            eval_count=args.eval_count,
            audit_only=args.audit_only,
        )
    )


if __name__ == "__main__":
    generation_cli()
