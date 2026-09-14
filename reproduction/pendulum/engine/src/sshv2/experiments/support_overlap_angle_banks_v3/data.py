"""Data generation for Support-Overlap angle-mechanism banks v3.

The high bank holds out ``34 deg remaining`` under a geometry flip, but the
same local mechanism is present in the low-bank ID examples.  This separates
the intended geometry-bound case retrieval test from the missing-mechanism
confound in Support v2.
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
from typing import Any

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

VERSION = "support_overlap_angle_banks_v3"
EXPERIMENT_NAME = VERSION
FPS, FRAMES, PREFIX, RESOLUTION = 15, 49, 5, 128
RADIUS = 0.040
BACKGROUND = (22, 22, 26)
SUPPORT = (158, 158, 158)
COLORS = {0: (235, 45, 45), 1: (45, 95, 235)}
COLOR_NAMES = {0: "red", 1: "blue"}
BANKS = {"low": (22.0, 34.0), "high": (34.0, 48.0)}  # shallow, steep
CONDITIONS = (("P", .50), ("C", 1.00), ("G", .95), ("G", .99), ("G", 1.00),
              ("GC", .95), ("GC", .99), ("GC", 1.00))


@dataclass(frozen=True)
class SupportOverlapDataConfig:
    root: Path
    train_count: int = 2048
    eval_count: int = 64
    calibration_count: int = 10_000


def _valid_positions(positions: np.ndarray) -> bool:
    return valid_positions(positions, radius=RADIUS)


def _tag(alpha: float) -> str:
    return f"a{int(round(100 * alpha)):03d}"


def _json_default(x: Any):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x).__name__)


def _unit(angle_deg: float) -> np.ndarray:
    r = math.radians(angle_deg)
    return np.asarray((math.cos(r), math.sin(r)), dtype=np.float64)


class SupportOverlapScene:
    ramp_length, ramp_thickness = .160, 4
    gravity, retreat_duration = .160, .060

    @staticmethod
    def nuisance(seed: int) -> dict[str, float]:
        r = _rng(seed, 411)
        return {
            "delta_t": float(r.uniform(.100, .135)),
            "first_retraction_time": float(r.uniform(.035, .050)),
            "scene_dx": float(r.uniform(-.035, .035)),
            "scene_dy": float(r.uniform(-.004, .004)),
            "light": float(r.uniform(.98, 1.02)),
        }

    @staticmethod
    def _mirror_points(x: np.ndarray, G: int) -> np.ndarray:
        y = np.asarray(x, dtype=np.float64).copy()
        if G:
            y[..., 0] = 1.0 - y[..., 0]
        return y

    @staticmethod
    def _screen_tangent(S: int, remaining_angle: float) -> np.ndarray:
        # Screen S is the first-removed side.  The ball moves away from it.
        return np.asarray(((-1.0 if S == 0 else 1.0) * math.cos(math.radians(remaining_angle)),
                           -math.sin(math.radians(remaining_angle))), dtype=np.float64)

    @staticmethod
    def _trajectory(p0: np.ndarray, q: dict[str, float], S: int, remaining_angle: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        tangent = SupportOverlapScene._screen_tangent(S, remaining_angle)
        acceleration = SupportOverlapScene.gravity * abs(tangent[1]) * tangent
        t_first = q["first_retraction_time"]
        t_slide = t_first + SupportOverlapScene.retreat_duration
        t_release = t_first + q["delta_t"] + SupportOverlapScene.retreat_duration
        p_release = p0 + .5 * acceleration * q["delta_t"] ** 2
        v_release = acceleration * q["delta_t"]
        gravity = np.asarray((0.0, -SupportOverlapScene.gravity), dtype=np.float64)
        positions, velocities = [], []
        for f in range(FRAMES):
            t = f / FPS
            if t <= t_slide:
                p, v = p0, np.zeros(2)
            elif t <= t_release:
                dt = t - t_slide
                p, v = p0 + .5 * acceleration * dt * dt, acceleration * dt
            else:
                dt = t - t_release
                p, v = p_release + v_release * dt + .5 * gravity * dt * dt, v_release + gravity * dt
            positions.append(p); velocities.append(v)
        return (np.asarray(positions), np.asarray(velocities), p_release, v_release, t_slide, t_release)

    @staticmethod
    def event(seed: int, S: int, G: int, C: int, angle_bank: str) -> dict[str, Any]:
        if angle_bank not in BANKS or S not in (0, 1) or G not in (0, 1):
            raise ValueError((angle_bank, S, G))
        shallow, steep = BANKS[angle_bank]
        q = SupportOverlapScene.nuisance(seed)
        bottom_canonical = np.asarray((.50 + q["scene_dx"], .815 + q["scene_dy"]), dtype=np.float64)
        p0_canonical = np.asarray((.50 + q["scene_dx"], .860 + q["scene_dy"]), dtype=np.float64)
        left_end = bottom_canonical + np.asarray((-math.cos(math.radians(shallow)), math.sin(math.radians(shallow)))) * SupportOverlapScene.ramp_length
        right_end = bottom_canonical + np.asarray((math.cos(math.radians(steep)), math.sin(math.radians(steep)))) * SupportOverlapScene.ramp_length
        canonical_state = S ^ G
        canonical_retract = "left" if canonical_state == 0 else "right"
        # In the canonical shallow-left chart, left removal leaves steep; right
        # removal leaves shallow.  Mirroring restores the requested screen S.
        remaining_angle = steep if canonical_state == 0 else shallow
        p0 = SupportOverlapScene._mirror_points(p0_canonical, G)
        positions, velocities, p_release, v_release, t_slide, t_release = SupportOverlapScene._trajectory(p0, q, S, remaining_angle)
        if t_release >= (PREFIX - 1) / FPS or not _valid_positions(positions):
            raise RuntimeError("invalid support-overlap trajectory")
        bottom = SupportOverlapScene._mirror_points(bottom_canonical, G)
        supports = {
            "left": {"endpoints": [_as_list(SupportOverlapScene._mirror_points(left_end, G)), _as_list(bottom)],
                     "first_removed": S == 0},
            "right": {"endpoints": [_as_list(bottom), _as_list(SupportOverlapScene._mirror_points(right_end, G))],
                      "first_removed": S == 1},
        }
        screen_left = steep if G else shallow
        screen_right = shallow if G else steep
        return {
            "scene_type": EXPERIMENT_NAME, "S": int(S), "C": int(C), "G": int(G), "angle_bank": angle_bank,
            "color": COLOR_NAMES[C], "color_rgb": list(COLORS[C]),
            "angle_pair": {"shallow_deg": shallow, "steep_deg": steep},
            "screen_state": {"first_removed_support": "left" if S == 0 else "right", "future_branch": "left" if S == 0 else "right",
                             "relative_shallow_side": "left" if G == 0 else "right", "left_slope_deg": screen_left,
                             "right_slope_deg": screen_right, "remaining_angle_deg": remaining_angle,
                             "canonical_retract_side": canonical_retract},
            "positions": _as_list(positions), "velocities": _as_list(velocities), "initial_position": _as_list(p0),
            "release_position": _as_list(p_release), "velocity_at_release": _as_list(v_release),
            "slide_acceleration": _as_list(SupportOverlapScene.gravity * abs(SupportOverlapScene._screen_tangent(S, remaining_angle)[1]) * SupportOverlapScene._screen_tangent(S, remaining_angle)),
            "support_base": _as_list(bottom), "supports": supports, "radius": RADIUS, "mass": 1.0, "friction": 0.0,
            "gravity": [0.0, -SupportOverlapScene.gravity], "support_timing": {"first_begin_time": q["first_retraction_time"],
                "first_absent_time": t_slide, "slide_start_time": t_slide, "second_begin_time": q["first_retraction_time"] + q["delta_t"],
                "second_absent_time": t_release, "release_time": t_release}, "nuisance": q,
        }

    @staticmethod
    def trajectory_payload(event: dict[str, Any], branch: int, payload_angle: float) -> np.ndarray:
        p0 = np.asarray(event["initial_position"], dtype=np.float64)
        return SupportOverlapScene._trajectory(p0, event["nuisance"], branch, payload_angle)[0]

    @staticmethod
    def render(event: dict[str, Any]) -> np.ndarray:
        frames = np.full((FRAMES, RESOLUTION, RESOLUTION, 3), BACKGROUND, dtype=np.uint8)
        timing = event["support_timing"]
        for f in range(FRAMES):
            t = f / FPS
            for support in event["supports"].values():
                start = timing["first_begin_time"] if support["first_removed"] else timing["second_begin_time"]
                if t >= start + SupportOverlapScene.retreat_duration:
                    continue
                offset = 0.0 if t <= start else -.125 * (t - start) / SupportOverlapScene.retreat_duration
                a, b = np.asarray(support["endpoints"])
                _segment(frames[f], a + (0.0, offset), b + (0.0, offset), SUPPORT, SupportOverlapScene.ramp_thickness)
            _circle(frames[f], np.asarray(event["positions"])[f], RADIUS, tuple(event["color_rgb"]))
        return frames


def _assignments(n: int, regime: str, alpha: float, seed: int) -> list[tuple[int, int, int, str]]:
    if n % 4:
        raise ValueError("n must be divisible by four for S x angle-bank quotas")
    if regime not in ("P", "C", "G", "GC"):
        raise ValueError(regime)
    n_stratum = n // 4
    ans: list[tuple[int, int, int, str]] = []
    for bank_i, bank in enumerate(("low", "high")):
        for S in (0, 1):
            def cue(name: str, reliability: float) -> np.ndarray:
                result = np.full(n_stratum, 1 - S, dtype=np.int8)
                agree = int(round(reliability * n_stratum))
                order = _rng(seed + bank_i * 97 + S * 17, 1001 if name == "C" else 1003).permutation(n_stratum)
                result[order[:agree]] = S
                return result
            c_rel = alpha if regime in ("C", "GC") else .5
            g_rel = alpha if regime in ("G", "GC") else .5
            C, G = cue("C", c_rel), cue("G", g_rel)
            ans.extend((S, int(C[i]), int(G[i]), bank) for i in range(n_stratum))
    order = _rng(seed, 1199).permutation(n)
    return [ans[int(i)] for i in order]


def _oracles(event: dict[str, Any]) -> dict[str, list[list[float]]]:
    return {f"branch{branch}_payload{payload:d}": _as_list(SupportOverlapScene.trajectory_payload(event, branch, float(payload)))
            for branch in (0, 1) for payload in (34, 48)}


def _record(seed: int, S: int, C: int, G: int, bank: str, regime: str, alpha: float | None, variant: str | None, sample_id: str, record_id: str, *, include_oracles: bool) -> dict[str, Any]:
    event = SupportOverlapScene.event(seed, S, G, C, bank)
    record = {"benchmark_version": VERSION, "experiment_name": EXPERIMENT_NAME, "scene_type": EXPERIMENT_NAME,
              "record_id": record_id, "sample_id": sample_id, "seed": seed, "S": S, "C": C, "G": G,
              "angle_bank": bank, "train_regime": regime, "alpha": alpha, "eval_variant": variant,
              "prefix": {"pixel_frames": PREFIX, "future_start_frame": PREFIX, "frames": FRAMES, "fps": FPS, "resolution": RESOLUTION},
              "nuisance": event["nuisance"], "ground_truth_trajectory": event["positions"],
              "renderer_metadata": {"color": event["color"], "color_rgb": event["color_rgb"], "radius": RADIUS, "event": event}}
    if include_oracles:
        record["oracle_trajectories"] = _oracles(event)
        record["correct_branch"] = S
        record["correct_payload_angle"] = int(event["screen_state"]["remaining_angle_deg"])
        record["shortcut_template"] = {"branch": 1 - S, "payload_angle": 48}
    return record


def _save(record: dict[str, Any], directory: Path, index: int) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"sample_{index:06d}"; video, meta = directory / f"{stem}.mp4", directory / f"{stem}.json"
    with imageio.get_writer(video, fps=FPS, codec="libx264", quality=10, macro_block_size=None) as writer:
        for frame in SupportOverlapScene.render(record["renderer_metadata"]["event"]): writer.append_data(frame)
    meta.write_text(json.dumps(record, indent=2, default=_json_default) + "\n")
    return {"video": video.name, "source": video.name, "metadata": meta.name, "record_id": record["record_id"],
            "sample_id": record["sample_id"], "seed": record["seed"], "S": record["S"], "C": record["C"], "G": record["G"],
            "angle_bank": record["angle_bank"], "train_regime": record["train_regime"], "alpha": record["alpha"], "eval_variant": record["eval_variant"]}


def _write_rows(directory: Path, rows: list[dict[str, Any]]) -> None:
    with (directory / "metadata.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def _build_train(root: Path, regime: str, alpha: float, n: int, seed_start: int) -> dict[str, Any]:
    directory = root / "videos" / "train" / f"train_{regime}_{_tag(alpha)}"; rows = []
    for i, (S, C, G, bank) in enumerate(_assignments(n, regime, alpha, seed_start + 13)):
        rows.append(_save(_record(seed_start + i, S, C, G, bank, regime, alpha, None, f"train:{regime}:{_tag(alpha)}:{i:06d}", f"train:{regime}:{_tag(alpha)}:{i:06d}", include_oracles=False), directory, i))
    _write_rows(directory, rows)
    return {"regime": regime, "alpha": alpha, "samples": n, "path": str(directory.relative_to(root))}


def _build_eval(root: Path, n_base: int, seed_start: int) -> dict[str, Any]:
    directory = root / "videos" / "eval_high"; low_dir = root / "videos" / "matched_low_reference"; rows=[]; low_rows=[]
    variants = (("ID", lambda S: (S, S)), ("C_flip", lambda S: (1-S, S)), ("G_flip", lambda S: (S, 1-S)), ("GC_flip", lambda S: (1-S, 1-S)))
    for i in range(n_base):
        seed=seed_start+i
        for S in (0,1):
            sid=f"eval:{i:06d}:S{S}"
            for variant, fn in variants:
                C,G=fn(S)
                rows.append(_save(_record(seed,S,C,G,"high","eval",None,variant,sid,f"{sid}:{variant}",include_oracles=True),directory,len(rows)))
            # The high G-flip has G=1-S,C=S. Low ID (G=S,C=S) matches its
            # remaining-34 local mechanism after the first support clears.
            low_rows.append(_save(_record(seed,S,S,S,"low","matched_low",None,"matched_low_ID",sid,f"{sid}:matched_low_ID",include_oracles=True),low_dir,len(low_rows)))
    _write_rows(directory,rows); _write_rows(low_dir,low_rows)
    return {"high_records":len(rows),"matched_low_records":len(low_rows)}


def calibrate(root: Path, count: int) -> dict[str, Any]:
    values=defaultdict(list)
    for i in range(count):
        for bank in ("low","high"):
            for S in (0,1):
                e=SupportOverlapScene.event(8_800_000+i,S,S,0,bank); values[(bank,int(e["screen_state"]["remaining_angle_deg"]))].append(e)
    out={}
    for key,xs in values.items():
        v=np.asarray([x["velocity_at_release"] for x in xs]); t=np.asarray([x["support_timing"]["release_time"] for x in xs]); p=np.asarray([x["positions"] for x in xs])
        out[f"{key[0]}_remaining{key[1]}"]={"n":len(xs),"release_v_mean":_as_list(v.mean(0)),"release_v_std":_as_list(v.std(0)),"release_time_mean":float(t.mean()),"release_time_range":[float(t.min()),float(t.max())],"future_endpoint_mean":_as_list(p[:,-1].mean(0))}
    path=root/"metadata"/"oracle_calibration_10000.json"; path.parent.mkdir(exist_ok=True); path.write_text(json.dumps(out,indent=2)+"\n")
    return out


def build(root: Path, train_count: int=2048, eval_base_count: int=64, calibration_count: int=10_000) -> dict[str, Any]:
    if train_count % 4: raise ValueError("train_count must divide S x bank")
    root.mkdir(parents=True,exist_ok=True); report={"experiment_name":EXPERIMENT_NAME,"train":[]}
    for j,(regime,alpha) in enumerate(CONDITIONS): report["train"].append(_build_train(root,regime,alpha,train_count,7_100_000+j*10_000))
    report["eval"]=_build_eval(root,eval_base_count,7_900_000); report["calibration"]=calibrate(root,calibration_count)
    cfg={"experiment_name":EXPERIMENT_NAME,"version":VERSION,"banks":BANKS,"train_count":train_count,"eval_base_count":eval_base_count,"calibration_count":calibration_count,"conditions":CONDITIONS,"video":{"resolution":RESOLUTION,"frames":FRAMES,"fps":FPS,"prefix":PREFIX}}
    cfg["config_hash"]=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:16]
    (root/"metadata").mkdir(exist_ok=True); (root/"metadata"/"generation_config.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False)); (root/"metadata"/"build_report.json").write_text(json.dumps(report,indent=2,default=_json_default)+"\n")
    common_samples = []
    for metadata_csv in sorted(root.glob("videos/**/metadata.csv")):
        bank_name = metadata_csv.parent.name
        split = (
            "eval"
            if "eval" in bank_name or "reference" in bank_name
            else "train"
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
        experiment="support_overlap",
        dataset=VERSION,
        samples=common_samples,
        extra={
            "generation_config": "metadata/generation_config.yaml",
            "build_report": "metadata/build_report.json",
        },
    )
    return report


def generate_dataset(config: SupportOverlapDataConfig) -> Path:
    build(
        config.root,
        config.train_count,
        config.eval_count,
        config.calibration_count,
    )
    return config.root


def _records(directory: Path) -> list[dict[str, Any]]:
    with (directory / "metadata.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        json.loads((directory / row["metadata"]).read_text())
        for row in rows
    ]


def audit_dataset(root: Path, out: Path) -> dict[str, Any]:
    """Check bank balance, matched references, and mechanism separation."""
    violations: list[list[object]] = []
    train_report: dict[str, Any] = {}
    train_root = root / "videos" / "train"
    for bank in sorted(train_root.glob("train_*")):
        records = _records(bank)
        strata = Counter(
            (record["S"], record["angle_bank"])
            for record in records
        )
        expected = len(records) // 4
        if len(strata) != 4 or any(
            count != expected for count in strata.values()
        ):
            violations.append([
                bank.name,
                "S_bank_quota",
                dict(strata),
            ])
        train_report[bank.name] = {
            "samples": len(records),
            "S_bank": {
                str(key): value for key, value in strata.items()
            },
        }

    high = _records(root / "videos" / "eval_high")
    low_records = _records(
        root / "videos" / "matched_low_reference"
    )
    low = {record["sample_id"]: record for record in low_records}
    gflip = [
        record
        for record in high
        if record["eval_variant"] == "G_flip"
    ]
    for record in gflip:
        reference = low.get(record["sample_id"])
        if reference is None:
            violations.append([
                record["sample_id"],
                "missing_low_reference",
            ])
            continue
        event = record["renderer_metadata"]["event"]
        reference_event = reference["renderer_metadata"]["event"]
        local = (
            np.asarray(event["initial_position"])
            - np.asarray(event["support_base"])
        )
        local_reference = (
            np.asarray(reference_event["initial_position"])
            - np.asarray(reference_event["support_base"])
        )
        if not np.allclose(local, local_reference, atol=1e-10):
            violations.append([
                record["sample_id"],
                "local_anchor_position",
            ])
        if not np.allclose(
            event["velocity_at_release"],
            reference_event["velocity_at_release"],
            atol=1e-10,
        ):
            violations.append([
                record["sample_id"],
                "local_release_velocity",
            ])

    calibration = json.loads(
        (
            root
            / "metadata"
            / "oracle_calibration_10000.json"
        ).read_text()
    )
    velocity_34 = abs(
        calibration["low_remaining34"]["release_v_mean"][1]
    )
    velocity_48 = abs(
        calibration["high_remaining48"]["release_v_mean"][1]
    )
    if velocity_48 - velocity_34 < 0.003:
        violations.append([
            "calibration",
            "34_vs_48_vy_separation",
            velocity_34,
            velocity_48,
        ])
    report = {
        "benchmark_version": VERSION,
        "train": train_report,
        "eval": {
            "high_records": len(high),
            "matched_low_records": len(low_records),
            "gflip_records": len(gflip),
        },
        "calibration": calibration,
        "violations": violations,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
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
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--train-count",type=int,default=2048); p.add_argument("--eval-count",type=int,default=64); p.add_argument("--calibration-count",type=int,default=10_000); a=p.parse_args()
    generate_dataset(
        SupportOverlapDataConfig(
            root=a.root,
            train_count=a.train_count,
            eval_count=a.eval_count,
            calibration_count=a.calibration_count,
        )
    )


if __name__ == "__main__":
    generation_cli()
