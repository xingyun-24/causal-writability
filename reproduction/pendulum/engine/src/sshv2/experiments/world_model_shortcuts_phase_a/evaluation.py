"""Unified generated-video route and coherence evaluation for Phase A."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import imageio.v2 as imageio
import numpy as np
import torch
from sshv2.common.results import (
    find_prediction,
    write_metrics,
    write_predictions,
)
from sshv2.experiments.world_model_shortcuts_phase_a.data import (
    FPS,
    FRAMES,
    PREFIX,
    RADIUS,
    VERSION,
)


@dataclass(frozen=True)
class BallWallEvaluationConfig:
    """Minimal options for evaluating an existing prediction directory."""

    limit: int = 0


@dataclass(frozen=True)
class PhaseAPredictionConfig:
    training_config: Path
    checkpoint: Path
    steps: int = 20
    limit: int = 0
    seed_offset: int = 1_700_000
    device: str = "cuda"
    dataset_id: str = VERSION

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "PhaseAPredictionConfig":
        return cls(
            training_config=Path(value["training_config"]),
            checkpoint=Path(value["checkpoint"]),
            steps=int(
                value.get("steps", value.get("diffusion_steps", 20))
            ),
            limit=int(value.get("limit") or 0),
            seed_offset=int(value.get("seed_offset", 1_700_000)),
            device=str(value.get("device", "cuda")),
            dataset_id=str(value.get("dataset_id") or VERSION),
        )


def read_video(path: Path) -> np.ndarray:
    reader = imageio.get_reader(path)
    try:
        return np.stack([reader.get_data(i) for i in range(FRAMES)])
    finally:
        reader.close()


def as_tensor(video: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(video).permute(3, 0, 1, 2).float() / 127.5 - 1.0


def detect(video: np.ndarray, color: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, g, b = video[..., 0], video[..., 1], video[..., 2]
    mask = (r > 105) & (r > g * 1.25) & (r > b * 1.25) if color == "red" else (b > 90) & (b > r * 1.15) & (b > g * 1.10)
    mass = mask.sum((1, 2))
    h, w = video.shape[1:3]
    xx, yy = np.arange(w)[None, None, :], np.arange(h)[None, :, None]
    x = np.divide((mask * xx).sum((1, 2)), mass, out=np.full(FRAMES, np.nan), where=mass > 10)
    y = np.divide((mask * yy).sum((1, 2)), mass, out=np.full(FRAMES, np.nan), where=mass > 10)
    return np.stack(((x + .5) / w, 1.0 - (y + .5) / h), -1), mass, mass > 10


def sample(pipe, raw: np.ndarray, seed: int, steps: int) -> np.ndarray:
    x = as_tensor(raw)
    with torch.no_grad():
        pipe.load_models_to_device(["vae"])
        condition = pipe.vae.encode([x[:, :PREFIX].to(dtype=pipe.torch_dtype)], device=pipe.device, tiled=False).to(dtype=pipe.torch_dtype, device=pipe.device)
        pipe.pre_encoded_(True)
        generated = pipe(height=128, width=128, num_frames=FRAMES, num_condition_frames=2, condition_frames=condition,
                         num_inference_steps=steps, num_samples=1, cfg_scale=1.0, tiled=False, seed=seed,
                         return_as_tensor=True, progress_bar_cmd=lambda timesteps: timesteps)[0]
    return ((generated.permute(1, 2, 3, 0).float().cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


def expected_visible(scene: str, event: dict[str, Any]) -> np.ndarray:
    """Frames where an object should be visible rather than behind the occluder."""
    visible = np.ones(FRAMES, dtype=bool)
    if scene == "occluded_uniform_motion":
        p = np.asarray(event["positions"])
        lo, hi = np.asarray(event["occluder"]["lo"]), np.asarray(event["occluder"]["hi"])
        hidden = ((p[:, 0] >= lo[0] - RADIUS) & (p[:, 0] <= hi[0] + RADIUS)
                  & (p[:, 1] >= lo[1] - RADIUS) & (p[:, 1] <= hi[1] + RADIUS))
        visible = ~hidden
    return visible


def scene_validity(scene: str, pred: np.ndarray, centres: np.ndarray, mass: np.ndarray, detected: np.ndarray, event: dict[str, Any]) -> dict[str, float]:
    future = slice(PREFIX, None)
    future_centres = centres[future]
    visible = expected_visible(scene, event)[future]
    observed = np.isfinite(future_centres).all(1) & visible
    consecutive = observed[1:] & observed[:-1]
    jumps = np.linalg.norm(np.diff(future_centres, axis=0), axis=1)[consecutive]
    teleport = float(np.nanmean(jumps > .040)) if np.isfinite(jumps).any() else 1.0
    out = ((future_centres[:, 0] < RADIUS) | (future_centres[:, 0] > 1 - RADIUS) |
           (future_centres[:, 1] < RADIUS) | (future_centres[:, 1] > 1 - RADIUS))
    out_of_frame = float(np.nanmean(out[observed])) if observed.any() else 1.0
    expected = np.pi * (RADIUS * 128) ** 2
    deformation = float(np.mean(((mass[future] < expected * .45) | (mass[future] > expected * 1.8))[visible])) if visible.any() else 1.0
    penetration = 0.0
    impossible = 0.0
    if scene == "ball_wall":
        p, n, c = centres[future], np.asarray(event["normal"]), np.asarray(event["contact_point"])
        penetration = float(np.nanmean(((p - c) @ n) < RADIUS - .01))
    elif scene == "occluded_uniform_motion":
        # Post-occluder motion should not exhibit a large unexplained turn.
        good = np.isfinite(future_centres).all(1) & visible
        if good.sum() >= 5:
            t = np.arange(PREFIX, FRAMES)[good] / FPS
            px = np.polyfit(t, future_centres[good, 0], 1)
            py = np.polyfit(t, future_centres[good, 1], 1)
            fitted = np.stack((np.polyval(px, t), np.polyval(py, t)), -1)
            impossible = float(np.sqrt(np.mean((future_centres[good] - fitted) ** 2)) > .012)
    extraction = float(detected[future][visible].mean()) if visible.any() else 0.0
    valid = float(extraction >= .90 and teleport == 0 and deformation == 0 and out_of_frame == 0 and penetration == 0 and impossible == 0)
    return {"trajectory_extraction_success": extraction, "disappearance_rate": float((~detected[future][visible]).mean()) if visible.any() else 1.0,
            "expected_occluded_rate": float((~visible).mean()),
            "object_count_consistency": float(np.mean(((mass[future] > expected * .45) & (mass[future] < expected * 1.8))[visible])) if visible.any() else 0.0,
            "teleportation_rate": teleport, "out_of_frame_rate": out_of_frame, "shape_deformation_rate": deformation,
            "penetration_or_impossible_rate": max(penetration, impossible), "valid": valid}


def evaluate_one(pred: np.ndarray, record: dict[str, Any]) -> dict[str, Any]:
    event = record["renderer_metadata"]["event"]
    centres, mass, detected = detect(pred, event["color"])
    true, cf = np.asarray(record["true_branch_oracle_trajectory"]), np.asarray(record["counterfactual_branch_trajectory"])
    valid = scene_validity(record["scene_type"], pred, centres, mass, detected, event)
    future = np.isfinite(centres[PREFIX:]).all(1) & expected_visible(record["scene_type"], event)[PREFIX:]
    if not valid["valid"] or not future.any():
        d_true = d_cf = ade = fde = rho = np.nan
        route = "invalid"
    else:
        p, t, c = centres[PREFIX:][future], true[PREFIX:][future], cf[PREFIX:][future]
        d_true = float(np.mean(np.linalg.norm(p - t, axis=1)))
        d_cf = float(np.mean(np.linalg.norm(p - c, axis=1)))
        ade = d_true
        last = np.flatnonzero(future)[-1] + PREFIX
        fde = float(np.linalg.norm(centres[last] - true[last]))
        rho = float((d_cf - d_true) / (d_cf + d_true + 1e-8))
        route = "physical" if rho > .05 else "wrong" if rho < -.05 else "ambiguous"
    variant = record["eval_variant"]
    appearance_follow = float(route == "wrong" and variant in ("C_flip", "GC_flip"))
    geometry_follow = float(route == "wrong" and variant in ("G_flip", "GC_flip"))
    return {"record_id": record["record_id"], "sample_id": record["sample_id"], "scene": record["scene_type"],
            "S": record["S"], "C": record["C"], "G": record["G"], "train_regime": record["train_regime"], "alpha": record["alpha"],
            "eval_variant": variant, "d_true": d_true, "d_cf": d_cf, "route_margin_rho": rho, "ADE": ade, "FDE": fde,
            "route": route, "physical_route": float(route == "physical"), "wrong_route": float(route == "wrong"),
            "ambiguous_route": float(route == "ambiguous"), "appearance_follow": appearance_follow, "geometry_follow": geometry_follow, **valid}


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(x[key]) for x in rows if isinstance(x.get(key), (int, float)) and np.isfinite(x[key])]
    return float(np.mean(values)) if values else float("nan")


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("d_true", "d_cf", "route_margin_rho", "ADE", "FDE", "physical_route", "wrong_route", "ambiguous_route", "appearance_follow", "geometry_follow",
            "trajectory_extraction_success", "disappearance_rate", "expected_occluded_rate", "object_count_consistency", "teleportation_rate", "out_of_frame_rate", "shape_deformation_rate", "penetration_or_impossible_rate", "valid")
    result = {"all": {k: _mean(rows, k) for k in keys}, "variants": {}}
    for variant in sorted({x["eval_variant"] for x in rows}):
        xs = [x for x in rows if x["eval_variant"] == variant]
        result["variants"][variant] = {k: _mean(xs, k) for k in keys}
        result["variants"][variant]["samples"] = len(xs)
    return result


def predict(
    dataset_dir: Path,
    prediction_dir: Path,
    config: PhaseAPredictionConfig | Mapping[str, Any],
) -> dict[str, Any]:
    """Generate Phase-A predictions without running scientific metrics."""
    if isinstance(config, Mapping):
        config = PhaseAPredictionConfig.from_mapping(config)
    if config.steps <= 0:
        raise ValueError("steps must be positive")
    from sshv2.wan.config import (
        CompatibilityTrainingConfig as TrainingConfig,
    )
    from sshv2.wan.trainer import (
        CompatibilityWanTrainingModule as WanTrainingModule,
    )

    training = TrainingConfig.from_file(config.training_config)
    training.model.dit.ckpt_file = config.checkpoint
    model = WanTrainingModule(
        training.model.dit,
        training.model.vae,
        no_encoding=False,
        num_condition_frames=2,
        num_inference_steps=config.steps,
        pipeline_type=training.model.pipe,
    )
    pipe = model.pipe
    pipe.to(config.device)
    pipe.load_models_to_device(("dit", "vae"))
    rows = list(
        csv.DictReader((dataset_dir / "metadata.csv").open())
    )[: config.limit or None]
    predictions: list[dict[str, Any]] = []
    for row in rows:
        record = json.loads(
            (dataset_dir / row["metadata"]).read_text()
        )
        raw = read_video(dataset_dir / row["video"])
        generation_seed = int(record["seed"]) + config.seed_offset
        generated = sample(
            pipe,
            raw,
            generation_seed,
            config.steps,
        )
        destination = (
            prediction_dir
            / "predictions"
            / record["eval_variant"]
            / row["video"]
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(
            destination,
            fps=FPS,
            codec="libx264",
            quality=10,
            macro_block_size=None,
        ) as writer:
            for frame in generated:
                writer.append_data(frame)
        predictions.append(
            {
                "prediction_id": record["record_id"],
                "sample_id": record["sample_id"],
                "prediction": destination.relative_to(
                    prediction_dir
                ).as_posix(),
                "attributes": {
                    "variant": record["eval_variant"],
                    "seed": generation_seed,
                },
            }
        )
    return write_predictions(
        prediction_dir,
        experiment="world_model_shortcuts",
        dataset=config.dataset_id,
        predictions=predictions,
        checkpoint=config.checkpoint,
        config=config.training_config,
        extra={
            "steps": config.steps,
            "seed_offset": config.seed_offset,
        },
    )


def evaluate(
    dataset_dir: Path,
    prediction_dir: Path,
    config: BallWallEvaluationConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate already-generated videos using the common experiment contract."""
    if isinstance(config, Mapping):
        config = BallWallEvaluationConfig(
            limit=int(config.get("limit") or 0)
        )
    config = config or BallWallEvaluationConfig()
    rows = list(csv.DictReader((dataset_dir / "metadata.csv").open()))
    rows = rows[: config.limit or None]
    metrics: list[dict[str, Any]] = []
    for row in rows:
        record = json.loads((dataset_dir / row["metadata"]).read_text())
        indexed = find_prediction(
            prediction_dir,
            sample_id=record["sample_id"],
            prediction_id=record["record_id"],
        )
        candidates = (
            indexed,
            prediction_dir / record["eval_variant"] / row["video"],
            prediction_dir / row["video"],
        )
        prediction = next(
            (
                path
                for path in candidates
                if path is not None and path.exists()
            ),
            None,
        )
        if prediction is None:
            raise FileNotFoundError(
                f"Missing prediction for {record['record_id']}: {candidates}"
            )
        metrics.append(evaluate_one(read_video(prediction), record))
    return {"summary": aggregate(metrics), "samples": metrics}


def write_evaluation(
    result: dict[str, Any],
    out: Path,
    *,
    dataset: str = VERSION,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist Ball-Wall metrics through the common result envelope."""
    return write_metrics(
        out,
        experiment="world_model_shortcuts",
        dataset=dataset,
        summary=result["summary"],
        samples=result["samples"],
        context=context,
        legacy={"samples": result["samples"]},
    )


def load_aggregate_metrics(
    root: Path,
    benchmark: str,
) -> list[dict[str, object]]:
    """Load every evaluator summary below one experiment result root."""
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("*/metrics.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        for variant, values in report["summary"]["variants"].items():
            rows.append(
                {
                    "benchmark": benchmark,
                    "train_condition": path.parent.name,
                    "variant": variant,
                    **values,
                }
            )
    return rows


def _format_route_cell(row: dict[str, object]) -> str:
    return "/".join(
        [
            f"{float(row['ADE']):.4f}",
            f"{float(row['FDE']):.4f}",
            f"{float(row['route_margin_rho']):.3f}",
            f"{100 * float(row['valid']):.1f}",
            f"{100 * float(row['physical_route']):.1f}",
            f"{100 * float(row['wrong_route']):.1f}",
            f"{100 * float(row['ambiguous_route']):.1f}",
        ]
    )


def export_result_tables(
    support_root: Path,
    ball_wall_root: Path,
    out: Path,
) -> list[dict[str, object]]:
    """Write the complete aggregate CSV and per-benchmark Markdown tables."""
    out.mkdir(parents=True, exist_ok=True)
    rows = load_aggregate_metrics(
        support_root,
        "support_screen_state_v2",
    ) + load_aggregate_metrics(
        ball_wall_root,
        "ball_wall_phase_a_v1",
    )
    if not rows:
        raise ValueError("No evaluator metrics.json files were found")
    with (out / "all_aggregate_metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for benchmark in sorted(
        {str(row["benchmark"]) for row in rows}
    ):
        group = [
            row for row in rows
            if row["benchmark"] == benchmark
        ]
        conditions = sorted(
            {str(row["train_condition"]) for row in group}
        )
        by_key = {
            (
                str(row["train_condition"]),
                str(row["variant"]),
            ): row
            for row in group
        }
        lines = [
            f"# {benchmark}",
            "",
            (
                "Cell: `ADE / FDE / rho / valid% / physical% / "
                "wrong% / ambiguous%`."
            ),
            "",
            "| train | ID | C-flip | G-flip | GC-flip |",
            "|---|---|---|---|---|",
        ]
        for condition in conditions:
            lines.append(
                "| "
                + condition
                + " | "
                + " | ".join(
                    _format_route_cell(by_key[(condition, variant)])
                    for variant in (
                        "ID",
                        "C_flip",
                        "G_flip",
                        "GC_flip",
                    )
                )
                + " |"
            )
        (out / f"{benchmark}_route_table.md").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
    return rows
