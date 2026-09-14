"""Four-oracle route/payload evaluation for Support-Overlap v3."""
from __future__ import annotations

import csv, json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import imageio.v2 as imageio
import numpy as np
import torch

from sshv2.common.video import read_video
from sshv2.common.results import (
    find_prediction,
    write_metrics,
    write_predictions,
)
from sshv2.experiments.support_overlap_angle_banks_v3.data import (
    FPS,
    FRAMES,
    PREFIX,
    RADIUS,
    SUPPORT,
    SupportOverlapScene,
)


def detect(
    video: np.ndarray,
    color: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Track the coloured ball without importing a Phase-A evaluator."""
    red, green, blue = video[..., 0], video[..., 1], video[..., 2]
    mask = (
        (red > 105)
        & (red > green * 1.25)
        & (red > blue * 1.25)
        if color == "red"
        else (
            (blue > 90)
            & (blue > red * 1.15)
            & (blue > green * 1.10)
        )
    )
    mass = mask.sum((1, 2))
    height, width = video.shape[1:3]
    xx = np.arange(width)[None, None, :]
    yy = np.arange(height)[None, :, None]
    x = np.divide(
        (mask * xx).sum((1, 2)),
        mass,
        out=np.full(FRAMES, np.nan),
        where=mass > 10,
    )
    y = np.divide(
        (mask * yy).sum((1, 2)),
        mass,
        out=np.full(FRAMES, np.nan),
        where=mass > 10,
    )
    centres = np.stack(
        ((x + 0.5) / width, 1.0 - (y + 0.5) / height),
        axis=-1,
    )
    return centres, mass, mass > 10


def _as_tensor(video: np.ndarray) -> torch.Tensor:
    return (
        torch.from_numpy(video)
        .permute(3, 0, 1, 2)
        .float()
        / 127.5
        - 1.0
    )


def sample(pipe, raw: np.ndarray, seed: int, steps: int) -> np.ndarray:
    """Generate one future with shared Wan but experiment-owned inputs."""
    video = _as_tensor(raw)
    with torch.no_grad():
        pipe.load_models_to_device(["vae"])
        condition = pipe.vae.encode(
            [
                video[:, :PREFIX].to(
                    dtype=pipe.torch_dtype,
                )
            ],
            device=pipe.device,
            tiled=False,
        ).to(
            dtype=pipe.torch_dtype,
            device=pipe.device,
        )
        pipe.pre_encoded_(True)
        generated = pipe(
            height=128,
            width=128,
            num_frames=FRAMES,
            num_condition_frames=2,
            condition_frames=condition,
            num_inference_steps=steps,
            num_samples=1,
            cfg_scale=1.0,
            tiled=False,
            seed=seed,
            return_as_tensor=True,
            progress_bar_cmd=lambda timesteps: timesteps,
        )[0]
    return (
        (generated.permute(1, 2, 3, 0).float().cpu().numpy() + 1.0)
        * 127.5
    ).clip(0, 255).astype(np.uint8)


def validity(video: np.ndarray, centres: np.ndarray, mass: np.ndarray, detected: np.ndarray) -> dict[str, float]:
    future=slice(PREFIX,None); expected=np.pi*(RADIUS*128)**2; obs=np.isfinite(centres[future]).all(1)
    jumps=np.linalg.norm(np.diff(centres[future],axis=0),axis=1); good=np.isfinite(jumps)
    teleport=float(np.mean(jumps[good]>.040)) if good.any() else 1.0
    deformation=float(np.mean((mass[future]<expected*.45)|(mass[future]>expected*1.8)))
    disappear=float(np.mean(~detected[future])); valid=float(obs.mean()>=.90 and teleport==0 and deformation==0)
    return {"valid":valid,"disappearance_rate":disappear,"teleportation_rate":teleport,"shape_deformation_rate":deformation}


def ballistic(centres: np.ndarray, release_time: float) -> tuple[float,float,float,float]:
    good=np.isfinite(centres[PREFIX:]).all(1); t=np.arange(PREFIX,FRAMES)[good]/FPS; p=centres[PREFIX:][good]
    if len(t)<5:return (float("nan"),)*4
    vx,bx=np.polyfit(t,p[:,0],1); sy,by=np.polyfit(t,p[:,1]+.5*SupportOverlapScene.gravity*t*t,1)
    vy_release=sy-SupportOverlapScene.gravity*release_time
    return float(vx),float(vy_release),float(bx),float(by)


def one_from_detection(pred: np.ndarray, record: dict[str,Any], centres: np.ndarray, mass: np.ndarray, detected: np.ndarray) -> dict[str,Any]:
    """Route metrics from an externally supplied ball tracker.

    The standard evaluator calls this through :func:`one` with its red/blue
    detector.  Palette-only OOD checks use the same route calculation but a
    prefix-calibrated detector, so colour threshold failure is not conflated
    with failure of the video model.
    """
    event=record["renderer_metadata"]["event"]; q=validity(pred,centres,mass,detected)
    result={"record_id":record["record_id"],"sample_id":record["sample_id"],"S":record["S"],"C":record["C"],"G":record["G"],"angle_bank":record["angle_bank"],"eval_variant":record["eval_variant"],**q}
    if not q["valid"]:
        return {**result,**{k:float("nan") for k in ("rho_branch","rho_payload","release_time_error","release_vx_error","release_vy_error","post_release_ballistic_ADE","post_release_ballistic_FDE")},"branch_correct":0.,"payload_correct":0.,"full_physics":0.,"branch_only":0.,"shortcut_wrong":0.,"ambiguous":0.,"invalid":1.,"nearest_oracle":"invalid"}
    good=np.isfinite(centres[PREFIX:]).all(1); p=centres[PREFIX:][good]
    ds={k:float(np.mean(np.linalg.norm(p-np.asarray(v)[PREFIX:][good],axis=1))) for k,v in record["oracle_trajectories"].items()}
    cb=int(record["correct_branch"]); cp=int(record["correct_payload_angle"]); op=48 if cp==34 else 34
    dc,do=ds[f"branch{cb}_payload{cp}"],ds[f"branch{cb}_payload{op}"]
    dw34,dw48=ds[f"branch{1-cb}_payload34"],ds[f"branch{1-cb}_payload48"]
    dbranch_c,dbranch_w=min(dc,do),min(dw34,dw48)
    rb=(dbranch_w-dbranch_c)/(dbranch_w+dbranch_c+1e-8); rp=(do-dc)/(do+dc+1e-8)
    bcorrect=rb>.05; pcorrect=rp>.05; branch_only=bcorrect and rp<-.05; shortcut=(rb<-.05 and min(ds,key=ds.get)==f"branch{1-cb}_payload48")
    ambiguous=not (bcorrect and pcorrect) and not branch_only and not shortcut
    vx,vy,_,_=ballistic(centres,float(event["support_timing"]["release_time"])); vtrue=np.asarray(event["velocity_at_release"]); atrue=np.asarray(event["slide_acceleration"])
    delta_hat=float(np.dot(np.asarray((vx,vy)),atrue)/max(float(atrue@atrue),1e-10)); trel_hat=float(event["support_timing"]["slide_start_time"]+delta_hat)
    return {**result,"d_correct_branch_payload":dc,"d_other_payload":do,"d_wrong_branch_payload34":dw34,"d_wrong_branch_payload48":dw48,"rho_branch":rb,"rho_payload":rp,"branch_correct":float(bcorrect),"payload_correct":float(pcorrect),"full_physics":float(bcorrect and pcorrect),"branch_only":float(branch_only),"shortcut_wrong":float(shortcut),"ambiguous":float(ambiguous),"invalid":0.,"nearest_oracle":min(ds,key=ds.get),"release_time_error":abs(trel_hat-float(event["support_timing"]["release_time"])),"release_vx_error":abs(vx-vtrue[0]),"release_vy_error":abs(vy-vtrue[1]),"post_release_ballistic_ADE":dc,"post_release_ballistic_FDE":float(np.linalg.norm(centres[FRAMES-1]-np.asarray(record["oracle_trajectories"][f"branch{cb}_payload{cp}"])[FRAMES-1]))}


def one(pred: np.ndarray, record: dict[str,Any]) -> dict[str,Any]:
    event=record["renderer_metadata"]["event"]
    return one_from_detection(pred, record, *detect(pred,event["color"]))


def aggregate(rows:list[dict[str,Any]])->dict[str,Any]:
    out={"all":{},"variants":{}}
    keys=[k for k in rows[0] if k not in ("record_id","sample_id","angle_bank","eval_variant","nearest_oracle")]
    for name,xs in [("all",rows),*[(v,[x for x in rows if x["eval_variant"]==v]) for v in sorted({x["eval_variant"] for x in rows})]]:
        dest=out["all"] if name=="all" else out["variants"].setdefault(name,{})
        for k in keys:
            vals=[float(x[k]) for x in xs if isinstance(x.get(k),(int,float)) and np.isfinite(x[k])]
            dest[k]=float(np.mean(vals)) if vals else float("nan")
        dest["samples"]=len(xs)
    return out


@dataclass(frozen=True)
class SupportOverlapEvaluationConfig:
    limit: int = 0


@dataclass(frozen=True)
class SupportOverlapPredictionConfig:
    training_config: Path
    checkpoint: Path
    steps: int = 20
    limit: int = 0
    seed_offset: int = 1_700_000
    device: str = "cuda"
    save_videos_limit: int = 16
    dataset_id: str = "support_overlap_angle_banks_v3"

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "SupportOverlapPredictionConfig":
        return cls(
            training_config=Path(value["training_config"]),
            checkpoint=Path(value["checkpoint"]),
            steps=int(
                value.get("steps", value.get("diffusion_steps", 20))
            ),
            limit=int(value.get("limit") or 0),
            seed_offset=int(value.get("seed_offset", 1_700_000)),
            device=str(value.get("device", "cuda")),
            save_videos_limit=int(
                value.get("save_videos_limit", 16)
            ),
            dataset_id=str(
                value.get("dataset_id")
                or "support_overlap_angle_banks_v3"
            ),
        )


def predict(
    dataset_dir: Path,
    prediction_dir: Path,
    config: SupportOverlapPredictionConfig | Mapping[str, Any],
) -> dict[str, Any]:
    """Generate pure Support-Overlap predictions and a common manifest."""
    if isinstance(config, Mapping):
        config = SupportOverlapPredictionConfig.from_mapping(config)
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
    source_rows = list(
        csv.DictReader((dataset_dir / "metadata.csv").open())
    )[: config.limit or None]
    predictions = []
    for index, row in enumerate(source_rows):
        record = json.loads(
            (dataset_dir / row["metadata"]).read_text()
        )
        generation_seed = int(record["seed"]) + config.seed_offset
        raw = read_video(dataset_dir / row["video"])
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
        if index < config.save_videos_limit:
            diagnostic = (
                prediction_dir
                / "diagnostics"
                / record["eval_variant"]
                / row["video"]
            )
            diagnostic.parent.mkdir(parents=True, exist_ok=True)
            condition = raw.copy()
            condition[PREFIX:] = 22
            with imageio.get_writer(
                diagnostic,
                fps=FPS,
                codec="libx264",
                quality=10,
                macro_block_size=None,
            ) as writer:
                for frame in np.concatenate(
                    (condition, raw, generated),
                    axis=1,
                ):
                    writer.append_data(frame)
        predictions.append({
            "prediction_id": record["record_id"],
            "sample_id": record["sample_id"],
            "prediction": destination.relative_to(
                prediction_dir
            ).as_posix(),
            "attributes": {
                "variant": record["eval_variant"],
                "seed": generation_seed,
                "diagnostic_video_selected": (
                    index < config.save_videos_limit
                ),
            },
        })
    return write_predictions(
        prediction_dir,
        experiment="support_overlap",
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
    config: SupportOverlapEvaluationConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(config, Mapping):
        config = SupportOverlapEvaluationConfig(
            limit=int(config.get("limit") or 0)
        )
    config = config or SupportOverlapEvaluationConfig()
    source_rows = list(
        csv.DictReader((dataset_dir / "metadata.csv").open())
    )[: config.limit or None]
    rows = []
    for row in source_rows:
        record = json.loads(
            (dataset_dir / row["metadata"]).read_text()
        )
        sample_id = record["record_id"]
        prediction = find_prediction(
            prediction_dir,
            sample_id=sample_id,
            prediction_id=record["record_id"],
        )
        if prediction is None:
            prediction = (
                prediction_dir
                / record["eval_variant"]
                / row["video"]
            )
        rows.append(one(read_video(prediction), record))
    return {"summary": aggregate(rows), "samples": rows}


def write_evaluation(
    result: dict[str, Any],
    out: Path,
    *,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return write_metrics(
        out,
        experiment="support_overlap",
        dataset="support_overlap_angle_banks_v3",
        summary=result["summary"],
        samples=result["samples"],
        context=context,
        legacy={"samples": result["samples"]},
    )
