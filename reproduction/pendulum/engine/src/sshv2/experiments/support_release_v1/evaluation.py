"""Evaluation for frozen ``support_release_v1``."""
from __future__ import annotations
import csv,json,math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import imageio.v2 as imageio
import numpy as np
import torch
from sshv2.wan.config import CompatibilityTrainingConfig as TrainingConfig
from sshv2.wan.trainer import CompatibilityWanTrainingModule as WanTrainingModule
from sshv2.common.results import (
    find_prediction,
    write_metrics,
    write_predictions,
)

FPS, FRAMES, RADIUS = 15,49,.040
def read(p):
    r=imageio.get_reader(p)
    try:return np.stack([r.get_data(i) for i in range(FRAMES)])
    finally:r.close()
def tensor(x):return torch.from_numpy(x).permute(3,0,1,2).float()/127.5-1
def ad(a,b):return abs((a-b+180)%360-180)
def ang(v):return math.degrees(math.atan2(v[1],v[0]))
def mask(video,color):
    r,g,b=video[...,0],video[...,1],video[...,2]
    return (r>105)&(r>g*1.25)&(r>b*1.25) if color=='red' else (b>90)&(b>r*1.15)&(b>g*1.10)
def detector(video,color):
    m=mask(video,color);mass=m.sum((1,2));xx=np.arange(128)[None,None,:];yy=np.arange(128)[None,:,None]
    x=np.divide((m*xx).sum((1,2)),mass,out=np.full(FRAMES,np.nan),where=mass>10);y=np.divide((m*yy).sum((1,2)),mass,out=np.full(FRAMES,np.nan),where=mass>10)
    return np.stack(((x+.5)/128,1-(y+.5)/128),-1),mass,mass>10
def sample(pipe,raw,seed,steps):
    v=tensor(raw)
    with torch.no_grad():
        pipe.load_models_to_device(['vae']);c=pipe.vae.encode([v[:,:5].to(dtype=pipe.torch_dtype)],device=pipe.device,tiled=False).to(dtype=pipe.torch_dtype,device=pipe.device);pipe.pre_encoded_(True)
        g=pipe(height=128,width=128,num_frames=49,num_condition_frames=2,condition_frames=c,num_inference_steps=steps,num_samples=1,cfg_scale=1.,tiled=False,seed=seed,return_as_tensor=True,progress_bar_cmd=lambda x:x)[0]
    return ((g.permute(1,2,3,0).float().cpu().numpy()+1)*127.5).clip(0,255).astype(np.uint8)
def support_mask(event):
    yy,xx=np.mgrid[:128,:128];region=np.zeros((128,128),bool)
    for s in event['supports'].values():
        a,b=np.asarray(s['endpoints']);a=np.array((a[0]*128,(1-a[1])*128));b=np.array((b[0]*128,(1-b[1])*128));d=b-a;den=max(float(d@d),1e-12);u=np.clip(((xx-a[0])*d[0]+(yy-a[1])*d[1])/den,0,1);region|=np.hypot(xx-(a[0]+u*d[0]),yy-(a[1]+u*d[1]))<=4
    return region
def opposite_index(root):
    out={}
    for r in csv.DictReader((root/'metadata.csv').open()):
        m=json.loads((root/r['metadata']).read_text());out[(m['octet_id'],m['geometry'],m['color'],m['state'])]=m
    return out
def metric(pred,m,other):
    e=m['event'];p,mass,ok=detector(pred,m['color']);gt=np.asarray(e['positions']);wrong=np.asarray(other[(m['octet_id'],m['geometry'],m['color'],'B' if m['state']=='A' else 'A')]['event']['positions']);f=slice(5,None);valid=ok[5:].any();rel=p-p[4];gr=gt-gt[4];wr=wrong-wrong[4];finite=np.isfinite(rel[5:]).all(1)
    err=np.linalg.norm(rel[5:]-gr[5:],axis=1);werr=np.linalg.norm(rel[5:]-wr[5:],axis=1)
    dcor=float(np.nanmean(err)) if valid else np.nan;dwrong=float(np.nanmean(werr)) if valid else np.nan
    dv=wr[5:]-gr[5:];res=rel[5:]-gr[5:];rho=float(np.nansum(res*dv)/np.sum(dv*dv)) if valid else np.nan
    # Fit the free-flight arc and recover the velocity at the last prefix
    # frame.  A late-frame finite difference would incorrectly penalize the
    # physically expected gravity-induced rotation of the velocity vector.
    tt=np.arange(5,FRAMES,dtype=float)/FPS-4/FPS
    good=np.isfinite(p[5:]).all(1)
    if good.sum()>=6:
        bx=np.polyfit(tt[good],p[5:,0][good],1)[0]
        cy=np.polyfit(tt[good],p[5:,1][good],2)
        vp=np.asarray((bx,cy[1]))
    else: vp=np.asarray((np.nan,np.nan))
    vo=np.asarray(e['velocity_at_release']);direction=float(np.sign(vp[0])==np.sign(vo[0])) if np.isfinite(vp).all() else 0.
    d=np.linalg.norm(np.diff(p[5:],axis=0),axis=1);tele=float(np.nanmean(d>.035)) if valid else 1.
    ddy=np.diff(p[5:,1],2);up=float(np.nanmedian(ddy)>.001) if np.isfinite(ddy).any() else 1.
    expected=math.pi*(RADIUS*128)**2;deform=float(np.mean((mass[5:]<expected*.45)|(mass[5:]>expected*1.75)))
    region=support_mask(e);gray=pred[5:,...,:3].mean(-1);ball_pixels=mask(pred,m['color'])[5:]
    # The ball initially passes through the old support region; its coloured
    # pixels must not be counted as a hallucinated grey support.
    persist=float(np.mean(((gray[:,region]>95)&(~ball_pixels[:,region])).mean(1)>.12))
    outframe=float(np.mean((p[5:,0]<RADIUS)|(p[5:,0]>1-RADIUS)|(p[5:,1]<RADIUS)|(p[5:,1]>1-RADIUS)))
    multi=float(np.mean(mass[5:]>expected*1.8));good=float(ok[5:].mean()>.95 and tele==0 and deform==0 and up==0 and persist==0 and outframe==0)
    return {'state':m['state'],'geometry':m['geometry'],'color':m['color'],'outgoing_angle_mae':ad(ang(vp),ang(vo)) if np.isfinite(vp).all() else np.nan,
      'horizontal_trajectory_rmse':float(np.sqrt(np.nanmean((rel[5:,0]-gr[5:,0])**2))) if valid else np.nan,
      'trajectory_rmse':float(np.sqrt(np.nanmean(err**2))) if valid else np.nan,'endpoint_error':float(np.linalg.norm(rel[-1]-gr[-1])) if np.isfinite(rel[-1]).all() else np.nan,
      'final_horizontal_direction_accuracy':direction,'d_correct':dcor,'d_wrong':dwrong,'correct_follow':float(dcor<dwrong) if np.isfinite(dcor+dwrong) else 0.,'shortcut_follow':float(dwrong<dcor) if np.isfinite(dcor+dwrong) else 0.,'rho':rho,
      'ball_detection_success':float(ok[5:].mean()),'disappearance_rate':float((~ok[5:]).mean()),'teleportation_rate':tele,'shape_deformation_rate':deform,'upward_acceleration_inconsistency':up,'future_support_persistence':persist,'out_of_frame_rate':outframe,'multiple_ball_hallucination':multi,'valid_trajectory_rate':good,
      'predicted_angle':float(ang(vp)) if np.isfinite(vp).all() else np.nan,'correct_angle':float(ang(vo)),'opposite_angle':float(ang(np.asarray(other[(m['octet_id'],m['geometry'],m['color'],'B' if m['state']=='A' else 'A')]['event']['velocity_at_release'])))}
def mean(xs,k):return float(np.nanmean([x[k] for x in xs]))


@dataclass(frozen=True)
class SupportReleaseEvaluationConfig:
    limit: int = 0


@dataclass(frozen=True)
class SupportReleasePredictionConfig:
    training_config: Path
    checkpoint: Path
    steps: int = 20
    limit: int = 0
    seed_offset: int = 1_700_000
    device: str = "cuda"
    dataset_id: str = "support_release_v1"

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "SupportReleasePredictionConfig":
        return cls(
            training_config=Path(value["training_config"]),
            checkpoint=Path(value["checkpoint"]),
            steps=int(
                value.get("steps", value.get("diffusion_steps", 20))
            ),
            limit=int(value.get("limit") or 0),
            seed_offset=int(value.get("seed_offset", 1_700_000)),
            device=str(value.get("device", "cuda")),
            dataset_id=str(
                value.get("dataset_id") or "support_release_v1"
            ),
        )


def _summary(allrows):
    keys=('outgoing_angle_mae','horizontal_trajectory_rmse','trajectory_rmse','endpoint_error','final_horizontal_direction_accuracy','d_correct','d_wrong','correct_follow','shortcut_follow','rho','ball_detection_success','disappearance_rate','teleportation_rate','shape_deformation_rate','upward_acceleration_inconsistency','future_support_persistence','out_of_frame_rate','multiple_ball_hallucination','valid_trajectory_rate')
    report={'all':{k:mean(allrows,k) for k in keys},'variants':{},'by_state':{}}
    for v in ('ID','G','C','GC'):
        xs=[x for x in allrows if x['variant']==v]
        if not xs:
            continue
        report['variants'][v]={k:mean(xs,k) for k in keys}
        report['variants'][v]['rho_median']=float(np.nanmedian([x['rho'] for x in xs]))
        report['variants'][v]['both_shortcut_follow']=float(np.mean([all(x['shortcut_follow'] for x in allrows if x['octet_id']==q and x['variant']==v) for q in sorted({x['octet_id'] for x in xs})]))
    for state in 'AB':
        xs=[x for x in allrows if x['state']==state]
        if xs:
            report['by_state'][state]={k:mean(xs,k) for k in keys}
    return report


def predict(
    dataset_dir: Path,
    prediction_dir: Path,
    config: SupportReleasePredictionConfig | Mapping[str, Any],
):
    """Generate pure Support-Release predictions and a common manifest."""
    if isinstance(config, Mapping):
        config = SupportReleasePredictionConfig.from_mapping(config)
    if config.steps <= 0:
        raise ValueError("steps must be positive")
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
    for row in source_rows:
        metadata = json.loads(
            (dataset_dir / row["metadata"]).read_text()
        )
        sample_id = row.get("sample_id") or (
            f"{metadata['octet_id']}:{metadata['track']}:"
            f"{metadata['state']}:{metadata['variant']}"
        )
        generation_seed = (
            int(metadata["base_seed"]) + config.seed_offset
        )
        generated = sample(
            pipe,
            read(dataset_dir / row["video"]),
            generation_seed,
            config.steps,
        )
        destination = (
            prediction_dir
            / "predictions"
            / metadata["variant"]
            / metadata["state"]
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
        predictions.append({
            "prediction_id": sample_id,
            "sample_id": sample_id,
            "prediction": destination.relative_to(
                prediction_dir
            ).as_posix(),
            "attributes": {
                "variant": metadata["variant"],
                "state": metadata["state"],
                "seed": generation_seed,
            },
        })
    return write_predictions(
        prediction_dir,
        experiment="support_release",
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
    config: SupportReleaseEvaluationConfig | Mapping[str, Any] | None = None,
):
    if isinstance(config, Mapping):
        config = SupportReleaseEvaluationConfig(
            limit=int(config.get("limit") or 0)
        )
    config = config or SupportReleaseEvaluationConfig()
    source_rows = list(
        csv.DictReader((dataset_dir / 'metadata.csv').open())
    )[: config.limit or None]
    opposites = opposite_index(dataset_dir)
    all_rows = []
    for row in source_rows:
        metadata = json.loads(
            (dataset_dir / row['metadata']).read_text()
        )
        sample_id = row.get('sample_id') or (
            f"{metadata['octet_id']}:{metadata['track']}:"
            f"{metadata['state']}:{metadata['variant']}"
        )
        prediction = find_prediction(
            prediction_dir,
            sample_id=sample_id,
        )
        if prediction is None:
            prediction = (
                prediction_dir
                / metadata['variant']
                / metadata['state']
                / row['video']
            )
        item = metric(pred=read(prediction),m=metadata,other=opposites)
        item.update({
            'sample_id': sample_id,
            'sample': row['video'],
            'octet_id': metadata['octet_id'],
            'variant': metadata['variant'],
        })
        all_rows.append(item)
    return {'summary': _summary(all_rows), 'samples': all_rows}


def write_evaluation(result, out: Path, *, context=None):
    return write_metrics(
        out,
        experiment='support_release',
        dataset='support_release_v1',
        summary=result['summary'],
        samples=result['samples'],
        context=context,
        legacy={'samples': result['samples']},
    )
