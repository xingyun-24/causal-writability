"""Evaluation for the frozen dual-wall shortcut benchmark."""
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

def read(p):
    r=imageio.get_reader(p)
    try:return np.stack([r.get_data(i) for i in range(49)])
    finally:r.close()
def tensor(x): return torch.from_numpy(x).permute(3,0,1,2).float()/127.5-1
def ad(a,b): return abs((a-b+180)%360-180)
def angle(v): return math.degrees(math.atan2(v[1],v[0]))
def detector(video,color):
    r,g,b=video[...,0],video[...,1],video[...,2]
    m=(r>105)&(r>g*1.25)&(r>b*1.25) if color=='red' else (b>90)&(b>r*1.15)&(b>g*1.1)
    mass=m.sum((1,2));xx=np.arange(128)[None,None,:];yy=np.arange(128)[None,:,None]
    x=np.divide((m*xx).sum((1,2)),mass,out=np.full(49,np.nan),where=mass>10);y=np.divide((m*yy).sum((1,2)),mass,out=np.full(49,np.nan),where=mass>10)
    return np.stack(((x+.5)/128,1-(y+.5)/128),-1),mass>10
def sample(pipe,raw,seed,steps):
    v=tensor(raw)
    with torch.no_grad():
        pipe.load_models_to_device(['vae']); c=pipe.vae.encode([v[:,:5].to(dtype=pipe.torch_dtype)],device=pipe.device,tiled=False).to(dtype=pipe.torch_dtype,device=pipe.device);pipe.pre_encoded_(True)
        g=pipe(height=128,width=128,num_frames=49,num_condition_frames=2,condition_frames=c,num_inference_steps=steps,num_samples=1,cfg_scale=1.,tiled=False,seed=seed,return_as_tensor=True,progress_bar_cmd=lambda x:x)[0]
    return ((g.permute(1,2,3,0).float().cpu().numpy()+1)*127.5).clip(0,255).astype(np.uint8)
def opposite_index(root):
    out=defaultdict(dict)
    for r in csv.DictReader((root/'metadata.csv').open()):
        m=json.loads((root/r['metadata']).read_text())
        for e in m['interactions'].values(): out[m['quartet_id']][(e['wall_instance'],e['state'])]=e
    return out
def one_metrics(pred,m,other):
    ans=[]
    for side,e in m['interactions'].items():
        p,ok=detector(pred,e['color']);gt=np.asarray(e['positions']);wrong=np.asarray(other[(e['wall_instance'],'B' if e['state']=='A' else 'A')]['positions']);future=ok[5:]
        valid=future.any(); err=np.linalg.norm(p[5:]-gt[5:],axis=1);werr=np.linalg.norm(p[5:]-wrong[5:],axis=1)
        d_correct=float(np.nanmean(err)) if valid else np.nan;d_wrong=float(np.nanmean(werr)) if valid else np.nan
        vp=np.nanmedian(np.diff(p[8:],axis=0)*15,axis=0);vo=np.asarray(e['velocity_out']);oa=ad(angle(vp),angle(vo)) if np.isfinite(vp).all() else np.nan
        n,c=np.asarray(e['normal']),np.asarray(e['contact_point']);signed=(p-c)@n
        ans.append({'side':side,'wall_instance':e['wall_instance'],'state':e['state'],'color':e['color'],'outgoing_angle_mae':float(oa),'center_trajectory_rmse':float(np.sqrt(np.nanmean(err**2))) if valid else np.nan,'endpoint_error':float(np.linalg.norm(p[-1]-gt[-1])) if np.isfinite(p[-1]).all() else np.nan,'collision_validity':float(future.mean()),'disappearance_rate':float((~future).mean()),'penetration_rate':float((signed[5:]<e['radius']-.01).mean()),'d_correct':d_correct,'d_wrong':d_wrong,'correct_follow':float(d_correct<d_wrong) if np.isfinite(d_correct+d_wrong) else 0.,'shortcut_follow':float(d_wrong<d_correct) if np.isfinite(d_correct+d_wrong) else 0.,'predicted_angle':float(angle(vp)) if np.isfinite(vp).all() else np.nan,'correct_angle':float(e['theta_out']),'opposite_angle':float(other[(e['wall_instance'],'B' if e['state']=='A' else 'A')]['theta_out'])})
    # Future identities are assigned by their current visual colour, so a
    # collapse of colour masks measures either a merge or disappearance.
    a,b=ans;pa,oa0=detector(pred,a['color']);pb,ob0=detector(pred,b['color']);merge=float((np.linalg.norm(pa[5:]-pb[5:],axis=1)<.07).mean())
    return ans,merge
def mean(xs,key): return float(np.nanmean([x[key] for x in xs]))


@dataclass(frozen=True)
class DualWallEvaluationConfig:
    limit: int = 0


@dataclass(frozen=True)
class DualWallPredictionConfig:
    training_config: Path
    checkpoint: Path
    steps: int = 20
    limit: int = 0
    seed_offset: int = 1_700_000
    device: str = "cuda"
    dataset_id: str = "dual_wall_shortcuts_v1"

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "DualWallPredictionConfig":
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
                value.get("dataset_id") or "dual_wall_shortcuts_v1"
            ),
        )


def _summary(all_rows):
    keys=('outgoing_angle_mae','center_trajectory_rmse','endpoint_error','collision_validity','disappearance_rate','penetration_rate','merge_rate','d_correct','d_wrong','correct_follow','shortcut_follow')
    report={'all':{k:mean(all_rows,k) for k in keys},'variants':{},'by_slot':{}}
    for v in ('ID','G','C','GC'):
        xs=[q for q in all_rows if q['variant']==v]
        if not xs:
            continue
        report['variants'][v]={k:mean(xs,k) for k in keys}
        report['variants'][v]['both_interactions_shortcut_follow']=float(np.mean([all(q['shortcut_follow'] for q in all_rows if q['quartet_id']==qid and q['variant']==v) for qid in sorted({q['quartet_id'] for q in xs})]))
    for side in ('left','right'):
        xs=[q for q in all_rows if q['side']==side]
        if xs:
            report['by_slot'][side]={k:mean(xs,k) for k in keys}
    return report


def predict(
    dataset_dir: Path,
    prediction_dir: Path,
    config: DualWallPredictionConfig | Mapping[str, Any],
):
    """Generate pure Dual-Wall predictions and a common manifest."""
    if isinstance(config, Mapping):
        config = DualWallPredictionConfig.from_mapping(config)
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
            f"{metadata['quartet_id']}:{metadata['track']}:"
            f"{metadata['variant']}"
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
            / row["video"]
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(
            destination,
            fps=15,
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
                "seed": generation_seed,
            },
        })
    return write_predictions(
        prediction_dir,
        experiment="dual_wall",
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
    config: DualWallEvaluationConfig | Mapping[str, Any] | None = None,
):
    if isinstance(config, Mapping):
        config = DualWallEvaluationConfig(
            limit=int(config.get("limit") or 0)
        )
    config = config or DualWallEvaluationConfig()
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
            f"{metadata['quartet_id']}:{metadata['track']}:"
            f"{metadata['variant']}"
        )
        prediction = find_prediction(
            prediction_dir,
            sample_id=sample_id,
        )
        if prediction is None:
            prediction = (
                prediction_dir
                / metadata['variant']
                / row['video']
            )
        items, merge = one_metrics(
            read(prediction),
            metadata,
            opposites[metadata['quartet_id']],
        )
        for item in items:
            item.update({
                'sample_id': sample_id,
                'sample': row['video'],
                'quartet_id': metadata['quartet_id'],
                'variant': metadata['variant'],
                'pos_perm': metadata['pos_perm'],
                'merge_rate': merge,
            })
            all_rows.append(item)
    return {'summary': _summary(all_rows), 'samples': all_rows}


def write_evaluation(result, out: Path, *, context=None):
    return write_metrics(
        out,
        experiment='dual_wall',
        dataset='dual_wall_shortcuts_v1',
        summary=result['summary'],
        samples=result['samples'],
        context=context,
        legacy={'interactions': result['samples']},
    )
