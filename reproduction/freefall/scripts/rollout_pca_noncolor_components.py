#!/usr/bin/env python3
"""Inject fitted PCA components 2..k while fixing PC1 to zero."""
from __future__ import annotations

import argparse, csv, json
from pathlib import Path
import numpy as np
import torch, yaml
from full_token_pca_recovery import load_matrix
from layer_residual_replacement import colour_name, condition_token_count, detect_ball, load_condition, measure
from projectile_block_pca_fit import capture_residual, cfg_from, generate

def color_value(v: str) -> float: return {"red":1.0,"blue":-1.0}[v]
def color_name_frames(frames: np.ndarray, radius: int, height: int) -> str:
    probe=type("Config",(),{"radius":radius,"height":height})(); _,rgb,_=detect_ball(frames,probe); return colour_name(rgb)
def main() -> None:
    p=argparse.ArgumentParser()
    for n in ("dataset-dir","data-config","training-config","checkpoint","pair-manifest","fit-json","pca-root","out"):p.add_argument(f"--{n}",type=Path,required=True)
    p.add_argument("--scale",type=float,required=True);p.add_argument("--block",type=int,default=1);p.add_argument("--observed-window",type=int,default=32);p.add_argument("--steps",type=int,default=20);p.add_argument("--device",default="cuda")
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True); fit=json.loads(a.fit_json.read_text());k=int(fit["components"])
    matrix,_,vectors,eigs,shape=load_matrix(a.pca_root,torch.device(a.device)); coef=torch.tensor(fit["coefficients_rows_feature_order"],dtype=torch.float32,device=a.device)
    pairs=json.loads(a.pair_manifest.read_text())["pairs"]; data=yaml.safe_load(a.data_config.read_text());cfg=cfg_from(a.data_config);cfg.gravity_bands={"low":tuple(data["physics"]["low_gravity_range"]),"high":tuple(data["physics"]["high_gravity_range"])};cfg.radius=int(data["render"]["ball_radius_px"]);cfg.gravity_accuracy_threshold=float(data["evaluation"]["E3_threshold"])
    from sshv2.wan.config import StandardTrainingConfig
    from sshv2.wan.trainer import WanTrainingModule
    tr=StandardTrainingConfig.from_file(a.training_config);tr.model.dit.ckpt_file=a.checkpoint; module=WanTrainingModule(dit_config=tr.model.dit,vae_config=tr.model.vae,no_encoding=False,num_condition_frames=tr.model.num_condition_frames,num_inference_steps=a.steps,pipeline_type=tr.model.pipe,pipeline_kwargs=tr.model.pipe_kwargs);pipe=module.pipe;pipe.to(a.device);pipe.load_models_to_device(("dit","vae"));pipe.pre_encoded_(False);cfg.condition_latents=int(tr.model.num_condition_frames)
    meta={(r["pair_id"],r["variant"]):r for r in csv.DictReader((a.dataset_dir/"metadata.csv").open(newline=""))};out=[]
    for n,pair in enumerate(pairs,1):
      pid=pair["pair_id"];cur=meta[(pid,"conflict")];target=meta[(pid,"aligned")];g=float(pair["gravity_true"]);dc=color_value(target["color_label"])-color_value(cur["color_label"]);feature=torch.tensor([dc,dc*g,dc*np.sqrt(g)],dtype=torch.float32,device=a.device);score=feature@coef;score[0]=0;score[1:]*=a.scale;weights=vectors[:,:k]@(score/torch.sqrt(eigs[:k].float()));delta=(weights.to(matrix.dtype)@matrix).reshape(shape)
      old=cfg.short_masked_prefix;cfg.short_masked_prefix=cfg.prediction_start-a.observed_window;condition=load_condition(cur,a.dataset_dir,cfg,pipe.torch_dtype,a.device,True);cfg.short_masked_prefix=old;seed=int(cur["base_seed"])+17_000_000;res=torch.from_numpy(capture_residual(pipe,condition,cfg,a.steps,seed,a.block)).to(a.device);call=0
      def hook(_m,_i,h):
       nonlocal call
       tokens=condition_token_count(int(h.shape[1]),cfg);edited=h.clone();edited[:,:tokens]=(res[call].unsqueeze(0)+delta[call].unsqueeze(0)).to(h.dtype);call+=1;return edited
      handle=pipe.dit.blocks[a.block].register_forward_hook(hook)
      try:frames=generate(pipe,condition,cfg,a.steps,seed)
      finally:handle.remove()
      measured=measure(frames,cur,cfg);detected=color_name_frames(frames,cfg.radius,cfg.height);out.append({"pair_id":pid,"input_colour":cur["color_label"],"detected_colour_majority":detected,**measured});print(f"pairs={n}/{len(pairs)}",flush=True)
    changed=[r for r in out if r["detected_colour_majority"]!=r["input_colour"]];summary={"method":f"fitted PCA components 2..{k}; PC1 fixed zero","components":k,"scale":a.scale,"pairs":len(out),"E3_accuracy":float(np.mean([r["E3_correct"] for r in out])),"E0_accuracy":float(np.mean([r["E0_correct"] for r in out])),"mean_abs_g_error":float(np.mean([r["gravity_error"] for r in out if r["gravity_error"] is not None])),"colour_changed_pairs":len(changed),"colour_preserved_pairs":len(out)-len(changed),"results":out};(a.out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");(a.out/"outcomes.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({k:summary[k] for k in ("components","scale","E3_accuracy","mean_abs_g_error","colour_changed_pairs")},indent=2))
if __name__=="__main__":main()
