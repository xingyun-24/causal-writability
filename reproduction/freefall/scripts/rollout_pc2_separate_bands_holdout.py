#!/usr/bin/env python3
"""Roll out band-specific PC2-only colour-preserving fit on held-out pairs."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np, torch, yaml
from full_token_pca_recovery import load_matrix
from layer_residual_replacement import colour_name,condition_token_count,detect_ball,load_condition,measure
from projectile_block_pca_fit import capture_residual,cfg_from,generate
def cv(v): return {'red':1.,'blue':-1.}[v]
def main():
 p=argparse.ArgumentParser()
 for n in ('dataset-dir','data-config','training-config','checkpoint','pair-manifest','fit-json','eval-rows','pca-root','out'): p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--scale',type=float,default=2.);p.add_argument('--block',type=int,default=1);p.add_argument('--observed-window',type=int,default=32);p.add_argument('--steps',type=int,default=20);p.add_argument('--device',default='cuda')
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True); fit=json.loads(a.fit_json.read_text()); pairs={x['pair_id']:x for x in json.loads(a.pair_manifest.read_text())['pairs']}; hold=set(fit['holdout_pair_ids']); z=fit['bands']; device=torch.device(a.device);matrix,_,vectors,eigs,shape=load_matrix(a.pca_root,device)
 data=yaml.safe_load(a.data_config.read_text());cfg=cfg_from(a.data_config);cfg.gravity_bands={'low':tuple(data['physics']['low_gravity_range']),'high':tuple(data['physics']['high_gravity_range'])};cfg.radius=int(data['render']['ball_radius_px']);cfg.gravity_accuracy_threshold=float(data['evaluation']['E3_threshold'])
 from sshv2.wan.config import StandardTrainingConfig
 from sshv2.wan.trainer import WanTrainingModule
 tr=StandardTrainingConfig.from_file(a.training_config);tr.model.dit.ckpt_file=a.checkpoint;mod=WanTrainingModule(dit_config=tr.model.dit,vae_config=tr.model.vae,no_encoding=False,num_condition_frames=tr.model.num_condition_frames,num_inference_steps=a.steps,pipeline_type=tr.model.pipe,pipeline_kwargs=tr.model.pipe_kwargs);pipe=mod.pipe;pipe.to(a.device);pipe.load_models_to_device(('dit','vae'));pipe.pre_encoded_(False);cfg.condition_latents=int(tr.model.num_condition_frames)
 meta={(r['pair_id'],r['variant']):r for r in csv.DictReader((a.dataset_dir/'metadata.csv').open(newline=''))};out=[]
 for n,pair in enumerate([x for x in json.loads(a.pair_manifest.read_text())['pairs'] if x['pair_id'] in hold],1):
  pid=pair['pair_id'];band=pair['gravity_interval'];coef=torch.tensor(z[band]['coefficients_rows_feature_order'],dtype=torch.float32,device=device);cur=meta[(pid,'conflict')];tar=meta[(pid,'aligned')];g=float(pair['gravity_true']);dc=cv(tar['color_label'])-cv(cur['color_label']);feature=torch.tensor([dc,dc*g,dc*np.sqrt(g)],dtype=torch.float32,device=device);score=feature@coef;score[0]=0;score[1]*=a.scale;weights=vectors[:,:2]@(score/torch.sqrt(eigs[:2].float()));delta=(weights.to(matrix.dtype)@matrix).reshape(shape);old=cfg.short_masked_prefix;cfg.short_masked_prefix=cfg.prediction_start-a.observed_window;cond=load_condition(cur,a.dataset_dir,cfg,pipe.torch_dtype,a.device,True);cfg.short_masked_prefix=old;seed=int(cur['base_seed'])+17000000;res=torch.from_numpy(capture_residual(pipe,cond,cfg,a.steps,seed,a.block)).to(device);call=0
  def hook(_m,_i,h):
   nonlocal call
   t=condition_token_count(int(h.shape[1]),cfg);e=h.clone();e[:,:t]=(res[call][None]+delta[call][None]).to(h.dtype);call+=1;return e
  h=pipe.dit.blocks[a.block].register_forward_hook(hook)
  try: frames=generate(pipe,cond,cfg,a.steps,seed)
  finally:h.remove()
  _,rgb,valid=detect_ball(frames,type('C',(),{'radius':cfg.radius,'height':cfg.height})()); detected=colour_name(rgb);out.append({'pair_id':pid,'band':band,'input_colour':cur['color_label'],'detected_colour_majority':detected,**measure(frames,cur,cfg)});print(f'pairs={n}/{len(hold)}',flush=True)
 changed=sum(x['detected_colour_majority']!=x['input_colour'] for x in out);payload={'method':'separate low/high band fit; PC1 zero; PC2-only colour-preserving rollout on heldout half','scale':a.scale,'pairs':len(out),'E3_accuracy':float(np.mean([x['E3_correct'] for x in out])),'E0_accuracy':float(np.mean([x['E0_correct'] for x in out])),'mean_abs_g_error':float(np.mean([x['gravity_error'] for x in out])),'colour_changed_pairs':changed,'colour_preserved_pairs':len(out)-changed,'results':out};(a.out/'summary.json').write_text(json.dumps(payload,indent=2)+'\n');(a.out/'outcomes.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:payload[k] for k in ('pairs','E3_accuracy','E0_accuracy','mean_abs_g_error','colour_changed_pairs')},indent=2))
if __name__=='__main__':main()
