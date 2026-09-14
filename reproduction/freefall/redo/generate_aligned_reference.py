#!/usr/bin/env python3
import argparse,csv,sys
from pathlib import Path
import numpy as np,torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,required=True);p.add_argument('--pair-id',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
    root=a.project_root;sys.path.insert(0,str(root/'scripts'))
    from layer_residual_replacement import generate,load_condition
    from projectile_block_pca_fit import cfg_from
    from sshv2.wan.config import StandardTrainingConfig
    from sshv2.wan.trainer import WanTrainingModule
    dataset=root/'data/raw-v3-fixedpos-freefall-eval320/eval';config=root/'config/data-v3-fixedpos-freefall.yaml';cfg=cfg_from(config)
    with (dataset/'metadata.csv').open(newline='') as f:rows={(r['pair_id'],r['variant']):r for r in csv.DictReader(f)}
    row=rows[(a.pair_id,'aligned')];training=StandardTrainingConfig.from_file(root/'config/Train-short-v3-large-fixedpos-freefall-hist32.yaml');training.model.dit.ckpt_file=root/'checkpoints/short-v3-large-fixedpos-freefall-hist32/hist32/step-100000.safetensors'
    module=WanTrainingModule(dit_config=training.model.dit,vae_config=training.model.vae,no_encoding=False,num_condition_frames=training.model.num_condition_frames,num_inference_steps=20,pipeline_type=training.model.pipe,pipeline_kwargs=training.model.pipe_kwargs)
    pipe=module.pipe;pipe.to(a.device);pipe.load_models_to_device(('dit','vae'));pipe.pre_encoded_(False);cfg.condition_latents=int(training.model.num_condition_frames);old=cfg.short_masked_prefix;cfg.short_masked_prefix=cfg.prediction_start-32
    try:condition=load_condition(row,dataset,cfg,pipe.torch_dtype,a.device,True)
    finally:cfg.short_masked_prefix=old
    frames=generate(pipe,condition,cfg,20,int(row['base_seed'])+17000000);np.savez_compressed(a.out,frames=frames);print(a.out,frames.shape)

if __name__=='__main__':main()
