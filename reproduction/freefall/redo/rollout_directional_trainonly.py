#!/usr/bin/env python3
"""Generate natural and fit-only 2-D controller outputs on held-out receivers."""
import argparse,csv,hashlib,json,sys,time
from pathlib import Path

import numpy as np
import torch,yaml


def write(path,data):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n'); tmp.replace(path)


def longest_false_run(mask):
    best=run=0
    for value in mask:
        run=0 if value else run+1; best=max(best,run)
    return best


def sha(path):
    h=hashlib.sha256();
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project-root',type=Path,required=True); p.add_argument('--pca-dir',type=Path,required=True)
    p.add_argument('--controller-json',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--shard',type=int,required=True); p.add_argument('--num-shards',type=int,default=3); p.add_argument('--device',default='cuda')
    a=p.parse_args(); root=a.project_root; sys.path.insert(0,str(root/'scripts'))
    from layer_residual_replacement import condition_token_count,generate,load_condition,measure,detect_ball
    from audit_v1_estimator import audit_track
    from projectile_block_pca_fit import cfg_from
    from sshv2.wan.config import StandardTrainingConfig
    from sshv2.wan.trainer import WanTrainingModule
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    out=a.out/f'shard{a.shard}'; (out/'frames').mkdir(parents=True,exist_ok=True)
    controller=json.loads(a.controller_json.read_text()); packed=np.load(a.pca_dir/'train_only_pca.npz')
    ids=np.asarray(packed['pair_ids'],dtype=str); heldout=packed['heldout_mask'].astype(bool); train=packed['train_mask'].astype(bool)
    wanted=np.flatnonzero(heldout)[a.shard::a.num_shards]
    run_spec={'controller_sha256':sha(a.controller_json),'pca_sha256':sha(a.pca_dir/'train_only_pca.npz'),
              'script_sha256':sha(Path(__file__)),'shard':a.shard,'num_shards':a.num_shards,'pair_ids':ids[wanted].tolist(),
              'pca_rank':2,'block':1,'token_scope':'condition-prefix only','fm_calls':20,'observed_window':32,
              'generation_seed':'base_seed + 17000000','global_scale':controller['global_scale']}
    if (out/'run_spec.json').exists(): assert json.loads((out/'run_spec.json').read_text())==run_spec
    else: write(out/'run_spec.json',run_spec)
    results=json.loads((out/'outcomes.json').read_text()) if (out/'outcomes.json').exists() else []
    done={(r['receiver_id'],r['condition']) for r in results}
    sources={}
    for shard in sorted((root/'results/freefall-hist32-step100000-block1-pca128').glob('shard*')):
        array=np.load(shard/'deltas.npy',mmap_mode='r'); rows=json.loads((shard/'metadata.json').read_text())
        for j,row in enumerate(rows): sources[row['pair_id']]=(array,j,row)
    shape=tuple(int(v) for v in packed['delta_shape']); device=torch.device(a.device)
    matrix=torch.empty((train.sum(),int(np.prod(shape))),dtype=torch.float64,device=device)
    for j,i in enumerate(np.flatnonzero(train)):
        array,row,_=sources[ids[i]]; matrix[j].copy_(torch.from_numpy(np.array(array[row],copy=True)).flatten())
    weights=torch.tensor(packed['train_vectors']/np.sqrt(packed['eigenvalues'])[None,:],dtype=torch.float64,device=device)
    directions=(weights[:,:2].T@matrix).float(); del weights,matrix; torch.cuda.empty_cache()
    dataset=root/'data/raw-v3-fixedpos-freefall-eval320/eval'
    with (dataset/'metadata.csv').open(newline='') as f: metadata={(r['pair_id'],r['variant']):r for r in csv.DictReader(f)}
    evals={(r['pair_id'],r['condition']):r for r in json.loads((root/'results/freefall-hist32-step100000-eval320-strict/rows_merged.json').read_text())}
    config=root/'config/data-v3-fixedpos-freefall.yaml'; raw=yaml.safe_load(config.read_text()); cfg=cfg_from(config)
    cfg.gravity_bands={'low':tuple(raw['physics']['low_gravity_range']),'high':tuple(raw['physics']['high_gravity_range'])}
    cfg.radius=int(raw['render']['ball_radius_px']); cfg.gravity_accuracy_threshold=float(raw['evaluation']['E3_threshold'])
    training=StandardTrainingConfig.from_file(root/'config/Train-short-v3-large-fixedpos-freefall-hist32.yaml')
    training.model.dit.ckpt_file=root/'checkpoints/short-v3-large-fixedpos-freefall-hist32/hist32/step-100000.safetensors'
    module=WanTrainingModule(dit_config=training.model.dit,vae_config=training.model.vae,no_encoding=False,
        num_condition_frames=training.model.num_condition_frames,num_inference_steps=20,pipeline_type=training.model.pipe,pipeline_kwargs=training.model.pipe_kwargs)
    pipe=module.pipe; pipe.to(a.device); pipe.load_models_to_device(('dit','vae')); pipe.pre_encoded_(False); cfg.condition_latents=int(training.model.num_condition_frames)
    def enrich(frames,row):
        simple=measure(frames,row,cfg); detailed=audit_track(frames,row,cfg); y,rgb,valid=detect_ball(frames,cfg)
        gaps=np.abs(np.diff(y[np.isfinite(y)])) if np.isfinite(y).sum()>1 else np.asarray([])
        future_rgb=np.nanmedian(rgb,axis=0) if np.isfinite(rgb).any() else np.asarray([np.nan]*3)
        return {**simple,**detailed,'valid':simple['valid'],'gravity_hat':simple['gravity_hat'],'gravity_hat_e0':simple['gravity_hat_e0'],
                'fit_rmse':detailed['y_rmse_E3'],'detector_coverage':float(valid.mean()),'max_missing_run':int(longest_false_run(valid)),
                'max_adjacent_y_jump':float(gaps.max()) if len(gaps) else None,
                'future_r':float(future_rgb[0]),'future_g':float(future_rgb[1]),'future_b':float(future_rgb[2])}
    start=time.monotonic()
    for count,i in enumerate(wanted,1):
        pid=ids[i]; conflict=metadata[(pid,'conflict')]; band=controller['direction'][i]; g=float(controller['gravity_target'][i]); seed=int(conflict['base_seed'])+17000000
        old=cfg.short_masked_prefix; cfg.short_masked_prefix=cfg.prediction_start-32
        try: condition=load_condition(conflict,dataset,cfg,pipe.torch_dtype,a.device,True)
        finally: cfg.short_masked_prefix=old
        captures=[]
        def capture(_m,_x,hidden):
            n=condition_token_count(int(hidden.shape[1]),cfg); captures.append(hidden[0,:n].detach().clone())
        handle=pipe.dit.blocks[1].register_forward_hook(capture)
        try: natural_frames=generate(pipe,condition,cfg,20,seed)
        finally: handle.remove()
        assert len(captures)==20; current=torch.stack(captures).float(); del captures
        for condition_name,frames in [('natural_conflict',natural_frames)]:
            if (pid,condition_name) not in done:
                path=out/'frames'/f'{pid}__{condition_name}.npz'; np.savez_compressed(path,frames=frames)
                m=enrich(frames,conflict); results.append({'receiver_id':pid,'pair_id':pid,'condition':condition_name,'direction':band,
                    'target_direction':band,'g_target':g,'gravity_hat_conflict':float(evals[(pid,'conflict')]['g_E3']),
                    'gravity_hat_aligned':float(evals[(pid,'aligned')]['g_E3']),'gravity_hat_edited':m['gravity_hat'],
                    'recovery_denominator':float(evals[(pid,'aligned')]['g_E3'])-float(evals[(pid,'conflict')]['g_E3']),
                    'normalized_recovery_g':0.0,'outcome':'E3_pass' if m['E3_correct'] else 'E3_fail','intervention_site':'after_block_1',
                    'fm_calls_observed':20,'fm_calls_expected':20,'condition_prefix_tokens_edited':0,'target_suffix_tokens_edited':0,
                    'intervention_scale':0.0,'activation_delta_l2':0.0,'generation_seed':seed,'output_frames_path':str(path),**m}); done.add((pid,condition_name)); write(out/'outcomes.json',results)
        coef=torch.tensor(controller['models'][band]['coefficients_rows_[1_g_target]'],dtype=torch.float32,device=device)
        score=torch.tensor([1.,g],device=device)@coef
        delta=(score@directions*float(controller['global_scale'])).reshape(shape)
        call=0
        def replace(_m,_x,hidden):
            nonlocal call
            n=condition_token_count(int(hidden.shape[1]),cfg); edited=hidden.clone(); edited[:,:n]=(current[call]+delta[call]).unsqueeze(0).to(hidden.dtype); call+=1; return edited
        handle=pipe.dit.blocks[1].register_forward_hook(replace)
        try: frames=generate(pipe,condition,cfg,20,seed)
        finally: handle.remove()
        assert call==20
        condition_name='fit_only_controller'
        if (pid,condition_name) not in done:
            path=out/'frames'/f'{pid}__{condition_name}.npz'; np.savez_compressed(path,frames=frames); m=enrich(frames,conflict)
            gc=float(evals[(pid,'conflict')]['g_E3']); ga=float(evals[(pid,'aligned')]['g_E3']); edit=m['gravity_hat']; denom=ga-gc
            results.append({'receiver_id':pid,'pair_id':pid,'condition':condition_name,'direction':band,'target_direction':band,'g_target':g,
                'gravity_hat_conflict':gc,'gravity_hat_aligned':ga,'gravity_hat_edited':edit,'recovery_denominator':denom,
                'normalized_recovery_g':None if edit is None else (edit-gc)/denom,'outcome':'E3_pass' if m['E3_correct'] else 'E3_fail',
                'intervention_site':'after_block_1','fm_calls_observed':call,'fm_calls_expected':20,'condition_prefix_tokens_edited':1088,
                'target_suffix_tokens_edited':0,'intervention_scale':float(controller['global_scale']),
                'activation_delta_l2':float(torch.linalg.vector_norm(delta)),'generation_seed':seed,'output_frames_path':str(path),**m})
            done.add((pid,condition_name)); write(out/'outcomes.json',results)
        print(f'shard={a.shard} pair={count}/{len(wanted)} id={pid} elapsed={time.monotonic()-start:.1f}',flush=True)
    write(out/'complete.json',{'pairs':len(wanted),'rows':len(results),'elapsed_seconds':time.monotonic()-start})


if __name__=='__main__':main()
