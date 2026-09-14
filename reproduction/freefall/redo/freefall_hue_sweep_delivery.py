#!/usr/bin/env python3
"""Regenerate F1 futures with measured RGB and lossless video delivery."""
import argparse,csv,hashlib,json,math,sys,time
from pathlib import Path

import cv2,imageio.v2 as imageio,numpy as np,torch,yaml


def finite(x):
    if isinstance(x,dict): return {k:finite(v) for k,v in x.items()}
    if isinstance(x,list): return [finite(v) for v in x]
    if isinstance(x,np.generic): return finite(x.item())
    if isinstance(x,float) and not math.isfinite(x): return None
    return x


def write(path,data):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(finite(data),indent=2)+'\n'); tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def recolour(frames,rgb,background):
    out=frames.copy(); bg=np.asarray(background,dtype=np.int16); target=np.asarray(rgb,dtype=np.float32)
    for i,frame in enumerate(out):
        d=np.linalg.norm(frame.astype(np.int16)-bg[None,None,:],axis=-1); mask=d>10; alpha=np.clip(d[mask,None]/80.,0,1)
        out[i][mask]=np.rint((1-alpha)*bg+alpha*target).astype(np.uint8)
    return out


def longest_false(mask):
    best=run=0
    for v in mask: run=0 if v else run+1; best=max(best,run)
    return best


def main():
    p=argparse.ArgumentParser(); p.add_argument('--project-root',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--skip-json',type=Path)
    p.add_argument('--shard',type=int,required=True); p.add_argument('--num-shards',type=int,default=3); p.add_argument('--device',default='cuda'); a=p.parse_args()
    root=a.project_root; sys.path.insert(0,str(root/'scripts'))
    from audit_v1_estimator import audit_track,make_cfg,track_ball
    from layer_residual_replacement import generate,load_condition
    from sshv2.wan.config import StandardTrainingConfig
    from sshv2.wan.trainer import WanTrainingModule
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    dataset=root/'data/raw-v3-fixedpos-freefall-eval320/eval'; config=root/'config/data-v3-fixedpos-freefall.yaml'; raw=yaml.safe_load(config.read_text()); cfg=make_cfg(config)
    cfg.gravity_bands={'low':tuple(raw['physics']['low_gravity_range']),'high':tuple(raw['physics']['high_gravity_range'])}; cfg.radius=int(raw['render']['ball_radius_px']); cfg.gravity_accuracy_threshold=float(raw['evaluation']['E3_threshold']); cfg.condition_latents=17
    with (dataset/'metadata.csv').open(newline='') as f: rows=list(csv.DictReader(f))
    aligned={r['pair_id']:r for r in rows if r['variant']=='aligned'}
    physical=sorted(aligned.values(),key=lambda r:(r['gravity_interval'],int(r['base_seed'])))
    physical=[r for r in physical if r['gravity_interval']=='low'][:32]+[r for r in physical if r['gravity_interval']=='high'][:32]
    red=np.asarray(raw['render']['red_rgb']); blue=np.asarray(raw['render']['blue_rgb'])
    colours=[(j,j/10.,tuple(np.rint((1-j/10.)*red+(j/10.)*blue).astype(int))) for j in range(11)]
    jobs=[(window,label,row,j,u,rgb) for window,label in ((32,'Short'),(64,'Long')) for row in physical for j,u,rgb in colours]
    if a.skip_json:
        skipped={tuple(x) for x in json.loads(a.skip_json.read_text())}
        jobs=[job for job in jobs if (job[1],job[2]['pair_id'],job[3]) not in skipped]
    jobs=jobs[a.shard::a.num_shards]
    out=a.out/f'shard{a.shard}'; videos=out/'videos'; inputs=out/'inputs'; input_videos=inputs/'videos'; videos.mkdir(parents=True,exist_ok=True); input_videos.mkdir(parents=True,exist_ok=True)
    results=json.loads((out/'rows.json').read_text()) if (out/'rows.json').exists() else []; done={(r['history_regime'],r['pair_id'],r['hue_index']) for r in results}
    training=StandardTrainingConfig.from_file(root/'config/Train-short-v3-large-fixedpos-freefall-hist32.yaml'); training.model.dit.ckpt_file=root/'checkpoints/short-v3-large-fixedpos-freefall-hist32/hist32/step-100000.safetensors'
    module=WanTrainingModule(dit_config=training.model.dit,vae_config=training.model.vae,no_encoding=False,num_condition_frames=training.model.num_condition_frames,num_inference_steps=20,pipeline_type=training.model.pipe,pipeline_kwargs=training.model.pipe_kwargs)
    pipe=module.pipe; pipe.to(a.device); pipe.load_models_to_device(('dit','vae')); pipe.pre_encoded_(False); cfg.condition_latents=int(training.model.num_condition_frames)
    started=time.monotonic()
    for count,(window,label,source,j,u,rgb) in enumerate(jobs,1):
        key=(label,source['pair_id'],j)
        if key in done: continue
        input_path=input_videos/f"{source['pair_id']}_u{j:02d}.mp4"
        if not input_path.exists():
            frames=np.asarray(imageio.mimread(dataset/'videos'/source['video']),dtype=np.uint8); imageio.mimsave(input_path,recolour(frames,rgb,raw['render']['background_rgb']),fps=20,codec='libx264',ffmpeg_params=['-preset','ultrafast','-crf','0','-pix_fmt','yuv444p'])
        row=dict(source); row['video']=input_path.name; row['color_label']='red' if u<.5 else 'blue'
        old=cfg.short_masked_prefix; cfg.short_masked_prefix=cfg.prediction_start-window
        try: condition=load_condition(row,inputs,cfg,pipe.torch_dtype,a.device,True)
        finally: cfg.short_masked_prefix=old
        future=generate(pipe,condition,cfg,20,int(source['base_seed'])+17000000)
        measured=audit_track(future,row,cfg); x,y,rgbs=track_ball(future,cfg.radius); valid=np.isfinite(x)&np.isfinite(y)
        median=np.nanmedian(rgbs,axis=0) if np.isfinite(rgbs).any() else np.asarray([np.nan]*3); gap=np.abs(np.diff(y[np.isfinite(y)])) if valid.sum()>1 else np.asarray([])
        measured['E3_correct']=bool(measured['g_E3'] is not None and abs(measured['g_E3']-measured['gravity_true'])<cfg.gravity_accuracy_threshold); measured['E0_correct']=bool(measured['g_E0_strict'] is not None and abs(measured['g_E0_strict']-measured['gravity_true'])<cfg.gravity_accuracy_threshold)
        future_path=videos/f"{source['pair_id']}__{label.lower()}__u{j:02d}.mp4"; imageio.mimsave(future_path,future,fps=20,codec='libx264',ffmpeg_params=['-preset','ultrafast','-crf','0','-pix_fmt','yuv444p'])
        results.append({'task':'free_fall','model_seed':3407,'checkpoint_step':100000,'history_regime':label,'trajectory_id':source['base_pair_id'],'pair_id':source['pair_id'],'generation_seed':int(source['base_seed'])+17000000,
            'hue_index':j,'hue_u':u,'input_r':rgb[0],'input_g':rgb[1],'input_b':rgb[2],'true_gravity_band':source['gravity_interval'],'gravity_true':float(source['gravity']),
            'gravity_hat_e3':measured['g_E3'],'gravity_hat_e0':measured['g_E0_strict'],'fit_rmse':measured['y_rmse_E3'],'valid':measured['valid_track'],'invalid_reasons':'' if measured['valid_track'] else 'detected_frames_below_58',
            'outcome':'E3_pass' if measured['E3_correct'] else 'E3_fail','future_r':float(median[0]),'future_g':float(median[1]),'future_b':float(median[2]),
            'future_color_coordinate':float((median[2]-median[0])/(median[2]+median[0]+1e-12)),'detected_colour':measured['detected_colour'],'detector_coverage':float(valid.mean()),
            'max_missing_run':longest_false(valid),'max_adjacent_y_jump':float(gap.max()) if len(gap) else None,'output_video_path':str(future_path),'output_video_sha256':sha(future_path),**measured})
        done.add(key); write(out/'rows.json',results)
        if count%8==0: print(f'shard={a.shard} jobs={count}/{len(jobs)} elapsed={time.monotonic()-started:.1f}',flush=True)
    write(out/'complete.json',{'rows':len(results),'expected':len(jobs),'elapsed_seconds':time.monotonic()-started})


if __name__=='__main__':main()
