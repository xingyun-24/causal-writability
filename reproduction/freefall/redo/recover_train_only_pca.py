#!/usr/bin/env python3
"""Held-out direct residual reconstruction with a frozen train-only PCA basis."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml


def write_json(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--pca-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--num-shards', type=int, default=2)
    parser.add_argument('--ranks', type=int, nargs='+', default=[1,2,4,8,16,32,64])
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    root = args.project_root
    sys.path.insert(0, str(root/'scripts'))
    from layer_residual_replacement import condition_token_count, generate, load_condition, measure
    from projectile_block_pca_fit import cfg_from
    from sshv2.wan.config import StandardTrainingConfig
    from sshv2.wan.trainer import WanTrainingModule
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    out = args.out/f'shard{args.shard}'
    out.mkdir(parents=True, exist_ok=True)
    packed_path = args.pca_dir/'train_only_pca.npz'
    p = np.load(packed_path)
    ids = [str(pid) for pid in p['pair_ids']]
    train, heldout = p['train_mask'],p['heldout_mask']
    assert train.sum()==heldout.sum()==64 and np.all(train^heldout)
    eigenvalues = p['eigenvalues']
    assert all(1<=rank<=len(eigenvalues) for rank in args.ranks)
    selected = [i for j,i in enumerate(np.flatnonzero(heldout)) if j%args.num_shards==args.shard]
    spec = {'method':'direct heldout delta projection onto train-only uncentered PCA basis',
            'pca_sha256':hashlib.sha256(packed_path.read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'fit_pairs':int(train.sum()),'total_heldout_pairs':int(heldout.sum()),
            'pair_ids':[ids[i] for i in selected], 'ranks':args.ranks,
            'block':1,'observed_window':32,'steps':20,'seed_offset':17000000,
            'num_shards':args.num_shards,'shard':args.shard,
            'energy_scaling':'sqrt(total training eigenvalue energy / retained training eigenvalue energy)',
            'full_delta':'unprojected heldout aligned-minus-conflict residual reference; not a PCA rank',
            'scope':'oracle residual reconstruction; scores use heldout aligned-minus-conflict residual, not regression-predicted scores'}
    if (out/'run_spec.json').exists():
        assert json.loads((out/'run_spec.json').read_text())==spec, 'Resume configuration mismatch'
    else:
        write_json(out/'run_spec.json',spec)
    results = json.loads((out/'outcomes.json').read_text()) if (out/'outcomes.json').exists() else []
    done = {(r['pair_id'],r['rank']) for r in results}
    rank_names = [str(k) for k in args.ranks]+['full_delta']
    if all((ids[i],rank) in done for i in selected for rank in rank_names):
        print('All assigned outputs already complete',flush=True)
        return
    sources = {}
    for shard in sorted((root/'results/freefall-hist32-step100000-block1-pca128').glob('shard*')):
        array = np.load(shard/'deltas.npy',mmap_mode='r')
        rows = json.loads((shard/'metadata.json').read_text())
        for i,row in enumerate(rows):
            assert row['pair_id'] not in sources
            sources[row['pair_id']] = (array,i,row)
    assert {sources[ids[i]][2]['base_seed'] for i in np.flatnonzero(train)}.isdisjoint(
        {sources[ids[i]][2]['base_seed'] for i in np.flatnonzero(heldout)})
    shape = tuple(int(v) for v in p['delta_shape'])
    matrix = torch.empty((int(train.sum()),int(np.prod(shape))),dtype=torch.float64,device=device)
    for j,i in enumerate(np.flatnonzero(train)):
        a,row,_ = sources[ids[i]]
        matrix[j].copy_(torch.from_numpy(np.array(a[row],copy=True)).flatten())
    weights = torch.tensor(p['train_vectors']/np.sqrt(eigenvalues)[None,:],dtype=torch.float64,device=device)
    # These directions contain only the 64 fitting samples. Never substitute an
    # all-sample projector; rank64 still projects heldout and is not full delta.
    directions = (weights.T @ matrix).float()
    del matrix,weights
    torch.cuda.empty_cache()
    scores = torch.tensor(p['scores'],dtype=torch.float32,device=device)
    scales = {str(rank):float(np.sqrt(eigenvalues.sum()/eigenvalues[:rank].sum())) for rank in args.ranks}
    scales['full_delta'] = 1.
    dataset = root/'data/raw-v3-fixedpos-freefall-eval320/eval'
    with (dataset/'metadata.csv').open(newline='') as stream:
        rows = {(r['pair_id'],r['variant']):r for r in csv.DictReader(stream)}
    baseline_rows = json.loads((root/'results/freefall-hist32-step100000-eval320-strict/rows_merged.json').read_text())
    baseline = {(r['pair_id'],r['condition']):r for r in baseline_rows}
    data_config = root/'config/data-v3-fixedpos-freefall.yaml'
    cfg = cfg_from(data_config)
    data = yaml.safe_load(data_config.read_text())
    cfg.gravity_bands = {'low':tuple(data['physics']['low_gravity_range']), 'high':tuple(data['physics']['high_gravity_range'])}
    cfg.radius = int(data['render']['ball_radius_px'])
    cfg.gravity_accuracy_threshold = float(data['evaluation']['E3_threshold'])
    training = StandardTrainingConfig.from_file(root/'config/Train-short-v3-large-fixedpos-freefall-hist32.yaml')
    training.model.dit.ckpt_file = root/'checkpoints/short-v3-large-fixedpos-freefall-hist32/hist32/step-100000.safetensors'
    module = WanTrainingModule(dit_config=training.model.dit,vae_config=training.model.vae,no_encoding=False,
                              num_condition_frames=training.model.num_condition_frames,num_inference_steps=20,
                              pipeline_type=training.model.pipe,pipeline_kwargs=training.model.pipe_kwargs)
    pipe = module.pipe
    pipe.to(args.device); pipe.load_models_to_device(('dit','vae')); pipe.pre_encoded_(False)
    cfg.condition_latents = int(training.model.num_condition_frames)
    print(f'Model loaded; {len(selected)} heldout pairs, {len(rank_names)} recovery settings each',flush=True)
    baseline_checks = json.loads((out/'baseline_checks.json').read_text()) if (out/'baseline_checks.json').exists() else {}
    started = time.monotonic()
    for pair_count,i in enumerate(selected,1):
        pid = ids[i]
        pending = [rank for rank in rank_names if (pid,rank) not in done]
        if not pending:
            continue
        conflict = rows[(pid,'conflict')]
        seed = int(conflict['base_seed'])+17000000
        old_prefix = cfg.short_masked_prefix
        cfg.short_masked_prefix = cfg.prediction_start-32
        try:
            condition = load_condition(conflict,dataset,cfg,pipe.torch_dtype,args.device,True)
        finally:
            cfg.short_masked_prefix = old_prefix
        block = pipe.dit.blocks[1]
        captures = []
        def capture_hook(_module,_inputs,output):
            count = condition_token_count(int(output.shape[1]),cfg)
            captures.append(output[0,:count].detach().clone())
        handle = block.register_forward_hook(capture_hook)
        try:
            baseline_frames = generate(pipe,condition,cfg,20,seed)
        finally:
            handle.remove()
        assert len(captures)==20
        conflict_residual = torch.stack(captures).float()
        del captures
        measured_baseline = measure(baseline_frames,conflict,cfg)
        old_conflict_g = float(baseline[(pid,'conflict')]['g_E3'])
        aligned_g = float(baseline[(pid,'aligned')]['g_E3'])
        current_conflict_g = measured_baseline['gravity_hat']
        baseline_checks[pid] = {'previous_conflict_g_E3':old_conflict_g,'current':measured_baseline,
                                'absolute_g_drift':abs(current_conflict_g-old_conflict_g) if current_conflict_g is not None else None}
        write_json(out/'baseline_checks.json',baseline_checks)
        for rank in pending:
            if rank=='full_delta':
                array,row,_ = sources[pid]
                delta = torch.tensor(np.array(array[row],copy=True),device=device)
            else:
                k = int(rank)
                delta = ((scores[i,:k] @ directions[:k])*scales[rank]).reshape(shape)
            call = 0
            def replacement_hook(_module,_inputs,output):
                nonlocal call
                assert call<20
                count = condition_token_count(int(output.shape[1]),cfg)
                edited = output.clone()
                edited[:,:count] = (conflict_residual[call]+delta[call]).unsqueeze(0).to(dtype=output.dtype)
                call += 1
                return edited
            handle = block.register_forward_hook(replacement_hook)
            try:
                frames = generate(pipe,condition,cfg,20,seed)
            finally:
                handle.remove()
            assert call==20
            measured = measure(frames,conflict,cfg)
            g = measured['gravity_hat']
            denominator = aligned_g-old_conflict_g
            recovery = (g-old_conflict_g)/denominator if g is not None and abs(denominator)>1e-12 else None
            frame_dir = out/'frames'/rank
            frame_dir.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(frame_dir/f'{pid}.npz',frames=frames)
            result = {'pair_id':pid,'rank':rank,'components':None if rank=='full_delta' else int(rank),
                      'block':1,'split':'heldout','energy_scale':scales[rank],**measured,
                      'baseline_aligned_g_E3':aligned_g,'baseline_conflict_g_E3':old_conflict_g,
                      'recovery_R':recovery,'recovery_success':bool(measured['valid'] and recovery is not None and .9<=recovery<=1.1),
                      'strict_E3_correct':bool(measured['valid'] and measured['E3_correct']),
                      'strict_E0_correct':bool(measured['valid'] and measured['E0_correct'])}
            results.append(result)
            write_json(out/'outcomes.json',results)
            done.add((pid,rank))
            print(f'pair={pair_count}/{len(selected)} id={pid} rank={rank} E3={measured["E3_correct"]} R={recovery} outputs={len(results)} elapsed_s={time.monotonic()-started:.1f}',flush=True)
            del delta,frames
        del conflict_residual,condition
    write_json(out/'complete.json',{'pairs':len(selected),'outputs':len(results),'elapsed_seconds':time.monotonic()-started})
    print('COMPLETE',flush=True)


if __name__=='__main__':
    main()
