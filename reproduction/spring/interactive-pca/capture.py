"""Project matched natural activations onto the paper's frozen fit-only PCA basis."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from causal_writability.physics import candidate_definition, decode_latent, encode_condition, evaluate_future
from sshv2.diffsynth.trainers.wan_video_train import WanTrainingModule
from sshv2.interpretability.condition_residual_patching import ResidualPatchController, sample_final_latents
from sshv2.simulation.spring_shortcuts_v1 import dataclass_config_from_dict
from sshv2.utils.spring_configs import SpringTrainingConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--vae', type=Path, required=True)
    p.add_argument('--basis', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--block', type=int, default=6)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    a = p.parse_args()
    if not 0 <= a.shard < a.shards:
        p.error('Invalid shard')
    torch.set_num_threads(2)
    cfg = dataclass_config_from_dict(yaml.safe_load((a.project / 'configs/data.yaml').read_text()))
    train = SpringTrainingConfig.from_file(a.project / 'configs/spring_short.yaml')
    train.model.dit.ckpt_file = a.checkpoint.resolve()
    train.model.vae.ckpt_file = a.vae.resolve()
    pipe = WanTrainingModule(dit_config=train.model.dit, vae_config=train.model.vae,
                             no_encoding=False, num_condition_frames=17, num_inference_steps=20,
                             pipeline_type=train.model.pipe, pipeline_kwargs=train.model.pipe_kwargs).pipe
    pipe.to('cuda')
    pipe.load_models_to_device(('dit', 'vae'))
    pipe.pre_encoded_(False)
    basis = [torch.from_numpy(np.load(a.basis / f'step_{i:02d}.npy', allow_pickle=False)[:3].copy()).to('cuda')
             for i in range(20)]
    assert all(x.shape == (3, 1088, 1152) for x in basis)
    split = json.loads((a.project / 'data/strict_split.json').read_text())
    held = {r['trajectory_id'] for r in split if r['split'] == 'held_out'}
    rows = [r for r in json.loads((a.project / 'data/strict_bank.json').read_text()) if r['trajectory_id'] in held]
    assert len(rows) == 128
    rows = sorted(rows, key=lambda r:r['trajectory_id'])[a.shard::a.shards]
    a.out.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        dest = a.out / (row['trajectory_id'] + '.json')
        if dest.exists():
            continue
        candidate = candidate_definition(direction=row['direction'], candidate_index=row['candidate_index'], cfg=cfg)
        assert candidate['trajectory_id'] == row['trajectory_id']
        points = []
        captured = []
        for condition in ['aligned', 'conflict']:
            prefix = encode_condition(pipe, candidate[f'_{condition}_short_history_frames'], prediction_start=65, num_condition_frames=17)
            ctl = ResidualPatchController(20, 17, record_condition_layers=(a.block,))
            latent = sample_final_latents(pipe=pipe, condition_latents=prefix, num_frames=129,
                                          height=128, width=128, num_condition_frames=17,
                                          num_inference_steps=20, seed=row['generation_seed'], controller=ctl)
            activations = [ctl.bank.get('condition', a.block, i).squeeze(0) for i in range(20)]
            captured.append(activations)
            coordinates = sum(torch.einsum('nd,knd->k', x.to('cuda').float(), b)
                              for x, b in zip(activations, basis)).double().cpu().numpy()
            frames = decode_latent(pipe, latent, device='cuda', expected_frames=129)
            metric = evaluate_future(frames[65:], metadata=row, color_label_for_route=row[condition + '_color'], cfg=cfg)
            points.append({'pair_id':row['pair_id'], 'trajectory_id':row['trajectory_id'],
                           'condition':condition, 'split':'heldout', 'input_colour':row[condition + '_color'],
                           'true_band':row['true_band'], 'omega_true':row['omega_true'],
                           'omega_hat':metric['omega_pixel'] if metric['valid'] else None,
                           'valid':bool(metric['valid']), 'generation_seed':row['generation_seed'],
                           'pc1':float(coordinates[0]), 'pc2':float(coordinates[1]), 'pc3':float(coordinates[2])})
        delta = sum(torch.einsum('nd,knd->k', x.to('cuda').float() - y.to('cuda').float(), b)
                    for x, y, b in zip(captured[0], captured[1], basis)).double().cpu().numpy()
        raw_difference = np.array([points[0][k]-points[1][k] for k in ['pc1','pc2','pc3']])
        error = float(np.max(np.abs(delta - raw_difference)))
        assert error < .01, error
        payload = {'points':points, 'difference_pc':delta.tolist(), 'projection_subtraction_error':error}
        tmp = dest.with_suffix('.tmp')
        tmp.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
        tmp.replace(dest)
        print(f'shard {a.shard}: {index+1}/{len(rows)} {row["trajectory_id"]}', flush=True)


if __name__ == '__main__':
    main()
