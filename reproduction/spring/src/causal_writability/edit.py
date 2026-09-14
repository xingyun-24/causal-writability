"""Generate matched natural futures, residual edits and optional condition K/V edits."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from causal_writability.attention import ConditionQKVController
from causal_writability.physics import candidate_definition, decode_latent, encode_condition, evaluate_future
from sshv2.diffsynth.trainers.wan_video_train import WanTrainingModule
from sshv2.interpretability.condition_residual_patching import ResidualPatchController, sample_final_latents
from sshv2.interpretability.positive_condition_residual_all_layer_scan_128 import AuditedSingleSiteLiveAdditiveDirectionController
from sshv2.simulation.spring_shortcuts_v1 import dataclass_config_from_dict, write_video
from sshv2.utils.spring_configs import SpringTrainingConfig


def phase_features(row):
    theta = np.arctan2(-float(row['v_star']) / float(row['omega_true']), float(row['x_star']))
    sign = 1.0 if row['direction'] == 'true_fast_red_conflict' else -1.0
    return np.asarray([1, np.cos(theta), np.sin(theta), sign,
                       sign * np.cos(theta), sign * np.sin(theta)], dtype=np.float64)


def choose_rows(bank, split, direction, limit, trajectory=None):
    held = {r['trajectory_id'] for r in split if r['split'] == 'held_out'}
    rows = [r for r in bank if r['trajectory_id'] in held and r['direction'] == direction]
    if trajectory:
        rows = [r for r in rows if r['trajectory_id'] == trajectory]
    if not rows:
        raise ValueError('No held-out receiver matches the request')
    return rows if limit == 0 else rows[:limit]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('configs/spring_short.yaml'))
    p.add_argument('--data-config', type=Path, default=Path('configs/data.yaml'))
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--vae', type=Path, required=True)
    p.add_argument('--bank', type=Path, default=Path('data/strict_bank.json'))
    p.add_argument('--split', type=Path, default=Path('data/strict_split.json'))
    p.add_argument('--direction', choices=['fast', 'slow'], default='fast')
    p.add_argument('--history', choices=['short', 'long'], default='short')
    p.add_argument('--sites', default='3', help='Comma-separated zero-based blocks, or all')
    p.add_argument('--limit', type=int, default=1, help='0 runs all matching held-out receivers')
    p.add_argument('--trajectory')
    p.add_argument('--controller', type=Path, help='Frozen controller directory: model.json and basis/step_XX.npy')
    p.add_argument('--kv-block', type=int)
    p.add_argument('--components', choices=['k', 'v', 'kv'], default='kv')
    p.add_argument('--head', type=int)
    p.add_argument('--gain', type=float, default=1.0)
    p.add_argument('--fm-window', choices=['all', 'early', 'late'], default='all')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--cpu-threads', type=int, default=2)
    a = p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        p.error('Output directory must be new or empty')
    if a.kv_block is not None and a.controller is None:
        p.error('K/V control requires a donor-free frozen controller')
    if a.history != 'short' and a.controller is not None:
        p.error('Released phase controllers were fitted for Short only')
    torch.set_num_threads(a.cpu_threads)
    cfg = dataclass_config_from_dict(yaml.safe_load(a.data_config.read_text()))
    training = SpringTrainingConfig.from_file(a.config)
    training.model.dit.ckpt_file = a.checkpoint.resolve()
    training.model.vae.ckpt_file = a.vae.resolve()
    module = WanTrainingModule(dit_config=training.model.dit, vae_config=training.model.vae,
                              no_encoding=False, num_condition_frames=17, num_inference_steps=20,
                              pipeline_type=training.model.pipe, pipeline_kwargs=training.model.pipe_kwargs)
    pipe = module.pipe
    pipe.to(a.device)
    pipe.load_models_to_device(('dit', 'vae'))
    pipe.pre_encoded_(False)
    direction = 'true_fast_red_conflict' if a.direction == 'fast' else 'true_slow_blue_conflict'
    rows = choose_rows(json.loads(a.bank.read_text()), json.loads(a.split.read_text()),
                       direction, a.limit, a.trajectory)
    sites = list(range(-1, 30)) if a.sites == 'all' else [int(s) for s in a.sites.split(',')]
    if not sites or any(s < -1 or s >= 30 for s in sites):
        p.error('Sites must be between -1 and 29')
    model = json.loads((a.controller / 'model.json').read_text()) if a.controller else None
    if model and sites != [int(model['site'])]:
        p.error('Frozen controller must be used at its fitted site')
    if a.kv_block is not None and not 0 <= a.kv_block < 30:
        p.error('K/V block must be between 0 and 29')
    if a.head is not None and not 0 <= a.head < training.model.dit.num_heads:
        p.error('Head index out of range')
    a.out.mkdir(parents=True, exist_ok=True)
    result = []
    for row in rows:
        candidate = candidate_definition(direction=direction, candidate_index=row['candidate_index'], cfg=cfg)
        if candidate['trajectory_id'] != row['trajectory_id']:
            raise ValueError('The data configuration does not reproduce the frozen receiver')
        folder = a.out / row['trajectory_id']
        folder.mkdir()

        def condition(color):
            suffix = '_short_history_frames' if a.history == 'short' else '_frames'
            return encode_condition(pipe, candidate['_' + color + suffix],
                                    prediction_start=65, num_condition_frames=17)

        aligned, conflict = condition('aligned'), condition('conflict')

        def sample(c, controller, attention=None):
            if attention:
                attention.attach(pipe.dit.blocks[a.kv_block])
            try:
                value = sample_final_latents(pipe=pipe, condition_latents=c, num_frames=129,
                                             height=128, width=128, num_condition_frames=17,
                                             num_inference_steps=20, seed=row['generation_seed'], controller=controller)
            finally:
                if attention:
                    attention.remove()
            if attention:
                attention.finish()
            return value

        def measure(latent, name, color):
            frames = decode_latent(pipe, latent, device=a.device, expected_frames=129)
            metric = evaluate_future(frames[65:], metadata=row, color_label_for_route=color, cfg=cfg)
            write_video(folder / (name + '.mp4'), frames[65:], 20)
            return {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in metric.items()}

        # The aligned activation is an oracle reference; phase prediction below never reads it.
        donor = ResidualPatchController(20, 17, record_condition_layers=tuple(sites))
        reference = ResidualPatchController(20, 17, record_condition_layers=tuple(sites) if model else ())
        natural_a = measure(sample(aligned, donor), 'aligned', row['aligned_color'])
        natural_c = measure(sample(conflict, reference), 'conflict', row['conflict_color'])
        metrics = {'aligned': natural_a, 'conflict': natural_c}
        for site in sites:
            patch = ResidualPatchController(20, 17, inject_bank=donor.bank, inject_layer=site)
            metrics[f'paired_B{site}'] = measure(sample(conflict, patch), f'paired_B{site}', row['conflict_color'])
        if model:
            z = phase_features(row) @ np.asarray(model['ols_b'], dtype=np.float64)
            tensors = [torch.from_numpy(np.load(a.controller / 'basis' / f'step_{i:02d}.npy', allow_pickle=False)).to(a.device)
                       for i in range(20)]
            coords = torch.as_tensor(z, dtype=torch.float32, device=a.device)
            delta = [torch.einsum('k,knd->nd', coords, tensor) for tensor in tensors]
            patch = AuditedSingleSiteLiveAdditiveDirectionController(
                expected_steps=20, num_condition_frames=17, direction_by_step=delta,
                conflict_reference_by_step=[reference.bank.get('condition', sites[0], i).squeeze(0) for i in range(20)],
                inject_site=sites[0], arm='phase')
            recorder = ConditionQKVController(20, 17, 33) if a.kv_block is not None else None
            metrics['phase'] = measure(sample(conflict, patch, recorder), 'phase', row['conflict_color'])
            if recorder:
                active = None if a.fm_window == 'all' else tuple(range(10) if a.fm_window == 'early' else range(10, 20))
                attention = ConditionQKVController(20, 17, 33, mode='inject', components=tuple(a.components),
                                                   source=recorder.bank, active_steps=active,
                                                   head_indices=None if a.head is None else (a.head,),
                                                   num_heads=9, blend_alpha=a.gain)
                metrics['kv'] = measure(sample(conflict, ResidualPatchController(20, 17), attention), 'kv', row['conflict_color'])
        wa, wc = natural_a['omega_pixel'], natural_c['omega_pixel']
        for name, metric in metrics.items():
            value = metric['omega_pixel']
            metric['R'] = (value - wc) / (wa - wc) if None not in (wa, wc, value) and abs(wa - wc) > 1e-12 else None
        record = {'trajectory_id': row['trajectory_id'], 'generation_seed': row['generation_seed'], 'metrics': metrics}
        (folder / 'metrics.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
        result.append(record)
        print(row['trajectory_id'], {k: (v['omega_pixel'], v['R']) for k, v in metrics.items()}, flush=True)
    (a.out / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
