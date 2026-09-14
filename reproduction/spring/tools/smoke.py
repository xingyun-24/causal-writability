"""Small GPU exercise of the shipped pipeline, not a population-level paper rerun."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--vae', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('runs/smoke'))
    p.add_argument('--device', default='cuda')
    p.add_argument('--skip-training', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    checkpoint, vae = a.checkpoint.resolve(), a.vae.resolve()
    data_config = yaml.safe_load(Path('configs/data.yaml').read_text())
    data_config.update(train_base_seeds=1, eval_base_seeds=1, sanity_base_seeds=1,
                       tiny_train_videos=2, pilot_train_videos=2)
    config = a.out / 'data.yaml'
    config.write_text(yaml.safe_dump(data_config, sort_keys=False))
    data = a.out / 'data'
    training = yaml.safe_load(Path('configs/spring_short.yaml').read_text())
    training['model']['vae']['ckpt_file'] = str(vae)
    training['loader'].update(num_training_steps=1, batch_size=1, num_workers=0)
    training['data'].update(dataset=str(data / 'latents/train_short'),
                            labels=str(data / 'latents/train_short/metadata.csv'))
    training['log'].update(ckpt_dir=str(a.out / 'checkpoints'), out_dir=str(a.out / 'train'), save_at=[1])
    train_config = a.out / 'train.yaml'
    train_config.write_text(yaml.safe_dump(training, sort_keys=False))
    env = dict(os.environ, WANDB_MODE='disabled', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    env['PATH'] = str(Path(sys.executable).parent) + ':/usr/local/bin:/usr/bin:/bin'
    env.pop('PYTHONPATH', None)
    results = []

    def run(stage, arguments):
        command = [sys.executable, '-m', 'causal_writability.cli', *map(str, arguments)]
        with (a.out / f'{stage}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
        results.append(stage)
        print(stage, 'passed', flush=True)

    run('data', ['data', '--config', config, '--root', data, '--split', 'all'])
    run('encode', ['encode', '--source', data / 'videos/train', '--out-root', data / 'latents',
                   '--vae', vae, '--data-config', config, '--batch-size', 1, '--device', a.device])
    if not a.skip_training:
        run('train', ['train', '--config', train_config, '--steps', 1, '--no-wandb'])
    run('generate', ['generate', '--config', train_config, '--checkpoint', checkpoint,
                     '--dataset', data / 'videos/eval', '--data-config', config, '--history', 'short',
                     '--out', a.out / 'natural', '--limit', 1, '--device', a.device])
    run('evaluate', ['evaluate', '--dataset', data / 'videos/eval', '--predictions', a.out / 'natural',
                     '--data-config', config, '--history', 'short', '--out', a.out / 'evaluation', '--limit', 1])
    run('paired', ['edit', '--checkpoint', checkpoint, '--vae', vae, '--sites', 3,
                   '--out', a.out / 'paired', '--device', a.device])
    (a.out / 'status.json').write_text(json.dumps({'completed_stages': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
