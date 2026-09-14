"""Replot three numerical summaries from the shipped tables (not the paper layout)."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=Path('results'))
    p.add_argument('--out', type=Path, default=Path('runs/replot'))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout='constrained')
    for history, style in [('short', '--'), ('long', '-')]:
        rows = json.loads((a.results / f'behavior/{history}-3407.json').read_text())
        for band, color in [('slow', '#d86659'), ('fast', '#507cb8')]:
            cues = sorted({r['color_label'] for r in rows})
            rates = []
            for cue in cues:
                group = [r for r in rows if r['valid'] and r['true_band'] == band and r['color_label'] == cue]
                rates.append(np.mean([r['route_label'] == 'physics_frequency' for r in group]))
            axes[0].plot(np.linspace(0, 1, len(cues)), rates, style, color=color, label=f'{history}, {band}')
    axes[0].set(xlabel='Observed cue', ylabel='Physics-follow rate', title='Cue-specific behavior')
    axes[0].legend(fontsize=7)
    with (a.results / 'training_behavior.csv').open() as f:
        rows = list(csv.DictReader(f))
    steps = sorted({int(r['step_k']) for r in rows})
    means = [np.mean([float(r['P']) for r in rows if int(r['step_k']) == step]) for step in steps]
    axes[1].plot(steps, means, 'o-', color='#333333')
    axes[1].set(xlabel='Training steps (K)', ylabel='Full-grid physics-follow rate', title='15-seed mean')
    with (a.results / 'figure5_single_head_gain_sweep.csv').open() as f:
        rows = list(csv.DictReader(f))
    axes[2].plot([float(r['gain']) for r in rows], [float(r['P']) for r in rows], 'o-', color='#36978c')
    axes[2].set(xlabel='V-head gain', ylabel='Physics-follow rate', title='48 strict failures')
    for ax in axes:
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_ylim(0, 1.02)
    fig.savefig(a.out / 'summaries.pdf')
    fig.savefig(a.out / 'summaries.png', dpi=150)


if __name__ == '__main__':
    main()
