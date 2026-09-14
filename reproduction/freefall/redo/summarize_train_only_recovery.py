#!/usr/bin/env python3
"""Recompute held-out recovery metrics and same-cohort legacy comparison."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def summarize(rows):
    valid_r = [r['recovery_R'] for r in rows if r['recovery_R'] is not None and r['valid']]
    return {'pairs':len(rows),'valid_pairs':sum(bool(r['valid']) for r in rows),
            'E3_correct_count':sum(bool(r['E3_correct']) for r in rows),
            'E3_accuracy':float(np.mean([r['E3_correct'] for r in rows])),
            'E0_correct_count':sum(bool(r['E0_correct']) for r in rows),
            'E0_accuracy':float(np.mean([r['E0_correct'] for r in rows])),
            'strict_E3_correct_count':sum(bool(r['E3_correct'] and r['valid']) for r in rows),
            'strict_E3_accuracy':float(np.mean([r['E3_correct'] and r['valid'] for r in rows])),
            'strict_E0_accuracy':float(np.mean([r['E0_correct'] and r['valid'] for r in rows])),
            'recovery_success_count':sum(bool(r['recovery_success']) for r in rows),
            'recovery_success_rate':float(np.mean([r['recovery_success'] for r in rows])),
            'median_recovery_R':float(np.median(valid_r)) if valid_r else None,
            'mean_recovery_R':float(np.mean(valid_r)) if valid_r else None,
            'gravity_MAE':float(np.mean([r['gravity_error'] for r in rows if r['gravity_error'] is not None]))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--project-root',type=Path,required=True)
    args = parser.parse_args()
    root = args.run_root
    split = json.loads((root/'decomposition/split_manifest.json').read_text())
    heldout = set(split['heldout_pair_ids'])
    assert len(heldout)==64 and heldout.isdisjoint(split['train_pair_ids'])
    ranks = ['1','2','4','8','16','32','64','full_delta']
    outcomes = []
    drift = []
    for path in sorted((root/'recovery').glob('shard*')):
        assert (path/'complete.json').exists(), f'Incomplete shard: {path}'
        outcomes.extend(json.loads((path/'outcomes.json').read_text()))
        checks = json.loads((path/'baseline_checks.json').read_text())
        drift.extend(row['absolute_g_drift'] for row in checks.values() if row['absolute_g_drift'] is not None)
    assert len(outcomes)==64*len(ranks)
    assert len({(r['pair_id'],r['rank']) for r in outcomes})==len(outcomes)
    baseline_path = args.project_root/'results/freefall-hist32-step100000-eval320-strict/rows_merged.json'
    baseline = {(r['pair_id'],r['condition']):r for r in json.loads(baseline_path.read_text())}
    pca = json.loads((root/'decomposition/summary.json').read_text())
    summary = []
    old_root = args.project_root/'results/freefall-hist32-step100000-block1-pca128-recovery'
    for rank in ranks:
        rows = [r for r in outcomes if r['rank']==rank]
        assert {r['pair_id'] for r in rows}==heldout
        # Recompute normalized recovery from saved measurements rather than
        # treating the worker's booleans as the authoritative aggregate.
        for r in rows:
            pid = r['pair_id']; g = r['gravity_hat']
            ga = baseline[(pid,'aligned')]['g_E3']; gc = baseline[(pid,'conflict')]['g_E3']
            value = (g-gc)/(ga-gc) if g is not None and abs(ga-gc)>1e-12 else None
            assert value is None or np.isclose(value,r['recovery_R'],atol=1e-12)
            r['recovery_R'] = value
            r['recovery_success'] = bool(r['valid'] and value is not None and .9<=value<=1.1)
        result = {'rank':rank,**summarize(rows),
                  'by_band':{band:summarize([r for r in rows if r['true_band']==band]) for band in ('low','high')},
                  'energy_scale':rows[0]['energy_scale']}
        assert all(result['by_band'][band]['pairs']==32 for band in ('low','high'))
        if rank!='full_delta':
            result['train_retained_energy_fraction'] = pca['rank_metrics'][rank]['train']['retained_energy_fraction']
            result['heldout_retained_energy_fraction'] = pca['rank_metrics'][rank]['heldout']['retained_energy_fraction']
        old_rank = '128' if rank=='full_delta' else rank
        old = [r for r in json.loads((old_root/f'rank{old_rank}/outcomes.json').read_text()) if r['pair_id'] in heldout]
        assert len(old)==64
        for r in old:
            pid = r['pair_id']; g = r['gravity_hat']
            ga = baseline[(pid,'aligned')]['g_E3']; gc = baseline[(pid,'conflict')]['g_E3']
            r['recovery_R'] = (g-gc)/(ga-gc) if g is not None and abs(ga-gc)>1e-12 else None
            r['recovery_success'] = bool(r['valid'] and r['recovery_R'] is not None and .9<=r['recovery_R']<=1.1)
        result['legacy_same_64_pairs'] = summarize(old)
        summary.append(result)
    payload = {'pca_fit_pairs':64,'heldout_pairs':64,'outcomes':len(outcomes),
               'method':'direct residual reconstruction; heldout actual delta projected into train-only basis',
               'energy_scale_source':'training eigenvalues only',
               'E3_definition':'abs(g_edit - g_true) < 0.002; strict E3 additionally requires valid track',
               'R_definition':'(g_edit - g_conflict) / (g_aligned - g_conflict); E3 g estimates',
               'recovery_success_definition':'valid track and 0.9 <= R <= 1.1',
               'invalid_handling':'all 64 pairs remain in accuracy denominators; invalid tracks fail strict metrics and recovery success',
               'baseline_recheck_pairs':len(drift),'max_baseline_conflict_g_drift':max(drift),
               'rank_results':summary,
               'limitations':['Oracle delta reconstruction, not prediction from colour/gravity regression.',
                              'The selected 128-pair cohort was screened by baseline E3 performance; evaluation is conditional on that cohort.',
                              'The heldout examples had been examined in earlier analyses, so these are not fresh untouched test examples.']}
    out = root/'summary'; out.mkdir(exist_ok=True)
    save(out/'recovery_summary.json',payload)
    save(out/'outcomes_merged.json',sorted(outcomes,key=lambda r:(r['pair_id'],r['rank'])))
    fields = ['rank','pairs','valid_pairs','E3_correct_count','E3_accuracy','E0_accuracy','strict_E3_accuracy','recovery_success_count','recovery_success_rate','median_recovery_R','gravity_MAE']
    with (out/'recovery_summary.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader(); writer.writerows({k:r[k] for k in fields} for r in summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    x = np.arange(len(ranks))
    for key,label in [('strict_E3_accuracy','E3 accuracy'),('recovery_success_rate','Recovery R in [0.9, 1.1]')]:
        axes[0].plot(x,[100*r[key] for r in summary],marker='o',label=label)
    axes[0].set(xticks=x,xticklabels=ranks[:-1]+['Full delta'],ylim=(-3,103),ylabel='Success (%)',xlabel='PCA components / reference',title='64 heldout pairs; basis fitted on 64 train pairs')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.25)
    axes[1].plot(x,[100*r['strict_E3_accuracy'] for r in summary],marker='o',label='Train-only PCA basis')
    axes[1].plot(x,[100*r['legacy_same_64_pairs']['strict_E3_accuracy'] for r in summary],marker='x',label='Legacy basis; same 64 evaluation pairs')
    axes[1].set(xticks=x,xticklabels=ranks[:-1]+['Full delta'],ylim=(-3,103),ylabel='E3 accuracy (%)',xlabel='PCA components / reference',title='Comparison on the same evaluation cohort')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=.25)
    fig.savefig(out/'recovery_comparison.png',dpi=160); plt.close(fig)
    print(json.dumps(payload,indent=2))


if __name__=='__main__':
    main()
