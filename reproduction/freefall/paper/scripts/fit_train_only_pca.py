#!/usr/bin/env python3
"""Train-only uncentered residual PCA; explicit held-out projection and regression.

The feature-space basis is X_train.T @ train_vectors / sqrt(eigenvalues).
Intentionally omit the legacy `vectors` field: old consumers assume all samples
are basis-fitting samples and must not silently consume this new artifact.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fit_basis(x):
    gram = (x @ x.T).cpu().numpy()
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    keep = values > values[0] * 1e-12
    values, vectors = values[keep], vectors[:, keep]
    pivots = np.argmax(np.abs(vectors), axis=0)
    vectors *= np.where(vectors[pivots, np.arange(len(values))] < 0, -1, 1)
    weights = torch.as_tensor(vectors / np.sqrt(values)[None, :], device=x.device)
    return values, vectors, weights


def project(x, train, weights):
    return ((x @ train.T) @ weights).cpu().numpy()


def self_test():
    rng = np.random.default_rng(27)
    x = torch.tensor(rng.normal(size=(7, 19)), dtype=torch.float64)
    h = torch.tensor(rng.normal(size=(4, 19)), dtype=torch.float64)
    e, u, w = fit_basis(x)
    basis = x.T @ w
    np.testing.assert_allclose((basis.T @ basis).numpy(), np.eye(7), atol=1e-12)
    np.testing.assert_allclose(project(h, x, w), (h @ basis).numpy(), atol=1e-12)
    np.testing.assert_allclose(project(x, x, w), u * np.sqrt(e), atol=1e-12)
    _, s, vh = np.linalg.svd(x.numpy(), full_matrices=False)
    np.testing.assert_allclose(e, s*s, atol=1e-12)
    np.testing.assert_allclose((basis @ basis.T).numpy(), vh.T @ vh, atol=1e-12)
    before = e.copy(), u.copy()
    h.mul_(100).add_(23)
    project(h, x, w)
    after_e, after_u, _ = fit_basis(x)
    np.testing.assert_array_equal(before[0], after_e)
    np.testing.assert_array_equal(before[1], after_u)
    print('PASS: SVD equivalence, orthonormal basis, projections, heldout perturbation isolation', flush=True)


def quality(actual, predicted, mean):
    mse = np.mean((actual-predicted)**2, axis=0)
    baseline = np.mean((actual-mean)**2, axis=0)
    return {'joint_rmse': float(np.sqrt(mse.mean())),
            'component_rmse': np.sqrt(mse).tolist(),
            'component_R2': [float(1-a/b) if b > 0 else None for a,b in zip(mse,baseline)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pca-root', type=Path)
    parser.add_argument('--dataset-dir', type=Path)
    parser.add_argument('--pair-manifest', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--device', default='cuda:3')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    args.out.mkdir(parents=True, exist_ok=False)
    arrays, entries, source_info = [], [], []
    for shard in sorted(args.pca_root.glob('shard*')):
        if not shard.is_dir():
            continue
        path = shard/'metadata.json'
        rows = json.loads(path.read_text())
        array = np.load(shard/'deltas.npy', mmap_mode='r')
        assert len(rows) == len(array)
        source_info.append({'path': str(shard), 'shape': list(array.shape),
                            'metadata_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        entries.extend((row, len(arrays), i) for i,row in enumerate(rows))
        arrays.append(array)
    entries.sort(key=lambda item: item[0]['pair_id'])
    ids = np.asarray([item[0]['pair_id'] for item in entries])
    bands = np.asarray([item[0]['gravity_interval'] for item in entries])
    assert len(ids) == len(set(ids)) == 128
    holdout = np.zeros(len(ids), dtype=bool)
    for band in ('low','high'):
        indices = np.flatnonzero(bands == band)
        assert len(indices) == 64
        holdout[indices[::2]] = True
    original_holdout = holdout.copy()
    seeds = np.asarray([entry[0]['base_seed'] for entry in entries])
    adjustments = []
    # Keep shared-seed pairs together. Move the heldout sibling into training,
    # balancing within its gravity band using the closest-g singleton train pair.
    for seed in sorted(set(seeds)):
        siblings = np.flatnonzero(seeds == seed)
        if not (np.any(holdout[siblings]) and np.any(~holdout[siblings])):
            continue
        for index in siblings[holdout[siblings]]:
            candidates = [i for i in range(len(ids)) if not holdout[i]
                          and bands[i] == bands[index] and np.sum(seeds == seeds[i]) == 1]
            swap = min(candidates, key=lambda i: (abs(entries[i][0]['gravity_true']-entries[index][0]['gravity_true']), str(ids[i])))
            holdout[index] = False
            holdout[swap] = True
            adjustments.append({'to_train':str(ids[index]),'to_heldout':str(ids[swap]),
                                'reason':f'keep shared base_seed {seed} together; closest-g singleton swap within band'})
    train = ~holdout
    # Match the established regression split, and prevent sibling variants of a
    # shared simulated trajectory crossing the split.
    train_seeds = {entries[i][0]['base_seed'] for i in np.flatnonzero(train)}
    heldout_seeds = {entries[i][0]['base_seed'] for i in np.flatnonzero(holdout)}
    assert train_seeds.isdisjoint(heldout_seeds), 'base_seed leakage across splits'
    split = {'train_pair_ids': ids[train].tolist(), 'heldout_pair_ids': ids[holdout].tolist(),
             'train_pairs': int(train.sum()), 'heldout_pairs': int(holdout.sum()),
             'rule': 'sort pair_id within low/high; even positions heldout, odd positions train; repair shared-seed crossings by same-band closest-g singleton swaps',
             'adjustments_from_legacy_split':adjustments,
             'original_heldout_pair_ids':ids[original_holdout].tolist(),
             'base_seeds_disjoint': True}
    write_json(args.out/'split_manifest.json', split)
    print('Frozen split: train=64, heldout=64; base seeds disjoint', flush=True)
    shape = arrays[0].shape[1:]
    def load(indices):
        result = torch.empty((len(indices), int(np.prod(shape))), dtype=torch.float64, device=args.device)
        for target, index in enumerate(indices):
            _, a, row = entries[int(index)]
            result[target].copy_(torch.from_numpy(np.array(arrays[a][row], copy=True)).flatten())
        return result
    x = load(np.flatnonzero(train))
    eigenvalues, vectors, weights = fit_basis(x)
    scores = np.empty((len(ids),len(eigenvalues)), dtype=np.float64)
    scores[train] = project(x,x,weights)
    norms = np.zeros(len(ids))
    norms[train] = torch.einsum('ij,ij->i',x,x).cpu().numpy()
    np.testing.assert_allclose(scores[train],vectors*np.sqrt(eigenvalues)[None,:],rtol=1e-6,atol=1e-7)
    np.testing.assert_allclose(np.sum(scores[train]**2), norms[train].sum(), rtol=1e-8)
    print('Basis fixed from training data. Loading heldout only for projection.',flush=True)
    heldout_indices = np.flatnonzero(holdout)
    for start in range(0,len(heldout_indices),16):
        indices = heldout_indices[start:start+16]
        h = load(indices)
        scores[indices] = project(h,x,weights)
        norms[indices] = torch.einsum('ij,ij->i',h,h).cpu().numpy()
        del h
        print(f'Projected heldout {start+len(indices)}/64',flush=True)
    assert np.all(np.sum(scores**2,axis=1) <= norms*(1+1e-7))
    np.savez_compressed(args.out/'train_only_pca.npz',schema_version=np.asarray(2),
                        eigenvalues=eigenvalues,train_vectors=vectors,scores=scores,
                        pair_ids=ids,train_mask=train,heldout_mask=holdout,delta_shape=np.asarray(shape),
                        squared_norms=norms,centered=np.asarray(False))
    ranks = [1,2,3,4,8,16,32,64]
    metrics = {}
    for rank in ranks:
        if rank > len(eigenvalues):
            continue
        metrics[str(rank)] = {}
        for name,mask in [('train',train),('heldout',holdout)]:
            retained = np.sum(scores[mask,:rank]**2)
            total = norms[mask].sum()
            metrics[str(rank)][name] = {'retained_energy_fraction': float(retained/total),
                                      'relative_reconstruction_error': float(np.sqrt(max(0,1-retained/total)))}
    pairs = {p['pair_id']:p for p in json.loads(args.pair_manifest.read_text())['pairs']}
    with (args.dataset_dir/'metadata.csv').open(newline='') as stream:
        metadata = {(r['pair_id'],r['variant']):r for r in csv.DictReader(stream)}
    g = np.asarray([float(pairs[pid]['gravity_true']) for pid in ids])
    colors = {'red':1.,'blue':-1.}
    dc = np.asarray([colors[metadata[(pid,'aligned')]['color_label']]-colors[metadata[(pid,'conflict')]['color_label']] for pid in ids])
    design = np.column_stack((dc,dc*g,dc*np.sqrt(g)))
    regression = {}
    for rank in (2,3,4):
        actual = scores[:,:rank]
        coefficients = np.linalg.lstsq(design[train],actual[train],rcond=None)[0]
        predicted = design @ coefficients
        payload = {'basis':'color * [1,g,sqrt(g)] difference; no intercept',
                   'pca_artifact':'train_only_pca.npz','components':rank,
                   **split,'coefficients_rows_feature_order':coefficients.tolist(),
                   'train_quality':quality(actual[train],predicted[train],actual[train].mean(0)),
                   'holdout_quality':quality(actual[holdout],predicted[holdout],actual[train].mean(0)),
                   'pair_ids':ids.tolist(),'actual_scores':actual.tolist(),'fitted_scores':predicted.tolist(),
                   'train_pca_energy_fraction':metrics[str(rank)]['train']['retained_energy_fraction'],
                   'heldout_pca_energy_fraction':metrics[str(rank)]['heldout']['retained_energy_fraction']}
        write_json(args.out/f'pca{rank}_colour_1_g_sqrtg_fit.json',payload)
        regression[str(rank)] = {key:payload[key] for key in ('train_quality','holdout_quality')}
    summary = {'method':'uncentered PCA (SVD) of aligned-minus-conflict residual; training-only basis',
               'samples':len(ids),'fit_pairs':int(train.sum()),'heldout_pairs':int(holdout.sum()),
               'split_counts':{name:{band:int(np.sum(mask & (bands==band))) for band in ('low','high')} for name,mask in [('train',train),('heldout',holdout)]},
               'delta_shape':list(shape),'numerical_rank':len(eigenvalues),'dtype':'float64',
               'sources':source_info,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'rank_metrics':metrics,'regression':regression,
               'rank_selection':'fixed diagnostic ranks; no selection or tuning using heldout',
               'limitations':['Existing 128 pairs were selected by baseline aligned/conflict E3 performance; heldout is within this selected cohort.',
                              'Prior analyses have already examined these heldout pairs; this is a corrected split evaluation, not a fresh untouched test set.',
                              'No video generation or intervention rollout is performed by this script.'],
               'validation':{'unique_pairs':True,'base_seeds_disjoint':True,'train_score_identity':True,'projection_energy_bound':True}}
    write_json(args.out/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    for name,mask,marker in [('Train',train,'o'),('Held-out',holdout,'x')]:
        for band,color in [('low','#2979b8'),('high','#dc652a')]:
            selected = mask & (bands==band)
            axes[0].scatter(scores[selected,0],scores[selected,1],c=color,marker=marker,s=28,label=f'{name} / {band}')
    axes[0].set(xlabel='PC1 score',ylabel='PC2 score',title='Basis fitted on 64 training pairs')
    axes[0].legend(fontsize=8)
    for name in ('train','heldout'):
        axes[1].plot([int(k) for k in metrics],[metrics[k][name]['retained_energy_fraction']*100 for k in metrics],marker='o',label=name)
    axes[1].set(xlabel='Components',ylabel='Retained residual energy (%)',xscale='log',title='Projection onto the fixed training basis')
    axes[1].legend(); axes[1].grid(alpha=.25)
    fig.savefig(args.out/'train_heldout_pca.png',dpi=160)
    plt.close(fig)
    print(json.dumps({'fit_pairs':64,'heldout_pairs':64,'rank2':metrics['2'],'regression2':regression['2']},indent=2),flush=True)


if __name__ == '__main__':
    main()
