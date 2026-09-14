#!/usr/bin/env python3
"""Fit direction-specific [1, g_target] models to train-only PCA scores."""
import argparse
import json
from pathlib import Path

import numpy as np


def r2(actual, predicted):
    error = np.sum((actual-predicted)**2,axis=0)
    total = np.sum((actual-actual.mean(axis=0))**2,axis=0)
    return [float(1-e/t) if t>0 else None for e,t in zip(error,total)]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pca-dir',type=Path,required=True)
    p.add_argument('--pair-manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    packed=np.load(a.pca_dir/'train_only_pca.npz')
    ids=np.asarray(packed['pair_ids'],dtype=str)
    scores=packed['scores'][:,:2].astype(np.float64)
    train=packed['train_mask'].astype(bool); heldout=packed['heldout_mask'].astype(bool)
    split=json.loads((a.pca_dir/'split_manifest.json').read_text())
    assert set(ids[train])==set(split['train_pair_ids'])
    assert set(ids[heldout])==set(split['heldout_pair_ids'])
    pairs={row['pair_id']:row for row in json.loads(a.pair_manifest.read_text())['pairs']}
    gravity=np.asarray([float(pairs[pid]['gravity_true']) for pid in ids])
    bands=np.asarray([pairs[pid]['gravity_interval'] for pid in ids])
    models={}; predictions=[]
    for band in ('low','high'):
        fit=train&(bands==band); test=heldout&(bands==band)
        assert fit.sum()==test.sum()==32
        x=np.column_stack((np.ones(fit.sum()),gravity[fit]))
        coef=np.linalg.lstsq(x,scores[fit],rcond=None)[0]
        train_pred=x@coef
        heldout_pred=np.column_stack((np.ones(test.sum()),gravity[test]))@coef
        models[band]={'train_pair_ids':ids[fit].tolist(),'heldout_pair_ids':ids[test].tolist(),
                      'coefficients_rows_[1_g_target]':coef.tolist(),
                      'train_condition_number':float(np.linalg.cond(x)),
                      'train_coordinate_R2':r2(scores[fit],train_pred),
                      'heldout_coordinate_R2':r2(scores[test],heldout_pred),
                      'heldout_joint_R2':float(1-np.sum((scores[test]-heldout_pred)**2)/np.sum((scores[test]-scores[test].mean(0))**2)),
                      'heldout_rmse':float(np.sqrt(np.mean((scores[test]-heldout_pred)**2)))}
    labels=np.where(train,'train','heldout')
    for i,pid in enumerate(ids):
        band=bands[i]; coef=np.asarray(models[band]['coefficients_rows_[1_g_target]'])
        pred=np.asarray([1.,gravity[i]])@coef
        for c in range(2):
            predictions.append({'receiver_id':pid,'pair_id':pid,'split':labels[i],'direction':band,
                                'target_direction':band,'g_target':float(gravity[i]),'delta_gravity':None,
                                'y_boundary':0.65,'vy_boundary_scaled':0.0,'coordinate_rank':2,
                                'coordinate':c+1,'observed_score':float(scores[i,c]),'predicted_score':float(pred[c]),
                                'residual':float(scores[i,c]-pred[c]),'absolute_error':float(abs(scores[i,c]-pred[c]))})
    energy_scale=float(np.sqrt(packed['eigenvalues'].sum()/packed['eigenvalues'][:2].sum()))
    payload={'controller':'direction_specific_absolute_target_[1,g_target]','basis_source':'train_only_pca',
             'pca_rank':2,'fit_pairs':64,'heldout_pairs':64,'direction_specific':True,
             'coordinate_difference_model':False,'feature_order':['1','g_target'],
             'input_contract':'target direction plus absolute target gravity only; no generated gravity or heldout activation input',
             'global_scale':energy_scale,'global_scale_selection':'fixed from training PCA retained energy only',
             'pair_ids':ids.tolist(),'gravity_target':gravity.tolist(),'direction':bands.tolist(),
             'split_labels':labels.tolist(),'models':models,'coordinate_predictions':predictions}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({b:models[b] for b in ('low','high')},indent=2))


if __name__=='__main__':main()
