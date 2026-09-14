#!/usr/bin/env python3
"""Fit color*[1,g,sqrt(g)] separately within low/high gravity bands."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

def cv(v: str) -> float: return {"red": 1.0, "blue": -1.0}[v]
def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--pca-npz',type=Path,required=True); p.add_argument('--pair-manifest',type=Path,required=True); p.add_argument('--dataset-dir',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--components',type=int,default=2)
    a=p.parse_args(); z=np.load(a.pca_npz); ids=[str(x) for x in z['pair_ids']]; ev=z['eigenvalues'].astype(float); scores=z['vectors'][:,:a.components].astype(float)*np.sqrt(ev[:a.components])[None,:]
    pairs={x['pair_id']:x for x in json.loads(a.pair_manifest.read_text())['pairs']}; rows={(x['pair_id'],x['variant']):x for x in csv.DictReader((a.dataset_dir/'metadata.csv').open(newline=''))}
    g=np.asarray([float(pairs[i]['gravity_true']) for i in ids]); bands=np.asarray([pairs[i]['gravity_interval'] for i in ids]); dc=np.asarray([cv(rows[(i,'aligned')]['color_label'])-cv(rows[(i,'conflict')]['color_label']) for i in ids]); design=np.column_stack((dc,dc*g,dc*np.sqrt(g)))
    result={}
    holdout_ids=[]
    for band in ('low','high'):
        idx=np.flatnonzero(bands==band); hold=idx[::2]; train=np.ones(len(idx),bool); train[::2]=False; coef,*_=np.linalg.lstsq(design[idx][train],scores[idx][train],rcond=None); pred=design[idx]@coef; holdout_ids.extend(ids[i] for i in hold)
        result[band]={'pair_ids':[ids[i] for i in idx],'train_pair_ids':[ids[i] for i in idx if i not in set(hold)],'holdout_pair_ids':[ids[i] for i in hold],'coefficients_rows_feature_order':coef.tolist(),'train_joint_rmse':float(np.sqrt(np.mean((scores[idx][train]-pred[train])**2))),'holdout_joint_rmse':float(np.sqrt(np.mean((scores[idx][~train]-pred[~train])**2)))}
    payload={'basis':'f(color,g)=color*[1,g,sqrt(g)]*B; separate fit for low/high; no intercept','feature_order':['delta_color','delta_color*g','delta_color*sqrt(g)'],'components':a.components,'pair_ids':ids,'holdout_pair_ids':holdout_ids,'bands':result,'pca_energy_fraction':float(ev[:a.components].sum()/ev.sum())}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
