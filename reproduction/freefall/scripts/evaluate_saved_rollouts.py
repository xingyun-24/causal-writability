#!/usr/bin/env python3
"""Re-evaluate saved RGB futures using the released E3/validity implementation."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from audit_v1_estimator import audit_track, make_cfg
from layer_residual_replacement import measure


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames-root',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--data-config',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    meta={(r['pair_id'],r['variant']):r for r in csv.DictReader(a.metadata.open())}
    cfg=make_cfg(a.data_config)
    cfg.gravity_bands=cfg.bands
    cfg.gravity_accuracy_threshold=float(cfg.evaluation.get('E3_threshold',0.002))
    rows=[]
    for path in sorted(a.frames_root.glob('shard*/frames/*.npz')):
        pid,condition=path.stem.split('__',1)
        frames=np.load(path)['frames']
        m=measure(frames,meta[pid,'conflict'],cfg)
        detailed=audit_track(frames,meta[pid,'conflict'],cfg)
        rows.append({'pair_id':pid,'condition':condition,'source':str(path),**detailed,
                     **m})
    if not rows: raise ValueError('No saved frame arrays found')
    summary={}
    for condition in sorted({r['condition'] for r in rows}):
        group=[r for r in rows if r['condition']==condition]
        summary[condition]={'n':len(group),'valid':sum(bool(r['valid']) for r in group),
                            'E3_correct':sum(bool(r['E3_correct']) for r in group)}
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
