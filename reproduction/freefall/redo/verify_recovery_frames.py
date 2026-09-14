#!/usr/bin/env python3
"""Independently remeasure one saved video per rank and gravity band."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--project-root',type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.project_root/'scripts'))
    from layer_residual_replacement import measure
    from projectile_block_pca_fit import cfg_from
    path=args.project_root/'config/data-v3-fixedpos-freefall.yaml'
    cfg=cfg_from(path); data=yaml.safe_load(path.read_text())
    cfg.gravity_bands={'low':tuple(data['physics']['low_gravity_range']),'high':tuple(data['physics']['high_gravity_range'])}
    cfg.radius=int(data['render']['ball_radius_px'])
    cfg.gravity_accuracy_threshold=float(data['evaluation']['E3_threshold'])
    with (args.project_root/'data/raw-v3-fixedpos-freefall-eval320/eval/metadata.csv').open(newline='') as stream:
        metadata={(r['pair_id'],r['variant']):r for r in csv.DictReader(stream)}
    selected={}
    for shard in sorted((args.run_root/'recovery').glob('shard*')):
        for row in json.loads((shard/'outcomes.json').read_text()):
            selected.setdefault((row['rank'],row['true_band']),(shard,row))
    assert len(selected)==16
    checks=[]
    for key,(shard,row) in selected.items():
        pid=row['pair_id']
        path=shard/'frames'/row['rank']/f'{pid}.npz'
        frames=np.load(path)['frames']
        assert frames.shape==(64,128,128,3) and frames.dtype==np.uint8
        new=measure(frames,metadata[(pid,'conflict')],cfg)
        for field,value in new.items():
            if isinstance(value,float):
                assert np.isclose(value,row[field],rtol=0,atol=1e-12),(pid,field)
            else:
                assert value==row[field],(pid,field)
        checks.append({'rank':key[0],'gravity_band':key[1],'pair_id':pid,'passed':True})
    result={'verified_saved_videos':len(checks),'selection':'first available video per rank and gravity band',
            'frame_shape':[64,128,128,3],'checks':checks,'all_measurement_fields_match':True}
    (args.run_root/'summary/frame_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
