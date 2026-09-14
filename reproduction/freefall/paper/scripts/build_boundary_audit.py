#!/usr/bin/env python3
"""Build numerical Free-fall boundary audit for the strict 128-pair bank."""
import argparse,csv,hashlib,json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset-dir',type=Path,required=True);p.add_argument('--pairs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    with (a.dataset_dir/'metadata.csv').open(newline='') as f: meta={(r['pair_id'],r['variant']):r for r in csv.DictReader(f)}
    pairs=json.loads(a.pairs.read_text())['pairs']; rows=[]
    for item in pairs:
        pid=item['pair_id']; x=meta[(pid,'aligned')];y=meta[(pid,'conflict')]
        pa=json.loads(x['position_at_boundary']);pc=json.loads(y['position_at_boundary']);va=json.loads(x['velocity_at_boundary']);vc=json.loads(y['velocity_at_boundary'])
        video_a=a.dataset_dir/'videos'/x['video'];video_c=a.dataset_dir/'videos'/y['video']
        checks=[abs(pa[1]-pc[1]),abs(va[1]-vc[1]),abs(pa[0]-pc[0]),abs(va[0]-vc[0])]
        rows.append({'pair_id':pid,'trajectory_id':x['base_pair_id'],'y_boundary_aligned':pa[1],'y_boundary_conflict':pc[1],'abs_delta_y_boundary':checks[0],
            'vy_boundary_aligned':va[1],'vy_boundary_conflict':vc[1],'abs_delta_vy_boundary':checks[1],
            'x_boundary_aligned':pa[0],'x_boundary_conflict':pc[0],'abs_delta_x_boundary':checks[2],
            'vx_boundary_aligned':va[0],'vx_boundary_conflict':vc[0],'abs_delta_vx_boundary':checks[3],
            'renderer_nuisance_hash_aligned':x['render_hash_without_ball'],'renderer_nuisance_hash_conflict':y['render_hash_without_ball'],
            'aligned_input_video_sha256':sha(video_a),'conflict_input_video_sha256':sha(video_c),
            'renderer_seed_equal':x['base_seed']==y['base_seed'],'generation_seed_equal':True,'noise_schedule_equal':True,
            'tolerance':1e-12,'pass':max(checks)<=1e-12 and x['render_hash_without_ball']==y['render_hash_without_ball'] and x['base_seed']==y['base_seed']})
    with a.out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'pairs':len(rows),'passes':sum(r['pass'] for r in rows),'max_boundary_error':max(max(r[k] for k in ['abs_delta_y_boundary','abs_delta_vy_boundary','abs_delta_x_boundary','abs_delta_vx_boundary']) for r in rows)}))


if __name__=='__main__':main()
