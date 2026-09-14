#!/usr/bin/env python3
"""Export the first two feature-space components from the implicit train PCA."""
import argparse,json
from pathlib import Path
import numpy as np,torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,required=True);p.add_argument('--pca-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
    packed=np.load(a.pca_dir/'train_only_pca.npz');ids=np.asarray(packed['pair_ids'],dtype=str);train=packed['train_mask'].astype(bool);shape=tuple(int(v) for v in packed['delta_shape'])
    sources={}
    for shard in sorted((a.project_root/'results/freefall-hist32-step100000-block1-pca128').glob('shard*')):
        array=np.load(shard/'deltas.npy',mmap_mode='r');rows=json.loads((shard/'metadata.json').read_text())
        for i,row in enumerate(rows):sources[row['pair_id']]=(array,i)
    x=torch.empty((train.sum(),int(np.prod(shape))),dtype=torch.float64,device=a.device)
    for j,i in enumerate(np.flatnonzero(train)):
        array,row=sources[ids[i]];x[j].copy_(torch.from_numpy(np.array(array[row],copy=True)).flatten())
    weights=torch.tensor(packed['train_vectors'][:,:2]/np.sqrt(packed['eigenvalues'][:2])[None,:],dtype=torch.float64,device=a.device)
    basis_tensor=(weights.T@x).reshape((2,)+shape).float()
    flat=basis_tensor.double().reshape(2,-1);gram=(flat@flat.T).cpu().numpy();basis=basis_tensor.cpu().numpy()
    np.testing.assert_allclose(gram,np.eye(2),rtol=1e-5,atol=1e-5)
    np.save(a.out,basis)
    print({'shape':basis.shape,'dtype':str(basis.dtype),'gram':gram.tolist()})


if __name__=='__main__':main()
