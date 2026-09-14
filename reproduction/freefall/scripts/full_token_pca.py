#!/usr/bin/env python3
"""Compute exact sample-space PCA for saved full condition-token residual deltas."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    shards = sorted(path for path in args.pca_root.glob("shard*") if path.is_dir())
    arrays = [np.load(path / "deltas.npy", mmap_mode="r") for path in shards]
    metadata = sum((json.loads((path / "metadata.json").read_text()) for path in shards), [])
    locations = [(array_index, index) for array_index, array in enumerate(arrays) for index in range(len(array))]
    order = np.argsort([item["pair_id"] for item in metadata])
    shape = tuple(arrays[0].shape[1:])
    samples = len(metadata)
    features = int(np.prod(shape))
    device = torch.device(args.device)
    matrix_dtype = torch.float32 if arrays[0].dtype == np.float32 else torch.float16
    matrix = torch.empty((samples, features), dtype=matrix_dtype, device=device)
    pair_ids = []
    for target, source in enumerate(order):
        array_index, index = locations[int(source)]
        matrix[target].copy_(torch.from_numpy(np.array(arrays[array_index][index], copy=True)).flatten().to(device))
        pair_ids.append(metadata[int(source)]["pair_id"])
    matrix_float = matrix.float()
    gram = (matrix_float @ matrix_float.T).cpu().numpy().astype(np.float64)
    del matrix_float, matrix
    eigenvalues, vectors = np.linalg.eigh(gram)
    order_pc = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order_pc], 0.0)
    vectors = vectors[:, order_pc]
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "sample_pca.npz", eigenvalues=eigenvalues, vectors=vectors, pair_ids=np.asarray(pair_ids), delta_shape=np.asarray(shape))
    payload = {"samples": samples, "delta_shape": list(shape), "features": features, "cumulative_energy": {str(k): float(eigenvalues[:k].sum() / eigenvalues.sum()) for k in (1, 2, 4, 8, 16, 32, 64, 96, 124, 125)}}
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
