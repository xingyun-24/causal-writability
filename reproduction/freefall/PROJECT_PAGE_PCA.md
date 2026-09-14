# Current project-page PCA

This supplements the submitted train-only results; it does not replace the
paper controller or alter the paper. Both endpoint activations were recaptured
with the provided hist32/100K checkpoint, original renderer and codec, seed
`base_seed + 17000000`, 32 observed frames, zero-based block 1, and all 20 calls.

The first three uncentered feature-space PCs are reconstructed from the **64
fit differences only**, using the archived sample eigenvectors/eigenvalues.
Held-out endpoints never enter that reconstruction. The paper controller
continues to use rank 2; rank 3 is for visualization only.

The 128 projected endpoints come from 64 held-out pairs, with 32 points per
red/blue x low/high-gravity combination. Their 64 aligned-minus-conflict
projections agree with the archived PC1/2/3 scores to a maximum absolute error
of `5.54e-11`. The reconstructed first two PCs match the released dense basis
to relative error `2.54e-8`, within its float32 precision.

## Reproduce

Use the environment, checkpoint and VAE described in `README.md`. Materialize
`redo/results-grouped/train_only_pca.npz` through Git LFS before running. From
this directory, set `PYTHONPATH=src:lib/diffsynth` and run:

```bash
python scripts/capture_project_pca.py --root . --out runs/page-pca/fit \
  --split fit --device cuda:0
python scripts/capture_project_pca.py --root . --out runs/page-pca/heldout \
  --split heldout --device cuda:0
python scripts/build_project_pca.py --captures runs/page-pca \
  --pca redo/results-grouped/train_only_pca.npz \
  --basis-reference paper/pca_components.npy \
  --plotly /path/to/plotly-2.35.2.min.js \
  --out runs/page-pca/html --device cuda:0
```

Capture can be sharded with `--shard I --shards N`. Input videos are rendered
only for the frozen selected pairs from `redo/delivery_inputs/metadata.csv`.
Large raw activation arrays are local intermediates, not Git assets. The
site embeds and small data table live in `website/public/interactives/` at
repository root. Plotly is embedded offline and preserves its MIT header.

Raw-view color is newly decoded gravity measured with E3; difference-view
color is the paired aligned rollout's decoded gravity. Neither is input RGB.
This run verified natural activation extraction and projections, not a new
full controller-intervention reproduction or a new training run.
