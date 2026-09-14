# Model and Data Downloads

Model weights are listed in [CHECKPOINTS.md](CHECKPOINTS.md). The Hugging Face
repository is private during release preparation; use an account with access.

Free Fall also needs its frozen PCA arrays and archived decoded RGB frames for
controller inference and saved-video re-evaluation. These 646 files (about
741 MB when restored, including the basis at two required paths) are hosted
under `artifacts/` in the same model repository. They are not
new fits or regenerated experimental results.

From this repository root:

```bash
python -m pip install 'huggingface_hub>=1.0,<2'
hf auth login
python scripts/download_freefall.py
```

This restores the original relative paths expected by the Free Fall scripts,
downloads its hist32/100K checkpoint, and obtains the VAE from Wan's official
repository. To restore only the PCA arrays and saved frames, use
`--artifacts-only`. Already cached downloads are reused.

Spring and Pendulum model download commands remain in their task READMEs.
Spring controller bases and the Pendulum full held-out Top-4 basis must still
be supplied or regenerated following those READMEs; uploading core model
weights does not replace this step. The full 15-seed checkpoint history is not
part of the selected release.
