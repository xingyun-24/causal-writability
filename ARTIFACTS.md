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
python -m pip install 'huggingface_hub==0.36.2'
hf auth login
python scripts/download_freefall.py
```

This restores the original relative paths expected by the Free Fall scripts,
downloads its hist32/100K checkpoint, and obtains the VAE from Wan's official
repository. To restore only the PCA arrays and saved frames, use
`--artifacts-only`. Already cached downloads are reused.

Spring's frozen Short controllers are available under `controllers/spring/`:
seed 3407 at block 3, seed 3408 at block 6, and seed 3409 at block 4. Each
bundle contains `model.json` and 20 per-call basis arrays (about 401 MB).
See [the Spring download and edit commands](reproduction/spring/README.md).
No full activation bank is needed to use these controllers.

Pendulum provides frozen split and coordinate tables plus code to reconstruct
the fit-only PCA basis and refit its coefficients. The original dense basis is
not distributed; this is a rebuild workflow, not a ready-to-run controller
download. See [the Pendulum instructions](reproduction/pendulum/project-page/README.md).
Do not combine a rebuilt basis with coefficients from the old basis.

The full 15-seed checkpoint history and raw residual banks are not part of the
selected release.
