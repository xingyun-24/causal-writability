# Pendulum

## Contents

- `engine/`: Pendulum runtime with the new Stage 3/4 helpers and final Short/Long configs already integrated.
- `project-page/data/`: frozen 64-fit/64-held-out lists and controller reference tables.
- `engine/scripts/sshv2/fit_pendulum_harmonic_coordinates.py`: final direction-specific Top-4 phase law.
- `engine/scripts/sshv2/run_pendulum_top4_controller.py`: held-out controller generation at B12.
- `engine/scripts/sshv2/reevaluate_pendulum_decoded_recovery.py`: current video-evaluation contract.
- `paper/`: corrected result tables and the exact Figure 15-18 assets.

The original Rank-1 Stage 4 program supplies helper functions but is not the
final paper controller. Follow [the detailed commands](project-page/README.md)
for residual/PCA generation and the Top-4 workflow.

## Setup and Smoke Test

From this directory, use a separate environment. The integration repository
includes the matching DiffSynth dependency under `../freefall/lib/diffsynth`
(the dependency revision is recorded in `../sources.json`). This does not use
the separate Spring vendored runtime.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r engine/requirements.txt
pip install --no-deps -e engine
export PYTHONPATH="$PWD/engine/src:$PWD/../freefall/lib/diffsynth:$PWD/engine/scripts/sshv2"

hf download xingyun-24/causal-writability --include 'pendulum/*' --local-dir weights
hf download Wan-AI/Wan2.1-T2V-1.3B Wan2.1_VAE.pth --local-dir weights
python smoke.py --history short \
  --checkpoint weights/pendulum/pendulum-short-3407-50k.safetensors \
  --vae weights/Wan2.1_VAE.pth --out runs/smoke-short --device cuda:0
```

For Long, use `--history long` and the corresponding Long checkpoint. Both
models passed this B_082 runtime test: input rendering, all 20 generation calls,
full-matched editing and decoded-video evaluation. All six generated outputs
were evaluator-valid. This is not a full 64-receiver Top-4 reproduction; that
requires the regenerated PCA basis and frozen controller inputs.

## Rebuild Existing Figures

```bash
python rebuild_figures.py --out runs/rebuilt
python verify_data.py
```

These commands use the stored tables, not a new model run. No original paper
result is overwritten. See [model downloads](../../CHECKPOINTS.md); both final
Short/Long checkpoints are publicly downloadable.
