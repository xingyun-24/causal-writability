# Causal Writability in Video Models

Code for **A Chosen Future Can Still Be Rewritten: Causal Writability in Video Models**.

This local release candidate contains the Spring renderer, training pipeline,
decoded-video evaluator, residual patching and frozen-controller K/V interventions.
Pendulum and Free Fall live in the neighboring task directories and use separate
environments. Core model downloads are available through the Hugging
Face repository; see `../../CHECKPOINTS.md`. `cw weights --model ...` uses the
logged-in HF client for those checkpoints.

## Setup

Use Python 3.12.14 and a CUDA-capable PyTorch environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
pip install --no-deps -e .
pytest -q
```

Run commands from the repository root. Paths in the training configs are relative
to this directory, not the original server. Use `cw COMMAND --help` for options.

After supplying the seed-3407 Short checkpoint and VAE, the small end-to-end test is:

```bash
python tools/smoke.py --checkpoint weights/spring-short-3407-50k.safetensors \
  --vae weights/vae.pth --out runs/smoke
```

This generates a tiny dataset and runs one optimizer step, not a 50K retraining.

## Models

The core models are Spring Short and Long at 50K steps, each with seeds 3407,
3408 and 3409. The registry in `models/registry.json` provides their download
locations. No Hugging Face login is required for these public downloads.
The following commands install a Short model and the upstream VAE:

```bash
cw weights --model spring-short-3407-50k \
  --out weights/spring-short-3407-50k.safetensors
cw weights --model wan21-vae --out weights/vae.pth
```

See [Models](docs/models.md) for intended uses and limitations.

## Train

The renderer and training implementation are retained from the original Spring
code, without the later separate-RNG training rewrite.

```bash
cw data --config configs/data.yaml --root data --split train
cw encode --source data/videos/train --out-root data/latents \
  --vae weights/vae.pth --data-config configs/data.yaml --device cuda
cw train --config configs/spring_short.yaml --seed 3407 --no-wandb
```

For Long, use `configs/spring_long.yaml`. The default is 50K training steps;
`--steps 100000` extends the schedule. Training the complete model is not part
of the small release smoke test.

## Generate and Evaluate

```bash
cw data --config configs/data.yaml --root data --split eval
cw generate --config configs/spring_short.yaml \
  --checkpoint weights/spring-short-3407-50k.safetensors \
  --dataset data/videos/eval --data-config configs/data.yaml \
  --history short --out runs/natural --limit 4
cw evaluate --dataset data/videos/eval --predictions runs/natural \
  --data-config configs/data.yaml --history short --out runs/evaluation --limit 4
```

This generated evaluation set contains aligned/conflict endpoint examples;
it is not the paper's fixed 64-by-11 cue grid. Archived fixed-grid per-video
results are separately provided in `results/behavior/`.

## Causal Interventions

The frozen shared bank and fit/held-out membership are in `data/`. The runner
reconstructs the original input pixels from their physical parameters and checks
receiver identity. By default, it selects one held-out fast-target receiver.

```bash
cw edit --checkpoint weights/spring-short-3407-50k.safetensors \
  --vae weights/vae.pth --sites 3 --out runs/paired
```

Use `--sites all` for a 31-site residual scan and `--limit 0` for all 64 held-out
receivers in one target direction. This shared-bank scan is distinct from the
checkpoint-local failure banks used for the training-time study.

Download the frozen controller for the same Short model seed. This bundle
contains only the fitted coefficients and the 20 per-call top-four basis
tensors, not the full residual bank. Run from this directory:

```bash
hf download xingyun-24/causal-writability \
  --include 'controllers/spring/3409/*' --local-dir weights

cw weights --model spring-short-3409-50k --out weights/spring-short-3409-50k.safetensors
cw edit --checkpoint weights/spring-short-3409-50k.safetensors \
  --vae weights/vae.pth --sites 4 --controller weights/controllers/spring/3409 \
  --out runs/controlled
```

The other supplied Short controllers are seed 3407 at site 3 and seed 3408 at
site 6 (all sites are zero-based blocks). Change the download prefix, checkpoint,
`--sites` and `--controller` together. Each seed's bundle is about 401 MB.
Skip the weight-install command if that checkpoint is already installed.
These are not Long-model controllers. The command above performs a phase edit.
Adding `--kv-block 9 --components v --head 8 --gain 8` also runs a targeted
attention-write intervention. This gain can overshoot on some receivers; it is
not a universal restoration setting for the shared bank.

`--fm-window early` or `late` restricts the K/V write to calls 0-9 or 10-19.
The demo selects from the shared held-out bank; reproducing the paper's specific
48-failure dose sweep additionally requires its selection-clean receiver list.
Do not compare a demo fraction directly to that cohort's reported rate.

Each receiver produces aligned/conflict, paired-edit and optional phase/KV
future videos plus `metrics.json`. The phase prediction uses only receiver state
and frozen coefficients/bases, not held-out aligned activations. Aligned generation
is used solely for the separate oracle and normalized-recovery reference.

## Results

`results/index.json` maps paper figure numbers to the supplied plots and supporting
tables. It currently contains selected supporting data, not every appendix input.
Physics-follow rate and normalized recovery are different metrics; see
[Evaluation](docs/evaluation.md).

```bash
python tools/plot_results.py --out runs/replot
```

This replots numerical summaries, not the exact paper artwork. Original decoded
examples and their model/trajectory associations are in `examples/videos.json`.

## License and Publication Status

The authors' original code is licensed under [MIT](LICENSE).
Included third-party code retains its applicable terms; see [NOTICE](NOTICE.md).
The existing DiffSynth license is preserved in `third_party/`. This code license
does not automatically license separately distributed weights, paper text or media.
No remote repository has been created or changed by this preparation.
