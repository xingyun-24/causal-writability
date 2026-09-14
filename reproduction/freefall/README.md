# Free Fall: hist32 / 100K / B1 / train-only

This package matches `freefall_pca_trainonly_redo.zip`.
`redo/` preserves its scripts, split and fitted controller. PCA arrays and saved
RGB futures are restored by the download command below, unchanged from the
handoff; duplicate nested tarballs are omitted. `paper/` contains
the later corrected panel-kit tables and figures. Its `source_kit_manifest.json`
describes the full source kit, including videos not copied here.

The older desktop benchmark snapshot and pooled-controller results are **not**
the current paper protocol. This adds the missing runtime, latest 64/64 split,
train-only fit/recovery code, dense rank-2 basis, checkpoint and frame evidence.
The source runtime was copied from the experiment server, whose checkout was
dirty. Unused older scripts/configs remain
as source history; use only the entry points below for the current result.

## Files and frozen protocol

| Item | Path relative to this directory |
|---|---|
| Final checkpoint | `checkpoints/short-v3-large-fixedpos-freefall-hist32/hist32/step-100000.safetensors` |
| Model/training config | `config/Train-short-v3-large-fixedpos-freefall-hist32.yaml` |
| Exact archived resolved config | `paper/resolved_config.yaml` |
| Renderer/data config | `config/data-v3-fixedpos-freefall.yaml` |
| Wan2.1 VAE | `models/Wan2.1_VAE.pth` |
| Strict bank, physical parameters and generation seeds | `paper/strict_bank.csv`, `redo/delivery_inputs/metadata.csv` |
| Exact split | `redo/results-grouped/split_manifest.json` |
| Source pair manifest | `results/freefall-hist32-step100000-eval320-strict/recovery_pairs_strict128.json` |
| Implicit PCA, including higher components | `redo/results-grouped/train_only_pca.npz` |
| Explicit rank-2 PCA basis | `paper/pca_components.npy` |
| Frozen controller | `redo/delivery_inputs/directional_controller_fit.json` |
| Original producer and saved RGB futures | `redo/rollout_directional_trainonly.py`, `redo/controller_extracted/controller_rollout/` |
| Source dependency inventory | `environment.source.freeze.txt` |
| Four existing MP4 exports | `../../website/public/videos/freefall/originals/` |

Model downloads are listed in the repository's `CHECKPOINTS.md`.

There are 64 fit pairs and 64 held-out pairs, with 32 low/32 high in each split.
The split repair keeps shared base seeds together: `eval_50208_high` moved to fit
and `eval_50101_high` moved to held-out. Splitting is deterministic, using the
archived pair order and the documented repair, rather than a new random seed.
Input base seeds are recorded in metadata. Generation seed = base seed +
17000000. All 20 FM calls of the 1088 condition-prefix tokens are concatenated:
each residual is `(20,1088,1536)`, after zero-based block 1.
Uncentered SVD fits only the 64 fit matched differences, aligned minus conflict.
The controller is direction-specific `[1,g_target]`, rank 2, with frozen global
scale `1.0022815498887996`. Held-out aligned activations and generated gravity
are not controller inputs.

## Install

From the project repository root, download the model and artifacts, then use a separate
environment; Spring and Pendulum also define `sshv2` and must not share it.
The experiment source environment is Python 3.10.12 on Linux/CUDA.

```bash
python -m pip install 'huggingface_hub==0.36.2'
python scripts/download_freefall.py
cd reproduction/freefall
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
export ROOT="$PWD"
export PYTHONPATH="$ROOT/src:$ROOT/lib/diffsynth:$ROOT/scripts"
python scripts/rollout_trainonly_release.py --help
python redo/fit_train_only_pca.py --self-test
```

`requirements.source-pinned.txt` records the top-level versions actually found
on the server. The full freeze includes OS packages and should not be blindly
installed into a fresh environment. The vendored DiffSynth source is included
with its Apache-2.0 license; its source commit is
`afd101f3452c9ecae0c87b79adfa2e22d65ffdc3`.
The original training config retains archival absolute training paths; inference
uses the supplied checkpoint and relative `models/Wan2.1_VAE.pth`. Set the data
and log paths to local locations before retraining. Third-party model terms
remain applicable; the code license does not replace separate model terms.

## Generate exact evaluation inputs

Run all following commands from `reproduction/freefall`, with the environment
above. Outputs must be new directories. The complete 320-seed generation is
included to preserve the original bank context; the frozen 128-pair manifest
selects its inputs without reselecting based on new outcomes.

```bash
python scripts/build_dataset_v2.py \
  --config config/data-v3-fixedpos-freefall.yaml \
  --out data/raw-v3-fixedpos-freefall-eval320 --split eval \
  --base-seeds 320 --seed-offset 50000

# Original 2048 training examples, if needed for retraining:
python scripts/build_dataset_v2.py \
  --config config/data-v3-fixedpos-freefall.yaml \
  --out data/raw-v3-fixedpos-freefall-2048 --split train --base-seeds 1024
```

## Natural generation and evaluation

This command generates and evaluates the original 640 pairs / 1280 conditions
using the E3 boundary-state estimator. It writes measurement rows, not MP4s.
For saved natural-conflict futures on the final held-out bank, use the controller
command below, which generates the paired natural reference with identical noise.

```bash
python scripts/audit_v1_estimator.py \
  --dataset-dir data/raw-v3-fixedpos-freefall-eval320/eval \
  --data-config config/data-v3-fixedpos-freefall.yaml \
  --training-config config/Train-short-v3-large-fixedpos-freefall-hist32.yaml \
  --checkpoint checkpoints/short-v3-large-fixedpos-freefall-hist32/hist32/step-100000.safetensors \
  --out runs/natural-e3 --source model --observed-window 32 \
  --steps 20 --pair-limit 0 --device cuda:0
```

## Controller edit without residual regeneration

`scripts/rollout_trainonly_release.py` is a copy of the archived producer with
only basis loading changed to the supplied dense export, plus `--limit` for smoke
checks. Injection and evaluation are unchanged, and it
never needs held-out source residuals. The exact archived producer remains in
`redo/rollout_directional_trainonly.py` for use after residual regeneration.

```bash
python scripts/rollout_trainonly_release.py \
  --project-root "$ROOT" --pca-dir redo/results-grouped \
  --controller-json redo/delivery_inputs/directional_controller_fit.json \
  --basis paper/pca_components.npy --out runs/controller \
  --shard 0 --num-shards 1 --device cuda:0

# Re-evaluate all archived natural-conflict/controller saved frames, no GPU inference:
python scripts/evaluate_saved_rollouts.py \
  --frames-root redo/controller_extracted/controller_rollout \
  --metadata redo/delivery_inputs/metadata.csv \
  --data-config config/data-v3-fixedpos-freefall.yaml --out runs/reevaluated

# Refit the controller from the already frozen train-only PCA:
python redo/fit_directional_trainonly.py --pca-dir redo/results-grouped \
  --pair-manifest results/freefall-hist32-step100000-eval320-strict/recovery_pairs_strict128.json \
  --out runs/controller-refit.json
```

To evaluate newly generated futures, replace `--frames-root` with
`runs/controller`. E3 uses `abs(g_hat-g_true)<0.002`; track validity and all samples
remain in the denominator. Re-evaluation of all 128 archived futures reproduced
64/64 valid in each condition and E3 0/64 natural versus 61/64 controller.

## Residual regeneration and PCA refit

Source residual arrays are deliberately not committed (roughly 17 GB for 128
float32 differences); the supplied dense basis makes controller use independent
of them. To refit, generate the same inputs, then run these four shards
sequentially or assign each to a separate device.

```bash
python scripts/split_recovery_manifest.py \
  --manifest results/freefall-hist32-step100000-eval320-strict/recovery_pairs_strict128.json \
  --out-dir results/freefall-hist32-step100000-eval320-strict/manifests4 --shards 4
for shard in 0 1 2 3; do
  python scripts/projectile_block_pca_fit.py capture \
    --dataset-dir data/raw-v3-fixedpos-freefall-eval320/eval \
    --data-config config/data-v3-fixedpos-freefall.yaml \
    --training-config config/Train-short-v3-large-fixedpos-freefall-hist32.yaml \
    --checkpoint checkpoints/short-v3-large-fixedpos-freefall-hist32/hist32/step-100000.safetensors \
    --pair-manifest "results/freefall-hist32-step100000-eval320-strict/manifests4/recovery_pairs_shard${shard}.json" \
    --out "results/freefall-hist32-step100000-block1-pca128/shard${shard}" \
    --block 1 --observed-window 32 --steps 20 --seed-offset 17000000 \
    --delta-dtype float32 --device cuda:0
done
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python redo/fit_train_only_pca.py \
  --pca-root results/freefall-hist32-step100000-block1-pca128 \
  --dataset-dir data/raw-v3-fixedpos-freefall-eval320/eval \
  --pair-manifest results/freefall-hist32-step100000-eval320-strict/recovery_pairs_strict128.json \
  --out runs/refit-pca --device cuda:0
python redo/export_trainonly_basis.py --project-root "$ROOT" \
  --pca-dir runs/refit-pca --out runs/refit-components.npy --device cuda:0
```

Refitting uses substantial CPU/GPU memory (fit matrix alone is about 17 GB in
float64). The larger original PCA command's `analyze` subcommand is not the
train-only refit entry; use `redo/fit_train_only_pca.py`.

## Validation and limits

The original delivery included saved-frame evaluation. During integration,
natural aligned/conflict rollouts and B1 activations were regenerated for the
128-pair bank; the resulting held-out projections reproduce the archived
coordinates. This is not a new full controller or training reproduction.
The bank was originally selected using baseline behavior and
had been seen in earlier analyses; the repaired split guarantees fitting
isolation, not a wholly unseen test population. This delivery does not substitute
an old B2 HTML for a current B1 visualization, or supply the missing Pendulum
raw activations as if they were Free Fall data.
