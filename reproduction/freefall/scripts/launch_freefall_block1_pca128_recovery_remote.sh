#!/usr/bin/env bash
set -euo pipefail

COMPONENTS=${1:?usage: launch_freefall_block1_pca128_recovery_remote.sh COMPONENTS GPU}
GPU=${2:?usage: launch_freefall_block1_pca128_recovery_remote.sh COMPONENTS GPU}
ROOT=/data/home/yuanman/spring-mechanism-run/projectile-gravity-continuous-v2
REPO=/data/home/yuanman/physics-shortcuts-benchmarks
PCA_RUN=freefall-hist32-step100000-block1-pca128
RECOVERY_RUN=freefall-hist32-step100000-block1-pca128-recovery

cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$REPO/src:$REPO/lib/diffsynth"
python3 "$ROOT/scripts/full_token_pca_recovery.py" \
  --dataset-dir "$ROOT/data/raw-v3-fixedpos-freefall-eval320/eval" \
  --data-config "$ROOT/config/data-v3-fixedpos-freefall.yaml" \
  --training-config "$ROOT/config/Train-short-v3-large-fixedpos-freefall-hist32.yaml" \
  --checkpoint "$ROOT/checkpoints/short-v3-large-fixedpos-freefall-hist32/hist32/step-100000.safetensors" \
  --pair-manifest "$ROOT/results/freefall-hist32-step100000-eval320-strict/recovery_pairs_strict128.json" \
  --pca-root "$ROOT/results/$PCA_RUN" \
  --out "$ROOT/results/$RECOVERY_RUN/rank${COMPONENTS}" \
  --components "$COMPONENTS" --block 1 --observed-window 32 --steps 20 --device cuda
