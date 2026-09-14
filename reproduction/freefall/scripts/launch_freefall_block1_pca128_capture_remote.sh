#!/usr/bin/env bash
set -euo pipefail

SHARD=${1:?usage: launch_freefall_block1_pca128_capture_remote.sh SHARD GPU}
GPU=${2:?usage: launch_freefall_block1_pca128_capture_remote.sh SHARD GPU}
ROOT=/data/home/yuanman/spring-mechanism-run/projectile-gravity-continuous-v2
REPO=/data/home/yuanman/physics-shortcuts-benchmarks
RUN=freefall-hist32-step100000-block1-pca128

cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$REPO/src:$REPO/lib/diffsynth"
python3 "$ROOT/scripts/projectile_block_pca_fit.py" capture \
  --dataset-dir "$ROOT/data/raw-v3-fixedpos-freefall-eval320/eval" \
  --data-config "$ROOT/config/data-v3-fixedpos-freefall.yaml" \
  --training-config "$ROOT/config/Train-short-v3-large-fixedpos-freefall-hist32.yaml" \
  --checkpoint "$ROOT/checkpoints/short-v3-large-fixedpos-freefall-hist32/hist32/step-100000.safetensors" \
  --pair-manifest "$ROOT/results/freefall-hist32-step100000-eval320-strict/manifests4/recovery_pairs_shard${SHARD}.json" \
  --out "$ROOT/results/$RUN/shard${SHARD}" \
  --block 1 --observed-window 32 --steps 20 --delta-dtype float32 --device cuda
