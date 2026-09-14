#!/usr/bin/env bash
set -euo pipefail

SHARD=${1:?usage: launch_freefall_fulltoken_capture_remote.sh SHARD GPU}
GPU=${2:?usage: launch_freefall_fulltoken_capture_remote.sh SHARD GPU}
BLOCK=${3:-1}
ROOT=/data/home/yuanman/spring-mechanism-run/projectile-gravity-continuous-v2
REPO=/data/home/yuanman/physics-shortcuts-benchmarks
RUN=freefall-hist32-step100000-block${BLOCK}-fulltoken-pca-fp32
cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$REPO/src:$REPO/lib/diffsynth"
python3 "$ROOT/scripts/projectile_block_pca_fit.py" capture \
  --dataset-dir "$ROOT/data/raw-v3-fixedpos-freefall-2048/eval" \
  --data-config "$ROOT/config/data-v3-fixedpos-freefall.yaml" \
  --training-config "$ROOT/config/Train-short-v3-large-fixedpos-freefall-hist32.yaml" \
  --checkpoint "$ROOT/checkpoints/short-v3-large-fixedpos-freefall-hist32/hist32/step-100000.safetensors" \
  --pair-manifest "$ROOT/results/freefall-hist32-step100000-eval/manifests/recovery_pairs_shard$SHARD.json" \
  --out "$ROOT/results/$RUN/shard$SHARD" \
  --block "$BLOCK" --observed-window 32 --steps 20 --delta-dtype float32 --device cuda
