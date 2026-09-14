#!/usr/bin/env bash
set -euo pipefail

SHARD=${1:?usage: run_freefall_layer_recovery_remote.sh SHARD GPU}
GPU=${2:?usage: run_freefall_layer_recovery_remote.sh SHARD GPU}
ROOT=/data/home/yuanman/spring-mechanism-run/projectile-gravity-continuous-v2
REPO=/data/home/yuanman/physics-shortcuts-benchmarks
RUN=freefall-hist32-step100000-recovery95-e3-par8
OUT="$ROOT/results/$RUN/shard$SHARD"
cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$REPO/src:$REPO/lib/diffsynth"
python3 "$ROOT/scripts/layer_residual_replacement.py" \
  --dataset-dir "$ROOT/data/raw-v3-fixedpos-freefall-2048/eval" \
  --out "$OUT" \
  --training-config "$ROOT/config/Train-short-v3-large-fixedpos-freefall-hist32.yaml" \
  --checkpoint "$ROOT/checkpoints/short-v3-large-fixedpos-freefall-hist32/hist32/step-100000.safetensors" \
  --data-config "$ROOT/config/data-v3-fixedpos-freefall.yaml" \
  --pair-manifest "$ROOT/results/freefall-hist32-step100000-eval/manifests8/recovery_pairs_shard$SHARD.json" \
  --observed-window 32 --steps 20 --device cuda
