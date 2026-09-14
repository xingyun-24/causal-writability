#!/usr/bin/env bash
set -euo pipefail

SHARD=${1:?usage: run_recovery_parallel_remote.sh SHARD GPU}
GPU=${2:?usage: run_recovery_parallel_remote.sh SHARD GPU}
ROOT=/data/home/yuanman/spring-mechanism-run/projectile-gravity-continuous-v2
REPO=/data/home/yuanman/physics-shortcuts-benchmarks
RUN=hist16-step50000-recovery128-e3-par32
OUT="$ROOT/results/$RUN/shard$SHARD"
MANIFEST="$ROOT/results/$RUN/manifests/recovery_pairs_shard$SHARD.json"
CHECKPOINT="$ROOT/checkpoints/short-v3-large-fixedpos-horizontal-hist16/hist16/step-50000.safetensors"

test -s "$CHECKPOINT"
test -s "$MANIFEST"
mkdir -p "$OUT"
cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$REPO/src:$REPO/lib/diffsynth"
python3 "$ROOT/scripts/layer_residual_replacement.py" \
  --dataset-dir "$ROOT/data/raw-v3-fixedpos-horizontal-eval-256/eval" \
  --out "$OUT" \
  --training-config "$ROOT/config/Train-short-v3-large-fixedpos-horizontal-hist16.yaml" \
  --checkpoint "$CHECKPOINT" \
  --data-config "$ROOT/config/data-v3-fixedpos-horizontal.yaml" \
  --pair-manifest "$MANIFEST" \
  --observed-window 16 \
  --steps 20 \
  --device cuda
