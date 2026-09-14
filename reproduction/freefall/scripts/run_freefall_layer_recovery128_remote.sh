#!/usr/bin/env bash
set -euo pipefail

# Recompute the write-layer scan using the frozen 128-pair strict manifest.
ROOT=/data/home/yuanman/spring-mechanism-run/projectile-gravity-continuous-v2
REPO=/data/home/yuanman/physics-shortcuts-benchmarks
RUN=freefall-hist32-step100000-recovery128-e3-par4
OUT="$ROOT/results/$RUN"
MANIFESTS="$ROOT/results/freefall-hist32-step100000-eval320-strict/manifests4"
GPUS=(1 3 4 6)

test ! -e "$OUT"
mkdir -p "$OUT/logs"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/lib/diffsynth"

for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" python3 "$ROOT/scripts/layer_residual_replacement.py" \
    --dataset-dir "$ROOT/data/raw-v3-fixedpos-freefall-eval320/eval" \
    --out "$OUT/shard$shard" \
    --training-config "$ROOT/config/Train-short-v3-large-fixedpos-freefall-hist32.yaml" \
    --checkpoint "$ROOT/checkpoints/short-v3-large-fixedpos-freefall-hist32/hist32/step-100000.safetensors" \
    --data-config "$ROOT/config/data-v3-fixedpos-freefall.yaml" \
    --pair-manifest "$MANIFESTS/recovery_pairs_shard$shard.json" \
    --observed-window 32 --steps 20 --device cuda \
    >"$OUT/logs/shard$shard.log" 2>&1 &
  pids[$shard]=$!
done

for shard in 0 1 2 3; do
  wait "${pids[$shard]}"
done

python3 "$ROOT/scripts/summarize_recovery_layers.py" \
  --outcomes "$OUT"/shard*/outcomes.json --out-dir "$OUT/summary"

