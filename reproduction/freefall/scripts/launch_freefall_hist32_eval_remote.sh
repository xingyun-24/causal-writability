#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/home/yuanman/spring-mechanism-run/projectile-gravity-continuous-v2
REPO=/data/home/yuanman/physics-shortcuts-benchmarks
RUN=freefall-hist32-step100000-eval
OUT="$ROOT/results/$RUN"
mkdir -p "$OUT"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/lib/diffsynth"

for shard in 0 1 2; do
  gpu=$((shard + 1))
  CUDA_VISIBLE_DEVICES="$gpu" python3 "$ROOT/scripts/audit_v1_estimator.py" \
    --dataset-dir "$ROOT/data/raw-v3-fixedpos-freefall-2048/eval" \
    --data-config "$ROOT/config/data-v3-fixedpos-freefall.yaml" \
    --out "$OUT/shard$shard" --source model \
    --training-config "$ROOT/config/Train-short-v3-large-fixedpos-freefall-hist32.yaml" \
    --checkpoint "$ROOT/checkpoints/short-v3-large-fixedpos-freefall-hist32/hist32/step-100000.safetensors" \
    --observed-window 32 --steps 20 --pair-limit 128 --pair-offset "$shard" --pair-stride 3 --device cuda \
    >"$OUT/shard$shard.log" 2>&1 &
done
wait
