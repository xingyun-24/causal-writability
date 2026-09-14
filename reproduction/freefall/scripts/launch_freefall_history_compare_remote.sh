#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/home/yuanman/spring-mechanism-run/projectile-gravity-continuous-v2
REPO=/data/home/yuanman/physics-shortcuts-benchmarks
RAW="$ROOT/data/raw-v3-fixedpos-freefall-2048/eval"
CONFIG="$ROOT/config/data-v3-fixedpos-freefall.yaml"
export PYTHONPATH="$REPO/src:$REPO/lib/diffsynth"
cd "$REPO"

run_one() {
  local history="$1" gpu="$2" shard="$3"
  local out="$ROOT/results/freefall-hist${history}-step100000-eval128"
  CUDA_VISIBLE_DEVICES="$gpu" python3 "$ROOT/scripts/audit_v1_estimator.py" \
    --dataset-dir "$RAW" --data-config "$CONFIG" --out "$out/shard$shard" --source model \
    --training-config "$ROOT/config/Train-short-v3-large-fixedpos-freefall-hist${history}.yaml" \
    --checkpoint "$ROOT/checkpoints/short-v3-large-fixedpos-freefall-hist${history}/hist${history}/step-100000.safetensors" \
    --observed-window "$history" --steps 20 --pair-limit 128 --pair-offset "$shard" --pair-stride 2 --device cuda \
    >"$out/shard$shard.log" 2>&1
}

rm -rf "$ROOT/results/freefall-hist8-step100000-eval128" "$ROOT/results/freefall-hist16-step100000-eval128"
mkdir -p "$ROOT/results/freefall-hist8-step100000-eval128" "$ROOT/results/freefall-hist16-step100000-eval128"
run_one 8 0 0 & run_one 8 1 1 &
run_one 16 2 0 & run_one 16 3 1 &
run_one 32 4 0 & run_one 32 5 1 &
wait
