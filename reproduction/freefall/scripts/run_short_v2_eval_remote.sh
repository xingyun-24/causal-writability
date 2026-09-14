#!/usr/bin/env bash
# Evaluate the completed short-v2 checkpoint on held-out V2 pairs.
set -euo pipefail

ROOT=/data/home/yuanman/spring-mechanism-run/projectile-gravity-continuous-v2
REPO=/data/home/yuanman/physics-shortcuts-benchmarks
EVAL_DIR="$ROOT/data/eval-v2"
OUT_DIR="$ROOT/results/short-v2-step50000-eval"
CHECKPOINT="$ROOT/checkpoints/short-v2/short-v2/step-50000.safetensors"

test ! -e "$EVAL_DIR/eval"
test ! -e "$OUT_DIR"
test -s "$CHECKPOINT"
mkdir -p "$ROOT/data" "$ROOT/results"

cd "$REPO"
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="$REPO/src:$REPO/lib/diffsynth"

python3 "$ROOT/scripts/build_dataset_v2.py" \
  --config "$ROOT/config/data-v2.yaml" \
  --out "$EVAL_DIR" \
  --split eval \
  --base-seeds 128 \
  --seed-offset 50000

python3 "$ROOT/scripts/audit_v2_estimator.py" \
  --dataset-dir "$EVAL_DIR/eval" \
  --data-config "$ROOT/config/data-v2.yaml" \
  --out "$OUT_DIR" \
  --source model \
  --pair-limit 256 \
  --training-config "$ROOT/config/Train-short-v2.yaml" \
  --checkpoint "$CHECKPOINT" \
  --short-history \
  --steps 20 \
  --device cuda
