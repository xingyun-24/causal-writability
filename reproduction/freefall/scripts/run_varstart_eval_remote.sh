#!/usr/bin/env bash
# Build held-out V2/V3 pairs and evaluate both completed variable-start models.
set -euo pipefail

ROOT=/data/home/yuanman/spring-mechanism-run/projectile-gravity-continuous-v2
REPO=/data/home/yuanman/physics-shortcuts-benchmarks
V2_EVAL="$ROOT/data/eval-v2-varstart"
V3_EVAL="$ROOT/data/eval-v3-varstart"
V2_OUT="$ROOT/results/short-v2-varstart-step50000-eval"
V3_OUT="$ROOT/results/short-v3-varstart-step50000-eval"

test ! -e "$V2_EVAL/eval"
test ! -e "$V3_EVAL/eval"
test ! -e "$V2_OUT"
test ! -e "$V3_OUT"
mkdir -p "$ROOT/data" "$ROOT/results" "$ROOT/logs"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/lib/diffsynth"

python3 "$ROOT/scripts/build_dataset_v2.py" --config "$ROOT/config/data-v2.yaml" --out "$V2_EVAL" --split eval --base-seeds 128 --seed-offset 50000
python3 "$ROOT/scripts/build_dataset_v2.py" --config "$ROOT/config/data-v3.yaml" --out "$V3_EVAL" --split eval --base-seeds 128 --seed-offset 50000

CUDA_VISIBLE_DEVICES=5 python3 "$ROOT/scripts/audit_v2_estimator.py" \
  --dataset-dir "$V2_EVAL/eval" --data-config "$ROOT/config/data-v2.yaml" \
  --out "$V2_OUT" --source model --pair-limit 256 \
  --training-config "$ROOT/config/Train-short-v2-varstart.yaml" \
  --checkpoint "$ROOT/checkpoints/short-v2-varstart/short-v2-varstart/step-50000.safetensors" \
  --short-history --steps 20 --device cuda &
V2_PID=$!
CUDA_VISIBLE_DEVICES=6 python3 "$ROOT/scripts/audit_v2_estimator.py" \
  --dataset-dir "$V3_EVAL/eval" --data-config "$ROOT/config/data-v3.yaml" \
  --out "$V3_OUT" --source model --pair-limit 256 \
  --training-config "$ROOT/config/Train-short-v3-varstart.yaml" \
  --checkpoint "$ROOT/checkpoints/short-v3-varstart/short-v3-varstart/step-50000.safetensors" \
  --short-history --steps 20 --device cuda &
V3_PID=$!
wait "$V2_PID"
wait "$V3_PID"
