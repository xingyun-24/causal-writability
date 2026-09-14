#!/usr/bin/env bash
set -euo pipefail

SPLIT=${1:?usage: run_f1_f2_colour_offset_rollout_remote.sh TRAIN_OR_HOLDOUT_SHARD GPU}
GPU=${2:?usage: run_f1_f2_colour_offset_rollout_remote.sh TRAIN_OR_HOLDOUT_SHARD GPU}
MODE=${3:?usage: run_f1_f2_colour_offset_rollout_remote.sh TRAIN_OR_HOLDOUT_SHARD GPU train_or_holdout}
ROOT=/data/home/yuanman/spring-mechanism-run/projectile-gravity-continuous-v2
REPO=/data/home/yuanman/physics-shortcuts-benchmarks
RUN=${RUN:-hist32-step100000-block2-f1f2-colouroffset-rollout}
FIT_JSON=${FIT_JSON:-$ROOT/results/hist32-step100000-block2-fulltoken-pca-fp32/decomposition/f1_f2_colour_offset_fit.json}
OUT="$ROOT/results/$RUN/$MODE/shard$SPLIT"

test -f "$ROOT/checkpoints/short-v3-large-fixedpos-horizontal-hist32/hist32/step-100000.safetensors"
mkdir -p "$OUT"
cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$REPO/src:$REPO/lib/diffsynth"
python3 "$ROOT/scripts/rollout_pca_f1_f2_colour_offset.py" \
  --dataset-dir "$ROOT/data/raw-v3-fixedpos-horizontal-eval-256/eval" \
  --data-config "$ROOT/config/data-v3-fixedpos-horizontal.yaml" \
  --training-config "$ROOT/config/Train-short-v3-large-fixedpos-horizontal-hist32.yaml" \
  --checkpoint "$ROOT/checkpoints/short-v3-large-fixedpos-horizontal-hist32/hist32/step-100000.safetensors" \
  --eval-rows "$ROOT/results/hist32-step100000-block2-fulltoken-pca-fp32/decomposition/eval_rows.json" \
  --fit-json "$FIT_JSON" \
  --pca-root "$ROOT/results/hist32-step100000-block2-fulltoken-pca-fp32" \
  --out "$OUT" \
  --block 2 --observed-window 32 --steps 20 \
  --split "$MODE" --num-shards "${NUM_SHARDS:-4}" --shard-index "$SPLIT" --device cuda
