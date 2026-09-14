#!/usr/bin/env bash
set -euo pipefail

# The 128-seed evaluation has only 31 strict low-g recovery candidates.
# Use 320 base seeds so the final balanced manifest can contain 64 low-g and
# 64 high-g pairs under the current absolute-E3 selection rule.
ROOT=/data/home/yuanman/spring-mechanism-run/projectile-gravity-continuous-v2
REPO=/data/home/yuanman/physics-shortcuts-benchmarks
RAW="$ROOT/data/raw-v3-fixedpos-freefall-eval320"
RUN=freefall-hist32-step100000-eval320-strict
OUT="$ROOT/results/$RUN"
CONFIG="$ROOT/config/data-v3-fixedpos-freefall.yaml"
TRAIN="$ROOT/config/Train-short-v3-large-fixedpos-freefall-hist32.yaml"
CHECKPOINT="$ROOT/checkpoints/short-v3-large-fixedpos-freefall-hist32/hist32/step-100000.safetensors"
GPUS=(1 3 6)

test ! -e "$RAW"
test ! -e "$OUT"
test -s "$CHECKPOINT"
mkdir -p "$ROOT/logs"

cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/lib/diffsynth"

python3 "$ROOT/scripts/build_dataset_v2.py" \
  --config "$CONFIG" --out "$RAW" --split eval --base-seeds 320 --seed-offset 50000 \
  >"$ROOT/logs/$RUN-build.log" 2>&1

mkdir -p "$OUT"
for shard in 0 1 2; do
  CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" python3 "$ROOT/scripts/audit_v1_estimator.py" \
    --dataset-dir "$RAW/eval" --data-config "$CONFIG" --out "$OUT/shard$shard" --source model \
    --training-config "$TRAIN" --checkpoint "$CHECKPOINT" --observed-window 32 --steps 20 \
    --pair-limit 0 --pair-offset "$shard" --pair-stride 3 --device cuda \
    >"$OUT/shard$shard.log" 2>&1 &
  pids[$shard]=$!
done
for shard in 0 1 2; do
  wait "${pids[$shard]}"
done

python3 "$ROOT/scripts/summarize_history_eval.py" --root "$OUT" --out "$OUT/summary_merged.json"

