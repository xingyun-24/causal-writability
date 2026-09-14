#!/usr/bin/env bash
# Encode V3 samples with varied prediction starts, then train a separate run.
set -euo pipefail

ROOT=/data/home/yuanman/spring-mechanism-run/projectile-gravity-continuous-v2
REPO=/data/home/yuanman/physics-shortcuts-benchmarks
RAW_DIR="$ROOT/data/raw-v3"
LATENT_DIR="$ROOT/data/latents/train_short_v3_varstart"
RUN=short-v3-varstart
STARTS=(32 36 40 44 48 52 56 60 64 68 72 76 80 84 88 92 96)
GPUS=(2 5 6)

test -d "$RAW_DIR/train"
# This directory may contain a compatible partial encode after an interrupted run.
mkdir -p "$LATENT_DIR" "$ROOT/logs" "$ROOT/training/$RUN/resolved" "$ROOT/checkpoints/$RUN" "$ROOT/output/$RUN" "$REPO/data"

cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/lib/diffsynth"

for SHARD in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$SHARD]}" python3 "$ROOT/scripts/prepare_window_latents.py" \
    --source "$RAW_DIR/train" \
    --out "$LATENT_DIR" \
    --data-config "$ROOT/config/data-v3.yaml" \
    --window 32 \
    --prediction-starts "${STARTS[@]}" \
    --batch-size 4 \
    --num-shards "${#GPUS[@]}" \
    --shard-index "$SHARD" \
    --no-metadata \
    --device cuda &
  PIDS[$SHARD]=$!
done
for PID in "${PIDS[@]}"; do wait "$PID"; done
python3 "$ROOT/scripts/prepare_window_latents.py" \
  --source "$RAW_DIR/train" \
  --out "$LATENT_DIR" \
  --data-config "$ROOT/config/data-v3.yaml" \
  --window 32 \
  --prediction-starts "${STARTS[@]}" \
  --metadata-only

test "$(find "$LATENT_DIR" -maxdepth 1 -name '*.pt' | wc -l)" -eq 2048
test -s "$LATENT_DIR/metadata.csv"
ln -sfn "$ROOT/data" "$REPO/data/projectile_gravity_v3"
export CUDA_VISIBLE_DEVICES=2

python3 "$ROOT/scripts/train_projectile.py" \
  --config "$ROOT/config/Train-short-v3-varstart.yaml" \
  --resolved-dir "$ROOT/training/$RUN/resolved" \
  --ckpt-dir "$ROOT/checkpoints/$RUN" \
  --out-dir "$ROOT/output/$RUN" \
  --no-wandb
