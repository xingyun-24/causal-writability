#!/usr/bin/env bash
# Build the frozen V2 short-history training input, then start training.
set -euo pipefail

ROOT=/data/home/yuanman/spring-mechanism-run/projectile-gravity-continuous-v2
REPO=/data/home/yuanman/physics-shortcuts-benchmarks
RAW_DIR="$ROOT/data/raw-v2"
LATENT_DIR="$ROOT/data/latents/train_short_v2"
RUN=short-v2

test ! -e "$RAW_DIR/train"
test ! -e "$LATENT_DIR"
mkdir -p "$ROOT/data" "$ROOT/data/latents" "$ROOT/logs" "$ROOT/training/$RUN/resolved" "$ROOT/checkpoints/$RUN" "$ROOT/output/$RUN" "$REPO/data"

cd "$REPO"
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="$REPO/src:$REPO/lib/diffsynth"

python3 "$ROOT/scripts/build_dataset_v2.py" \
  --config "$ROOT/config/data-v2.yaml" \
  --out "$RAW_DIR" \
  --split train \
  --base-seeds 1024

python3 "$ROOT/scripts/prepare_window_latents.py" \
  --source "$RAW_DIR/train" \
  --out "$LATENT_DIR" \
  --data-config "$ROOT/config/data-v2.yaml" \
  --window 32 \
  --device cuda

test "$(find "$LATENT_DIR" -maxdepth 1 -name '*.pt' | wc -l)" -eq 2048
test -s "$LATENT_DIR/metadata.csv"
ln -sfn "$ROOT/data" "$REPO/data/projectile_gravity_v2"

python3 "$ROOT/scripts/train_projectile.py" \
  --config "$ROOT/config/Train-short-v2.yaml" \
  --resolved-dir "$ROOT/training/$RUN/resolved" \
  --ckpt-dir "$ROOT/checkpoints/$RUN" \
  --out-dir "$ROOT/output/$RUN" \
  --no-wandb
