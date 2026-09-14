#!/usr/bin/env bash
# Encode V3 short-history latents, then start an independent 50k-step run.
set -euo pipefail

ROOT=/data/home/yuanman/spring-mechanism-run/projectile-gravity-continuous-v2
REPO=/data/home/yuanman/physics-shortcuts-benchmarks
RAW_DIR="$ROOT/data/raw-v3"
LATENT_DIR="$ROOT/data/latents/train_short_v3"
RUN=short-v3

test -d "$RAW_DIR/train"
test ! -e "$LATENT_DIR"
mkdir -p "$ROOT/data/latents" "$ROOT/logs" "$ROOT/training/$RUN/resolved" "$ROOT/checkpoints/$RUN" "$ROOT/output/$RUN" "$REPO/data"

cd "$REPO"
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="$REPO/src:$REPO/lib/diffsynth"

python3 "$ROOT/scripts/prepare_window_latents.py" \
  --source "$RAW_DIR/train" \
  --out "$LATENT_DIR" \
  --data-config "$ROOT/config/data-v3.yaml" \
  --window 32 \
  --device cuda

test "$(find "$LATENT_DIR" -maxdepth 1 -name '*.pt' | wc -l)" -eq 2048
test -s "$LATENT_DIR/metadata.csv"
ln -sfn "$ROOT/data" "$REPO/data/projectile_gravity_v3"

python3 "$ROOT/scripts/train_projectile.py" \
  --config "$ROOT/config/Train-short-v3.yaml" \
  --resolved-dir "$ROOT/training/$RUN/resolved" \
  --ckpt-dir "$ROOT/checkpoints/$RUN" \
  --out-dir "$ROOT/output/$RUN" \
  --no-wandb
