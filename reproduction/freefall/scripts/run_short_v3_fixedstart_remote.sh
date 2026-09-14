#!/usr/bin/env bash
# Train the fixed-frame-65 V3 baseline from the existing complete latent set.
set -euo pipefail

ROOT=/data/home/yuanman/spring-mechanism-run/projectile-gravity-continuous-v2
REPO=/data/home/yuanman/physics-shortcuts-benchmarks
LATENT_DIR="$ROOT/data/latents/train_short_v3"
RUN=short-v3-fixedstart

test "$(find "$LATENT_DIR" -maxdepth 1 -name '*.pt' | wc -l)" -eq 2048
test -s "$LATENT_DIR/metadata.csv"
mkdir -p "$ROOT/logs" "$ROOT/training/$RUN/resolved" "$ROOT/checkpoints/$RUN" "$ROOT/output/$RUN" "$REPO/data"

cd "$REPO"
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH="$REPO/src:$REPO/lib/diffsynth"
ln -sfn "$ROOT/data" "$REPO/data/projectile_gravity_v3"

python3 "$ROOT/scripts/train_projectile.py" \
  --config "$ROOT/config/Train-short-v3-fixedstart.yaml" \
  --resolved-dir "$ROOT/training/$RUN/resolved" \
  --ckpt-dir "$ROOT/checkpoints/$RUN" \
  --out-dir "$ROOT/output/$RUN" \
  --no-wandb
