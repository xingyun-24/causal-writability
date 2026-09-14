#!/usr/bin/env bash
# Train one fixed-start V3 large model with the requested visible-history length.
set -euo pipefail

WINDOW=${1:?usage: run_v3_large_fixedstart_history_remote.sh WINDOW GPU}
GPU=${2:?usage: run_v3_large_fixedstart_history_remote.sh WINDOW GPU}
case "$WINDOW" in 8|16|32|64) ;; *) echo "WINDOW must be 8, 16, 32, or 64" >&2; exit 2;; esac

ROOT=/data/home/yuanman/spring-mechanism-run/projectile-gravity-continuous-v2
REPO=/data/home/yuanman/physics-shortcuts-benchmarks
RAW_DIR="$ROOT/data/raw-v3"
LATENT_DIR="$ROOT/data/latents/train_short_v3_large_hist$WINDOW"
RUN="short-v3-large-hist$WINDOW"

test -d "$RAW_DIR/train"
test ! -e "$LATENT_DIR"
mkdir -p "$ROOT/data/latents" "$ROOT/logs" "$ROOT/training/$RUN/resolved" "$ROOT/checkpoints/$RUN" "$ROOT/output/$RUN" "$REPO/data"
cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$REPO/src:$REPO/lib/diffsynth"

python3 "$ROOT/scripts/prepare_window_latents.py" \
  --source "$RAW_DIR/train" --out "$LATENT_DIR" --data-config "$ROOT/config/data-v3.yaml" \
  --window "$WINDOW" --allow-window-override --batch-size 4 --device cuda

test "$(find "$LATENT_DIR" -maxdepth 1 -name '*.pt' | wc -l)" -eq 2048
test -s "$LATENT_DIR/metadata.csv"
ln -sfn "$ROOT/data" "$REPO/data/projectile_gravity_v3"
python3 "$ROOT/scripts/train_projectile.py" \
  --config "$ROOT/config/Train-short-v3-large-hist$WINDOW.yaml" \
  --resolved-dir "$ROOT/training/$RUN/resolved" \
  --ckpt-dir "$ROOT/checkpoints/$RUN" --out-dir "$ROOT/output/$RUN" --no-wandb
