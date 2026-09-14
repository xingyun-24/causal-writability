#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/home/yuanman/spring-mechanism-run/projectile-gravity-continuous-v2
REPO=/data/home/yuanman/physics-shortcuts-benchmarks
RAW_DIR="$ROOT/data/raw-v3-fixedpos-freefall-2048/train"
CONFIG="$ROOT/config/data-v3-fixedpos-freefall.yaml"
RUNS=(short-v3-large-fixedpos-freefall-hist8 short-v3-large-fixedpos-freefall-hist16)
WINDOWS=(8 16)
ENCODE_GPUS=(0 1 2 3 4 6 7 0)
TRAIN_GPUS=(0 1)

test "$(find "$RAW_DIR/videos" -maxdepth 1 -name '*.mp4' | wc -l)" -eq 2048
mkdir -p "$ROOT/logs" "$ROOT/data/latents" "$REPO/data"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/lib/diffsynth"

for setting in 0 1; do
  window=${WINDOWS[$setting]}
  run=${RUNS[$setting]}
  latent="$ROOT/data/latents/train_short_v3_large_fixedpos_freefall_hist$window"
  test ! -e "$latent"
  mkdir -p "$latent"
  for shard in 0 1 2 3; do
    gpu=${ENCODE_GPUS[$((setting * 4 + shard))]}
    CUDA_VISIBLE_DEVICES="$gpu" python3 "$ROOT/scripts/prepare_window_latents.py" \
      --source "$RAW_DIR" --out "$latent" --data-config "$CONFIG" --window "$window" \
      --allow-window-override --batch-size 4 --device cuda --num-shards 4 --shard-index "$shard" --no-metadata \
      > "$ROOT/logs/$run-encode-shard$shard.log" 2>&1 &
    encode_pids[$((setting * 4 + shard))]=$!
  done
done
for index in 0 1 2 3 4 5 6 7; do wait "${encode_pids[$index]}"; done

for setting in 0 1; do
  window=${WINDOWS[$setting]}
  run=${RUNS[$setting]}
  latent="$ROOT/data/latents/train_short_v3_large_fixedpos_freefall_hist$window"
  python3 "$ROOT/scripts/prepare_window_latents.py" --source "$RAW_DIR" --out "$latent" --data-config "$CONFIG" --window "$window" --allow-window-override --metadata-only
  test "$(find "$latent" -maxdepth 1 -name '*.pt' | wc -l)" -eq 2048
  test -s "$latent/metadata.csv"
  mkdir -p "$ROOT/training/$run/resolved" "$ROOT/checkpoints/$run" "$ROOT/output/$run"
  gpu=${TRAIN_GPUS[$setting]}
  CUDA_VISIBLE_DEVICES="$gpu" nohup python3 "$ROOT/scripts/train_projectile.py" \
    --config "$ROOT/config/Train-short-v3-large-fixedpos-freefall-hist$window.yaml" \
    --resolved-dir "$ROOT/training/$run/resolved" --ckpt-dir "$ROOT/checkpoints/$run" \
    --out-dir "$ROOT/output/$run" --no-wandb > "$ROOT/logs/$run.log" 2>&1 &
  echo "started $run gpu=$gpu pid=$!"
done
