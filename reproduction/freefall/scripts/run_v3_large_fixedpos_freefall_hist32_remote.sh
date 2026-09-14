#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/home/yuanman/spring-mechanism-run/projectile-gravity-continuous-v2
REPO=/data/home/yuanman/physics-shortcuts-benchmarks
RAW_DIR="$ROOT/data/raw-v3-fixedpos-freefall-2048"
LATENT_DIR="$ROOT/data/latents/train_short_v3_large_fixedpos_freefall_hist32"
RUN=short-v3-large-fixedpos-freefall-hist32
CONFIG="$ROOT/config/data-v3-fixedpos-freefall.yaml"
TRAIN_GPU=5
ENCODE_GPUS=(0 2 3 4)

test ! -e "$RAW_DIR"
test ! -e "$LATENT_DIR"
mkdir -p "$ROOT/data" "$ROOT/data/latents" "$ROOT/logs" "$ROOT/training/$RUN/resolved" "$ROOT/checkpoints/$RUN" "$ROOT/output/$RUN" "$REPO/data"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/lib/diffsynth"

python3 "$ROOT/scripts/build_dataset_v2.py" --config "$CONFIG" --out "$RAW_DIR" --split train --base-seeds 1024 > "$ROOT/logs/$RUN-build-train.log" 2>&1 &
train_build_pid=$!
python3 "$ROOT/scripts/build_dataset_v2.py" --config "$CONFIG" --out "$RAW_DIR" --split eval --base-seeds 128 --seed-offset 50000 > "$ROOT/logs/$RUN-build-eval.log" 2>&1 &
eval_build_pid=$!
wait "$train_build_pid"

for shard in 0 1 2 3; do
  gpu=${ENCODE_GPUS[$shard]}
  CUDA_VISIBLE_DEVICES="$gpu" python3 "$ROOT/scripts/prepare_window_latents.py" \
    --source "$RAW_DIR/train" --out "$LATENT_DIR" --data-config "$CONFIG" \
    --window 32 --batch-size 4 --device cuda --num-shards 4 --shard-index "$shard" --no-metadata \
    > "$ROOT/logs/$RUN-encode-shard$shard.log" 2>&1 &
  encode_pids[$shard]=$!
done
for shard in 0 1 2 3; do wait "${encode_pids[$shard]}"; done

python3 "$ROOT/scripts/prepare_window_latents.py" --source "$RAW_DIR/train" --out "$LATENT_DIR" --data-config "$CONFIG" --window 32 --metadata-only
test "$(find "$LATENT_DIR" -maxdepth 1 -name '*.pt' | wc -l)" -eq 2048
test -s "$LATENT_DIR/metadata.csv"
ln -sfn "$ROOT/data" "$REPO/data/projectile_gravity_v3_fixedpos_freefall"

CUDA_VISIBLE_DEVICES="$TRAIN_GPU" python3 "$ROOT/scripts/train_projectile.py" \
  --config "$ROOT/config/Train-short-v3-large-fixedpos-freefall-hist32.yaml" \
  --resolved-dir "$ROOT/training/$RUN/resolved" --ckpt-dir "$ROOT/checkpoints/$RUN" \
  --out-dir "$ROOT/output/$RUN" --no-wandb

wait "$eval_build_pid"
