#!/usr/bin/env bash
# Generate the V3 raw training videos only. It does not start a training run.
set -euo pipefail

ROOT=/data/home/yuanman/spring-mechanism-run/projectile-gravity-continuous-v2
REPO=/data/home/yuanman/physics-shortcuts-benchmarks
RAW_DIR="$ROOT/data/raw-v3"

test ! -e "$RAW_DIR/train"
mkdir -p "$ROOT/data" "$ROOT/logs"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/lib/diffsynth"

python3 "$ROOT/scripts/build_dataset_v2.py" \
  --config "$ROOT/config/data-v3.yaml" \
  --out "$RAW_DIR" \
  --split train \
  --base-seeds 1024
