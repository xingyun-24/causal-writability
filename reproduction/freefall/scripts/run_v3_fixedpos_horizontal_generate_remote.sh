#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/home/yuanman/spring-mechanism-run/projectile-gravity-continuous-v2
REPO=/data/home/yuanman/physics-shortcuts-benchmarks
OUT="$ROOT/data/raw-v3-fixedpos-horizontal-2048"
test ! -e "$OUT/train"
mkdir -p "$ROOT/data" "$ROOT/logs"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/lib/diffsynth"
python3 "$ROOT/scripts/build_dataset_v2.py" --config "$ROOT/config/data-v3-fixedpos-horizontal.yaml" --out "$OUT" --split train --base-seeds 1024
