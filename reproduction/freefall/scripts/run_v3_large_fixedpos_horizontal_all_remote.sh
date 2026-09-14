#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/home/yuanman/spring-mechanism-run/projectile-gravity-continuous-v2
RAW="$ROOT/data/raw-v3-fixedpos-horizontal-2048/train"
while [ ! -d "$RAW/videos" ] || [ "$(find "$RAW/videos" -maxdepth 1 -type f -name '*.mp4' | wc -l)" -lt 2048 ]; do
  sleep 30
done
mkdir -p "$ROOT/logs"
for spec in '8 1' '16 2' '32 3' '64 4'; do
  set -- $spec
  W=$1; G=$2
  nohup bash "$ROOT/scripts/run_v3_large_fixedpos_horizontal_history_remote.sh" "$W" "$G" > "$ROOT/logs/v3-large-fixedpos-horizontal-hist$W.log" 2>&1 &
  echo "started hist$W pid=$! gpu=$G"
done
