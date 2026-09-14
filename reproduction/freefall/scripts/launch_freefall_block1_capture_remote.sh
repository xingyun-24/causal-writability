#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/home/yuanman/spring-mechanism-run/projectile-gravity-continuous-v2
RUN=freefall-hist32-step100000-block1-fulltoken-pca-fp32
LOG="$ROOT/results/$RUN/logs"
mkdir -p "$LOG"
for spec in "0 1" "1 2" "2 3" "3 4"; do
  set -- $spec
  nohup bash "$ROOT/scripts/launch_freefall_fulltoken_capture_remote.sh" "$1" "$2" 1 >"$LOG/shard$1.log" 2>&1 &
done
wait
