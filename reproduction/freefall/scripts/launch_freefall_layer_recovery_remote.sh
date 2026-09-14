#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/home/yuanman/spring-mechanism-run/projectile-gravity-continuous-v2
RUN=freefall-hist32-step100000-recovery95-e3-par8
LOG="$ROOT/results/$RUN/logs"
mkdir -p "$LOG"
for shard in 0 1 2 3 4 5; do
  nohup bash "$ROOT/scripts/run_freefall_layer_recovery_remote.sh" "$shard" "$shard" >"$LOG/shard$shard.log" 2>&1 &
done
nohup bash "$ROOT/scripts/run_freefall_layer_recovery_remote.sh" 6 6 >"$LOG/shard6.log" 2>&1 &
wait
bash "$ROOT/scripts/run_freefall_layer_recovery_remote.sh" 7 6 >"$LOG/shard7.log" 2>&1
