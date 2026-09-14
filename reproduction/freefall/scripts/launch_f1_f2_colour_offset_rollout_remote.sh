#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/home/yuanman/spring-mechanism-run/projectile-gravity-continuous-v2
RUN=hist32-step100000-block2-f1f2-colouroffset-rollout
LOG="$ROOT/results/$RUN/logs"
mkdir -p "$LOG"

for gpu in 0 1 2 3; do
  bash "$ROOT/scripts/run_f1_f2_colour_offset_rollout_remote.sh" "$gpu" "$gpu" train >"$LOG/train-shard$gpu.log" 2>&1 &
done
for shard in 0 1 2 3; do
  gpu=$((shard + 4))
  bash "$ROOT/scripts/run_f1_f2_colour_offset_rollout_remote.sh" "$shard" "$gpu" holdout >"$LOG/holdout-shard$shard.log" 2>&1 &
done
wait
