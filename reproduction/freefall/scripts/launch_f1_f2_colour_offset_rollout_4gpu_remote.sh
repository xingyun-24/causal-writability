#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/home/yuanman/spring-mechanism-run/projectile-gravity-continuous-v2
RUN=hist32-step100000-block2-f1f2-colouroffset-rollout
LOG="$ROOT/results/$RUN/logs"
mkdir -p "$LOG"

NUM_SHARDS=2 bash "$ROOT/scripts/run_f1_f2_colour_offset_rollout_remote.sh" 0 4 train >"$LOG/train-shard0.log" 2>&1 &
NUM_SHARDS=2 bash "$ROOT/scripts/run_f1_f2_colour_offset_rollout_remote.sh" 1 5 train >"$LOG/train-shard1.log" 2>&1 &
NUM_SHARDS=2 bash "$ROOT/scripts/run_f1_f2_colour_offset_rollout_remote.sh" 0 6 holdout >"$LOG/holdout-shard0.log" 2>&1 &
NUM_SHARDS=2 bash "$ROOT/scripts/run_f1_f2_colour_offset_rollout_remote.sh" 1 7 holdout >"$LOG/holdout-shard1.log" 2>&1 &
wait
