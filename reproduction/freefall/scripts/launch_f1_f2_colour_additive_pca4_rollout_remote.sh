#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/home/yuanman/spring-mechanism-run/projectile-gravity-continuous-v2
RUN=hist32-step100000-block2-f1f2-colouradditive-pca4-rollout
FIT_JSON="$ROOT/results/hist32-step100000-block2-fulltoken-pca-fp32/decomposition/f1_f2_colour_additive_pca4_fit.json"
LOG="$ROOT/results/$RUN/logs"
mkdir -p "$LOG"

RUN="$RUN" FIT_JSON="$FIT_JSON" NUM_SHARDS=2 bash "$ROOT/scripts/run_f1_f2_colour_offset_rollout_remote.sh" 0 1 train >"$LOG/train-shard0.log" 2>&1 &
RUN="$RUN" FIT_JSON="$FIT_JSON" NUM_SHARDS=2 bash "$ROOT/scripts/run_f1_f2_colour_offset_rollout_remote.sh" 1 2 train >"$LOG/train-shard1.log" 2>&1 &
RUN="$RUN" FIT_JSON="$FIT_JSON" NUM_SHARDS=2 bash "$ROOT/scripts/run_f1_f2_colour_offset_rollout_remote.sh" 0 3 holdout >"$LOG/holdout-shard0.log" 2>&1 &
RUN="$RUN" FIT_JSON="$FIT_JSON" NUM_SHARDS=2 bash "$ROOT/scripts/run_f1_f2_colour_offset_rollout_remote.sh" 1 4 holdout >"$LOG/holdout-shard1.log" 2>&1 &
wait
