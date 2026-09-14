#!/usr/bin/env bash
set -euo pipefail

SHARD=${1:?usage: launch_f1_f2_colour_additive_pca8_shard_remote.sh SHARD GPU SPLIT}
GPU=${2:?usage: launch_f1_f2_colour_additive_pca8_shard_remote.sh SHARD GPU SPLIT}
SPLIT=${3:?usage: launch_f1_f2_colour_additive_pca8_shard_remote.sh SHARD GPU SPLIT}
ROOT=/data/home/yuanman/spring-mechanism-run/projectile-gravity-continuous-v2
RUN=hist32-step100000-block2-f1f2-colouradditive-pca8-rollout
FIT_JSON="$ROOT/results/hist32-step100000-block2-fulltoken-pca-fp32/decomposition/f1_f2_colour_additive_pca8_fit.json"
LOG="$ROOT/results/$RUN/logs"
mkdir -p "$LOG"
exec env RUN="$RUN" FIT_JSON="$FIT_JSON" NUM_SHARDS=2 \
  bash "$ROOT/scripts/run_f1_f2_colour_offset_rollout_remote.sh" "$SHARD" "$GPU" "$SPLIT" \
  >"$LOG/$SPLIT-shard$SHARD.log" 2>&1
