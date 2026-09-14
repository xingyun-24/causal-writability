#!/usr/bin/env bash
set -euo pipefail

SHARD=${1:?usage: queue_recovery_hist32_parallel_remote.sh SHARD GPU}
GPU=${2:?usage: queue_recovery_hist32_parallel_remote.sh SHARD GPU}
ROOT=/data/home/yuanman/spring-mechanism-run/projectile-gravity-continuous-v2
MIN_FREE_MB=${MIN_FREE_MB:-50000}
while true; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -dc '0-9')
  TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU" | tr -dc '0-9')
  FREE=$((TOTAL - USED))
  if [ "$FREE" -ge "$MIN_FREE_MB" ]; then
    exec bash "$ROOT/scripts/run_recovery_hist32_parallel_remote.sh" "$SHARD" "$GPU"
  fi
  printf 'shard=%s gpu=%s free_mb=%s waiting\n' "$SHARD" "$GPU" "$FREE"
  sleep 60
done
