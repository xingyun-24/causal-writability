#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/home/yuanman/spring-mechanism-run/projectile-gravity-continuous-v2
REPO=/data/home/yuanman/physics-shortcuts-benchmarks
RUN=hist16-step50000-recovery128
OUT="$ROOT/results/$RUN"
MANIFEST="$ROOT/results/hist16-step50000-recovery_pairs-128.json"
test -s "$MANIFEST"
test ! -e "$OUT"
mkdir -p "$OUT/manifests" "$ROOT/logs"
cd "$REPO"
python3 "$ROOT/scripts/split_recovery_manifest.py" --manifest "$MANIFEST" --out-dir "$OUT/manifests" --shards 3
for spec in '0 5' '1 6' '2 7'; do
  set -- $spec
  SHARD=$1; GPU=$2
  nohup bash "$ROOT/scripts/queue_recovery_worker_remote.sh" "$SHARD" "$GPU" > "$ROOT/logs/$RUN-shard$SHARD.log" 2>&1 &
  echo "queued shard=$SHARD gpu=$GPU pid=$!"
done
