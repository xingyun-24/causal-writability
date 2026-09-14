#!/usr/bin/env bash
set -euo pipefail

project_root="/data/home/yangleqian/workspace/physics-shortcuts-benchmarks"
python_bin="/data/home/yangleqian/venvs/physics-shortcuts-benchmarks/bin/python"
config="configs/pendulum/frequency_color_circle_frequency_scan.yaml"
dataset_base="data/pendulum/frequency_color_circle_frequency_scan_11x13"
run_root="runs/pendulum/frequency_color_circle_frequency_scan_11x13"
training_run="runs/pendulum/frequency_color_circle__train_5932a18b5983"
analysis_root="${run_root}/analysis"
triptych_root="${run_root}/triptych_videos"
shards_per_history=6
threads_per_shard=8
steps=20
seed_offset=23000000

cd "${project_root}"
mkdir -p "${run_root}/logs" "${run_root}/cpu_shards"

exec 9>"${run_root}/.cpu_launch.lock"
if ! flock -n 9; then
  echo "$(date --iso-8601=seconds) CPU scan launcher is already running" >&2
  exit 9
fi

write_launcher_state() {
  local status="$1"
  local stage="$2"
  local detail="${3:-}"
  "${python_bin}" -c \
    'import json,os,sys,tempfile; p=sys.argv[1]; value={"status":sys.argv[2],"stage":sys.argv[3],"detail":sys.argv[4]}; os.makedirs(os.path.dirname(p),exist_ok=True); fd,t=tempfile.mkstemp(dir=os.path.dirname(p),prefix=".cpu_launcher_",suffix=".tmp"); f=os.fdopen(fd,"w"); json.dump(value,f,indent=2,sort_keys=True); f.write("\n"); f.close(); os.replace(t,p)' \
    "${run_root}/cpu_launcher_progress.json" \
    "${status}" \
    "${stage}" \
    "${detail}"
}

shard_complete() {
  local history="$1"
  local shard_index="$2"
  local progress
  progress="${run_root}/cpu_shards/${history}/shard_$(printf '%02d' "${shard_index}")/progress.json"
  [[ -f "${progress}" ]] || return 1
  "${python_bin}" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); raise SystemExit(0 if p.get("status")=="complete" else 1)' \
    "${progress}"
}

echo "$(date --iso-8601=seconds) auditing CPU shard coverage"
write_launcher_state running audit "12 disjoint workers; 6 short and 6 long"
env PYTHONPATH=src "${python_bin}" -u -m \
  sshv2.experiments.pendulum.frequency_color_frequency_scan_cpu \
  audit \
  --dataset-root "${dataset_base}/circle" \
  --experiment-config "${config}" \
  --num-shards "${shards_per_history}" \
  > "${run_root}/logs/cpu_shard_audit.log" 2>&1

declare -A worker_pid=()
worker_count=0
for history_index in 0 1; do
  if (( history_index == 0 )); then
    history="short"
    cpu_base=0
  else
    history="long"
    cpu_base=48
  fi
  training_config="configs/pendulum/Train-frequency_color_circle-50k-${history}.yaml"
  checkpoint="${training_run}/${history}/ckpt/frequency_color_circle-50k-${history}/step-50000.safetensors"
  for ((shard_index=0; shard_index<shards_per_history; shard_index++)); do
    if shard_complete "${history}" "${shard_index}"; then
      echo "$(date --iso-8601=seconds) shard already complete history=${history} shard=${shard_index}"
      continue
    fi
    cpu_start=$((cpu_base + shard_index * threads_per_shard))
    cpu_end=$((cpu_start + threads_per_shard - 1))
    shard_label="$(printf '%02d' "${shard_index}")"
    shard_root="${run_root}/cpu_shards/${history}/shard_${shard_label}"
    log="${run_root}/logs/cpu_${history}_shard_${shard_label}.log"
    mkdir -p "${shard_root}"
    echo "$(date --iso-8601=seconds) starting CPU shard history=${history} shard=${shard_index} cores=${cpu_start}-${cpu_end}"
    env \
      OMP_NUM_THREADS="${threads_per_shard}" \
      MKL_NUM_THREADS="${threads_per_shard}" \
      OPENBLAS_NUM_THREADS="${threads_per_shard}" \
      PYTHONPATH="src:lib/diffsynth" \
      taskset -c "${cpu_start}-${cpu_end}" \
      "${python_bin}" -u -m \
      sshv2.experiments.pendulum.frequency_color_frequency_scan_cpu \
      predict-shard \
      --dataset-root "${dataset_base}/circle" \
      --experiment-config "${config}" \
      --training-config "${training_config}" \
      --checkpoint "${checkpoint}" \
      --history "${history}" \
      --output-root "${shard_root}" \
      --device cpu \
      --steps "${steps}" \
      --seed-offset "${seed_offset}" \
      --num-shards "${shards_per_history}" \
      --shard-index "${shard_index}" \
      > "${log}" 2>&1 &
    worker_pid["${history}_${shard_index}"]=$!
    worker_count=$((worker_count + 1))
  done
done

write_launcher_state running predictions "${worker_count} CPU shard processes started"
failed=0
for worker in "${!worker_pid[@]}"; do
  pid="${worker_pid[${worker}]}"
  if wait "${pid}"; then
    echo "$(date --iso-8601=seconds) CPU shard complete worker=${worker}"
  else
    status=$?
    echo "$(date --iso-8601=seconds) CPU shard failed worker=${worker} status=${status}" >&2
    failed=1
  fi
done
if (( failed != 0 )); then
  write_launcher_state failed predictions "inspect cpu_*_shard_*.log"
  exit 5
fi

echo "$(date --iso-8601=seconds) all CPU shards complete; merging canonical predictions"
write_launcher_state running merge "hard-linking validated disjoint shard outputs"
for history in short long; do
  training_config="configs/pendulum/Train-frequency_color_circle-50k-${history}.yaml"
  checkpoint="${training_run}/${history}/ckpt/frequency_color_circle-50k-${history}/step-50000.safetensors"
  env PYTHONPATH=src "${python_bin}" -u -m \
    sshv2.experiments.pendulum.frequency_color_frequency_scan_cpu \
    merge \
    --dataset-root "${dataset_base}/circle" \
    --experiment-config "${config}" \
    --training-config "${training_config}" \
    --checkpoint "${checkpoint}" \
    --history "${history}" \
    --shards-root "${run_root}/cpu_shards/${history}" \
    --output-root "${run_root}/circle/${history}/prediction" \
    --steps "${steps}" \
    --seed-offset "${seed_offset}" \
    --num-shards "${shards_per_history}" \
    > "${run_root}/logs/cpu_${history}_merge.log" 2>&1
  echo "$(date --iso-8601=seconds) merge complete history=${history}"
done

echo "$(date --iso-8601=seconds) starting metrics, plots, and triptychs"
write_launcher_state running pipeline "metrics, color, ten plots, and 143 triptychs"
env PYTHONPATH=src "${python_bin}" -u -m \
  sshv2.experiments.pendulum.frequency_color_frequency_scan_pipeline \
  --run-root "${run_root}" \
  --dataset-base "${dataset_base}" \
  --config "${config}" \
  --analysis-root "${analysis_root}" \
  --triptych-root "${triptych_root}" \
  --poll-seconds 30 \
  > "${run_root}/logs/cpu_pipeline.log" 2>&1

printf '%s\n' complete > "${run_root}/EXPERIMENT_COMPLETE"
write_launcher_state complete all "frequency_color_circle 11x13 CPU scan complete"
echo "$(date --iso-8601=seconds) frequency_color_circle 11x13 CPU scan complete"
