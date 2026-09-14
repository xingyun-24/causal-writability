#!/usr/bin/env bash
set -euo pipefail

project_root="/data/home/yangleqian/workspace/physics-shortcuts-benchmarks"
python_bin="/data/home/yangleqian/venvs/physics-shortcuts-benchmarks/bin/python"
config="configs/pendulum/frequency_color_circle_frequency_scan.yaml"
dataset_base="data/pendulum/frequency_color_circle_frequency_scan_11x13"
run_root="runs/pendulum/frequency_color_circle_frequency_scan_11x13"
training_run="runs/pendulum/frequency_color_circle__train_5932a18b5983"
gpu_root="${run_root}/gpu_acceleration"
analysis_root="${run_root}/analysis"
triptych_root="${run_root}/triptych_videos"
expected_predictions=2288
steps=20
seed_offset=23000000

cd "${project_root}"
mkdir -p "${run_root}/logs" "${gpu_root}"

exec 9>"${run_root}/.hybrid_gpu.lock"
if ! flock -n 9; then
  echo "$(date --iso-8601=seconds) hybrid GPU monitor is already running" >&2
  exit 9
fi

write_state() {
  local status="$1"
  local stage="$2"
  local detail="${3:-}"
  "${python_bin}" -c \
    'import json,os,sys,tempfile; p=sys.argv[1]; value={"status":sys.argv[2],"stage":sys.argv[3],"detail":sys.argv[4]}; os.makedirs(os.path.dirname(p),exist_ok=True); fd,t=tempfile.mkstemp(dir=os.path.dirname(p),prefix=".hybrid_gpu_",suffix=".tmp"); f=os.fdopen(fd,"w"); json.dump(value,f,indent=2,sort_keys=True); f.write("\n"); f.close(); os.replace(t,p)' \
    "${run_root}/hybrid_gpu_progress.json" \
    "${status}" \
    "${stage}" \
    "${detail}"
}

prediction_complete() {
  local history="$1"
  local progress="${gpu_root}/${history}/prediction/progress.json"
  [[ -f "${progress}" ]] || return 1
  "${python_bin}" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); raise SystemExit(0 if p.get("status")=="complete" and int(p.get("completed_predictions",-1))==int(sys.argv[2]) else 1)' \
    "${progress}" "${expected_predictions}"
}

canonical_complete() {
  local history="$1"
  local progress="${run_root}/circle/${history}/prediction/progress.json"
  [[ -f "${progress}" ]] || return 1
  "${python_bin}" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); raise SystemExit(0 if p.get("status")=="complete" and int(p.get("completed_predictions",-1))==int(sys.argv[2]) else 1)' \
    "${progress}" "${expected_predictions}"
}

history_running() {
  local history="$1"
  ps -eo comm=,args= | awk \
    -v history="--history ${history}" \
    -v output="${gpu_root}/${history}/prediction" \
    '$1 ~ /^python/ && index($0,"frequency_color_frequency_scan_runner predict") && index($0,history) && index($0,output) {found=1} END {exit(found ? 0 : 1)}'
}

required_resume_gpu() {
  local history="$1"
  local state="${gpu_root}/${history}/prediction/run_state.json"
  if [[ -f "${state}" ]]; then
    "${python_bin}" -c \
      'import json,sys; print(json.load(open(sys.argv[1])).get("cuda_visible_devices",""))' \
      "${state}"
  fi
}

healthy_free_gpus() {
  local app_uuids index uuid memory_used utilization volatile_ecc app_count
  app_uuids="$(
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits \
      2>/dev/null || true
  )"
  nvidia-smi \
    --query-gpu=index,uuid,memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total \
    --format=csv,noheader,nounits \
    | while IFS=',' read -r index uuid memory_used utilization volatile_ecc; do
        index="$(xargs <<<"${index}")"
        uuid="$(xargs <<<"${uuid}")"
        memory_used="$(xargs <<<"${memory_used}")"
        utilization="$(xargs <<<"${utilization}")"
        volatile_ecc="$(xargs <<<"${volatile_ecc}")"
        app_count="$(
          awk -v target="${uuid}" \
            '$0 == target {count++} END {print count+0}' \
            <<<"${app_uuids}"
        )"
        if [[ "${index}" != "0" && "${volatile_ecc}" == "0" ]] \
          && (( memory_used < 512 && utilization < 5 && app_count == 0 )); then
          printf '%s\n' "${index}"
        fi
      done
}

stop_cpu_history() {
  local history="$1"
  pkill -TERM -f \
    "^[^ ]*python[^ ]* -u -m sshv2.experiments.pendulum.frequency_color_frequency_scan_cpu predict-shard .*--history ${history} .*cpu_shards/${history}" \
    2>/dev/null || true
}

start_prediction() {
  local history="$1"
  local gpu="$2"
  local resume_gpu training_config checkpoint output log
  resume_gpu="$(required_resume_gpu "${history}")"
  if [[ -n "${resume_gpu}" && "${resume_gpu}" != "${gpu}" ]]; then
    return 2
  fi
  training_config="configs/pendulum/Train-frequency_color_circle-50k-${history}.yaml"
  checkpoint="${training_run}/${history}/ckpt/frequency_color_circle-50k-${history}/step-50000.safetensors"
  output="${gpu_root}/${history}/prediction"
  log="${run_root}/logs/gpu_acceleration_${history}.log"
  mkdir -p "${output}"
  echo "$(date --iso-8601=seconds) starting/resuming GPU history=${history} gpu=${gpu}"
  stop_cpu_history "${history}"
  env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="src:lib/diffsynth" \
    "${python_bin}" -u -m \
    sshv2.experiments.pendulum.frequency_color_frequency_scan_runner \
    predict \
    --dataset-root "${dataset_base}/circle" \
    --experiment-config "${config}" \
    --training-config "${training_config}" \
    --checkpoint "${checkpoint}" \
    --history "${history}" \
    --output-root "${output}" \
    --device cuda \
    --steps "${steps}" \
    --seed-offset "${seed_offset}" \
    >> "${log}" 2>&1 &
}

echo "$(date --iso-8601=seconds) hybrid GPU monitor active"
write_state running monitor "waiting for healthy zero-ECC GPUs; GPU 0 excluded"
while ! prediction_complete short || ! prediction_complete long; do
  if canonical_complete short && canonical_complete long; then
    write_state complete canonical "CPU canonical predictions completed first"
    exit 0
  fi

  mapfile -t free_gpus < <(healthy_free_gpus)
  for history in short long; do
    if prediction_complete "${history}" || history_running "${history}"; then
      continue
    fi
    for gpu in "${free_gpus[@]}"; do
      [[ -n "${gpu}" ]] || continue
      if start_prediction "${history}" "${gpu}"; then
        free_gpus=("${free_gpus[@]/${gpu}/}")
        break
      fi
    done
  done

  short_count=0
  long_count=0
  for history in short long; do
    progress="${gpu_root}/${history}/prediction/progress.json"
    if [[ -f "${progress}" ]]; then
      count="$("${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("completed_predictions",0))' "${progress}" 2>/dev/null || printf 0)"
      if [[ "${history}" == short ]]; then short_count="${count}"; else long_count="${count}"; fi
    fi
  done
  write_state running predictions "short=${short_count}/${expected_predictions}; long=${long_count}/${expected_predictions}"
  sleep 30
done

echo "$(date --iso-8601=seconds) both GPU histories complete; stopping CPU fallbacks"
write_state running takeover "both isolated GPU outputs complete"
stop_cpu_history short
stop_cpu_history long
pkill -TERM -f \
  '^bash scripts/sshv2/run_pendulum_frequency_color_circle_frequency_scan_50k_cpu.sh$' \
  2>/dev/null || true
sleep 5

for history in short long; do
  training_config="configs/pendulum/Train-frequency_color_circle-50k-${history}.yaml"
  checkpoint="${training_run}/${history}/ckpt/frequency_color_circle-50k-${history}/step-50000.safetensors"
  env PYTHONPATH=src "${python_bin}" -u -m \
    sshv2.experiments.pendulum.frequency_color_frequency_scan_cpu \
    promote \
    --dataset-root "${dataset_base}/circle" \
    --experiment-config "${config}" \
    --source-root "${gpu_root}/${history}/prediction" \
    --output-root "${run_root}/circle/${history}/prediction" \
    --checkpoint "${checkpoint}" \
    --history "${history}" \
    --steps "${steps}" \
    --seed-offset "${seed_offset}" \
    > "${run_root}/logs/gpu_${history}_promotion.log" 2>&1
  echo "$(date --iso-8601=seconds) promoted GPU history=${history}"
done

echo "$(date --iso-8601=seconds) starting metrics, plots, and triptychs"
write_state running pipeline "metrics, color, ten plots, and 143 triptychs"
env PYTHONPATH=src "${python_bin}" -u -m \
  sshv2.experiments.pendulum.frequency_color_frequency_scan_pipeline \
  --run-root "${run_root}" \
  --dataset-base "${dataset_base}" \
  --config "${config}" \
  --analysis-root "${analysis_root}" \
  --triptych-root "${triptych_root}" \
  --poll-seconds 30 \
  > "${run_root}/logs/gpu_pipeline.log" 2>&1

printf '%s\n' complete > "${run_root}/EXPERIMENT_COMPLETE"
write_state complete all "frequency_color_circle 11x13 hybrid GPU scan complete"
echo "$(date --iso-8601=seconds) frequency_color_circle 11x13 hybrid GPU scan complete"
