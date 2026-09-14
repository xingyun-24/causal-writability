#!/usr/bin/env bash
set -euo pipefail

project_root="/data/home/yangleqian/workspace/physics-shortcuts-benchmarks"
python_bin="/data/home/yangleqian/venvs/physics-shortcuts-benchmarks/bin/python"
config="configs/pendulum/frequency_color_circle_frequency_scan.yaml"
dataset_base="data/pendulum/frequency_color_circle_frequency_scan_11x13"
run_root="runs/pendulum/frequency_color_circle_frequency_scan_11x13"
analysis_root="${run_root}/analysis"
triptych_root="${run_root}/triptych_videos"
training_run="runs/pendulum/frequency_color_circle__train_5932a18b5983"

cd "${project_root}"
mkdir -p "${run_root}/logs"

exec 9>"${run_root}/.launch.lock"
if ! flock -n 9; then
  echo "$(date --iso-8601=seconds) scan launcher is already running" >&2
  exit 9
fi

echo "$(date --iso-8601=seconds) generating/resuming circle scan dataset"
env PYTHONPATH=src "${python_bin}" -u -m \
  sshv2.experiments.pendulum.frequency_color_frequency_scan_data \
  --config "${config}" \
  --shape circle \
  --output-root "${dataset_base}/circle" \
  > "${run_root}/logs/dataset_circle.log" 2>&1
echo "$(date --iso-8601=seconds) circle scan dataset complete"

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
          awk -v target="${uuid}" '$0 == target {count++} END {print count+0}' \
            <<<"${app_uuids}"
        )"
        if [[ "${index}" != "0" && "${volatile_ecc}" == "0" ]] \
          && (( memory_used < 512 && utilization < 5 && app_count == 0 )); then
          printf '%s\n' "${index}"
        fi
      done
}

prediction_complete() {
  local history="$1"
  local progress="${run_root}/circle/${history}/prediction/progress.json"
  [[ -f "${progress}" ]] || return 1
  "${python_bin}" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); raise SystemExit(0 if p.get("status")=="complete" and int(p.get("completed_predictions",-1))==2288 else 1)' \
    "${progress}"
}

run_prediction() {
  local history="$1"
  local gpu="$2"
  local training_config checkpoint output_root
  training_config="configs/pendulum/Train-frequency_color_circle-50k-${history}.yaml"
  checkpoint="${training_run}/${history}/ckpt/frequency_color_circle-50k-${history}/step-50000.safetensors"
  output_root="${run_root}/circle/${history}/prediction"
  env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="src:lib/diffsynth" \
    "${python_bin}" -u -m \
    sshv2.experiments.pendulum.frequency_color_frequency_scan_runner \
    predict \
    --dataset-root "${dataset_base}/circle" \
    --experiment-config "${config}" \
    --training-config "${training_config}" \
    --checkpoint "${checkpoint}" \
    --history "${history}" \
    --output-root "${output_root}" \
    --device cuda \
    --steps 20 \
    --seed-offset 23000000
}

prediction_jobs=()
for history in short long; do
  if prediction_complete "${history}"; then
    echo "$(date --iso-8601=seconds) prediction already complete history=${history}"
  else
    prediction_jobs+=("${history}")
  fi
done

declare -A running_pid=()
declare -A running_gpu=()
next_job=0
while (( next_job < ${#prediction_jobs[@]} || ${#running_pid[@]} > 0 )); do
  for history in "${!running_pid[@]}"; do
    pid="${running_pid[${history}]}"
    if ! kill -0 "${pid}" 2>/dev/null; then
      set +e
      wait "${pid}"
      status=$?
      set -e
      gpu="${running_gpu[${history}]}"
      unset 'running_pid['"${history}"']'
      unset 'running_gpu['"${history}"']'
      if (( status != 0 )); then
        echo "$(date --iso-8601=seconds) prediction failed history=${history} gpu=${gpu} status=${status}" >&2
        exit 5
      fi
      echo "$(date --iso-8601=seconds) prediction complete history=${history} gpu=${gpu}"
    fi
  done

  while IFS= read -r gpu; do
    [[ -n "${gpu}" ]] || continue
    assigned=false
    for history in "${!running_gpu[@]}"; do
      if [[ "${running_gpu[${history}]}" == "${gpu}" ]]; then
        assigned=true
        break
      fi
    done
    if [[ "${assigned}" == true ]] || (( next_job >= ${#prediction_jobs[@]} )); then
      continue
    fi
    history="${prediction_jobs[${next_job}]}"
    echo "$(date --iso-8601=seconds) starting prediction history=${history} gpu=${gpu}"
    run_prediction "${history}" "${gpu}" \
      > "${run_root}/logs/circle_${history}_prediction.log" 2>&1 &
    running_pid["${history}"]=$!
    running_gpu["${history}"]="${gpu}"
    next_job=$((next_job + 1))
  done < <(healthy_free_gpus)

  if (( next_job < ${#prediction_jobs[@]} || ${#running_pid[@]} > 0 )); then
    sleep 30
  fi
done

echo "$(date --iso-8601=seconds) all predictions complete; starting metrics, plots, and triptychs"
env PYTHONPATH=src "${python_bin}" -u -m \
  sshv2.experiments.pendulum.frequency_color_frequency_scan_pipeline \
  --run-root "${run_root}" \
  --dataset-base "${dataset_base}" \
  --config "${config}" \
  --analysis-root "${analysis_root}" \
  --triptych-root "${triptych_root}" \
  --poll-seconds 30 \
  > "${run_root}/logs/pipeline.log" 2>&1

printf '%s\n' complete > "${run_root}/EXPERIMENT_COMPLETE"
echo "$(date --iso-8601=seconds) frequency_color_circle 11x13 scan complete"
