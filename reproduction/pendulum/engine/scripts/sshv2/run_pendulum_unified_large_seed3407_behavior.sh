#!/usr/bin/env bash
set -euo pipefail

project_root="${1:-$(pwd)}"
python_bin="${PENDULUM_VENV_PYTHON:-python}"
run_root="runs/pendulum/unified_mechanism_v2/behavior/large_seed3407"
dataset_root="data/pendulum/frequency_color_sweeps"
calibration_root="runs/pendulum/frequency_color_evaluation"
seed_offset=23000000
steps=20

conditions=(
  "short:${PENDULUM_SHORT_GPU:-1}:configs/pendulum/Train-frequency_color_circle-large-short-50k.yaml:runs/pendulum/frequency_color_circle_large_short_long_50k_seed3407/short/ckpt/frequency-color-circle-large-short-50k/step-50000.safetensors:0dd907676adebe24fd44b1e8feae808a2c7864615ac965a742e422f0cd721392"
  "long:${PENDULUM_LONG_GPU:-2}:configs/pendulum/Train-frequency_color_circle-large-long-50k.yaml:runs/pendulum/frequency_color_circle_large_short_long_50k_seed3407/long/ckpt/frequency-color-circle-large-long-50k/step-50000.safetensors:b54a508499040bc0591f90f583a7d531d737d6237ae3ef5d6064df8a9b8ab423"
)

cd "$project_root"
mkdir -p "$run_root/controller" "$run_root/logs"
exec 9>"$run_root/controller/pipeline.lock"
if ! flock -n 9; then
  echo "Another unified Large seed3407 behavior controller is active." >&2
  exit 9
fi

progress_complete() {
  local path="$1" key="$2" expected="$3"
  [[ -f "$path" ]] || return 1
  "$python_bin" -c 'import json,sys; p=json.load(open(sys.argv[1])); raise SystemExit(0 if p.get("status")=="complete" and int(p.get(sys.argv[2],-1))==int(sys.argv[3]) else 1)' "$path" "$key" "$expected"
}

gpu_is_free() {
  local gpu="$1" uuid apps
  uuid="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader,nounits | xargs)"
  apps="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits 2>/dev/null | awk -v u="$uuid" '$0==u{n++}END{print n+0}')"
  [[ "$apps" == "0" ]]
}

run_history() {
  local history="$1" gpu="$2" config="$3" checkpoint="$4" expected_sha="$5"
  local history_root="$run_root/$history" actual_sha band module dataset output calibration
  mkdir -p "$history_root"
  actual_sha="$(sha256sum "$checkpoint" | awk '{print $1}')"
  [[ "$actual_sha" == "$expected_sha" ]] || { echo "checkpoint hash mismatch: $history" >&2; return 10; }
  printf '%s\n' "$actual_sha" > "$history_root/checkpoint.sha256"
  printf '%s\n' "$config" > "$history_root/training_config.txt"
  printf '%s\n' "$gpu" > "$history_root/gpu.txt"
  for band in low high; do
    module="sshv2.experiments.pendulum.frequency_color_${band}"
    dataset="$dataset_root/${band}_frequency_11color_64states"
    output="$history_root/${band}_frequency_11color_64states"
    calibration="$calibration_root/${band}_frequency_11color_64states/detector_calibration"
    mkdir -p "$output"
    if ! progress_complete "$output/prediction/progress.json" completed_predictions 704; then
      env PYTHONPATH="src:lib/diffsynth" CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u -m "${module}_predict" \
        --dataset-root "$dataset" --output-root "$output/prediction" \
        --training-config "$config" --checkpoint "$checkpoint" --history "$history" \
        --device cuda --steps "$steps" --seed-offset "$seed_offset"
    fi
    if ! progress_complete "$output/metrics/progress.json" completed_samples 704; then
      env PYTHONPATH=src "$python_bin" -u -m "${module}_evaluate" \
        --dataset-root "$dataset" --prediction-root "$output/prediction" \
        --calibration-root "$calibration" --output-root "$output/metrics" --history "$history"
    fi
    if ! progress_complete "$output/color_metrics/progress.json" completed_samples 704; then
      env PYTHONPATH=src "$python_bin" -u -m "${module}_color_measure" \
        --dataset-root "$dataset" --prediction-root "$output/prediction" \
        --frequency-metrics-root "$output/metrics" --output-root "$output/color_metrics" --history "$history"
    fi
    touch "$output/EXPERIMENT_COMPLETE"
  done
  touch "$history_root/EXPERIMENT_COMPLETE"
}

for item in "${conditions[@]}"; do
  IFS=: read -r history gpu config checkpoint expected_sha <<< "$item"
  [[ -f "$config" && -f "$checkpoint" ]] || { echo "missing input for $history" >&2; exit 2; }
  gpu_is_free "$gpu" || { echo "GPU $gpu is busy" >&2; exit 3; }
done

declare -a pids=() histories=()
for item in "${conditions[@]}"; do
  IFS=: read -r history gpu config checkpoint expected_sha <<< "$item"
  run_history "$history" "$gpu" "$config" "$checkpoint" "$expected_sha" > "$run_root/logs/$history.log" 2>&1 &
  pids+=("$!")
  histories+=("$history")
done

status=0
set +e
for i in "${!pids[@]}"; do
  wait "${pids[$i]}"
  code=$?
  printf '%s\n' "$code" > "$run_root/${histories[$i]}/exit_status.txt"
  (( code == 0 )) || status=1
done
set -e
(( status == 0 )) || exit 1
touch "$run_root/controller/PIPELINE_COMPLETE"
