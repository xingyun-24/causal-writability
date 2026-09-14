#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-/data/home/yangleqian/workspace/physics-shortcuts-benchmarks}"
cd "$repo_root"

venv_python="/data/home/yangleqian/venvs/physics-shortcuts-benchmarks/bin/python"
manifest_id="frequency_color_square__train_85a5892318fc"
circle_manifest="runs/pendulum/weight_integrity/frequency_color_circle__train_e273989068c3.sha256"
data_root="data/pendulum/${manifest_id}"
latent_root="${data_root}/latents"
run_root="runs/pendulum/${manifest_id}"
log_root="${run_root}/pipeline_logs"
experiment_config="configs/pendulum/frequency_color_square.yaml"

mkdir -p "$log_root"
exec 9>"${log_root}/pipeline.lock"
if ! flock -n 9; then
  echo "Another frequency_color_square pipeline is already active."
  exit 1
fi

timestamp() {
  date "+%F %T %Z"
}

gpu_is_free() {
  local gpu="$1"
  local pids
  pids="$(
    nvidia-smi \
      -i "$gpu" \
      --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null \
      | tr -d "[:space:]"
  )"
  [[ -z "$pids" ]]
}

first_free_gpu() {
  local gpu
  for gpu in 0 1 2 3 4 5 6 7; do
    if gpu_is_free "$gpu"; then
      printf "%s\n" "$gpu"
      return 0
    fi
  done
  return 1
}

wait_for_stable_free_gpu() {
  local candidate=""
  local previous=""
  local stable_checks=0
  while (( stable_checks < 3 )); do
    candidate="$(first_free_gpu || true)"
    if [[ -n "$candidate" && "$candidate" == "$previous" ]]; then
      stable_checks=$((stable_checks + 1))
    elif [[ -n "$candidate" ]]; then
      previous="$candidate"
      stable_checks=1
    else
      previous=""
      stable_checks=0
    fi
    echo "$(timestamp) waiting_for_gpu candidate=${candidate:-none} stable_checks=${stable_checks}/3"
    if (( stable_checks < 3 )); then
      sleep 60
    fi
  done
  printf "%s\n" "$candidate"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1"
    exit 1
  fi
}

require_file "$circle_manifest"
require_file "${data_root}/build_summary.json"
require_file "$experiment_config"
require_file "configs/pendulum/Train-frequency_color_square-short.yaml"
require_file "configs/pendulum/Train-frequency_color_square-long.yaml"
require_file "models/Wan2.1_VAE.pth"

echo "$(timestamp) verifying_frequency_color_circle_weights"
sha256sum -c --quiet "$circle_manifest"
echo "$(timestamp) circle_weights_ok"

selected_gpu="$(wait_for_stable_free_gpu | tee "${log_root}/gpu_wait.log" | tail -1)"
echo "$selected_gpu" > "${log_root}/selected_gpu.txt"
echo "$(timestamp) selected_gpu=${selected_gpu}"

if [[ ! -f "${latent_root}/encoding_audit.json" ]]; then
  if [[ -d "$latent_root" ]] && find "$latent_root" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to overwrite incomplete latent directory: $latent_root"
    exit 1
  fi
  echo "$(timestamp) latent_encoding_started gpu=${selected_gpu}"
  env \
    CUDA_VISIBLE_DEVICES="$selected_gpu" \
    PYTHONPATH="src:lib/diffsynth" \
    "$venv_python" -u scripts/sshv2/pendulum_prepare_history_latents.py \
      --source "$data_root" \
      --experiment-config "$experiment_config" \
      --batch-size 8 \
      --device cuda \
      > "${log_root}/latent_encoding.log" 2>&1
  echo "$(timestamp) latent_encoding_complete"
else
  echo "$(timestamp) latent_encoding_already_complete"
fi

for history in short long; do
  metadata="${latent_root}/train_${history}/metadata.csv"
  require_file "$metadata"
  rows="$(awk 'END { print NR - 1 }' "$metadata")"
  if [[ "$rows" != "2048" ]]; then
    echo "Unexpected ${history} latent row count: $rows"
    exit 1
  fi
done

for history in short long; do
  config="configs/pendulum/Train-frequency_color_square-${history}.yaml"
  checkpoint="${run_root}/${history}/ckpt/frequency_color_square-${history}/step-10000.safetensors"
  checkpoint_dir="${run_root}/${history}/ckpt/frequency_color_square-${history}"
  if [[ -f "$checkpoint" ]]; then
    echo "$(timestamp) training_${history}_already_complete"
    continue
  fi
  if [[ -d "$checkpoint_dir" ]] && find "$checkpoint_dir" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to overwrite partial checkpoint directory: $checkpoint_dir"
    exit 1
  fi
  echo "$(timestamp) training_${history}_started gpu=${selected_gpu}"
  env \
    CUDA_VISIBLE_DEVICES="$selected_gpu" \
    PYTHONPATH="src:lib/diffsynth" \
    "$venv_python" -u -m sshv2.cli.train \
      --experiment pendulum \
      --config "$config" \
      --resolved-dir "${run_root}/${history}/resolved" \
      --no-wandb \
      > "${log_root}/train_${history}.log" 2>&1
  require_file "$checkpoint"
  echo "$(timestamp) training_${history}_complete"
done

echo "$(timestamp) verifying_frequency_color_circle_weights_after_training"
sha256sum -c --quiet "$circle_manifest"
echo "$(timestamp) circle_weights_still_ok"
echo "$(timestamp) PIPELINE_COMPLETE"
