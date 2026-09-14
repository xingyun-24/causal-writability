#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-/data/home/yangleqian/workspace/physics-shortcuts-benchmarks}"
cd "$repo_root"

venv_python="/data/home/yangleqian/venvs/physics-shortcuts-benchmarks/bin/python"
source_manifest_id="frequency_color_shape__train_6318725ad1b0"
manifest_id="frequency_color_shape__train_46578f66aa81"
source_weight_manifest="runs/pendulum/weight_integrity/${source_manifest_id}.sha256"
source_latent_root="data/pendulum/${source_manifest_id}/latents"
run_root="runs/pendulum/${manifest_id}"
log_root="${run_root}/pipeline_logs"
short_gpu=6
long_gpu=7
short_config="configs/pendulum/Train-frequency_color_shape-50k-short.yaml"
long_config="configs/pendulum/Train-frequency_color_shape-50k-long.yaml"

mkdir -p "$log_root"
exec 9>"${log_root}/pipeline.lock"
if ! flock -n 9; then
  echo "Another frequency_color_shape 50k pipeline is already active."
  exit 1
fi

timestamp() {
  date "+%F %T %Z"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1"
    exit 1
  fi
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

gpu_ecc() {
  nvidia-smi \
    -i "$1" \
    --query-gpu=ecc.errors.uncorrected.volatile.total \
    --format=csv,noheader,nounits \
    | tr -d "[:space:]"
}

require_file "$source_weight_manifest"
require_file "$short_config"
require_file "$long_config"
require_file "models/Wan2.1_VAE.pth"

for history in short long; do
  metadata="${source_latent_root}/train_${history}/metadata.csv"
  require_file "$metadata"
  rows="$(awk 'END { print NR - 1 }' "$metadata")"
  if [[ "$rows" != "2048" ]]; then
    echo "Unexpected source ${history} latent row count: $rows"
    exit 1
  fi
  checkpoint_dir="${run_root}/${history}/ckpt/frequency_color_shape-50k-${history}"
  if [[ -d "$checkpoint_dir" ]] && find "$checkpoint_dir" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to overwrite partial checkpoint directory: $checkpoint_dir"
    exit 1
  fi
done

if ! gpu_is_free "$short_gpu"; then
  echo "GPU ${short_gpu} is busy."
  exit 1
fi
if ! gpu_is_free "$long_gpu"; then
  echo "GPU ${long_gpu} is busy."
  exit 1
fi
short_ecc_baseline="$(gpu_ecc "$short_gpu")"
long_ecc_baseline="$(gpu_ecc "$long_gpu")"
if [[ "$short_ecc_baseline" != "0" ]]; then
  echo "GPU ${short_gpu} ECC changed from the accepted baseline of 0: ${short_ecc_baseline}"
  exit 1
fi
if [[ "$long_ecc_baseline" != "1" ]]; then
  echo "GPU ${long_gpu} ECC changed from the accepted baseline of 1: ${long_ecc_baseline}"
  exit 1
fi

echo "$(timestamp) verifying_protected_source_weights"
sha256sum -c --quiet "$source_weight_manifest"
echo "$(timestamp) protected_source_weights_ok"
echo "$short_gpu" > "${log_root}/short_gpu.txt"
echo "$long_gpu" > "${log_root}/long_gpu.txt"
echo "$short_ecc_baseline" > "${log_root}/short_gpu_ecc_baseline.txt"
echo "$long_ecc_baseline" > "${log_root}/long_gpu_ecc_baseline.txt"

echo "$(timestamp) training_short_started gpu=${short_gpu}"
env \
  CUDA_VISIBLE_DEVICES="$short_gpu" \
  PYTHONPATH="src:lib/diffsynth" \
  "$venv_python" -u -m sshv2.cli.train \
    --experiment pendulum \
    --config "$short_config" \
    --resolved-dir "${run_root}/short/resolved" \
    --no-wandb \
    > "${log_root}/train_short.log" 2>&1 &
short_pid=$!
echo "$short_pid" > "${log_root}/train_short.pid"

echo "$(timestamp) training_long_started gpu=${long_gpu}"
env \
  CUDA_VISIBLE_DEVICES="$long_gpu" \
  PYTHONPATH="src:lib/diffsynth" \
  "$venv_python" -u -m sshv2.cli.train \
    --experiment pendulum \
    --config "$long_config" \
    --resolved-dir "${run_root}/long/resolved" \
    --no-wandb \
    > "${log_root}/train_long.log" 2>&1 &
long_pid=$!
echo "$long_pid" > "${log_root}/train_long.pid"

set +e
wait "$short_pid"
short_status=$?
wait "$long_pid"
long_status=$?
set -e
echo "$(timestamp) training_processes_finished short_status=${short_status} long_status=${long_status}"
if (( short_status != 0 || long_status != 0 )); then
  exit 1
fi

require_file "${run_root}/short/ckpt/frequency_color_shape-50k-short/step-50000.safetensors"
require_file "${run_root}/long/ckpt/frequency_color_shape-50k-long/step-50000.safetensors"
if [[ "$(gpu_ecc "$short_gpu")" != "$short_ecc_baseline" ]]; then
  echo "GPU ${short_gpu} ECC increased during training."
  exit 1
fi
if [[ "$(gpu_ecc "$long_gpu")" != "$long_ecc_baseline" ]]; then
  echo "GPU ${long_gpu} ECC increased during training."
  exit 1
fi

echo "$(timestamp) verifying_protected_source_weights_after_training"
sha256sum -c --quiet "$source_weight_manifest"
echo "$(timestamp) protected_source_weights_still_ok"
echo "$(timestamp) PIPELINE_COMPLETE"
