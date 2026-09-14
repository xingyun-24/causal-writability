#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-/data/home/yangleqian/workspace/physics-shortcuts-benchmarks}"
cd "$repo_root"

venv_python="/data/home/yangleqian/venvs/physics-shortcuts-benchmarks/bin/python"
circle_id="amplitude_color_circle__train_9701981d8d7e"
shape_id="amplitude_color_shape__train_36a9e55c12a2"
circle_config="configs/pendulum/amplitude_color_circle.yaml"
shape_config="configs/pendulum/amplitude_color_shape.yaml"
circle_data_root="data/pendulum/${circle_id}"
shape_data_root="data/pendulum/${shape_id}"
circle_run_root="runs/pendulum/${circle_id}"
shape_run_root="runs/pendulum/${shape_id}"
circle_log_root="${circle_run_root}/pipeline_logs"
shape_log_root="${shape_run_root}/pipeline_logs"
controller_root="runs/pendulum/amplitude_color_circle_shape__pipeline_9701981d8d7e_36a9e55c12a2"
controller_log="${controller_root}/pipeline_master.log"

circle_short_gpu=4
circle_long_gpu=5
shape_short_gpu=6
shape_long_gpu=7

circle_short_config="configs/pendulum/Train-amplitude_color_circle-50k-short.yaml"
circle_long_config="configs/pendulum/Train-amplitude_color_circle-50k-long.yaml"
shape_short_config="configs/pendulum/Train-amplitude_color_shape-50k-short.yaml"
shape_long_config="configs/pendulum/Train-amplitude_color_shape-50k-long.yaml"

protected_manifests=(
  "runs/pendulum/weight_integrity/frequency_color_circle__train_e273989068c3.sha256"
  "runs/pendulum/weight_integrity/frequency_color_shape__train_6318725ad1b0.sha256"
)

mkdir -p "$controller_root" "$circle_log_root" "$shape_log_root"
exec 9>"${controller_root}/pipeline.lock"
if ! flock -n 9; then
  echo "Another amplitude color circle/shape pipeline is already active."
  exit 1
fi

timestamp() {
  date "+%F %T %Z"
}

log_event() {
  local line
  line="$(timestamp) $*"
  echo "$line"
  echo "$line" >> "$controller_log"
  echo "$line" >> "${circle_log_root}/pipeline_master.log"
  echo "$line" >> "${shape_log_root}/pipeline_master.log"
}

on_exit() {
  local status=$?
  if (( status != 0 )); then
    log_event "PIPELINE_FAILED status=${status}"
  fi
}
trap on_exit EXIT

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

require_free_bytes() {
  local minimum="$1"
  local available
  available="$(df --output=avail -B1 . | tail -n 1 | tr -d "[:space:]")"
  if (( available < minimum )); then
    echo "Insufficient free space: available=${available} required=${minimum}"
    exit 1
  fi
}

require_gpu_baselines() {
  local pairs=("4:1" "5:0" "6:0" "7:1")
  local pair gpu expected actual
  for pair in "${pairs[@]}"; do
    gpu="${pair%%:*}"
    expected="${pair##*:}"
    if ! gpu_is_free "$gpu"; then
      echo "GPU ${gpu} is busy."
      exit 1
    fi
    actual="$(gpu_ecc "$gpu")"
    if [[ "$actual" != "$expected" ]]; then
      echo "GPU ${gpu} ECC changed from baseline ${expected}: ${actual}"
      exit 1
    fi
  done
}

require_clean_data_slot() {
  local data_root="$1"
  if [[ -f "${data_root}/build_summary.json" ]]; then
    return
  fi
  if [[ -d "$data_root" ]] && find "$data_root" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to overwrite incomplete data directory: $data_root"
    exit 1
  fi
}

require_clean_checkpoint_slot() {
  local checkpoint_dir="$1"
  if [[ -d "$checkpoint_dir" ]] && find "$checkpoint_dir" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to overwrite partial checkpoint directory: $checkpoint_dir"
    exit 1
  fi
}

config_manifest_id() {
  local config="$1"
  PYTHONPATH="src:lib/diffsynth" "$venv_python" -c \
    'from pathlib import Path
import sys
import yaml
from sshv2.experiments.pendulum.data import config_from_mapping
p = Path(sys.argv[1])
print(config_from_mapping(yaml.safe_load(p.read_text())["data"]).training_manifest_id)' \
    "$config"
}

generate_dataset_one() {
  local label="$1"
  local config="$2"
  local data_root="$3"
  local log_root="$4"
  if [[ -f "${data_root}/build_summary.json" ]]; then
    log_event "${label}_data_generation_already_complete"
    return
  fi
  log_event "${label}_data_generation_started"
  env \
    PYTHONPATH="src:lib/diffsynth" \
    "$venv_python" -u -c \
    'from pathlib import Path
import sys
import yaml
from sshv2.experiments.pendulum.data import config_from_mapping, generate_dataset
p = Path(sys.argv[1])
c = config_from_mapping(yaml.safe_load(p.read_text())["data"])
generate_dataset(c, Path("data/pendulum") / c.training_manifest_id)
print("DATA_GENERATION_COMPLETE", flush=True)' \
    "$config" > "${log_root}/data_generation.log" 2>&1
  log_event "${label}_data_generation_complete"
}

audit_dataset_one() {
  local label="$1"
  local config="$2"
  local log_root="$3"
  log_event "${label}_data_audit_started"
  env \
    PYTHONPATH="src:lib/diffsynth" \
    "$venv_python" -u -c \
    'from pathlib import Path
import json
import sys
import yaml
from sshv2.experiments.pendulum.data import audit_dataset, config_from_mapping
p = Path(sys.argv[1])
c = config_from_mapping(yaml.safe_load(p.read_text())["data"])
r = audit_dataset(Path("data/pendulum") / c.training_manifest_id, c)
print(json.dumps(r, sort_keys=True), flush=True)' \
    "$config" > "${log_root}/data_audit.log" 2>&1
  log_event "${label}_data_audit_complete"
}

encode_latents_one() {
  local config="$1"
  local data_root="$2"
  local log_root="$3"
  local gpu="$4"
  env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="src:lib/diffsynth" \
    "$venv_python" -u scripts/sshv2/pendulum_prepare_history_latents.py \
      --source "$data_root" \
      --experiment-config "$config" \
      --batch-size 8 \
      --device cuda \
      > "${log_root}/latent_encoding.log" 2>&1
}

verify_latents() {
  local label="$1"
  local data_root="$2"
  local history metadata rows count
  require_file "${data_root}/latents/encoding_audit.json"
  for history in short long; do
    metadata="${data_root}/latents/train_${history}/metadata.csv"
    require_file "$metadata"
    rows="$(awk 'END { print NR - 1 }' "$metadata")"
    count="$(find "${data_root}/latents/train_${history}" -maxdepth 1 -type f -name "*.pt" | wc -l | tr -d "[:space:]")"
    if [[ "$rows" != "2048" || "$count" != "2048" ]]; then
      echo "Unexpected ${label} ${history} latent counts: files=${count} rows=${rows}"
      exit 1
    fi
  done
}

for path in \
  "$circle_config" \
  "$shape_config" \
  "$circle_short_config" \
  "$circle_long_config" \
  "$shape_short_config" \
  "$shape_long_config" \
  "models/Wan2.1_VAE.pth" \
  "scripts/sshv2/pendulum_prepare_history_latents.py" \
  "${protected_manifests[@]}"; do
  require_file "$path"
done

if [[ "$(config_manifest_id "$circle_config")" != "$circle_id" ]]; then
  echo "Circle config manifest ID does not match ${circle_id}."
  exit 1
fi
if [[ "$(config_manifest_id "$shape_config")" != "$shape_id" ]]; then
  echo "Shape config manifest ID does not match ${shape_id}."
  exit 1
fi

require_clean_data_slot "$circle_data_root"
require_clean_data_slot "$shape_data_root"
require_clean_checkpoint_slot "${circle_run_root}/short/ckpt/amplitude_color_circle-50k-short"
require_clean_checkpoint_slot "${circle_run_root}/long/ckpt/amplitude_color_circle-50k-long"
require_clean_checkpoint_slot "${shape_run_root}/short/ckpt/amplitude_color_shape-50k-short"
require_clean_checkpoint_slot "${shape_run_root}/long/ckpt/amplitude_color_shape-50k-long"
require_free_bytes $((90 * 1024 * 1024 * 1024))
require_gpu_baselines

log_event "pipeline_started circle_short_gpu=4 circle_long_gpu=5 shape_short_gpu=6 shape_long_gpu=7 checkpoint_interval=5000"
log_event "verifying_protected_frequency_weights"
for manifest in "${protected_manifests[@]}"; do
  sha256sum -c --quiet "$manifest"
done
log_event "protected_frequency_weights_ok"

generate_dataset_one "amplitude_color_circle" "$circle_config" "$circle_data_root" "$circle_log_root"
audit_dataset_one "amplitude_color_circle" "$circle_config" "$circle_log_root"
generate_dataset_one "amplitude_color_shape" "$shape_config" "$shape_data_root" "$shape_log_root"
audit_dataset_one "amplitude_color_shape" "$shape_config" "$shape_log_root"
log_event "all_data_generation_and_audits_complete"

require_free_bytes $((70 * 1024 * 1024 * 1024))
require_gpu_baselines
if [[ -d "${circle_data_root}/latents" ]] && find "${circle_data_root}/latents" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing to overwrite incomplete circle latent directory."
  exit 1
fi
if [[ -d "${shape_data_root}/latents" ]] && find "${shape_data_root}/latents" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing to overwrite incomplete shape latent directory."
  exit 1
fi

log_event "latent_encoding_started circle_gpu=4 shape_gpu=6"
encode_latents_one "$circle_config" "$circle_data_root" "$circle_log_root" "$circle_short_gpu" &
circle_encode_pid=$!
encode_latents_one "$shape_config" "$shape_data_root" "$shape_log_root" "$shape_short_gpu" &
shape_encode_pid=$!
set +e
wait "$circle_encode_pid"
circle_encode_status=$?
wait "$shape_encode_pid"
shape_encode_status=$?
set -e
log_event "latent_encoding_finished circle_status=${circle_encode_status} shape_status=${shape_encode_status}"
if (( circle_encode_status != 0 || shape_encode_status != 0 )); then
  exit 1
fi
verify_latents "amplitude_color_circle" "$circle_data_root"
verify_latents "amplitude_color_shape" "$shape_data_root"
log_event "all_latent_audits_complete"

require_free_bytes $((55 * 1024 * 1024 * 1024))
require_gpu_baselines
for gpu in 4 5 6 7; do
  echo "$(gpu_ecc "$gpu")" > "${controller_root}/gpu${gpu}_ecc_baseline.txt"
done
printf '%s\n' \
  "GPU4 amplitude_color_circle short 50000" \
  "GPU5 amplitude_color_circle long 50000" \
  "GPU6 amplitude_color_shape short 50000" \
  "GPU7 amplitude_color_shape long 50000" \
  > "${controller_root}/gpu_binding.txt"

log_event "training_amplitude_color_circle_short_started gpu=4"
env CUDA_VISIBLE_DEVICES=4 PYTHONPATH="src:lib/diffsynth" \
  "$venv_python" -u -m sshv2.cli.train \
    --experiment pendulum \
    --config "$circle_short_config" \
    --resolved-dir "${circle_run_root}/short/resolved" \
    --no-wandb > "${circle_log_root}/train_short.log" 2>&1 &
circle_short_pid=$!
echo "$circle_short_pid" > "${circle_log_root}/train_short.pid"

log_event "training_amplitude_color_circle_long_started gpu=5"
env CUDA_VISIBLE_DEVICES=5 PYTHONPATH="src:lib/diffsynth" \
  "$venv_python" -u -m sshv2.cli.train \
    --experiment pendulum \
    --config "$circle_long_config" \
    --resolved-dir "${circle_run_root}/long/resolved" \
    --no-wandb > "${circle_log_root}/train_long.log" 2>&1 &
circle_long_pid=$!
echo "$circle_long_pid" > "${circle_log_root}/train_long.pid"

log_event "training_amplitude_color_shape_short_started gpu=6"
env CUDA_VISIBLE_DEVICES=6 PYTHONPATH="src:lib/diffsynth" \
  "$venv_python" -u -m sshv2.cli.train \
    --experiment pendulum \
    --config "$shape_short_config" \
    --resolved-dir "${shape_run_root}/short/resolved" \
    --no-wandb > "${shape_log_root}/train_short.log" 2>&1 &
shape_short_pid=$!
echo "$shape_short_pid" > "${shape_log_root}/train_short.pid"

log_event "training_amplitude_color_shape_long_started gpu=7"
env CUDA_VISIBLE_DEVICES=7 PYTHONPATH="src:lib/diffsynth" \
  "$venv_python" -u -m sshv2.cli.train \
    --experiment pendulum \
    --config "$shape_long_config" \
    --resolved-dir "${shape_run_root}/long/resolved" \
    --no-wandb > "${shape_log_root}/train_long.log" 2>&1 &
shape_long_pid=$!
echo "$shape_long_pid" > "${shape_log_root}/train_long.pid"

set +e
wait "$circle_short_pid"; circle_short_status=$?
wait "$circle_long_pid"; circle_long_status=$?
wait "$shape_short_pid"; shape_short_status=$?
wait "$shape_long_pid"; shape_long_status=$?
set -e
log_event "training_processes_finished circle_short=${circle_short_status} circle_long=${circle_long_status} shape_short=${shape_short_status} shape_long=${shape_long_status}"
if (( circle_short_status != 0 || circle_long_status != 0 || shape_short_status != 0 || shape_long_status != 0 )); then
  exit 1
fi

for path in \
  "${circle_run_root}/short/ckpt/amplitude_color_circle-50k-short/step-50000.safetensors" \
  "${circle_run_root}/short/ckpt/amplitude_color_circle-50k-short/step-50000-state.pt" \
  "${circle_run_root}/long/ckpt/amplitude_color_circle-50k-long/step-50000.safetensors" \
  "${circle_run_root}/long/ckpt/amplitude_color_circle-50k-long/step-50000-state.pt" \
  "${shape_run_root}/short/ckpt/amplitude_color_shape-50k-short/step-50000.safetensors" \
  "${shape_run_root}/short/ckpt/amplitude_color_shape-50k-short/step-50000-state.pt" \
  "${shape_run_root}/long/ckpt/amplitude_color_shape-50k-long/step-50000.safetensors" \
  "${shape_run_root}/long/ckpt/amplitude_color_shape-50k-long/step-50000-state.pt"; do
  require_file "$path"
done

for gpu in 4 5 6 7; do
  baseline="$(<"${controller_root}/gpu${gpu}_ecc_baseline.txt")"
  if [[ "$(gpu_ecc "$gpu")" != "$baseline" ]]; then
    echo "GPU ${gpu} ECC increased during training."
    exit 1
  fi
done

log_event "verifying_protected_frequency_weights_after_training"
for manifest in "${protected_manifests[@]}"; do
  sha256sum -c --quiet "$manifest"
done
log_event "protected_frequency_weights_still_ok"
log_event "PIPELINE_COMPLETE"
