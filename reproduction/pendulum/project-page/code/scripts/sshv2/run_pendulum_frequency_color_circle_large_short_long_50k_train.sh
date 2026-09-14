#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-/data/home/yangleqian/workspace/physics-shortcuts-benchmarks}"
cd "$repo_root"

venv_python="/data/home/yangleqian/venvs/physics-shortcuts-benchmarks/bin/python"
source_manifest_id="frequency_color_circle__train_e273989068c3"
manifest_id="${source_manifest_id}"
run_root="runs/pendulum/frequency_color_circle_large_short_long_50k_seed3407"
log_root="${run_root}/pipeline_logs"
short_gpu="${SHORT_GPU:-4}"
long_gpu="${LONG_GPU:-5}"
short_config="configs/pendulum/Train-frequency_color_circle-large-short-50k.yaml"
long_config="configs/pendulum/Train-frequency_color_circle-large-long-50k.yaml"
source_weight_manifest="runs/pendulum/weight_integrity/${source_manifest_id}.sha256"
source_latent_root="data/pendulum/${source_manifest_id}/latents"

mkdir -p "$log_root"
exec 9>"${log_root}/pipeline.lock"
if ! flock -n 9; then
  echo "Another large frequency_color_circle short/long 50k pipeline is already active."
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

require_free_bytes() {
  local minimum="$1"
  local available
  available="$(df --output=avail -B1 . | tail -n 1 | tr -d "[:space:]")"
  if (( available < minimum )); then
    echo "Insufficient free space: available=${available} required=${minimum}"
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

validate_config() {
  local config="$1"
  local expected_history="$2"
  env PYTHONPATH="src:lib/diffsynth" \
    "$venv_python" - "$config" "$expected_history" <<'PY'
from pathlib import Path
import sys

from sshv2.wan.config import StandardTrainingConfig

path = Path(sys.argv[1])
expected_history = sys.argv[2]
cfg = StandardTrainingConfig.from_file(path)
assert cfg.seed == 3407
assert cfg.model.dit.dim == 1152
assert cfg.model.dit.num_heads == 9
assert cfg.model.dit.ffn_dim == 4608
assert cfg.model.dit.num_layers == 30
assert cfg.model.num_condition_frames == 17
assert cfg.loader.num_training_steps == 50000
assert cfg.loader.batch_size == 32
assert cfg.optimizer.learning_rate == 2e-4
assert cfg.optimizer.weight_decay == 0.01
assert cfg.optimizer.gradient_accumulation_steps == 1
assert cfg.optimizer.with_ema is False
assert cfg.log.save_at == [50000]
assert cfg.log.save_last_every == 1000
assert str(cfg.data.dataset).endswith(f"latents/train_{expected_history}")
print(f"CONFIG_OK {path} history={expected_history}")
PY
}

require_file "$venv_python"
require_file "$source_weight_manifest"
require_file "$short_config"
require_file "$long_config"
require_file "models/Wan2.1_VAE.pth"
require_free_bytes $((12 * 1024 * 1024 * 1024))

validate_config "$short_config" short
validate_config "$long_config" long

for history in short long; do
  metadata="${source_latent_root}/train_${history}/metadata.csv"
  require_file "$metadata"
  rows="$(awk 'END { print NR - 1 }' "$metadata")"
  if [[ "$rows" != "2048" ]]; then
    echo "Unexpected source ${history} latent row count: $rows"
    exit 1
  fi
  checkpoint_dir="${run_root}/${history}/ckpt/frequency-color-circle-large-${history}-50k"
  if [[ -f "${checkpoint_dir}/last.safetensors" && -f "${checkpoint_dir}/last-state.pt" ]]; then
    echo "history=${history} will resume from last checkpoint: $checkpoint_dir"
  elif [[ -d "$checkpoint_dir" ]] && find "$checkpoint_dir" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing ambiguous partial checkpoint directory: $checkpoint_dir"
    exit 1
  else
    echo "history=${history} will start fresh"
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
echo "$short_gpu" > "${log_root}/short_gpu.txt"
echo "$long_gpu" > "${log_root}/long_gpu.txt"
echo "$short_ecc_baseline" > "${log_root}/short_gpu_ecc_baseline.txt"
echo "$long_ecc_baseline" > "${log_root}/long_gpu_ecc_baseline.txt"

echo "$(timestamp) verifying_protected_source_weights"
sha256sum -c --quiet "$source_weight_manifest"
echo "$(timestamp) protected_source_weights_ok"

echo "$(timestamp) training_short_started gpu=${short_gpu}"
short_resume_args=()
if [[ -f "${run_root}/short/ckpt/frequency-color-circle-large-short-50k/last.safetensors" && -f "${run_root}/short/ckpt/frequency-color-circle-large-short-50k/last-state.pt" ]]; then
  short_resume_args+=(--resume)
fi
env \
  CUDA_VISIBLE_DEVICES="$short_gpu" \
  PYTHONPATH="src:lib/diffsynth" \
  "$venv_python" -u -m sshv2.cli.train \
    --experiment pendulum \
    --config "$short_config" \
    --resolved-dir "${run_root}/short/resolved" \
    --no-wandb \
    "${short_resume_args[@]}" \
    > "${log_root}/train_short.log" 2>&1 &
short_pid=$!
echo "$short_pid" > "${log_root}/train_short.pid"

echo "$(timestamp) training_long_started gpu=${long_gpu}"
long_resume_args=()
if [[ -f "${run_root}/long/ckpt/frequency-color-circle-large-long-50k/last.safetensors" && -f "${run_root}/long/ckpt/frequency-color-circle-large-long-50k/last-state.pt" ]]; then
  long_resume_args+=(--resume)
fi
env \
  CUDA_VISIBLE_DEVICES="$long_gpu" \
  PYTHONPATH="src:lib/diffsynth" \
  "$venv_python" -u -m sshv2.cli.train \
    --experiment pendulum \
    --config "$long_config" \
    --resolved-dir "${run_root}/long/resolved" \
    --no-wandb \
    "${long_resume_args[@]}" \
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

require_file "${run_root}/short/ckpt/frequency-color-circle-large-short-50k/step-50000.safetensors"
require_file "${run_root}/long/ckpt/frequency-color-circle-large-long-50k/step-50000.safetensors"
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
