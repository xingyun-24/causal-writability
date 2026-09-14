#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-/data/home/yangleqian/workspace/physics-shortcuts-benchmarks}"
cd "$repo_root"

venv_python="/data/home/yangleqian/venvs/physics-shortcuts-benchmarks/bin/python"
manifest_id="frequency_color_shape__train_6318725ad1b0"
circle_manifest="runs/pendulum/weight_integrity/frequency_color_circle__train_e273989068c3.sha256"
experiment_config="configs/pendulum/frequency_color_shape.yaml"
data_root="data/pendulum/${manifest_id}"
latent_root="${data_root}/latents"
run_root="runs/pendulum/${manifest_id}"
log_root="${run_root}/pipeline_logs"
short_gpu=6
long_gpu=7

mkdir -p "$log_root"
exec 9>"${log_root}/pipeline.lock"
if ! flock -n 9; then
  echo "Another frequency_color_shape pipeline is already active."
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

require_file "$circle_manifest"
require_file "$experiment_config"
require_file "configs/pendulum/Train-frequency_color_shape-short.yaml"
require_file "configs/pendulum/Train-frequency_color_shape-long.yaml"
require_file "models/Wan2.1_VAE.pth"

echo "$(timestamp) pipeline_started short_gpu=${short_gpu} long_gpu=${long_gpu}"

if [[ ! -f "${data_root}/build_summary.json" ]]; then
  if [[ -d "$data_root" ]] && find "$data_root" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to overwrite incomplete data directory: $data_root"
    exit 1
  fi
  echo "$(timestamp) data_generation_started"
  env \
    PYTHONPATH="src:lib/diffsynth" \
    "$venv_python" -u -c \
    'from pathlib import Path
import yaml
from sshv2.experiments.pendulum.data import config_from_mapping, generate_dataset
p = Path("configs/pendulum/frequency_color_shape.yaml")
c = config_from_mapping(yaml.safe_load(p.read_text())["data"])
generate_dataset(c, Path("data/pendulum") / c.training_manifest_id)
print("DATA_GENERATION_COMPLETE", flush=True)' \
    > "${log_root}/data_generation.log" 2>&1
  echo "$(timestamp) data_generation_complete"
else
  echo "$(timestamp) data_generation_already_complete"
fi

echo "$(timestamp) data_audit_started"
env \
  PYTHONPATH="src:lib/diffsynth" \
  "$venv_python" -u -c \
  'from pathlib import Path
import json
import yaml
from sshv2.experiments.pendulum.data import audit_dataset, config_from_mapping
p = Path("configs/pendulum/frequency_color_shape.yaml")
c = config_from_mapping(yaml.safe_load(p.read_text())["data"])
r = audit_dataset(Path("data/pendulum") / c.training_manifest_id, c)
print(json.dumps(r, sort_keys=True), flush=True)' \
  > "${log_root}/data_audit.log" 2>&1
echo "$(timestamp) data_audit_complete"

if [[ ! -f "${latent_root}/encoding_audit.json" ]]; then
  if [[ -d "$latent_root" ]] && find "$latent_root" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to overwrite incomplete latent directory: $latent_root"
    exit 1
  fi
  if ! gpu_is_free "$short_gpu"; then
    echo "GPU ${short_gpu} became busy before latent encoding."
    exit 1
  fi
  if [[ "$(gpu_ecc "$short_gpu")" != "0" ]]; then
    echo "GPU ${short_gpu} has nonzero uncorrectable ECC."
    exit 1
  fi
  echo "$(timestamp) latent_encoding_started gpu=${short_gpu}"
  env \
    CUDA_VISIBLE_DEVICES="$short_gpu" \
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
  checkpoint_dir="${run_root}/${history}/ckpt/frequency_color_shape-${history}"
  if [[ -d "$checkpoint_dir" ]] && find "$checkpoint_dir" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to overwrite partial checkpoint directory: $checkpoint_dir"
    exit 1
  fi
done

if ! gpu_is_free "$short_gpu"; then
  echo "GPU ${short_gpu} became busy before short training."
  exit 1
fi
if ! gpu_is_free "$long_gpu"; then
  echo "GPU ${long_gpu} became busy before long training."
  exit 1
fi
if [[ "$(gpu_ecc "$short_gpu")" != "0" ]]; then
  echo "GPU ${short_gpu} has nonzero uncorrectable ECC."
  exit 1
fi
long_ecc_baseline="$(gpu_ecc "$long_gpu")"
if [[ "$long_ecc_baseline" != "1" ]]; then
  echo "GPU ${long_gpu} ECC changed from the accepted baseline of 1: ${long_ecc_baseline}"
  exit 1
fi

echo "$short_gpu" > "${log_root}/short_gpu.txt"
echo "$long_gpu" > "${log_root}/long_gpu.txt"
echo "$long_ecc_baseline" > "${log_root}/long_gpu_ecc_baseline.txt"

echo "$(timestamp) training_short_started gpu=${short_gpu}"
env \
  CUDA_VISIBLE_DEVICES="$short_gpu" \
  PYTHONPATH="src:lib/diffsynth" \
  "$venv_python" -u -m sshv2.cli.train \
    --experiment pendulum \
    --config configs/pendulum/Train-frequency_color_shape-short.yaml \
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
    --config configs/pendulum/Train-frequency_color_shape-long.yaml \
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

require_file "${run_root}/short/ckpt/frequency_color_shape-short/step-10000.safetensors"
require_file "${run_root}/long/ckpt/frequency_color_shape-long/step-10000.safetensors"
if [[ "$(gpu_ecc "$long_gpu")" != "$long_ecc_baseline" ]]; then
  echo "GPU ${long_gpu} ECC increased during training."
  exit 1
fi

echo "$(timestamp) verifying_frequency_color_circle_weights"
sha256sum -c --quiet "$circle_manifest"
echo "$(timestamp) circle_weights_still_ok"
echo "$(timestamp) PIPELINE_COMPLETE"
