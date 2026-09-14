#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-/data/home/yangleqian/workspace/physics-shortcuts-benchmarks}"
venv_python="${PENDULUM_VENV_PYTHON:-/data/home/yangleqian/venvs/physics-shortcuts-benchmarks/bin/python}"
cd "$repo_root"

run_root="runs/pendulum/frequency_color_circle_large_short_wd_sweep_seed3407"
controller_log_root="${run_root}/controller"
latent_root="data/pendulum/frequency_color_circle__train_e273989068c3/latents/train_short"
source_hash_manifest="runs/pendulum/weight_integrity/frequency_color_circle__train_e273989068c3.sha256"

conditions=(
  "0p001:0.001:4:configs/pendulum/wd_sweep/Train-frequency_color_circle-large-short-wd-0.001-50k.yaml"
  "0p01:0.01:5:configs/pendulum/wd_sweep/Train-frequency_color_circle-large-short-wd-0.01-50k.yaml"
  "0p03:0.03:6:configs/pendulum/wd_sweep/Train-frequency_color_circle-large-short-wd-0.03-50k.yaml"
)

mkdir -p "$controller_log_root"
exec 9>"${controller_log_root}/pipeline.lock"
if ! flock -n 9; then
  echo "Another large Pendulum WD sweep controller is active."
  exit 1
fi

timestamp() {
  date "+%F %T %Z"
}

log_event() {
  echo "$(timestamp) $*" | tee -a "${controller_log_root}/pipeline.log"
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

require_free_bytes() {
  local minimum="$1"
  local available
  available="$(df --output=avail -B1 . | tail -n 1 | tr -d "[:space:]")"
  if (( available < minimum )); then
    echo "Insufficient free space: available=${available} required=${minimum}"
    exit 1
  fi
}

validate_config() {
  local config="$1"
  local expected_wd="$2"
  env PYTHONPATH="src:lib/diffsynth" \
    "$venv_python" - "$config" "$expected_wd" <<'PY'
from pathlib import Path
import sys

from sshv2.wan.config import StandardTrainingConfig

path = Path(sys.argv[1])
expected_wd = float(sys.argv[2])
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
assert cfg.optimizer.weight_decay == expected_wd
assert cfg.optimizer.gradient_accumulation_steps == 1
assert cfg.optimizer.with_ema is False
assert cfg.log.save_at == [50000]
assert cfg.log.save_last_every == 1000
assert str(cfg.data.dataset).endswith("latents/train_short")
print(f"CONFIG_OK {path} wd={expected_wd}")
PY
}

require_file "$venv_python"
require_file "models/Wan2.1_VAE.pth"
require_file "${latent_root}/metadata.csv"
require_file "$source_hash_manifest"
require_free_bytes $((35 * 1024 * 1024 * 1024))

rows="$(awk 'END { print NR - 1 }' "${latent_root}/metadata.csv")"
files="$(find "$latent_root" -maxdepth 1 -type f -name "*.pt" | wc -l | tr -d "[:space:]")"
if [[ "$rows" != "2048" || "$files" != "2048" ]]; then
  echo "Unexpected short latent counts: rows=${rows} files=${files}"
  exit 1
fi

log_event "verifying protected source weights"
sha256sum -c --quiet "$source_hash_manifest"
log_event "protected source weights verified"

declare -a pids=()
declare -a tags=()

for condition in "${conditions[@]}"; do
  IFS=: read -r tag wd gpu config <<< "$condition"
  require_file "$config"
  validate_config "$config" "$wd"

  condition_root="${run_root}/wd-${tag}"
  checkpoint_dir="${condition_root}/ckpt/frequency-color-circle-large-short-wd-${tag}-50k"
  final_checkpoint="${checkpoint_dir}/step-50000.safetensors"
  log_root="${condition_root}/pipeline_logs"
  mkdir -p "$log_root"

  if [[ -f "$final_checkpoint" ]]; then
    log_event "wd=${wd} already complete; reusing ${final_checkpoint}"
    sha256sum "$final_checkpoint" > "${log_root}/checkpoint.sha256"
    touch "${log_root}/reused_existing_checkpoint"
    continue
  fi
  if ! gpu_is_free "$gpu"; then
    echo "GPU ${gpu} is busy."
    exit 1
  fi

  ecc="$(gpu_ecc "$gpu")"
  echo "$gpu" > "${log_root}/gpu.txt"
  echo "$ecc" > "${log_root}/gpu_ecc_baseline.txt"

  resume_args=()
  if [[ -f "${checkpoint_dir}/last.safetensors" && -f "${checkpoint_dir}/last-state.pt" ]]; then
    resume_args+=(--resume)
    log_event "resuming wd=${wd} on GPU ${gpu} from rotating last checkpoint"
  elif [[ -d "$checkpoint_dir" ]] && find "$checkpoint_dir" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing ambiguous partial checkpoint directory: $checkpoint_dir"
    exit 1
  else
    log_event "starting wd=${wd} from seed 3407 on GPU ${gpu}"
  fi

  env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="src:lib/diffsynth" \
    "$venv_python" -u -m sshv2.cli.train \
      --experiment pendulum \
      --config "$config" \
      --resolved-dir "${condition_root}/resolved" \
      --no-wandb \
      "${resume_args[@]}" \
      > "${log_root}/train.log" 2>&1 &
  pid=$!
  echo "$pid" > "${log_root}/train.pid"
  pids+=("$pid")
  tags+=("$tag")
done

set +e
overall_status=0
for index in "${!pids[@]}"; do
  pid="${pids[$index]}"
  tag="${tags[$index]}"
  wait "$pid"
  status=$?
  echo "$status" > "${run_root}/wd-${tag}/pipeline_logs/train.exit_status"
  log_event "wd=${tag} process finished status=${status}"
  if (( status != 0 )); then
    overall_status=1
  fi
done
set -e

if (( overall_status != 0 )); then
  log_event "PIPELINE_FAILED"
  exit 1
fi

for condition in "${conditions[@]}"; do
  IFS=: read -r tag wd gpu config <<< "$condition"
  condition_root="${run_root}/wd-${tag}"
  checkpoint_dir="${condition_root}/ckpt/frequency-color-circle-large-short-wd-${tag}-50k"
  final_checkpoint="${checkpoint_dir}/step-50000.safetensors"
  log_root="${condition_root}/pipeline_logs"
  require_file "$final_checkpoint"
  sha256sum "$final_checkpoint" > "${log_root}/checkpoint.sha256"
  if [[ -f "${log_root}/gpu_ecc_baseline.txt" ]]; then
    starting_ecc="$(cat "${log_root}/gpu_ecc_baseline.txt")"
    ending_ecc="$(gpu_ecc "$gpu")"
    echo "$ending_ecc" > "${log_root}/gpu_ecc_final.txt"
    if [[ "$starting_ecc" != "$ending_ecc" ]]; then
      echo "GPU ${gpu} ECC increased during wd=${wd}: ${starting_ecc} -> ${ending_ecc}"
      exit 1
    fi
  fi
done

log_event "re-verifying protected source weights"
sha256sum -c --quiet "$source_hash_manifest"
log_event "PIPELINE_COMPLETE"
touch "${controller_log_root}/PIPELINE_COMPLETE"
