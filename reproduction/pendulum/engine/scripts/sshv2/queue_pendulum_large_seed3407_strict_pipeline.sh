#!/usr/bin/env bash
# Reconstruct the bank and exploratory Stage 2/3 after a controller PID.
# For the paper's frozen bank and Top-4 controller, use README section 4.4.
set -euo pipefail

wait_pid="${1:-}"
repo_root="${2:-$(pwd)}"
gpu_a="${3:-1}"
gpu_b="${4:-2}"
cd "$repo_root"

python_bin="${PENDULUM_VENV_PYTHON:-python}"
export PYTHONPATH="src:lib/diffsynth"

checkpoint="runs/pendulum/frequency_color_circle_large_short_long_50k_seed3407/short/ckpt/frequency-color-circle-large-short-50k/step-50000.safetensors"
checkpoint_sha="0dd907676adebe24fd44b1e8feae808a2c7864615ac965a742e422f0cd721392"
training_config="configs/pendulum/Train-frequency_color_circle-large-short-50k.yaml"
experiment_config="configs/pendulum/frequency_color_circle_frequency_scan.yaml"
experiment_root="runs/pendulum/frequency_color_circle_frequency_scan_11x13"
scan_dataset="data/pendulum/frequency_color_circle_frequency_scan_11x13/circle"
bank_data_root="data/pendulum/strict_bank_128"
root="runs/pendulum/unified_mechanism_v2/mechanism/large_seed3407_short"
bank_root="$root/strict_bank"
stage2_root="$root/stage2_commitment_scan"
stage3_root="$root/stage3_low_rank_causal_route"
log_root="$root/logs"
mkdir -p "$log_root"

if [[ -n "$wait_pid" ]]; then
  while kill -0 "$wait_pid" 2>/dev/null; do sleep 30; done
fi

actual_sha="$(sha256sum "$checkpoint" | awk '{print $1}')"
[[ "$actual_sha" == "$checkpoint_sha" ]] || {
  echo "checkpoint hash mismatch: $actual_sha" >&2
  exit 10
}

run_bank_direction() {
  local direction="$1" gpu="$2"
  "$python_bin" -u -m sshv2.experiments.pendulum.strict_bank \
    --model-name frequency_color_circle \
    --checkpoint "$checkpoint" \
    --training-config "$training_config" \
    --experiment-config "$experiment_config" \
    --data-root "$bank_data_root" \
    --out-root "$bank_root" \
    --history short --device "cuda:$gpu" --hidden-size 1152 \
    --direction "$direction" > "$log_root/strict_bank_${direction}.log" 2>&1
}

run_bank_direction A "$gpu_a" & bank_a_pid=$!
run_bank_direction B "$gpu_b" & bank_b_pid=$!
wait "$bank_a_pid"
wait "$bank_b_pid"

"$python_bin" -u -m sshv2.experiments.pendulum.strict_bank \
  --model-name frequency_color_circle \
  --checkpoint "$checkpoint" \
  --training-config "$training_config" \
  --experiment-config "$experiment_config" \
  --data-root "$bank_data_root" --out-root "$bank_root" \
  --history short --device cuda --hidden-size 1152 --direction merge \
  > "$log_root/strict_bank_merge.log" 2>&1

run_stage2_direction() {
  local direction="$1" gpu="$2"
  "$python_bin" -u -m sshv2.experiments.pendulum.stage2_commitment_scan \
    --model-name frequency_color_circle \
    --checkpoint "$checkpoint" \
    --training-config "$training_config" \
    --experiment-config "$experiment_config" \
    --experiment-root "$experiment_root" --dataset-root "$scan_dataset" \
    --out-root "$stage2_root/$direction" \
    --history short --device "cuda:$gpu" --steps 20 \
    --condition-tokens 1088 --hidden-size 1152 --noop-block-index 13 \
    --manifest "$bank_root/strict_bank_128.jsonl" \
    --natural-metrics "$bank_root/natural_metrics.csv" \
    --direction "$direction" > "$log_root/stage2_${direction}.log" 2>&1
}

run_stage2_direction A "$gpu_a" & stage2_a_pid=$!
run_stage2_direction B "$gpu_b" & stage2_b_pid=$!
wait "$stage2_a_pid"
wait "$stage2_b_pid"

"$python_bin" -u -m sshv2.experiments.pendulum.stage2_commitment_scan \
  --model-name frequency_color_circle \
  --checkpoint "$checkpoint" \
  --training-config "$training_config" \
  --experiment-config "$experiment_config" \
  --experiment-root "$experiment_root" --dataset-root "$scan_dataset" \
  --out-root "$stage2_root" --history short --device cuda \
  --merge-a-root "$stage2_root/A" --merge-b-root "$stage2_root/B" \
  > "$log_root/stage2_merge.log" 2>&1

commitment_block="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["commitment"]["commitment_block_after"])' "$stage2_root/summary.json")"
printf '%s\n' "$commitment_block" > "$root/commitment_block_index.txt"

"$python_bin" -u -m sshv2.experiments.pendulum.stage3_low_rank_causal_route \
  --model-name frequency_color_circle \
  --checkpoint "$checkpoint" \
  --training-config "$training_config" \
  --experiment-config "$experiment_config" --dataset-root "$scan_dataset" \
  --out-root "$stage3_root" --history short --device "cuda:$gpu_a" \
  --seed-offset 23000000 --hidden-size 1152 \
  --commitment-block-index "$commitment_block" \
  --manifest "$bank_root/strict_bank_128.jsonl" \
  --natural-metrics "$bank_root/natural_metrics.csv" \
  > "$log_root/stage3.log" 2>&1

printf '%s\n' 'Stage 3 complete. This is not the final paper controller run.' \
  'Use README section 4.4 with the frozen 64-fit/64-held-out bank at block 12.'
touch "$root/STAGE3_PIPELINE_COMPLETE"
