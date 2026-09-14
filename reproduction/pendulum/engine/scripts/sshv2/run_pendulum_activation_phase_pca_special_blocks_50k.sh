#!/usr/bin/env bash
set -euo pipefail

model=${1:?usage: $0 circle|shape low|high [device]}
target=${2:?usage: $0 circle|shape low|high [device]}
device=${3:-cuda}
repo=${PENDULUM_REPO:-/data/home/yangleqian/workspace/physics-shortcuts-benchmarks}
python=${PENDULUM_PYTHON:-/data/home/yangleqian/venvs/physics-shortcuts-benchmarks/bin/python}

case "$model" in
  circle)
    model_name=frequency_color_circle
    experiment_config=configs/pendulum/frequency_color_circle_frequency_scan.yaml
    training_config=configs/pendulum/Train-frequency_color_circle-50k-long.yaml
    checkpoint=runs/pendulum/frequency_color_circle__train_5932a18b5983/long/ckpt/frequency_color_circle-50k-long/step-50000.safetensors
    ;;
  shape)
    model_name=frequency_color_shape
    experiment_config=configs/pendulum/frequency_color_shape_frequency_scan.yaml
    training_config=configs/pendulum/Train-frequency_color_shape-50k-long.yaml
    checkpoint=runs/pendulum/frequency_color_shape__train_46578f66aa81/long/ckpt/frequency_color_shape-50k-long/step-50000.safetensors
    ;;
  *)
    echo "model must be circle or shape" >&2
    exit 2
    ;;
esac

case "$target" in
  low) block_index=26 ;;
  high) block_index=22 ;;
  *)
    echo "target must be low or high" >&2
    exit 2
    ;;
esac

cd "$repo"
source_root="runs/pendulum/mechanism_pca_50k_long_block14_v3/${model_name}"
out_root="runs/pendulum/activation_phase_pca_special_blocks_50k_long_v1/${model_name}/${target}_after_block_${block_index}"
for required in \
  "$source_root/receiver_manifest.jsonl" \
  "$experiment_config" \
  "$training_config" \
  "$checkpoint"; do
  test -f "$required"
done

export PYTHONPATH="${repo}/src:${repo}/lib/diffsynth${PYTHONPATH:+:${PYTHONPATH}}"
exec "$python" -u -m sshv2.experiments.pendulum.activation_phase_pca \
  --model-name "$model_name" \
  --target-label "$target" \
  --source-root "$source_root" \
  --experiment-config "$experiment_config" \
  --training-config "$training_config" \
  --checkpoint "$checkpoint" \
  --out-root "$out_root" \
  --block-index "$block_index" \
  --history long \
  --device "$device" \
  --steps 20 \
  --condition-tokens 1088 \
  --hidden-size 768 \
  --ranks 1,2,3,4 \
  --geometry-phase-count 32
