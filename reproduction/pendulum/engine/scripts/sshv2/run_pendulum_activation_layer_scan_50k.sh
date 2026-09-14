#!/usr/bin/env bash
set -euo pipefail

model=${1:?usage: $0 circle|shape [device]}
device=${2:-cuda}
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

cd "$repo"
source_root="runs/pendulum/mechanism_pca_50k_long_block14_v3/${model_name}"
out_root="runs/pendulum/activation_layer_scan_50k_long_v1/${model_name}"
for required in \
  "$source_root/receiver_manifest.jsonl" \
  "$source_root/metrics.csv" \
  "$experiment_config" \
  "$training_config" \
  "$checkpoint"; do
  test -f "$required"
done

export PYTHONPATH="${repo}/src:${repo}/lib/diffsynth${PYTHONPATH:+:${PYTHONPATH}}"
exec "$python" -u -m sshv2.experiments.pendulum.activation_layer_scan \
  --model-name "$model_name" \
  --source-root "$source_root" \
  --experiment-config "$experiment_config" \
  --training-config "$training_config" \
  --checkpoint "$checkpoint" \
  --out-root "$out_root" \
  --history long \
  --device "$device" \
  --steps 20 \
  --condition-tokens 1088 \
  --hidden-size 768 \
  --block-index 13 \
  --noop-block-index 13
