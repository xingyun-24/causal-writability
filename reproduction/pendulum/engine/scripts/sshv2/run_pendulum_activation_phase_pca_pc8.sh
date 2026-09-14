#!/usr/bin/env bash
set -euo pipefail

model=${1:?usage: $0 circle|shape low|high}
target=${2:?usage: $0 circle|shape low|high}
repo=${PENDULUM_REPO:-/data/home/yangleqian/workspace/physics-shortcuts-benchmarks}
python=${PENDULUM_PYTHON:-/data/home/yangleqian/venvs/physics-shortcuts-benchmarks/bin/python}

case "$model" in
  circle) model_name=frequency_color_circle ;;
  shape) model_name=frequency_color_shape ;;
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
source_root="runs/pendulum/activation_phase_pca_special_blocks_50k_long_v1/${model_name}/${target}_after_block_${block_index}"
out_root="runs/pendulum/activation_phase_pca_special_blocks_50k_long_pc8_v1/${model_name}/${target}_after_block_${block_index}"
test -f "$source_root/directions.npy"
test -f "$source_root/geometry_dense/directions.npy"

export PYTHONPATH="${repo}/src:${repo}/lib/diffsynth${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${repo}/runs/pendulum/.matplotlib-cache"
mkdir -p "$MPLCONFIGDIR"
exec "$python" -u -m sshv2.experiments.pendulum.extend_activation_phase_pca_pc8 \
  --source-root "$source_root" \
  --out-root "$out_root" \
  --pca-chunk-values 262144
