#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-/data/home/yangleqian/workspace/physics-shortcuts-benchmarks}"
cd "${repo_root}"
export PYTHONPATH="${repo_root}/src:${repo_root}/lib/diffsynth"
python_bin="/data/home/yangleqian/venvs/physics-shortcuts-benchmarks/bin/python"
exporter="${2:-/tmp/capture_pendulum_project_pca.py}"
out_root="${repo_root}/runs/pendulum/project_page_pca_v1"
mechanism_root="${repo_root}/runs/pendulum/unified_mechanism_v2/mechanism/large_seed3407_short"
checkpoint="${repo_root}/runs/pendulum/frequency_color_circle_large_short_long_50k_seed3407/short/ckpt/frequency-color-circle-large-short-50k/step-50000.safetensors"
training_config="${repo_root}/configs/pendulum/Train-frequency_color_circle-large-short-50k.yaml"
experiment_config="${repo_root}/configs/pendulum/frequency_color_circle_frequency_scan.yaml"
manifest="${mechanism_root}/strict_bank/strict_bank_128.jsonl"
natural_metrics="${mechanism_root}/strict_bank/natural_metrics.csv"
components="${mechanism_root}/stage3_low_rank_causal_route/pca/components.npy"
pca_audit="${mechanism_root}/stage3_low_rank_causal_route/pca/audit.json"
checkpoint_sha="0dd907676adebe24fd44b1e8feae808a2c7864615ac965a742e422f0cd721392"
gpus=(1 2 4 5)

mkdir -p "${out_root}/logs" "${out_root}/shards"
exec 9>"${out_root}/capture.lock"
flock -n 9 || { echo "capture is already running" >&2; exit 9; }

pids=()
for shard in 0 1 2 3; do
  "${python_bin}" -u "${exporter}" capture \
    --manifest "${manifest}" \
    --natural-metrics "${natural_metrics}" \
    --pca-audit "${pca_audit}" \
    --components "${components}" \
    --checkpoint "${checkpoint}" \
    --training-config "${training_config}" \
    --experiment-config "${experiment_config}" \
    --out-dir "${out_root}/shards" \
    --num-shards 4 --shard-index "${shard}" \
    --device "cuda:${gpus[$shard]}" \
    > "${out_root}/logs/shard-${shard}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
if (( status != 0 )); then
  echo "one or more capture shards failed" >&2
  exit 10
fi

"${python_bin}" "${exporter}" merge \
  --manifest "${manifest}" \
  --pca-audit "${pca_audit}" \
  --components "${components}" \
  --out-dir "${out_root}/shards" \
  --num-shards 4 \
  --checkpoint-sha256 "${checkpoint_sha}" \
  --output-json "${out_root}/pendulum-project-pca.json"
touch "${out_root}/PROJECT_PAGE_PCA_COMPLETE"
