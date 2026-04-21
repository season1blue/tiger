#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$script_dir/.." && pwd)"
cd "$root_dir"

log_file="${LOG_FILE:-$script_dir/pope_all.log}"
mkdir -p "$(dirname "$log_file")"
touch "$log_file"

exec > >(tee -a "$log_file") 2>&1

export DATASET_PREFIXES="${DATASET_PREFIXES:-coco,gqa,aokvqa}"
export SPLITS="${SPLITS:-random,popular,adversarial}"

# Optional overrides for the baseline run.
# Examples:
#   LIMIT=100 MODEL_PATH=/mnt/data/ssz/llms/Qwen2.5-VL-7B-Instruct bash baseline_test/pope_all.sh
#   NUM_GPUS=4 GPU_IDS_CSV="0,1,2,3" bash baseline_test/pope_all.sh

num_gpus="${NUM_GPUS:-1}"
gpu_ids_csv="${GPU_IDS_CSV:-}"

printf '\n[POPE-ALL] start %s\n' "$(date '+%F %T')"
printf '[POPE-ALL] log_file=%s\n' "$log_file"
printf '[POPE-ALL] DATASET_PREFIXES=%s\n' "$DATASET_PREFIXES"
printf '[POPE-ALL] SPLITS=%s\n' "$SPLITS"
printf '[POPE-ALL] NUM_GPUS=%s\n' "$num_gpus"
printf '[POPE-ALL] GPU_IDS_CSV=%s\n' "${gpu_ids_csv:-none}"

if [[ -n "$gpu_ids_csv" ]]; then
    bash "$script_dir/pope.sh" "$num_gpus" "$gpu_ids_csv"
else
    bash "$script_dir/pope.sh" "$num_gpus"
fi

printf '\n[POPE-ALL] done %s\n' "$(date '+%F %T')"