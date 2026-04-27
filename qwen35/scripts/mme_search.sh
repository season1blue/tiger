#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$script_dir/../.." && pwd)"
cd "$root_dir"

# Usage examples:
#   nohup bash qwen35/scripts/mme_search.sh > results/Qwen3.5-VL/mme/search/layer_search.out 2>&1 &
#   nohup bash qwen35/scripts/mme_search.sh evo > results/Qwen3.5-VL/mme/search/nohup.out 2>&1 &
#   nohup bash qwen35/scripts/mme_search.sh evo 5 12 > results/Qwen3.5-VL/mme/search/nohup.out 2>&1 &

method="${1:-evo}"
start_min="${2:-1}"
start_max="${3:-4}"

if [[ "$method" != "base" && "$method" != "memvr" && "$method" != "evo" ]]; then
    echo "Invalid method: $method"
    echo "Usage: bash qwen35/scripts/mme_search.sh [base|memvr|evo] [start_min] [start_max]"
    exit 1
fi

if ! [[ "$start_min" =~ ^[0-9]+$ ]] || ! [[ "$start_max" =~ ^[0-9]+$ ]] || [[ "$start_min" -gt "$start_max" ]]; then
    echo "Invalid layer range: start_min=$start_min start_max=$start_max"
    exit 1
fi

search_root="$root_dir/results/Qwen3.5-VL/mme/search"
run_stamp="$(date +%Y%m%d_%H%M%S)"
batch_log="$search_root/search_${method}_${run_stamp}.log"
mkdir -p "$search_root"

state_drift_threshold="${STATE_DRIFT_THRESHOLD:-0.5}"

echo "[SEARCH] method=$method" | tee -a "$batch_log"
echo "[SEARCH] start_min=$start_min start_max=$start_max" | tee -a "$batch_log"
echo "[SEARCH] state_drift_threshold fixed: $state_drift_threshold" | tee -a "$batch_log"
echo "[SEARCH] batch_log=$batch_log" | tee -a "$batch_log"

for ((start_layer=start_min; start_layer<=start_max; start_layer++)); do
    end_layer=$((start_layer + 2))

    run_tag="search_s${start_layer}_e${end_layer}_t${state_drift_threshold}"
    run_log="$search_root/${run_tag}.log"

    echo "[SEARCH] >>> start=$start_layer end=$end_layer state_drift_threshold=$state_drift_threshold run_tag=$run_tag" | tee -a "$batch_log"

    STARTING_LAYER="$start_layer" \
    ENDING_LAYER="$end_layer" \
    STATE_DRIFT_THRESHOLD="$state_drift_threshold" \
    MME_RUN_TAG="$run_tag" \
    bash qwen35/scripts/mme.sh "$method" > "$run_log" 2>&1

    rc=$?
    echo "[SEARCH] <<< run_tag=$run_tag exit_code=$rc log=$run_log" | tee -a "$batch_log"

    if [[ "$rc" -ne 0 ]]; then
        echo "[SEARCH] stop on failure: run_tag=$run_tag" | tee -a "$batch_log"
        exit "$rc"
    fi
done

echo "[SEARCH] completed successfully" | tee -a "$batch_log"