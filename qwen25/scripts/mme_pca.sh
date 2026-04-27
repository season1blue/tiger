#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$script_dir/../.." && pwd)"
cd "$root_dir"
export PYTHONPATH="$root_dir:${PYTHONPATH:-}"

# Usage:
#   bash qwen25/scripts/mme_pca.sh
#   bash qwen25/scripts/mme_pca.sh 24
#   bash qwen25/scripts/mme_pca.sh 24 1 "0"
#   bash qwen25/scripts/mme_pca.sh 24 4 "0,1,2,3"

subset_size="${1:-24}"
num_gpus="${2:-1}"
gpu_ids_csv="${3:-}"

pick_top_gpus_by_free_mem() {
    local count="$1"
    local query=""

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 1
    fi

    query="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ -z "$query" ]]; then
        return 1
    fi

    mapfile -t picked < <(
        echo "$query" \
        | awk -F',' '{gsub(/ /, "", $1); gsub(/ /, "", $2); print $1 "," $2}' \
        | sort -t',' -k2,2nr \
        | head -n "$count" \
        | cut -d',' -f1
    )

    if [[ "${#picked[@]}" -lt "$count" ]]; then
        return 1
    fi

    printf '%s\n' "${picked[@]}"
}

if ! [[ "$subset_size" =~ ^[0-9]+$ ]] || [[ "$subset_size" -lt 2 ]]; then
    echo "Invalid subset_size: $subset_size"
    echo "Usage: bash qwen25/scripts/mme_pca.sh [subset_size>=2] [num_gpus] [gpu_ids_csv]"
    exit 1
fi

if ! [[ "$num_gpus" =~ ^[0-9]+$ ]] || [[ "$num_gpus" -lt 1 ]]; then
    echo "Invalid num_gpus: $num_gpus"
    exit 1
fi

mme_root_default="../Datasets/MME"
model_name_default="Qwen2.5-VL"
model_path_default="../llms/Qwen2.5-VL-7B-Instruct"
run_tag_default="mme_pca_subset${subset_size}"

mme_root="${mme_root:-${MME_ROOT:-$mme_root_default}}"
model_name="${model_name:-${MODEL_NAME:-$model_name_default}}"
model_path="${model_path:-${MODEL_PATH:-$model_path_default}}"
run_tag="${mme_pca_run_tag:-${MME_PCA_RUN_TAG:-$run_tag_default}}"
pca_layer_index="${pca_layer_index:-${PCA_LAYER_INDEX:--1}}"

if [[ "$mme_root" != /* ]]; then
    mme_root="$root_dir/$mme_root"
fi
if [[ "$model_path" != /* ]]; then
    model_path="$root_dir/$model_path"
fi

source_question_file="$mme_root/llava_mme.jsonl"
subset_dir="$root_dir/results/$model_name/mme/pca_subsets"
subset_question_file="$subset_dir/${run_tag}.jsonl"
mkdir -p "$subset_dir"

python - <<'PY' "$source_question_file" "$subset_question_file" "$subset_size"
import json
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
target_path = Path(sys.argv[2])
subset_size = int(sys.argv[3])

records = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if subset_size > len(records):
    subset_size = len(records)

target_path.parent.mkdir(parents=True, exist_ok=True)
with target_path.open("w", encoding="utf-8") as f:
    for record in records[:subset_size]:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"[MME_PCA] subset_question_file={target_path}")
print(f"[MME_PCA] subset_size={subset_size}")
PY

gpu_select_mode="manual"
if [[ -n "$gpu_ids_csv" && "$gpu_ids_csv" != "auto" ]]; then
    IFS=',' read -r -a gpu_ids <<< "$gpu_ids_csv"
else
    gpu_ids=()
    if mapfile -t gpu_ids < <(pick_top_gpus_by_free_mem "$num_gpus"); then
        gpu_select_mode="auto_free_mem"
    else
        gpu_select_mode="fallback_index"
        for ((i=0; i<num_gpus; i++)); do
            gpu_ids+=("$i")
        done
    fi
fi

if [[ "${#gpu_ids[@]}" -lt "$num_gpus" ]]; then
    echo "[MME_PCA] provided gpu ids fewer than num_gpus"
    exit 1
fi

echo "[MME_PCA] gpu_select_mode=$gpu_select_mode"
echo "[MME_PCA] gpu_ids=${gpu_ids[*]}"

methods=(base memvr evo)
for method in "${methods[@]}"; do
    results_root="$root_dir/results/$model_name/mme/$method/$run_tag"
    analysis_log_root="$results_root/analysis"
    rm -rf "$analysis_log_root" "$results_root/pca"
    mkdir -p "$analysis_log_root"

    echo "[MME_PCA] run method=$method run_tag=$run_tag"

    common_args=(
        -m qwen25.qwen_eval
        --model-path "$model_path"
        --question-file "$subset_question_file"
        --image-folder "$mme_root/MME_Benchmark_release_version"
        --dataset-name "MME"
        --analysis-log-dir "$analysis_log_root"
        --temperature 0.1
        --cuda-device "cuda:0"
        --method "$method"
        --collect-pca
        --num-chunks "$num_gpus"
        --max-new-tokens "2"
        --entropy-threshold "0.3"
        --starting-layer "8"
        --ending-layer "10"
        --retracing-ratio "0.12"
        --retrace-delay-layers "1"
        --retrace-target-layers ""
        --state-drift-threshold "0.4"
        --state-drift-pooling "mean"
    )

    answers_file="$results_root/answers.jsonl"
    mkdir -p "$results_root"
    rm -f "$answers_file"

    if [[ "$num_gpus" -eq 1 ]]; then
        gpu_id="${gpu_ids[0]}"
        CUDA_VISIBLE_DEVICES="$gpu_id" python "${common_args[@]}" \
            --answers-file "$answers_file" \
            --chunk-idx 0
    else
        pids=()
        part_files=()
        tmp_dir="$results_root/.tmp_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$tmp_dir"
        for ((chunk_idx=0; chunk_idx<num_gpus; chunk_idx++)); do
            gpu_id="${gpu_ids[$chunk_idx]}"
            part_file="$tmp_dir/chunk_${chunk_idx}.jsonl"
            part_files+=("$part_file")
            CUDA_VISIBLE_DEVICES="$gpu_id" python "${common_args[@]}" \
                --answers-file "$part_file" \
                --temperature 0 \
                --chunk-idx "$chunk_idx" &
            pids+=("$!")
        done

        for pid in "${pids[@]}"; do
            wait "$pid"
        done

        : > "$answers_file"
        for part_file in "${part_files[@]}"; do
            cat "$part_file" >> "$answers_file"
        done
    fi

    python -m qwen25.pca_analysis \
        --current-results-dir "$results_root" \
        --method "$method" \
        --run-tag "$run_tag" \
        --layer-index "$pca_layer_index"
done

compare_dir="$root_dir/results/$model_name/mme/pca_compare/$run_tag"
echo "[MME_PCA] done"
echo "[MME_PCA] subset_question_file=$subset_question_file"
echo "[MME_PCA] compare_dir=$compare_dir"