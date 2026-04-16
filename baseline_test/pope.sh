#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$script_dir/.." && pwd)"
cd "$root_dir"
export PYTHONPATH="$root_dir:${PYTHONPATH:-}"

# Usage:
#   bash baseline_test/pope.sh
#   bash baseline_test/pope.sh 1 "1"
#   bash baseline_test/pope.sh 4 "0,1,2,3"
num_gpus="${1:-1}"
gpu_ids_csv="${2:-}"

# -----------------------------------------------------------------------------
# Default parameters (edit here)
# -----------------------------------------------------------------------------
pope_root_default="../Datasets/POPE"
model_name_default="Qwen2.5-VL"
model_path_default="../llms/Qwen2.5-VL-7B-Instruct"
coco_image_folder_default="../Datasets/coco2014/images/val2014"
gqa_image_folder_default="../Datasets/GQA/images"

dataset_prefixes_default="coco"
splits_default="random"
limit_default="0"                   # 0 means full set.
max_new_tokens_default="2"
temperature_default="0"

# -----------------------------------------------------------------------------
# Resolve parameters (CLI/env override default)
# -----------------------------------------------------------------------------
pope_root="${pope_root:-${POPE_ROOT:-$pope_root_default}}"
model_name="${model_name:-${MODEL_NAME:-$model_name_default}}"
model_path="${model_path:-${MODEL_PATH:-$model_path_default}}"
coco_image_folder="${coco_image_folder:-${COCO_IMAGE_FOLDER:-$coco_image_folder_default}}"
gqa_image_folder="${gqa_image_folder:-${GQA_IMAGE_FOLDER:-$gqa_image_folder_default}}"
run_tag="${pope_run_tag:-${POPE_RUN_TAG:-}}"

dataset_prefixes_csv="${dataset_prefixes_csv:-${DATASET_PREFIXES:-$dataset_prefixes_default}}"
splits_csv="${splits_csv:-${SPLITS:-$splits_default}}"
limit="${limit:-${LIMIT:-$limit_default}}"
max_new_tokens="${max_new_tokens:-${MAX_NEW_TOKENS:-$max_new_tokens_default}}"
temperature="${temperature:-${TEMPERATURE:-$temperature_default}}"

# Normalize paths once so later `cd` does not affect path resolution.
if [[ "$pope_root" != /* ]]; then
    pope_root="$root_dir/$pope_root"
fi
if [[ "$model_path" != /* ]]; then
    model_path="$root_dir/$model_path"
fi
if [[ "$coco_image_folder" != /* ]]; then
    coco_image_folder="$root_dir/$coco_image_folder"
fi
if [[ "$gqa_image_folder" != /* ]]; then
    gqa_image_folder="$root_dir/$gqa_image_folder"
fi

if ! [[ "$num_gpus" =~ ^[0-9]+$ ]] || [[ "$num_gpus" -lt 1 ]]; then
    echo "Invalid num_gpus: $num_gpus"
    exit 1
fi

if ! [[ "$limit" =~ ^[0-9]+$ ]]; then
    echo "Invalid limit: $limit"
    exit 1
fi

if [[ -n "$run_tag" ]]; then
    results_root="$root_dir/results/$model_name/pope/baseline/$run_tag"
    experiment="$model_name/baseline/$run_tag"
else
    results_root="$root_dir/results/$model_name/pope/baseline"
    experiment="$model_name/baseline"
fi

mkdir -p "$results_root"

tmp_root="$results_root/.tmp_baseline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$tmp_root"

result_json="$results_root/result.json"
rm -f "$result_json"

IFS=',' read -r -a dataset_prefixes <<< "$dataset_prefixes_csv"
IFS=',' read -r -a splits <<< "$splits_csv"

if [[ -n "$gpu_ids_csv" ]]; then
    IFS=',' read -r -a gpu_ids <<< "$gpu_ids_csv"
else
    gpu_ids=()
    for ((i=0; i<num_gpus; i++)); do
        gpu_ids+=("$i")
    done
fi

if [[ "${#gpu_ids[@]}" -lt "$num_gpus" ]]; then
    echo "[POPE-Baseline] provided gpu ids fewer than num_gpus"
    exit 1
fi

get_image_folder() {
    local dataset_prefix="$1"
    case "$dataset_prefix" in
        coco)
            echo "$coco_image_folder"
            ;;
        gqa)
            echo "$gqa_image_folder"
            ;;
        aokvqa)
            echo "$coco_image_folder"
            ;;
        *)
            echo ""
            ;;
    esac
}

run_case() {
    local dataset_prefix="$1"
    local split="$2"

    local dataset_group="${dataset_prefix}_POPE"
    local dataset_dir="$pope_root/output/$dataset_group"
    local metric_key="${dataset_group}_${split}"

    if [[ ! -d "$dataset_dir" ]]; then
        echo "[POPE-Baseline] skip $metric_key: dataset dir not found ($dataset_dir)"
        return 0
    fi

    local image_folder
    image_folder="$(get_image_folder "$dataset_prefix")"
    if [[ -z "$image_folder" || ! -d "$image_folder" ]]; then
        echo "[POPE-Baseline] skip $metric_key: image folder not found ($image_folder)"
        return 0
    fi

    local full_question_file
    full_question_file="$(find "$dataset_dir" -maxdepth 1 -type f -name "${dataset_prefix}_pope*_${split}.json" | head -n 1)"
    if [[ -z "$full_question_file" ]]; then
        echo "[POPE-Baseline] skip $metric_key: question file not found"
        return 0
    fi

    local question_file="$tmp_root/${metric_key}.jsonl"
    local answers_file="$results_root/${metric_key}.jsonl"
    local case_tmp_dir="$tmp_root/${metric_key}"
    mkdir -p "$case_tmp_dir"

    python "$root_dir/llava/scripts/prepare_pope_questions.py" \
        --input "$full_question_file" \
        --output "$question_file" \
        --limit "$limit"

    local expected_count
    expected_count=$(wc -l < "$question_file")
    if [[ "$expected_count" -eq 0 ]]; then
        echo "[POPE-Baseline] skip $metric_key: no samples after prepare"
        return 0
    fi

    echo "[POPE-Baseline] metric_key=$metric_key"
    echo "[POPE-Baseline] question_file=$question_file"
    echo "[POPE-Baseline] image_folder=$image_folder"
    echo "[POPE-Baseline] expected_count=$expected_count"
    echo "[POPE-Baseline] gpu_ids=${gpu_ids[*]}"

    rm -f "$answers_file"

    if [[ "$num_gpus" -eq 1 ]]; then
        CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" python "$root_dir/baseline_test/qwen_eval.py" \
            --model-path "$model_path" \
            --question-file "$question_file" \
            --image-folder "$image_folder" \
            --answers-file "$answers_file" \
            --temperature "$temperature" \
            --max-new-tokens "$max_new_tokens" \
            --cuda-device 'cuda:0' \
            --num-chunks 1 \
            --chunk-idx 0
    else
        local pids=()
        local part_files=()
        for ((chunk_idx=0; chunk_idx<num_gpus; chunk_idx++)); do
            local gpu_id="${gpu_ids[$chunk_idx]}"
            local part_file="$case_tmp_dir/chunk_${chunk_idx}.jsonl"
            part_files+=("$part_file")
            echo "[POPE-Baseline] launch metric=$metric_key chunk=$chunk_idx gpu=$gpu_id -> $part_file"
            CUDA_VISIBLE_DEVICES="$gpu_id" python "$root_dir/baseline_test/qwen_eval.py" \
                --model-path "$model_path" \
                --question-file "$question_file" \
                --image-folder "$image_folder" \
                --answers-file "$part_file" \
                --temperature "$temperature" \
                --max-new-tokens "$max_new_tokens" \
                --cuda-device 'cuda:0' \
                --num-chunks "$num_gpus" \
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

    local actual_count
    actual_count=$(wc -l < "$answers_file")
    echo "[POPE-Baseline] actual_count=$actual_count"
    if [[ "$actual_count" -ne "$expected_count" ]]; then
        echo "[POPE-Baseline] mismatch count for $metric_key expected=$expected_count actual=$actual_count"
        echo "[POPE-Baseline] keep tmp dir for debugging: $case_tmp_dir"
        return 1
    fi

    python "$root_dir/llava/scripts/pope_eval_metrics.py" \
        --question-file "$question_file" \
        --answers-file "$answers_file" \
        --metric-key "$metric_key" \
        --result-json "$result_json"
}

echo "[POPE-Baseline] num_gpus=$num_gpus"
echo "[POPE-Baseline] run_tag=${run_tag:-none}"
echo "[POPE-Baseline] model_path=$model_path"
echo "[POPE-Baseline] pope_root=$pope_root"
echo "[POPE-Baseline] dataset_prefixes=$dataset_prefixes_csv"
echo "[POPE-Baseline] splits=$splits_csv"
echo "[POPE-Baseline] limit=$limit"
echo "[POPE-Baseline] temperature=$temperature"
echo "[POPE-Baseline] max_new_tokens=$max_new_tokens"
echo "[POPE-Baseline] results_root=$results_root"
echo "[POPE-Baseline] tmp_root=$tmp_root"
echo "[POPE-Baseline] result_json=$result_json"

for dataset_prefix in "${dataset_prefixes[@]}"; do
    dataset_prefix="${dataset_prefix// /}"
    [[ -z "$dataset_prefix" ]] && continue
    for split in "${splits[@]}"; do
        split="${split// /}"
        [[ -z "$split" ]] && continue
        run_case "$dataset_prefix" "$split"
    done
done

pope_answer_root="$pope_root/answer/$experiment"
mkdir -p "$pope_answer_root"
cp -f "$result_json" "$pope_answer_root/result.json"
find "$results_root" -maxdepth 1 -type f -name "*_POPE_*.jsonl" -exec cp -f {} "$pope_answer_root/" \;

echo ""
echo "POPE baseline evaluation finished."
echo "answers directory: $results_root"
echo "summary json: $result_json"
echo "copied answers to: $pope_answer_root"
