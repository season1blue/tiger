#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$script_dir/../.." && pwd)"
cd "$root_dir"
export PYTHONPATH="$root_dir:${PYTHONPATH:-}"

# Usage:
#   bash qwen25/scripts/mme.sh
#   bash qwen25/scripts/mme.sh memvr
#   bash qwen25/scripts/mme.sh memvr 4
#   bash qwen25/scripts/mme.sh memvr 4 "0,1,2,3"
#   bash qwen25/scripts/mme.sh memvr 1 "1"
mode="${1:-memvr}"
num_gpus="${2:-1}"
gpu_ids_csv="${3:-}"

# -----------------------------------------------------------------------------
# Default parameters (edit here)
# -----------------------------------------------------------------------------
mme_root_default="../Datasets/MME"
model_name_default="Qwen2.5-VL"
model_path_default="../llms/Qwen2.5-VL-7B-Instruct"

entropy_threshold_default="0.75"
starting_layer_default="5"
ending_layer_default="16"
retracing_ratio_default="0.12"
retrace_delay_layers_default="1"
retrace_target_layers_default="12"
max_new_tokens_default="2"

# -----------------------------------------------------------------------------
# Resolve parameters (CLI/env override default)
# -----------------------------------------------------------------------------
mme_root="${mme_root:-${MME_ROOT:-$mme_root_default}}"
model_name="${model_name:-${MODEL_NAME:-$model_name_default}}"
model_path="${model_path:-${MODEL_PATH:-$model_path_default}}"
run_tag="${mme_run_tag:-${MME_RUN_TAG:-}}"

# Normalize paths once so later `cd` does not affect path resolution.
if [[ "$mme_root" != /* ]]; then
    mme_root="$root_dir/$mme_root"
fi
if [[ "$model_path" != /* ]]; then
    model_path="$root_dir/$model_path"
fi

if [[ "$mode" != "memvr" && "$mode" != "none" ]]; then
    echo "Invalid mode: $mode"
    echo "Usage: bash qwen25/scripts/mme.sh [memvr|none] [num_gpus] [gpu_ids_csv]"
    exit 1
fi

if ! [[ "$num_gpus" =~ ^[0-9]+$ ]] || [[ "$num_gpus" -lt 1 ]]; then
    echo "Invalid num_gpus: $num_gpus"
    exit 1
fi

if [[ -n "$run_tag" ]]; then
    results_root="$root_dir/results/$model_name/mme/$mode/$run_tag"
else
    results_root="$root_dir/results/$model_name/mme/$mode"
fi
answers_file="$results_root/answers.jsonl"
eval_results_dir="$results_root/eval_answers"

if [[ "$mode" == "memvr" ]]; then
    apply_memvr="memvr"
else
    apply_memvr="none"
fi

entropy_threshold="${entropy_threshold:-${ENTROPY_THRESHOLD:-}}"
starting_layer="${starting_layer:-${STARTING_LAYER:-}}"
ending_layer="${ending_layer:-${ENDING_LAYER:-}}"
retracing_ratio="${retracing_ratio:-${RETRACING_RATIO:-$retracing_ratio_default}}"
retrace_delay_layers="${retrace_delay_layers:-${RETRACE_DELAY_LAYERS:-$retrace_delay_layers_default}}"
retrace_target_layers="${retrace_target_layers:-${RETRACE_TARGET_LAYERS:-$retrace_target_layers_default}}"
max_new_tokens="${max_new_tokens:-${MAX_NEW_TOKENS:-$max_new_tokens_default}}"

if [[ -z "$entropy_threshold" ]]; then
    entropy_threshold="$entropy_threshold_default"
fi

if [[ -z "$starting_layer" ]]; then
    starting_layer="$starting_layer_default"
fi

if [[ -z "$ending_layer" ]]; then
    ending_layer="$ending_layer_default"
fi

mkdir -p "$results_root"

echo "[MME] mode=$mode"
echo "[MME] num_gpus=$num_gpus"
echo "[MME] run_tag=${run_tag:-none}"
echo "[MME] entropy_threshold=$entropy_threshold"
echo "[MME] starting_layer=$starting_layer"
echo "[MME] ending_layer=$ending_layer"
echo "[MME] retracing_ratio=$retracing_ratio"
echo "[MME] retrace_delay_layers=$retrace_delay_layers"
echo "[MME] retrace_target_layers=${retrace_target_layers:-none}"
echo "[MME] max_new_tokens=$max_new_tokens"
echo "[MME] model_path=$model_path"
echo "[MME] answers_file=$answers_file"

base_question_file="$mme_root/llava_mme.jsonl"
tmp_dir="$results_root/.tmp_${mode}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$tmp_dir"
question_file="$base_question_file"

expected_count=$(wc -l < "$question_file")
if [[ "$expected_count" -eq 0 ]]; then
    echo "[MME] no samples to run"
    exit 1
fi

if [[ -n "$gpu_ids_csv" ]]; then
    IFS=',' read -r -a gpu_ids <<< "$gpu_ids_csv"
else
    gpu_ids=()
    for ((i=0; i<num_gpus; i++)); do
        gpu_ids+=("$i")
    done
fi

if [[ "${#gpu_ids[@]}" -lt "$num_gpus" ]]; then
    echo "[MME] provided gpu ids fewer than num_gpus"
    exit 1
fi

echo "[MME] question_file=$question_file"
echo "[MME] expected_count=$expected_count"
echo "[MME] gpu_ids=${gpu_ids[*]}"

rm -f "$answers_file"

if [[ "$num_gpus" -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" python -m qwen25.qwen_eval \
        --model-path "$model_path" \
        --question-file "$question_file" \
        --image-folder "$mme_root/MME_Benchmark_release_version" \
        --answers-file "$answers_file" \
        --temperature 0.1 \
        --cuda-device 'cuda:0' \
        --apply-memvr "$apply_memvr" \
        --retracing-ratio "$retracing_ratio" \
        --retrace-delay-layers "$retrace_delay_layers" \
        --retrace-target-layers "$retrace_target_layers" \
        --entropy-threshold "$entropy_threshold" \
        --max-new-tokens "$max_new_tokens" \
        --starting-layer "$starting_layer" \
        --ending-layer "$ending_layer" \
        --num-chunks 1 \
        --chunk-idx 0
else
    pids=()
    part_files=()
    for ((chunk_idx=0; chunk_idx<num_gpus; chunk_idx++)); do
        gpu_id="${gpu_ids[$chunk_idx]}"
        part_file="$tmp_dir/chunk_${chunk_idx}.jsonl"
        part_files+=("$part_file")
        echo "[MME] launch chunk=$chunk_idx gpu=$gpu_id -> $part_file"
        CUDA_VISIBLE_DEVICES="$gpu_id" python -m qwen25.qwen_eval \
            --model-path "$model_path" \
            --question-file "$question_file" \
            --image-folder "$mme_root/MME_Benchmark_release_version" \
            --answers-file "$part_file" \
            --temperature 0 \
            --cuda-device 'cuda:0' \
            --apply-memvr "$apply_memvr" \
            --retracing-ratio "$retracing_ratio" \
            --retrace-delay-layers "$retrace_delay_layers" \
            --retrace-target-layers "$retrace_target_layers" \
            --entropy-threshold "$entropy_threshold" \
            --max-new-tokens "$max_new_tokens" \
            --starting-layer "$starting_layer" \
            --ending-layer "$ending_layer" \
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

actual_count=$(wc -l < "$answers_file")
echo "[MME] actual_count=$actual_count"
if [[ "$actual_count" -ne "$expected_count" ]]; then
    echo "[MME] mismatch count expected=$expected_count actual=$actual_count"
    echo "[MME] keep tmp dir for debugging: $tmp_dir"
    exit 1
fi

# convert_answer_to_mme.py expects the canonical layout under $mme_root.
if [[ -n "$run_tag" ]]; then
    experiment="$model_name/$mode/$run_tag"
else
    experiment="$model_name/$mode"
fi
mme_answers_file="$mme_root/answers/${experiment}.jsonl"
mkdir -p "$(dirname "$mme_answers_file")"
cp "$answers_file" "$mme_answers_file"

echo "$experiment"
cd "$mme_root"
python convert_answer_to_mme.py --experiment "$experiment"

rm -rf "$eval_results_dir"
mkdir -p "$(dirname "$eval_results_dir")"
cp -r "$mme_root/eval_tool/answers/${experiment}" "$eval_results_dir"

cd eval_tool
python calculation.py --results_dir "$eval_results_dir"