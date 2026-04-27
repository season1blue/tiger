#!/bin/bash

set -euo pipefail

# Usage:
#   bash qwen35/scripts/llavabench.sh
#   bash qwen35/scripts/llavabench.sh evo 2
#   bash qwen35/scripts/llavabench.sh evo 1 auto
#   bash qwen35/scripts/llavabench.sh base 1 3
method="${1:-memvr}"
num_gpus="${2:-1}"
gpu_ids_csv="${3:-}"

if [[ "$method" != "base" && "$method" != "memvr" && "$method" != "evo" ]]; then
    echo "Invalid method: $method"
    echo "Usage: bash qwen35/scripts/llavabench.sh [base|memvr|evo] [num_gpus] [gpu_ids_csv|auto]"
    exit 1
fi

if ! [[ "$num_gpus" =~ ^[0-9]+$ ]] || [[ "$num_gpus" -lt 1 ]]; then
    echo "Invalid num_gpus: $num_gpus"
    exit 1
fi

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

dataset_root="${LLAVABENCH_ROOT:-../Datasets/llava-bench-in-the-wild}"
model_name="${MODEL_NAME:-Qwen3.5-VL}"
model_path="${MODEL_PATH:-../llms/Qwen3.5-9B}"
answers_file="$dataset_root/answers/$model_name/${method}.jsonl"
reviews_dir="$dataset_root/reviews"
review_output="$reviews_dir/${model_name}_${method}.review.jsonl"
summary_output="$reviews_dir/${model_name}_${method}.summary.json"

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
    echo "[LLaVABench] provided gpu ids fewer than num_gpus"
    exit 1
fi

selected_visible_gpus="$(IFS=,; echo "${gpu_ids[*]}")"

echo "[LLaVABench] method=$method"
echo "[LLaVABench] num_gpus=$num_gpus gpu_select_mode=$gpu_select_mode"
echo "[LLaVABench] visible_gpus=$selected_visible_gpus"

mkdir -p "$(dirname "$answers_file")" "$reviews_dir"

if [[ -s "$answers_file" ]]; then
    echo "[LLaVABench] found existing answers, skip generation: $answers_file"
else
    CUDA_VISIBLE_DEVICES="$selected_visible_gpus" python -m qwen35.qwen_eval \
        --model-path "$model_path" \
        --question-file "$dataset_root/questions.jsonl" \
        --image-folder "$dataset_root/images" \
        --answers-file "$answers_file" \
        --temperature 0 \
        --cuda-device 'cuda:0' \
        --method "$method" \
        --retracing-ratio "${RETRACING_RATIO:-0.12}" \
        --retrace-delay-layers "${RETRACE_DELAY_LAYERS:-1}" \
        --retrace-target-layers "${RETRACE_TARGET_LAYERS:-}" \
        --state-drift-threshold "${STATE_DRIFT_THRESHOLD:-0.5}" \
        --state-drift-pooling "${STATE_DRIFT_POOLING:-mean}" \
        --entropy-threshold "${ENTROPY_THRESHOLD:-0.75}" \
        --max-new-tokens "${MAX_NEW_TOKENS:-512}" \
        --starting-layer "${STARTING_LAYER:-8}" \
        --ending-layer "${ENDING_LAYER:-10}" \
        --disable-output-postprocess
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "[LLaVABench] OPENAI_API_KEY not set, skip GPT review."
    echo "[LLaVABench] you can run review later with utils/eval_llavabench.py"
    exit 0
fi

api_mode="${LLAVABENCH_API_MODE:-responses}"
overwrite_review="${LLAVABENCH_REVIEW_OVERWRITE:-1}"
base_url_arg=()
if [[ -n "${OPENAI_BASE_URL:-}" ]]; then
    base_url_arg=(--base-url "$OPENAI_BASE_URL")
fi

overwrite_arg=()
if [[ "$overwrite_review" == "1" ]]; then
    overwrite_arg=(--overwrite-review)
fi

echo "[LLaVABench] start GPT review: mode=$api_mode output=$review_output"
python utils/eval_llavabench.py \
    --question "$dataset_root/questions.jsonl" \
    --context "$dataset_root/context.jsonl" \
    --reference-answer "$dataset_root/answers_gpt4.jsonl" \
    --model-answer "$answers_file" \
    --review-output "$review_output" \
    --summary-output "$summary_output" \
    --reviewer-model "${LLAVABENCH_REVIEW_MODEL:-gpt-4o-mini}" \
    --api-mode "$api_mode" \
    "${overwrite_arg[@]}" \
    "${base_url_arg[@]}"

echo "[LLaVABench] review_file=$review_output"
echo "[LLaVABench] summary_file=$summary_output"
