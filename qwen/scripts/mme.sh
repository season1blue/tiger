#!/bin/bash

set -euo pipefail

MME_ROOT="/data/ssz/Datasets/MME"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

# Usage:
#   bash qwen/scripts/mme.sh
#   bash qwen/scripts/mme.sh memvr
#   bash qwen/scripts/mme.sh memvr 4
#   bash qwen/scripts/mme.sh memvr 6 "1,2,3,4,5,6"
MODE="${1:-memvr}"
NUM_GPUS="${2:-1}"
GPU_IDS_CSV="${3:-}"
if [[ "$MODE" != "memvr" && "$MODE" != "none" ]]; then
    echo "Invalid mode: $MODE"
    echo "Usage: bash qwen/scripts/mme.sh [memvr|none] [num_gpus] [gpu_ids_csv]"
    exit 1
fi

if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [[ "$NUM_GPUS" -lt 1 ]]; then
    echo "Invalid num_gpus: $NUM_GPUS"
    exit 1
fi

EXPERIMENT="Qwen-VL-Chat/${MODE}"
ANSWERS_FILE="$MME_ROOT/answers/Qwen-VL-Chat/${MODE}.jsonl"
if [[ "$MODE" == "memvr" ]]; then
    APPLY_MEMVR="memvr"
else
    APPLY_MEMVR="none"
fi

echo "[MME] mode=$MODE"
echo "[MME] num_gpus=$NUM_GPUS"
echo "[MME] answers_file=$ANSWERS_FILE"

BASE_QUESTION_FILE="$MME_ROOT/llava_mme.jsonl"
TMP_DIR="$MME_ROOT/answers/Qwen-VL-Chat/.tmp_${MODE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TMP_DIR"
QUESTION_FILE="$BASE_QUESTION_FILE"

EXPECTED_COUNT=$(wc -l < "$QUESTION_FILE")
if [[ "$EXPECTED_COUNT" -eq 0 ]]; then
    echo "[MME] no samples to run"
    exit 1
fi

if [[ -n "$GPU_IDS_CSV" ]]; then
    IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_CSV"
else
    GPU_IDS=()
    for ((i=0; i<NUM_GPUS; i++)); do
        GPU_IDS+=("$i")
    done
fi

if [[ "${#GPU_IDS[@]}" -lt "$NUM_GPUS" ]]; then
    echo "[MME] provided gpu ids fewer than num_gpus"
    exit 1
fi

echo "[MME] question_file=$QUESTION_FILE"
echo "[MME] expected_count=$EXPECTED_COUNT"
echo "[MME] gpu_ids=${GPU_IDS[*]}"

rm -f "$ANSWERS_FILE"

if [[ "$NUM_GPUS" -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" python -m qwen.qwen_eval \
        --model-path /data/ssz/llms/Qwen-VL-Chat \
        --question-file "$QUESTION_FILE" \
        --image-folder "$MME_ROOT/MME_Benchmark_release_version" \
        --answers-file "$ANSWERS_FILE" \
        --temperature 0.1 \
        --cuda-device 'cuda:0' \
        --apply-memvr "$APPLY_MEMVR" \
        --retracing-ratio 0.25 \
        --entropy-threshold 0.95 \
        --max-new-tokens 2 \
        --starting-layer 8 \
        --ending-layer 16 \
        --num-chunks 1 \
        --chunk-idx 0
else
    PIDS=()
    PART_FILES=()
    for ((chunk_idx=0; chunk_idx<NUM_GPUS; chunk_idx++)); do
        gpu_id="${GPU_IDS[$chunk_idx]}"
        part_file="$TMP_DIR/chunk_${chunk_idx}.jsonl"
        PART_FILES+=("$part_file")
        echo "[MME] launch chunk=$chunk_idx gpu=$gpu_id -> $part_file"
        CUDA_VISIBLE_DEVICES="$gpu_id" python -m qwen.qwen_eval \
            --model-path /data/ssz/llms/Qwen-VL-Chat \
            --question-file "$QUESTION_FILE" \
            --image-folder "$MME_ROOT/MME_Benchmark_release_version" \
            --answers-file "$part_file" \
            --temperature 0 \
            --cuda-device 'cuda:0' \
            --apply-memvr "$APPLY_MEMVR" \
            --retracing-ratio 0.25 \
            --entropy-threshold 0.75 \
            --max-new-tokens 2 \
            --starting-layer 10 \
            --ending-layer 16 \
            --num-chunks "$NUM_GPUS" \
            --chunk-idx "$chunk_idx" &
        PIDS+=("$!")
    done

    for pid in "${PIDS[@]}"; do
        wait "$pid"
    done

    : > "$ANSWERS_FILE"
    for part_file in "${PART_FILES[@]}"; do
        cat "$part_file" >> "$ANSWERS_FILE"
    done
fi

ACTUAL_COUNT=$(wc -l < "$ANSWERS_FILE")
echo "[MME] actual_count=$ACTUAL_COUNT"
if [[ "$ACTUAL_COUNT" -ne "$EXPECTED_COUNT" ]]; then
    echo "[MME] mismatch count expected=$EXPECTED_COUNT actual=$ACTUAL_COUNT"
    echo "[MME] keep tmp dir for debugging: $TMP_DIR"
    exit 1
fi

echo "$EXPERIMENT"
cd "$MME_ROOT"
python convert_answer_to_mme.py --experiment "$EXPERIMENT"

cd eval_tool
python calculation.py --results_dir "answers/${EXPERIMENT}"