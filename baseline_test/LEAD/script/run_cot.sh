#!/bin/bash
# ============================================================
# 使用 Chain-of-Thought 方法运行评测
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES

python main.py \
    --model_name Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset data/physunibench.jsonl \
    --output_dir output/cot \
    --method cot \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 20 \
    --max_new_tokens 25600 \
    --seed 42 \
    "$@"
