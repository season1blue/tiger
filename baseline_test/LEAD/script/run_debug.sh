#!/bin/bash
# ============================================================
# 调试模式：仅运行 5 条样本，限制生成长度
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
    --output_dir output/debug \
    --method lead \
    --limit 5 \
    --max_new_tokens 512 \
    --seed 42 \
    "$@"
