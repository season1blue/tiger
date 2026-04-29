#!/bin/bash
# ============================================================
# PhysUniBench 评测快捷启动脚本
#
# 用法：
#   bash script/run.sh                    # 默认运行全部样本
#   bash script/run.sh --limit 10         # 仅运行前 10 条
#   bash script/run.sh --method cot       # 使用 CoT 方法
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

export CUDA_VISIBLE_DEVICES

python main.py \
    --model_name /mnt/data/ssz/llms/Qwen2.5-VL-7B-Instruct \
    --dataset data/physunibench.jsonl \
    --output_dir output \
    --method lead \
    --alpha 0.6 \
    --max_switch_count 5 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 20 \
    --max_new_tokens 25600 \
    --seed 42 \
    "$@"
