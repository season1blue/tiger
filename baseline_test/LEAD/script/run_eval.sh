#!/bin/bash
# ============================================================
# 仅评估已有推理结果（不运行推理）
#
# 用法：
#   bash script/run_eval.sh
#   bash script/run_eval.sh --results output/cot/results.jsonl
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

python main.py \
    --eval_only \
    "$@"
