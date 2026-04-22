#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

model="Qwen3.5-VL-7B-Instruct"
method="${1:-base}"

if [[ "$method" != "base" && "$method" != "memvr" && "$method" != "evo" ]]; then
    echo "Invalid method: $method"
    echo "Usage: bash qwen35/scripts/chair.sh [base|memvr|evo]"
    exit 1
fi

python -m qwen35.qwen_eval \
    --task-type chair \
    --model-path /data/ssz/llms/Qwen3.5-VL-7B-Instruct \
    --question-file /data/ssz/Datasets/chair/annotations/instances_val2014.json \
    --image-folder /data/ssz/Datasets/coco2014-val/images/val2014 \
    --answers-file /data/ssz/Datasets/chair/answers/$model/${method}.jsonl \
    --temperature 0 \
    --cuda-device 'cuda:0' \
    --method "$method" \
    --retracing-ratio 0.29 \
    --entropy-threshold 0.75 \
    --max-new-tokens 1024 \
    --starting-layer 9 \
    --ending-layer 16 \
