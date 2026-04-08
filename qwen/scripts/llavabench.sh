#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

model="Qwen-VL-Chat"
method="memvr"
python -m qwen.eval.qwen_eval \
    --model-path $model \
    --question-file /data/ssz/Datasets/llava/llava-bench-in-the-wild/questions.jsonl \
    --image-folder /data/ssz/Datasets/llava/llava-bench-in-the-wild/images \
    --answers-file /data/ssz/Datasets/llava/llava-bench-in-the-wild/answers/$model/${method}.jsonl \
    --temperature 0 \
    --cuda-device 'cuda:0' \
    --apply-memvr 'memvr' \
    --retracing-ratio 0.28 \
    --entropy-threshold 0.75 \
    --max-new-tokens 1024 \
    --starting-layer 9 \
    --ending-layer 16 \