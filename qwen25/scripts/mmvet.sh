#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

model='Qwen2.5-VL-7B-Instruct'
method='memvr'
QUESTION_JSON="/data/ssz/Datasets/mmvet/mm-vet.json"
QUESTION_JSONL="/data/ssz/Datasets/mmvet/mm-vet.jsonl"

python "$ROOT_DIR/llava/scripts/prepare_mmvet_questions.py" \
    --src "$QUESTION_JSON" \
    --dst "$QUESTION_JSONL"

python -m qwen25.qwen_eval \
    --model-path $model \
    --question-file "$QUESTION_JSONL" \
    --image-folder /data/ssz/Datasets/mmvet/images \
    --answers-file /data/ssz/Datasets/mmvet/answers/$model/${method}.jsonl \
    --temperature 0 \
    --cuda-device 'cuda:0' \
    --apply-memvr 'memvr' \
    --retracing-ratio 0.28 \
    --entropy-threshold 0.75 \
    --max-new-tokens 1024 \
    --starting-layer 9 \
    --ending-layer 16 \

mkdir -p /data/ssz/Datasets/mmvet/results/$model
python "$ROOT_DIR/utils/convert_mmvet_for_eval.py" \
    --src /data/ssz/Datasets/mmvet/answers/$model/${method}.jsonl \
    --dst /data/ssz/Datasets/mmvet/results/$model/${method}.json \