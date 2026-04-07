#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

MMVET_ROOT="/data/ssz/Datasets/mmvet"
QUESTION_JSON="$MMVET_ROOT/mm-vet.json"
QUESTION_JSONL="$MMVET_ROOT/mm-vet.jsonl"

model='llava-v1.5-7b'
method='memvr'

python "$SCRIPT_DIR/prepare_mmvet_questions.py" \
    --src "$QUESTION_JSON" \
    --dst "$QUESTION_JSONL"

python -m llava.eval.llava_model_vqa \
    --model-path $model \
    --question-file "$QUESTION_JSONL" \
    --image-folder "$MMVET_ROOT/images" \
    --answers-file "$MMVET_ROOT/answers/$model/${method}.jsonl" \
    --temperature 0 \
    --cuda-device 'cuda:0' \
    --apply-memvr 'memvr' \
    --retracing-ratio 0.12 \
    --entropy-threshold 0.75 \
    --max-new-tokens 1024 \
    --starting-layer 5 \
    --ending-layer 16

mkdir -p "$MMVET_ROOT/results/$model"
python "$ROOT_DIR/utils/convert_mmvet_for_eval.py" \
    --src "$MMVET_ROOT/answers/$model/${method}.jsonl" \
    --dst "$MMVET_ROOT/results/$model/${method}.json"