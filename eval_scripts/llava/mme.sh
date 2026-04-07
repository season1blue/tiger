#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

MME_ROOT="/data/ssz/Datasets/MME"
EXPERIMENT="llava-v1.5-7b/memvr"
CUDA_VISIBLE_DEVICES=2

python -m llava.eval.llava_model_vqa_loader \
    --model-path /data/ssz/llms/llava-v1.5-7b \
    --question-file "$MME_ROOT/llava_mme.jsonl" \
    --image-folder "$MME_ROOT/MME_Benchmark_release_version" \
    --answers-file "$MME_ROOT/answers/llava-v1.5-7b/memvr.jsonl" \
    --temperature 0 \
    --cuda-device 'cuda:0' \
    --apply-memvr 'none' \
    --retracing-ratio 0.12 \
    --entropy-threshold 0.75 \
    --max-new-tokens 1 \
    --starting-layer 5 \
    --ending-layer 16 

cd "$MME_ROOT"
python "$MME_ROOT/convert_answer_to_mme.py" --experiment "$EXPERIMENT"

cd eval_tool
python calculation.py --results_dir "answers/${EXPERIMENT}"