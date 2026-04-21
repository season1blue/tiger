#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

MME_ROOT="/data/ssz/Datasets/MME"
MEMVR_MODE="evo"
EXPERIMENT="llava-v1.5-7b/${MEMVR_MODE}"
export CUDA_VISIBLE_DEVICES=2

case "$MEMVR_MODE" in
    base)
        python -m llava.eval.llava_model_vqa_loader \
            --model-path /data/ssz/llms/llava-v1.5-7b \
            --question-file "$MME_ROOT/llava_mme.jsonl" \
            --image-folder "$MME_ROOT/MME_Benchmark_release_version" \
            --answers-file "$MME_ROOT/answers/llava-v1.5-7b/${MEMVR_MODE}.jsonl" \
            --temperature 0 \
            --cuda-device 'cuda:0' \
            --memvr-mode base \
            --max-new-tokens 1 \
            --starting-layer 5 \
            --ending-layer 32
        ;;
    memvr)
        python -m llava.eval.llava_model_vqa_loader \
            --model-path /data/ssz/llms/llava-v1.5-7b \
            --question-file "$MME_ROOT/llava_mme.jsonl" \
            --image-folder "$MME_ROOT/MME_Benchmark_release_version" \
            --answers-file "$MME_ROOT/answers/llava-v1.5-7b/${MEMVR_MODE}.jsonl" \
            --temperature 0 \
            --cuda-device 'cuda:0' \
            --memvr-mode memvr \
            --retracing-ratio 0.12 \
            --entropy-threshold 0.75 \
            --retrace-delay-layers 1 \
            --max-new-tokens 1 \
            --starting-layer 5 \
            --ending-layer 32
        ;;
    evo)
        python -m llava.eval.llava_model_vqa_loader \
            --model-path /data/ssz/llms/llava-v1.5-7b \
            --question-file "$MME_ROOT/llava_mme.jsonl" \
            --image-folder "$MME_ROOT/MME_Benchmark_release_version" \
            --answers-file "$MME_ROOT/answers/llava-v1.5-7b/${MEMVR_MODE}.jsonl" \
            --temperature 0 \
            --cuda-device 'cuda:0' \
            --memvr-mode evo \
            --retracing-ratio 0.12 \
            --entropy-threshold 0.75 \
            --state-drift-threshold 0.5 \
            --state-drift-pooling 'mean' \
            --retrace-delay-layers 1 \
            --max-new-tokens 1 \
            --starting-layer 5 \
            --ending-layer 32
        ;;
    *)
        echo "Unknown MEMVR_MODE: $MEMVR_MODE. Use base, memvr, or evo."
        exit 1
        ;;
esac

cd "$MME_ROOT"
python "$MME_ROOT/convert_answer_to_mme.py" --experiment "$EXPERIMENT"

cd eval_tool
python calculation.py --results_dir "answers/${EXPERIMENT}"