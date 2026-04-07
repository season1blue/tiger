#!/bin/bash

MME_ROOT="/data/ssz/Datasets/MME"
CUDA_VISIBLE_DEVICES=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

# Usage:
#   bash eval_scripts/qwen/mme.sh          # default: memvr
#   bash eval_scripts/qwen/mme.sh memvr    # enable MemVR
#   bash eval_scripts/qwen/mme.sh none     # disable MemVR
MODE="${1:-memvr}"
if [[ "$MODE" != "memvr" && "$MODE" != "none" ]]; then
    echo "Invalid mode: $MODE"
    echo "Usage: bash eval_scripts/qwen/mme.sh [memvr|none]"
    exit 1
fi

EXPERIMENT="Qwen-VL-Chat/${MODE}"
ANSWERS_FILE="$MME_ROOT/answers/Qwen-VL-Chat/${MODE}.jsonl"
if [[ "$MODE" == "memvr" ]]; then
    APPLY_MEMVR="memvr"
else
    APPLY_MEMVR="none"
fi

echo "[MME] mode=$MODE"
echo "[MME] answers_file=$ANSWERS_FILE"

python -m qwen.eval.qwen_eval \
    --model-path /data/ssz/llms/Qwen-VL-Chat \
    --question-file "$MME_ROOT/llava_mme.jsonl" \
    --image-folder "$MME_ROOT/MME_Benchmark_release_version" \
    --answers-file "$ANSWERS_FILE" \
    --temperature 0 \
    --cuda-device 'cuda:0' \
    --apply-memvr "$APPLY_MEMVR" \
    --retracing-ratio 0.25 \
    --entropy-threshold 0.75 \
    --max-new-tokens 2 \
    --starting-layer 5 \
    --ending-layer 16

echo "$EXPERIMENT"
cd "$MME_ROOT"
python convert_answer_to_mme.py --experiment "$EXPERIMENT"

cd eval_tool
python calculation.py --results_dir "answers/${EXPERIMENT}"