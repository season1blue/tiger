#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$script_dir/../.." && pwd)"
cd "$root_dir"
export PYTHONPATH="$root_dir:${PYTHONPATH:-}"

# Usage:
#   bash qwen35/scripts/mmvet.sh
#   bash qwen35/scripts/mmvet.sh base
#   bash qwen35/scripts/mmvet.sh memvr 0
#   bash qwen35/scripts/mmvet.sh evo auto
method="${1:-base}"
gpu_id="${2:-auto}"

pick_top_gpu_by_free_mem() {
    local query=""

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 1
    fi

    query="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ -z "$query" ]]; then
        return 1
    fi

    echo "$query" \
        | awk -F',' '{gsub(/ /, "", $1); gsub(/ /, "", $2); print $1 "," $2}' \
        | sort -t',' -k2,2nr \
        | head -n 1 \
        | cut -d',' -f1
}

convert_mmvet_results() {
    local src_file="$1"
    local dst_file="$2"

    python - <<'PY' "$src_file" "$dst_file"
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("src", type=str)
parser.add_argument("dst", type=str)
args = parser.parse_args()

src_path = Path(args.src)
dst_path = Path(args.dst)

cur_result = {}
for line in src_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    data = json.loads(line)
    qid = data["question_id"]
    cur_result[f"v1_{qid}"] = data["text"]

dst_path.parent.mkdir(parents=True, exist_ok=True)
with dst_path.open("w", encoding="utf-8") as f:
    json.dump(cur_result, f, indent=2, ensure_ascii=False)

print(f"[MMVet] converted_count={len(cur_result)}")
print(f"[MMVet] converted_results_file={dst_path}")
PY
}

if [[ "$method" != "base" && "$method" != "memvr" && "$method" != "evo" ]]; then
    echo "Invalid method: $method"
    echo "Usage: bash qwen35/scripts/mmvet.sh [base|memvr|evo] [gpu_id]"
    exit 1
fi

gpu_select_mode="manual"
if [[ -z "$gpu_id" || "$gpu_id" == "auto" ]]; then
    if selected_gpu="$(pick_top_gpu_by_free_mem)"; then
        gpu_id="$selected_gpu"
        gpu_select_mode="auto_free_mem"
    else
        gpu_id="0"
        gpu_select_mode="fallback_index"
    fi
fi

mmvet_root_default="../Datasets/mm-vet"
model_name_default="Qwen3.5"
model_path_default="../llms/Qwen3.5-9B"

entropy_threshold_default="0.3"
starting_layer_default="5"
ending_layer_default="16"
retracing_ratio_default="0.12"
retrace_delay_layers_default="1"
retrace_target_layers_default=""
state_drift_threshold_default="0.4"
state_drift_pooling_default="mean"
max_new_tokens_default="1024"

mmvet_root="${mmvet_root:-${MMVET_ROOT:-$mmvet_root_default}}"
model_name="${model_name:-${MODEL_NAME:-$model_name_default}}"
model_path="${model_path:-${MODEL_PATH:-$model_path_default}}"
run_tag="${mmvet_run_tag:-${MMVET_RUN_TAG:-}}"

entropy_threshold="${entropy_threshold:-${ENTROPY_THRESHOLD:-$entropy_threshold_default}}"
starting_layer="${starting_layer:-${STARTING_LAYER:-$starting_layer_default}}"
ending_layer="${ending_layer:-${ENDING_LAYER:-$ending_layer_default}}"
retracing_ratio="${retracing_ratio:-${RETRACING_RATIO:-$retracing_ratio_default}}"
retrace_delay_layers="${retrace_delay_layers:-${RETRACE_DELAY_LAYERS:-$retrace_delay_layers_default}}"
retrace_target_layers="${retrace_target_layers:-${RETRACE_TARGET_LAYERS:-$retrace_target_layers_default}}"
state_drift_threshold="${state_drift_threshold:-${STATE_DRIFT_THRESHOLD:-$state_drift_threshold_default}}"
state_drift_pooling="${state_drift_pooling:-${STATE_DRIFT_POOLING:-$state_drift_pooling_default}}"
max_new_tokens="${max_new_tokens:-${MAX_NEW_TOKENS:-$max_new_tokens_default}}"

if [[ "$mmvet_root" != /* ]]; then
    mmvet_root="$root_dir/$mmvet_root"
fi
if [[ "$model_path" != /* ]]; then
    model_path="$root_dir/$model_path"
fi

question_json="$mmvet_root/mm-vet.json"
question_jsonl="$mmvet_root/mm-vet.jsonl"
image_folder="$mmvet_root/images"

if [[ ! -f "$question_json" ]]; then
    echo "MM-Vet source json not found: $question_json"
    exit 1
fi

if [[ ! -d "$image_folder" ]]; then
    echo "MM-Vet image folder not found: $image_folder"
    exit 1
fi

if [[ -n "$run_tag" ]]; then
    results_root="$root_dir/results/$model_name/mmvet/$method/$run_tag"
else
    results_root="$root_dir/results/$model_name/mmvet/$method"
fi

answers_file="$results_root/answers.jsonl"
converted_results_file="$results_root/mmvet_results.json"
mkdir -p "$results_root"

echo "[MMVet] method=$method"
echo "[MMVet] gpu_id=$gpu_id"
echo "[MMVet] gpu_select_mode=$gpu_select_mode"
echo "[MMVet] run_tag=${run_tag:-none}"
echo "[MMVet] model_path=$model_path"
echo "[MMVet] mmvet_root=$mmvet_root"
echo "[MMVet] answers_file=$answers_file"

python "$root_dir/llava/scripts/prepare_mmvet_questions.py" \
    --src "$question_json" \
    --dst "$question_jsonl"

CUDA_VISIBLE_DEVICES="$gpu_id" python -m qwen35.qwen_eval \
    --model-path "$model_path" \
    --question-file "$question_jsonl" \
    --image-folder "$image_folder" \
    --answers-file "$answers_file" \
    --dataset-name "MMVet" \
    --temperature 0 \
    --cuda-device 'cuda:0' \
    --method "$method" \
    --retracing-ratio "$retracing_ratio" \
    --retrace-delay-layers "$retrace_delay_layers" \
    --retrace-target-layers "$retrace_target_layers" \
    --state-drift-threshold "$state_drift_threshold" \
    --state-drift-pooling "$state_drift_pooling" \
    --entropy-threshold "$entropy_threshold" \
    --max-new-tokens "$max_new_tokens" \
    --starting-layer "$starting_layer" \
    --ending-layer "$ending_layer" \
    --print-live-output \
    --num-chunks 1 \
    --chunk-idx 0

convert_mmvet_results "$answers_file" "$converted_results_file"