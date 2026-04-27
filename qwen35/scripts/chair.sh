#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$script_dir/../.." && pwd)"
cd "$root_dir"
export PYTHONPATH="$root_dir:${PYTHONPATH:-}"

# Usage:
#   bash qwen35/scripts/chair.sh
#   bash qwen35/scripts/chair.sh base
#   bash qwen35/scripts/chair.sh base 1
#   bash qwen35/scripts/chair.sh memvr 1 auto
#   bash qwen35/scripts/chair.sh evo 2
method="${1:-base}"
num_gpus="${2:-1}"
gpu_ids_csv="${3:-}"

if [[ "$method" != "base" && "$method" != "memvr" && "$method" != "evo" ]]; then
    echo "Invalid method: $method"
    echo "Usage: bash qwen35/scripts/chair.sh [base|memvr|evo] [num_gpus] [gpu_ids_csv|auto]"
    exit 1
fi

if ! [[ "$num_gpus" =~ ^[0-9]+$ ]] || [[ "$num_gpus" -lt 1 ]]; then
    echo "Invalid num_gpus: $num_gpus"
    exit 1
fi

pick_top_gpus_by_free_mem() {
    local count="$1"
    local query=""

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 1
    fi

    query="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ -z "$query" ]]; then
        return 1
    fi

    mapfile -t picked < <(
        echo "$query" \
        | awk -F',' '{gsub(/ /, "", $1); gsub(/ /, "", $2); print $1 "," $2}' \
        | sort -t',' -k2,2nr \
        | head -n "$count" \
        | cut -d',' -f1
    )

    if [[ "${#picked[@]}" -lt "$count" ]]; then
        return 1
    fi

    printf '%s\n' "${picked[@]}"
}

# -----------------------------------------------------------------------------
# Default parameters (can be overridden via env)
# -----------------------------------------------------------------------------
model_name="${MODEL_NAME:-Qwen3.5-VL}"
model_path="${MODEL_PATH:-../llms/Qwen3.5-9B}"
coco_root="${COCO_ROOT:-../Datasets/coco2014}"
instances_json="${INSTANCES_JSON:-../Datasets/coco2014/annotations/instances_val2014.json}"
seed="${CHAIR_SEED:-2}"
sample_count="${CHAIR_SAMPLE_COUNT:-500}"
max_new_tokens="${MAX_NEW_TOKENS:-512}"

# MemVR/EVO params
retracing_ratio="${RETRACING_RATIO:-0.12}"
entropy_threshold="${ENTROPY_THRESHOLD:-0.75}"
starting_layer="${STARTING_LAYER:-8}"
ending_layer="${ENDING_LAYER:-10}"
retrace_delay_layers="${RETRACE_DELAY_LAYERS:-1}"
retrace_target_layers="${RETRACE_TARGET_LAYERS:-}"
state_drift_threshold="${STATE_DRIFT_THRESHOLD:-0.5}"
state_drift_pooling="${STATE_DRIFT_POOLING:-mean}"

gpu_select_mode="manual"
if [[ -n "$gpu_ids_csv" && "$gpu_ids_csv" != "auto" ]]; then
    IFS=',' read -r -a gpu_ids <<< "$gpu_ids_csv"
else
    gpu_ids=()
    if mapfile -t gpu_ids < <(pick_top_gpus_by_free_mem "$num_gpus"); then
        gpu_select_mode="auto_free_mem"
    else
        gpu_select_mode="fallback_index"
        for ((i=0; i<num_gpus; i++)); do
            gpu_ids+=("$i")
        done
    fi
fi

if [[ "${#gpu_ids[@]}" -lt "$num_gpus" ]]; then
    echo "[CHAIR] provided gpu ids fewer than num_gpus"
    exit 1
fi

selected_visible_gpus="$(IFS=,; echo "${gpu_ids[*]}")"

image_folder="$coco_root/images/val2014"
if [[ ! -d "$image_folder" ]]; then
    if [[ -d "$coco_root/images" ]]; then
        image_folder="$coco_root/images"
    else
        image_folder="$coco_root"
    fi
fi

results_root="$root_dir/results/$model_name/chair/$method"
mkdir -p "$results_root"

sample_file="$results_root/sampled_500.txt"
captions_jsonl="$results_root/captions_500.jsonl"
captions_json="$results_root/captions_500.json"
analysis_json="$results_root/chair_detailed_500.json"

echo "[CHAIR] method=$method"
echo "[CHAIR] model_path=$model_path"
echo "[CHAIR] image_folder=$image_folder"
echo "[CHAIR] instances_json=$instances_json"
echo "[CHAIR] seed=$seed sample_count=$sample_count"
echo "[CHAIR] num_gpus=$num_gpus gpu_select_mode=$gpu_select_mode"
echo "[CHAIR] visible_gpus=$selected_visible_gpus"
echo "[CHAIR] sampled_image_list=$sample_file"

# Step 1: sample images with fixed seed.
python - <<'PY' "$image_folder" "$sample_file" "$seed" "$sample_count"
import random
import sys
from pathlib import Path

image_folder = Path(sys.argv[1])
sample_file = Path(sys.argv[2])
seed = int(sys.argv[3])
sample_count = int(sys.argv[4])

imgs = sorted([p.name for p in image_folder.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
if len(imgs) < sample_count:
    raise ValueError(f"Not enough images to sample: found={len(imgs)} required={sample_count}")

rng = random.Random(seed)
sampled = rng.sample(imgs, sample_count)
sample_file.write_text("\n".join(sampled) + "\n")
print(f"[CHAIR] sampled {len(sampled)} images -> {sample_file}")
PY

# Step 2: generate captions.
CUDA_VISIBLE_DEVICES="$selected_visible_gpus" python -m qwen35.qwen_eval \
    --task-type chair \
    --model-path "$model_path" \
    --question-file "$instances_json" \
    --image-folder "$image_folder" \
    --answers-file "$captions_jsonl" \
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
    --chair-image-list "$sample_file" \
    --chair-max-samples "$sample_count"

# Step 3: convert jsonl to json list.
python - <<'PY' "$captions_jsonl" "$captions_json"
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
dst.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
print(f"[CHAIR] converted captions jsonl -> json: {dst}")
PY

# Step 4: compute CHAIR metrics.
python qwen25/scripts/chair_eval_coco.py \
    --captions-json "$captions_json" \
    --instances-json "$instances_json" \
    --analysis-json "$analysis_json"

echo "[CHAIR] captions_json=$captions_json"
echo "[CHAIR] detailed_analysis=$analysis_json"
