#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$script_dir/.." && pwd)"
cd "$root_dir"
export PYTHONPATH="$root_dir:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

python_cmd=(/mnt/data/miniconda3/bin/conda run -n dualpd --no-capture-output python)

method="lead"
model_path_default="/mnt/data/ssz/llms/Qwen2.5-VL-7B-Instruct"
model_tag_default="Qwen2.5-VL-7B-Instruct"
limit_default="0"
do_sample_default="0"

pope_root_default="/mnt/data/ssz/Datasets/POPE"
llava_root_default="/mnt/data/ssz/mr/llava"
coco_image_folder_default="/mnt/data/ssz/Datasets/coco2014/images/val2014"
gqa_image_folder_default="/mnt/data/ssz/Datasets/GQA/images"

dataset_prefixes_default="coco,gqa,aokvqa"
splits_default="random,popular,adversarial"

model_path="${MODEL_PATH:-$model_path_default}"
model_tag="${MODEL_TAG:-$model_tag_default}"
pope_root="${POPE_ROOT:-$pope_root_default}"
llava_root="${LLAVA_ROOT:-$llava_root_default}"
coco_image_folder="${COCO_IMAGE_FOLDER:-$coco_image_folder_default}"
gqa_image_folder="${GQA_IMAGE_FOLDER:-$gqa_image_folder_default}"
dataset_prefixes_csv="${DATASET_PREFIXES:-$dataset_prefixes_default}"
splits_csv="${SPLITS:-$splits_default}"
limit="${LIMIT:-$limit_default}"
do_sample="${DO_SAMPLE:-$do_sample_default}"
alpha="${ALPHA:-0.6}"
max_switch_count="${MAX_SWITCH_COUNT:-5}"
temperature="${TEMPERATURE:-1.0}"
top_p="${TOP_P:-1.0}"
top_k="${TOP_K:-0}"
min_p="${MIN_P:-0.0}"
max_new_tokens="${MAX_NEW_TOKENS:-64}"
seed="${SEED:-42}"

if [[ "$method" != "lead" && "$method" != "cot" && "$method" != "cot_greedy" ]]; then
    echo "Invalid method: $method"
    exit 1
fi

if [[ "$model_path" != /* ]]; then
    model_path="$root_dir/$model_path"
fi
if [[ "$pope_root" != /* ]]; then
    pope_root="$root_dir/$pope_root"
fi
if [[ "$llava_root" != /* ]]; then
    llava_root="$root_dir/$llava_root"
fi
if [[ "$coco_image_folder" != /* ]]; then
    coco_image_folder="$root_dir/$coco_image_folder"
fi
if [[ "$gqa_image_folder" != /* ]]; then
    gqa_image_folder="$root_dir/$gqa_image_folder"
fi

if ! [[ "$limit" =~ ^[0-9]+$ ]]; then
    echo "Invalid LIMIT: $limit"
    exit 1
fi

results_root="$root_dir/results/$model_tag/pope/$method"
tmp_root="$results_root/.tmp"
result_json="$results_root/result.json"
log_file="$results_root/log.log"

mkdir -p "$results_root" "$tmp_root"
: > "$result_json"

exec > >(tee -a "$log_file") 2>&1
echo "==== [$(date '+%Y-%m-%d %H:%M:%S')] START pope.sh method=$method model=$model_tag pid=$$ ===="

IFS=',' read -r -a dataset_prefixes <<< "$dataset_prefixes_csv"
IFS=',' read -r -a splits <<< "$splits_csv"

get_image_folder() {
    local dataset_prefix="$1"
    case "$dataset_prefix" in
        coco)
            echo "$coco_image_folder"
            ;;
        gqa)
            echo "$gqa_image_folder"
            ;;
        aokvqa)
            echo "$coco_image_folder"
            ;;
        *)
            echo ""
            ;;
    esac
}

run_case() {
    local dataset_prefix="$1"
    local split="$2"
    local dataset_group="${dataset_prefix}_POPE"
    local dataset_dir="$pope_root/output/$dataset_group"
    local metric_key="${dataset_group}_${split}"

    if [[ ! -d "$dataset_dir" ]]; then
        echo "[POPE] skip $metric_key: dataset dir not found ($dataset_dir)"
        return 0
    fi

    local image_folder
    image_folder="$(get_image_folder "$dataset_prefix")"
    if [[ -z "$image_folder" || ! -d "$image_folder" ]]; then
        echo "[POPE] skip $metric_key: image folder not found ($image_folder)"
        return 0
    fi

    local question_src
    question_src="$(find "$dataset_dir" -maxdepth 1 -type f -name "${dataset_prefix}_pope*_${split}.json" | head -n 1)"
    if [[ -z "$question_src" ]]; then
        echo "[POPE] skip $metric_key: question source not found"
        return 0
    fi

    local question_file="$tmp_root/${metric_key}.jsonl"
    local answers_file="$results_root/${metric_key}.jsonl"
    local case_result_json="$results_root/result.json"

    "${python_cmd[@]}" "$llava_root/scripts/prepare_pope_questions.py" \
        --input "$question_src" \
        --output "$question_file" \
        --limit "$limit"

    local expected_count
    expected_count=$(wc -l < "$question_file")
    if [[ "$expected_count" -eq 0 ]]; then
        echo "[POPE] skip $metric_key: no samples after prepare"
        return 0
    fi

    echo "[POPE] metric_key=$metric_key"
    echo "[POPE] question_file=$question_file"
    echo "[POPE] answers_file=$answers_file"
    echo "[POPE] image_folder=$image_folder"
    echo "[POPE] expected_count=$expected_count"

    rm -f "$answers_file"

    local do_sample_flag=()
    if [[ "$do_sample" == "0" ]]; then
        do_sample_flag=(--no-do-sample)
    else
        do_sample_flag=(--do-sample)
    fi

    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${python_cmd[@]}" -m lead.pope \
        --model-name "$model_path" \
        --question-file "$question_file" \
        --image-folder "$image_folder" \
        --answers-file "$answers_file" \
        --method "$method" \
        --alpha "$alpha" \
        --max-switch-count "$max_switch_count" \
        --temperature "$temperature" \
        --top-p "$top_p" \
        --top-k "$top_k" \
        --min-p "$min_p" \
        --max-new-tokens "$max_new_tokens" \
        --seed "$seed" \
        "${do_sample_flag[@]}"

    "${python_cmd[@]}" "$llava_root/scripts/pope_eval_metrics.py" \
        --question-file "$question_file" \
        --answers-file "$answers_file" \
        --metric-key "$metric_key" \
        --result-json "$case_result_json"

    echo "[POPE] done $metric_key"
}

for dataset_prefix in "${dataset_prefixes[@]}"; do
    for split in "${splits[@]}"; do
        run_case "$dataset_prefix" "$split"
        echo ""
    done
done

echo "==== [$(date '+%Y-%m-%d %H:%M:%S')] END pope.sh ===="