#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

POPE_ROOT="/data/ssz/Datasets/POPE"
COCO_IMAGE_FOLDER="/data/ssz/Datasets/coco2014-val/images/val2014"
GQA_IMAGE_FOLDER="/data/ssz/Datasets/GQA/images"
MODEL_PATH="/data/ssz/llms/llava-v1.5-7b"
CUDA_DEVICE="cuda:0"

# Set LIMIT=0 for full evaluation; set e.g. LIMIT=100 for quick smoke test.
LIMIT=0
EXPERIMENT="llava-v1.5-7b/memvr"
ANSWER_ROOT="$POPE_ROOT/answer/$EXPERIMENT"
TMP_DIR="$POPE_ROOT/answer/.tmp_pope"
RESULT_JSON="$ANSWER_ROOT/result.json"

# MemVR knobs
APPLY_MEMVR="none"
RETRACING_RATIO=0.31
ENTROPY_THRESHOLD=0.75
STARTING_LAYER=5
ENDING_LAYER=16
MAX_NEW_TOKENS=1

# Select which POPE dataset groups/splits to run.
# Example: DATASET_PREFIXES=(coco) or DATASET_PREFIXES=(coco gqa aokvqa)
DATASET_PREFIXES=(coco)
# Example: SPLITS=(random popular adversarial)
SPLITS=(random)

mkdir -p "$ANSWER_ROOT"
mkdir -p "$TMP_DIR"

# Refresh summary file for this run.
rm -f "$RESULT_JSON"

get_image_folder() {
  local dataset_prefix="$1"
  case "$dataset_prefix" in
    coco)
      echo "$COCO_IMAGE_FOLDER"
      ;;
    gqa)
      echo "$GQA_IMAGE_FOLDER"
      ;;
    aokvqa)
      echo "$COCO_IMAGE_FOLDER"
      ;;
    *)
      echo ""
      ;;
  esac
}

run_case() {
  local dataset_group="$1"
  local dataset_prefix="$2"
  local split="$3"
  local full_question_file="$4"
  local image_folder="$5"
  local metric_key="${dataset_group}_${split}"
  local question_file="$TMP_DIR/${metric_key}.jsonl"

  python "$SCRIPT_DIR/prepare_pope_questions.py" \
    --input "$full_question_file" \
    --output "$question_file" \
    --limit "$LIMIT"

  if [ "$LIMIT" -gt 0 ]; then
    echo "[$metric_key] using subset from: $full_question_file"
  else
    echo "[$metric_key] using full file: $full_question_file"
  fi

  local answers_file="$ANSWER_ROOT/${metric_key}.jsonl"

  python -m llava.eval.llava_model_vqa_loader \
    --model-path "$MODEL_PATH" \
    --question-file "$question_file" \
    --image-folder "$image_folder" \
    --answers-file "$answers_file" \
    --temperature 0 \
    --cuda-device "$CUDA_DEVICE" \
    --apply-memvr "$APPLY_MEMVR" \
    --retracing-ratio "$RETRACING_RATIO" \
    --entropy-threshold "$ENTROPY_THRESHOLD" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --starting-layer "$STARTING_LAYER" \
    --ending-layer "$ENDING_LAYER"

  python "$SCRIPT_DIR/pope_eval_metrics.py" \
    --question-file "$question_file" \
    --answers-file "$answers_file" \
    --metric-key "$metric_key" \
    --result-json "$RESULT_JSON"
}

for dataset_prefix in "${DATASET_PREFIXES[@]}"; do
  dataset_group="${dataset_prefix}_POPE"
  dataset_dir="$POPE_ROOT/output/$dataset_group"

  if [ ! -d "$dataset_dir" ]; then
    echo "Skipping $dataset_group: dataset directory not found ($dataset_dir)"
    continue
  fi

  image_folder="$(get_image_folder "$dataset_prefix")"

  if [ -z "$image_folder" ] || [ ! -d "$image_folder" ]; then
    echo "Skipping $dataset_group: image folder missing ($image_folder)"
    continue
  fi

  for split in "${SPLITS[@]}"; do
    full_question_file="$(find "$dataset_dir" -maxdepth 1 -type f -name "${dataset_prefix}_pope*_${split}.json" | head -n 1)"
    if [ -z "$full_question_file" ]; then
      echo "Skipping ${dataset_group}_${split}: question file not found"
      continue
    fi

    run_case "$dataset_group" "$dataset_prefix" "$split" "$full_question_file" "$image_folder"
  done
done

echo "\nPOPE evaluation finished."
echo "Answers are under: $ANSWER_ROOT"
echo "Summary json: $RESULT_JSON"
