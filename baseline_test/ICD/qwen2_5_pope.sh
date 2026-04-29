#!/usr/bin/env bash
set -euo pipefail
# nohup bash qwen2_5_pope.sh > log.log 2>&1 & disown

gpu_devices=${CUDA_DEVICES:-0}
export CUDA_VISIBLE_DEVICES="1"
echo "[VCD] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

seed=${SEED:-42}
model_path=${MODEL_PATH:-/mnt/data/ssz/llms/Qwen2.5-VL-7B-Instruct}
question_folder=${QUESTION_FOLDER:-/mnt/data/ssz/Baselines/ICD/experiments/data/pope}
save_folder=${SAVE_FOLDER:-/mnt/data/ssz/Baselines/ICD/pope_results/qwen2_5_vl}
image_root=${IMAGE_ROOT:-/mnt/data/ssz/Datasets}
gvqa_image_root=${GVQA_IMAGE_ROOT:-${image_root}/GQA/images}
coco_image_root=${COCO_IMAGE_ROOT:-${image_root}/coco2014/images/val2014}
format=${FORMAT:-yn_format}
use_cd=${USE_CD:-1}
cd_alpha=${CD_ALPHA:-1}
cd_beta=${CD_BETA:-0.1}
temperature=${TEMPERATURE:-1.0}
top_p=${TOP_P:-1.0}
top_k=${TOP_K:-0}
max_new_tokens=${MAX_NEW_TOKENS:-32}
device_map=${DEVICE_MAP:-auto}
torch_dtype=${TORCH_DTYPE:-auto}
python_bin=${PYTHON_BIN:-python}

gen_args=(
  --model-path "${model_path}"
  --question_folder "${question_folder}"
  --save_folder "${save_folder}"
  --gvqa_image_root "${gvqa_image_root}"
  --coco_image_root "${coco_image_root}"
  --format "${format}"
  --seed "${seed}"
  --device-map "${device_map}"
  --torch-dtype "${torch_dtype}"
  --cd_alpha "${cd_alpha}"
  --cd_beta "${cd_beta}"
  --temperature "${temperature}"
  --top_p "${top_p}"
  --top_k "${top_k}"
  --max_new_tokens "${max_new_tokens}"
)

if [[ "${use_cd}" == "1" ]]; then
  gen_args+=(--use_cd)
  echo "[ICD] Contrastive decoding enabled: alpha=${cd_alpha}, beta=${cd_beta}"
else
  echo "[ICD] Baseline decoding enabled"
fi

"${python_bin}" ./experiments/gen_scripts/icd_qwen2_5_pope.py "${gen_args[@]}"

"${python_bin}" ./experiments/eval_scripts/eval_pope.py \
  --label_folder "${question_folder}" \
  --ans_folder "${save_folder}/$([[ "${use_cd}" == "1" ]] && echo icd || echo baseline)" \
  --format "${format}"