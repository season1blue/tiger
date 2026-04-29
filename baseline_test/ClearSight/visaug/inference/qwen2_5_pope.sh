#!/usr/bin/env bash
set -euo pipefail

# bash visaug/inference/qwen2_5_pope.sh

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_dir}"

seed=${1:-42}
dataset_name=${2:-"coco"}
split_type=${3:-"random"}
model_path=${4:-"/mnt/data/ssz/llms/Qwen2.5-VL-7B-Instruct"}
use_visaug=${5:-1}
enh_para=${6:-1.15}
sup_para=${7:-0.95}

device_map=${DEVICE_MAP:-auto}
pope_root=${POPE_ROOT:-/mnt/data/ssz/Datasets/POPE/output}
image_root=${POPE_IMAGE_ROOT:-/mnt/data/ssz/Datasets}
gpu_devices=${CUDA_DEVICES:-0}
python_bin=${PYTHON_BIN:-python}

export CUDA_VISIBLE_DEVICES="${gpu_devices}"
echo "[ClearSight] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

if [[ $dataset_name == "coco" || $dataset_name == "aokvqa" ]]; then
  image_folder=${image_root}/coco2014/images/val2014
else
  image_folder=${image_root}/GQA/images
fi

if [[ $dataset_name == "coco" ]]; then
  pope_dir=${pope_root}/coco_POPE
  pope_file=${pope_dir}/coco_pope_${split_type}.json
elif [[ $dataset_name == "gqa" ]]; then
  pope_dir=${pope_root}/gqa_POPE
  pope_file=${pope_dir}/gqa_pope_seem_${split_type}.json
elif [[ $dataset_name == "aokvqa" ]]; then
  pope_dir=${pope_root}/aokvqa_POPE
  pope_file=${pope_dir}/aokvqa_pope_seem_${split_type}.json
else
  echo "Unsupported dataset_name: ${dataset_name}" >&2
  exit 1
fi

out_dir=./outputs/inference/qwen2_5_vl
mkdir -p "${out_dir}"

if [[ "${use_visaug}" == "1" ]]; then
  answers_file=${out_dir}/res_${dataset_name}_${split_type}_qwen2_5_vl_visaug_seed${seed}.jsonl
  visaug_args="--use-visaug --enh-para ${enh_para} --sup-para ${sup_para}"
  echo "[ClearSight] VISAUG flag enabled for Qwen2.5-VL CLI compatibility: enh=${enh_para}, sup=${sup_para}"
else
  answers_file=${out_dir}/res_${dataset_name}_${split_type}_qwen2_5_vl_base_seed${seed}.jsonl
  visaug_args=""
  echo "[ClearSight] Baseline mode"
fi

"${python_bin}" ./visaug/inference/infer_pope_qwen2_5_vl.py \
  --model-path "${model_path}" \
  --image-folder "${image_folder}" \
  --question-file "${pope_file}" \
  --answers-file "${answers_file}" \
  --device-map "${device_map}" \
  --seed "${seed}" \
  ${visaug_args}

"${python_bin}" ./visaug/inference/eval_pope.py \
  --annotation-file "${pope_file}" \
  --result-file "${answers_file}"

echo "[ClearSight] Result file: ${answers_file}"
