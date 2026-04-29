#!/usr/bin/env bash
set -euo pipefail
# nohup bash experiments/cd_scripts/qwen2_5_pope.sh > log.log 2>&1 & disown


# GPU selection: default to card 0, override by setting CUDA_DEVICES (e.g. 0,1)
gpu_devices=${CUDA_DEVICES:-0}
export CUDA_VISIBLE_DEVICES="2"
echo "[VCD] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"


script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exp_dir="$(cd "${script_dir}/.." && pwd)"
cd "${exp_dir}"

seed=${1:-55}
dataset_name=${2:-"coco"}
type=${3:-"random"}
model_path=${4:-"/mnt/data/ssz/llms/Qwen2.5-VL-7B-Instruct"}
cd_alpha=${5:-1}
cd_beta=${6:-0.1}
noise_step=${7:-500}
device_map=${DEVICE_MAP:-auto}
pope_root=${POPE_ROOT:-/mnt/data/ssz/Datasets/POPE/output}
image_root=${POPE_IMAGE_ROOT:-/mnt/data/ssz/Datasets}
use_cd=1


if [[ $dataset_name == 'coco' || $dataset_name == 'aokvqa' ]]; then
  image_folder=${image_root}/coco2014/images/val2014
else
  image_folder=${image_root}/GQA/images
fi

if [[ $dataset_name == 'coco' ]]; then
  pope_dir=${pope_root}/coco_POPE
  pope_file=${pope_dir}/coco_pope_${type}.json
elif [[ $dataset_name == 'gqa' ]]; then
  pope_dir=${pope_root}/gqa_POPE
  pope_file=${pope_dir}/gqa_pope_seem_${type}.json
elif [[ $dataset_name == 'aokvqa' ]]; then
  pope_dir=${pope_root}/aokvqa_POPE
  pope_file=${pope_dir}/aokvqa_pope_seem_${type}.json
else
  echo "Unsupported dataset_name: ${dataset_name}" >&2
  exit 1
fi

answers_file=./output/qwen2_5_vl_${dataset_name}_pope_${type}_answers_vcd_seed${seed}.jsonl

# cd_comment: paper mapping
# 1) noisy visual branch v' is created in eval script via add_diffusion_noise(...)
# 2) images_cd/cd_alpha/cd_beta are passed to model.generate(...)
# 3) vcd_utils/vcd_sample.py performs contrastive logits fusion during sampling

gen_extra_args=""
if [[ "${use_cd}" == "1" ]]; then
  gen_extra_args="--use_cd --cd_alpha ${cd_alpha} --cd_beta ${cd_beta} --noise_step ${noise_step}"
  echo "[VCD] Contrastive decoding enabled: alpha=${cd_alpha}, beta=${cd_beta}, noise_step=${noise_step}"
else
  answers_file=./output/qwen2_5_vl_${dataset_name}_pope_${type}_answers_base_seed${seed}.jsonl
  echo "[VCD] Baseline decoding enabled (no contrastive branch)"
fi

python ./eval/object_hallucination_vqa_qwen2_5_vl.py \
--model-path "${model_path}" \
--image-folder "${image_folder}" \
--question-file "${pope_file}" \
--answers-file "${answers_file}" \
--device-map "${device_map}" \
${gen_extra_args} \
--seed "${seed}"

python ./eval/eval_pope.py \
--gt_files "${pope_file}" \
--gen_files "${answers_file}"