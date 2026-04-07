#!/bin/bash

model="llava-v1.5-7b"
method="memvr"
python -m llava.eval.llava_model_vqa_loader_chair \
    --model-path $model \
    --question-file /data/ssz/Datasets/chair/annotations/instances_val2014.json \
    --image-folder /data/ssz/Datasets/coco2014-val/images/val2014 \
    --answers-file /data/ssz/Datasets/chair/answers/llava-v1.5-7b/memvr.jsonl \
    --temperature 0 \
    --cuda-device 'cuda:0' \
    --apply-memvr 'memvr' \
    --retracing-ratio 0.31 \
    --entropy-threshold 0.75 \
    --max-new-tokens 1024 \
    --starting-layer 5 \
    --ending-layer 16 \
