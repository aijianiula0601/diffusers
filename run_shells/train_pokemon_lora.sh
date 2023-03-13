#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

cd ../

BASE_DIR="/mnt/cephfs/hjh/train_record/images/text2image/lora/train_pokemon"

MODEL_NAME="runwayml/stable-diffusion-v1-5"
OUTPUT_DIR="${BASE_DIR}/sddata/finetune/lora/pokemon"
DATASET_NAME="lambdalabs/pokemon-blip-captions"

CUDA_VISIBLE_DEVICES=0,6 \
accelerate launch --mixed_precision="fp16" examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --report_to=wandb \
  --checkpointing_steps=500 \
  --validation_prompt="A pokemon with blue eyes." \
  --seed=1337
