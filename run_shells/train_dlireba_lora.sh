#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

cd ../


#--------------------------------------------------------------------------------
# 参考：https://huggingface.co/blog/lora
#--------------------------------------------------------------------------------


BASE_DIR="/mnt/cephfs/hjh/train_record/images/text2image/lora/train_dlireba"

MODEL_NAME="swl-models/chilloutmix-ni"
OUTPUT_DIR="${BASE_DIR}/sddata/finetune/lora/dlireba"
DATASET_DIR="/mnt/cephfs/hjh/train_record/images/text2image/cmd_train_lora"

CUDA_VISIBLE_DEVICES=6,7 \
accelerate launch --mixed_precision="fp16" examples/dreambooth/train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_DIR \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=8 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --report_to=wandb \
  --checkpointing_steps=500 \
  --validation_prompt="A woman was standing in the street with a clear face." \
  --seed=1337
