#!/bin/bash

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TOKENIZERS_PARALLELISM=true
export CUDA_HOME="/opt/share/cuda-12.1"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export WANDB_MODE=disabled
export NVIDIA_PYTORCH_VERSION=3.10
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/share/cudnn-linux-x86_64-8.9.7.29_cuda12/lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /proj/checkpoints/bathen/envs/run.env
. /u/bathen/miniconda3/etc/profile.d/conda.sh
conda activate /proj/checkpoints/bathen/envs/conda/longrope

GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -w)

CHECKPOINT=/proj/checkpoints/stallone/models/preview/granite-3b-instruct-preview-4k-r240917a
OUTPUT=granite-3b-instruct-preview-16k
DATA_ROOT=/proj/checkpoints/bathen/data/pile/long-llm

mkdir -p data/outputs/
torchrun --nproc_per_node 8 -m main.train \
    --data_root $DATA_ROOT \
    --output_dir data/outputs/${OUTPUT} \
    --model_name_or_path $CHECKPOINT \
    --train_data long-llm:gpt/one_detail_book.train.64K.json long-llm:gpt/one_detail_paper.train.64K.json long-llm:gpt/multi_detail_book.train.json long-llm:gpt/multi_detail_paper_short.train.json long-llm:gpt/multi_detail_paper_long.train.json long-llm:gpt/bio_book.train.json long-llm:longalpaca/train.json long-llm:redpajama/train.json[5000] \
    --max_length 81920 \
    --group_by_length \
    --rope_theta 200e6 \
    --attn_impl flash_attention_2 \
    --gradient_checkpointing \
    --use_reentrant True \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --save_only_model \
    --save_strategy epoch \
    --logging_steps 5 \
    --bf16 \
    --lora_tune \
    --lora_extra_params embed_tokens \
    --load_in_4_bit \
    --chat_template granite

