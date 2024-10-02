#!/bin/bash


CHECKPOINT=/proj/checkpoints/bathen/developer/FlagEmbedding/Long_LLM/longllm_qlora/data/outputs/granite-3b-instruct-preview-16k/merged
#31500 for 32K+
MAX_LEN=16150

#200e3 for 32, 200e6 for 80k
RT=200e3

torchrun --nproc_per_node 8 -m main.eval_longbench --max_length $MAX_LEN --model_name_or_path $CHECKPOINT --rope_theta $RT --attn_impl flash_attention_2 --chat_template granite --data_root /proj/checkpoints/bathen/data/pile/long-llm
torchrun --nproc_per_node 8 -m main.eval_topic --model_name_or_path $CHECKPOINT --rope_theta $RT --attn_impl flash_attention_2 --chat_template granite --data_root /proj/checkpoints/bathen/data/pile/long-llm 

#torchrun --nproc_per_node 8 -m main.eval_mmlu --model_name_or_path /proj/checkpoints/bathen/developer/FlagEmbedding/Long_LLM/longllm_qlora/data/outputs/granite-3b-instruct-preview-32k/merged --rope_theta 200e3 --attn_impl flash_attention_2 --chat_template granite --data_root /proj/checkpoints/bathen/data/pile/long-llm

