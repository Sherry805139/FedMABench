#!/bin/bash
# 多GPU训练脚本（模型并行，不使用DeepSpeed）
# 使用方法: bash run_central_internvl2_2_multi_gpu.sh dataset.jsonl "0,1"
# 注意：模型并行不支持DeepSpeed，如果需要DeepSpeed请使用 run_central_internvl2_2_deepspeed.sh

DATASET=$1
GPU_IDS=${2:-"0"}  # 默认使用 GPU 0，可以设置为 "0,1" 或 "0,1,2,3"

echo "[INFO] Using GPUs: $GPU_IDS (Model Parallelism)"
echo "[INFO] Note: DeepSpeed is NOT compatible with Model Parallelism"

CUDA_VISIBLE_DEVICES=$GPU_IDS \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
MAX_PIXELS=400000 \
  swift sft \
  --round 30 \
  --round_per_epoch 10 \
  --fed_alg central \
  --client_num 1 \
  --model_type qwen2-vl-2b-instruct \
  --model_id_or_path /home/hmpiao/hmpiao/Qwen2-VL-2B-Instruct \
  --lazy_tokenize True \
  --preprocess_num_proc 4 \
  --dataset "$DATASET" \
  --sft_type lora \
  --tuner_backend peft \
  --dtype bf16 \
  --output_dir output \
  --train_dataset_sample -1 \
  --dataset_test_ratio 0 \
  --max_steps -1 \
  --max_length 2048 \
  --check_dataset_strategy warning \
  --lora_rank 8 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --gradient_checkpointing true \
  --batch_size 1 \
  --weight_decay 0.1 \
  --learning_rate 5e-5 \
  --gradient_accumulation_steps 8 \
  --max_grad_norm 0.5 \
  --warmup_ratio 0.03 \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 3 \
  --save_only_model True \
  --logging_steps 100

