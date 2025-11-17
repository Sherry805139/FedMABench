#!/bin/bash
# 多GPU训练脚本（模型并行，不使用DeepSpeed）
# 使用方法: bash run_central_internvl2_2_multi_gpu.sh "0,1"
#         或: bash run_central_internvl2_2_multi_gpu.sh dataset.jsonl "0,1"
# 注意：模型并行不支持DeepSpeed，如果需要DeepSpeed请使用 run_central_internvl2_2_deepspeed.sh

# 智能检测参数：如果第一个参数包含逗号，则认为是GPU IDs；否则认为是dataset
if [[ "$1" == *","* ]]; then
    # 第一个参数包含逗号，认为是GPU IDs
    GPU_IDS=$1
    DATASET="/home/hmpiao/hmpiao/xuerong/FedMABench/android_control_unpack/episode-wise-conversations.jsonl"
else
    # 第一个参数是dataset，第二个参数是GPU IDs
    DATASET=${1:-"/home/hmpiao/hmpiao/xuerong/FedMABench/android_control_unpack/episode-wise-conversations.jsonl"}
    GPU_IDS=${2:-"0"}  # 默认使用 GPU 0
fi

# 验证GPU IDs格式
if [[ ! "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "[ERROR] Invalid GPU IDs format: $GPU_IDS. Expected format: '0,1' or '0,1,2,3'"
    exit 1
fi

# 计算GPU数量
GPU_COUNT=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
echo "[INFO] Using GPUs: $GPU_IDS (Model Parallelism, $GPU_COUNT GPUs)"
echo "[INFO] Dataset: $DATASET"
echo "[INFO] Note: DeepSpeed is NOT compatible with Model Parallelism"
echo "[INFO] Setting CUDA_VISIBLE_DEVICES=$GPU_IDS"

# 导出环境变量，确保在Python启动前设置
export CUDA_VISIBLE_DEVICES=$GPU_IDS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAX_PIXELS=400000

# 验证环境变量
echo "[INFO] Verifying CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "[ERROR] CUDA_VISIBLE_DEVICES is not set!"
    exit 1
fi

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

