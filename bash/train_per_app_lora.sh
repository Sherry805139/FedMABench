#!/bin/bash
# 为每个app训练独立的LoRA模型
# 使用方法: bash train_per_app_lora.sh [GPU_IDS]
# 例如: bash train_per_app_lora.sh "0,1"

# 配置参数
GPU_IDS=${1:-"0"}
APP_DATA_DIR="./data_by_app"
MODEL_TYPE="qwen2-vl-2b-instruct"
MODEL_PATH="/home/hmpiao/hmpiao/Qwen2-VL-2B-Instruct"
OUTPUT_BASE_DIR="./lora_app"

# APPS from Table 5
APPS=("amazon" "ebay" "flipkart" "gmail" "clock" "reminder" "youtube")

# 验证GPU IDs格式
if [[ ! "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "[ERROR] Invalid GPU IDs format: $GPU_IDS. Expected format: '0,1' or '0,1,2,3'"
    exit 1
fi

# 计算GPU数量
GPU_COUNT=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
echo "[INFO] Using GPUs: $GPU_IDS (Model Parallelism, $GPU_COUNT GPUs)"
echo "[INFO] Base dataset: $APP_DATA_DIR"
echo "[INFO] App data directory: $APP_DATA_DIR"
echo "[INFO] Output base directory: $OUTPUT_BASE_DIR"

# 导出环境变量
export CUDA_VISIBLE_DEVICES=$GPU_IDS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAX_PIXELS=600000

# 检查app数据目录是否存在
if [ ! -d "$APP_DATA_DIR" ]; then
    echo "[ERROR] App data directory not found: $APP_DATA_DIR"
    echo "[INFO] Please run data_process/split_by_app.py first"
    exit 1
fi

# 为每个app训练LoRA
for app in "${APPS[@]}"; do
    app_dataset="$APP_DATA_DIR/${app}_train.jsonl"
    output_dir="$OUTPUT_BASE_DIR/app_lora_${app}"
    
    echo ""
    echo "=========================================="
    echo "Training LoRA for app: $app"
    echo "Dataset: $app_dataset"
    echo "Output: $output_dir"
    echo "=========================================="
    
    # 检查数据集是否存在
    if [ ! -f "$app_dataset" ]; then
        echo "[WARNING] Dataset not found: $app_dataset, skipping..."
        continue
    fi
    
    # 检查是否已经训练过
    if [ -d "$output_dir" ] && [ -f "$output_dir/checkpoint-*.pth" ] 2>/dev/null; then
        echo "[INFO] Checkpoint found in $output_dir, skipping training..."
        continue
    fi
    
    # 训练命令
    swift sft \
      --round 30 \
      --round_per_epoch 10 \
      --fed_alg central \
      --client_num 1 \
      --model_type $MODEL_TYPE \
      --model_id_or_path $MODEL_PATH \
      --lazy_tokenize True \
      --preprocess_num_proc 4 \
      --dataset "$app_dataset" \
      --sft_type lora \
      --tuner_backend peft \
      --dtype bf16 \
      --output_dir "$output_dir" \
      --train_dataset_sample -1 \
      --dataset_test_ratio 0 \
      --max_steps -1 \
      --max_length 1024 \
      --check_dataset_strategy warning \
      --lora_rank 8 \
      --lora_alpha 32 \
      --lora_dropout 0.05 \
      --gradient_checkpointing true \
      --batch_size 1 \
      --weight_decay 0.1 \
      --learning_rate 5e-5 \
      --gradient_accumulation_steps 16 \
      --max_grad_norm 0.5 \
      --warmup_ratio 0.03 \
      --eval_strategy no \
      --save_strategy steps \
      --save_steps 500 \
      --save_total_limit 3 \
      --save_only_model True \
      --logging_steps 100
    
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Training completed for app: $app"
    else
        echo "[ERROR] Training failed for app: $app"
    fi
done

echo ""
echo "=========================================="
echo "All app LoRA training completed!"
echo "=========================================="



