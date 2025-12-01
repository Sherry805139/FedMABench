#!/bin/bash

# 在 Val_100.jsonl 上评估每个类别 LoRA（默认只评 Round 30）
# 用法示例：
#   bash evaluation/test_category_loras_val100.sh 0 ./Val_100.jsonl 30
#
# 参数：
#   $1: GPU_ID（默认 0）
#   $2: 验证集 JSONL 路径（默认 ./Val_100.jsonl）
#   $3: 评估的 round（默认 30）

GPU_ID=${1:-0}
VAL_DATASET=${2:-"./Val_100.jsonl"}
ROUND=${3:-30}

BASE_OUTPUT_DIR="./output"
MODEL_TYPE="qwen2-vl-2b-instruct"
MODEL_PATH="/home/hmpiao/hmpiao/Qwen2-VL-2B-Instruct"

CATEGORIES=("Shopping" "Traveling" "Office" "Lives" "Entertainment")

echo "[INFO] Evaluating category LoRAs on Val_100.jsonl"
echo "[INFO] GPU: $GPU_ID"
echo "[INFO] Val dataset: $VAL_DATASET"
echo "[INFO] Round: $ROUND"

if [ ! -f "$VAL_DATASET" ]; then
  echo "[ERROR] Val dataset not found: $VAL_DATASET"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAX_PIXELS=200000

# 汇总结果到一个简单的 txt，放在 ./output 下
SUMMARY_FILE="$BASE_OUTPUT_DIR/category_lora_val100_summary.txt"
mkdir -p "$BASE_OUTPUT_DIR"
{
  echo "Category LoRA Val_100 Evaluation Summary"
  echo "========================================"
  echo "Val dataset: $VAL_DATASET"
  echo "Round: $ROUND"
  echo ""
} > "$SUMMARY_FILE"

for model_cat in "${CATEGORIES[@]}"; do
  model_lower=$(echo "$model_cat" | tr '[:upper:]' '[:lower:]')

  echo ""
  echo "##########################################"
  echo "Model LoRA: $model_cat  |  Val set: Val_100"
  echo "##########################################"

  # 查找对应 round 的 checkpoint 目录（兼容嵌套结构）
  ckpt_dir_candidates=$(find "$BASE_OUTPUT_DIR/category_lora_${model_lower}" -type d -name "global_lora_${ROUND}" 2>/dev/null | sort)
  ckpt_dir=$(echo "$ckpt_dir_candidates" | tail -n 1)

  if [ -z "$ckpt_dir" ] || [ ! -d "$ckpt_dir" ]; then
    echo "[WARNING] No checkpoint directory found for $model_cat round $ROUND under $BASE_OUTPUT_DIR/category_lora_${model_lower}, skipping..."
    continue
  fi

  echo "[INFO] Using checkpoint directory: $ckpt_dir"

  infer_dir="$ckpt_dir/infer_result"
  combo_dir="$infer_dir/val100"

  mkdir -p "$infer_dir"

  # 如果该 LoRA 已经在 Val_100 上有推理结果，则跳过推理
  existing_jsonl=$(find "$combo_dir" -type f -name "*.jsonl" 2>/dev/null | head -n 1)
  if [ -z "$existing_jsonl" ]; then
    echo "[INFO] Running inference on Val_100 for model $model_cat, round $ROUND..."

    # 清理顶层旧的 jsonl（保留子目录）
    find "$infer_dir" -maxdepth 1 -type f -name "*.jsonl" -delete

    swift infer \
      --ckpt_dir "$ckpt_dir" \
      --val_dataset "$VAL_DATASET" \
      --model_type $MODEL_TYPE \
      --model_id_or_path $MODEL_PATH \
      --sft_type lora

    if [ $? -ne 0 ]; then
      echo "[ERROR] Inference failed for model $model_cat on Val_100, round $ROUND"
      continue
    fi

    mkdir -p "$combo_dir"
    new_jsonl_files=$(find "$infer_dir" -maxdepth 1 -type f -name "*.jsonl" 2>/dev/null)
    for f in $new_jsonl_files; do
      mv "$f" "$combo_dir"/
    done
  else
    echo "[INFO] Inference results already exist for model $model_cat on Val_100, round $ROUND, skipping inference."
  fi

  # 计算准确率（整体 Step-level accuracy）
  jsonl_files=$(find "$combo_dir" -type f -name "*.jsonl" 2>/dev/null)
  if [ -z "$jsonl_files" ]; then
    echo "[WARNING] No inference results found in $combo_dir"
    continue
  fi

  for jsonl_file in $jsonl_files; do
    echo "[INFO] Processing: $jsonl_file"
    base_name="$(basename "$jsonl_file" .jsonl)"
    # 详细评测结果仍保存在各自的 infer_result 子目录，便于排查
    output_file="$combo_dir/${base_name}_val100_model-${model_lower}_result.txt"

    python evaluation/test_swift.py --data_path "$jsonl_file" > "$output_file" 2>&1
    echo "[INFO] Val_100 result saved to: $output_file"

    # 从评测结果中提取 Step-level accuracy，汇总到 ./output/category_lora_val100_summary.txt
    acc_line=$(grep "Step-level accuracy" "$output_file" | head -n 1)
    if [ -n "$acc_line" ]; then
      echo "Model LoRA: ${model_cat}, Val_100 -> ${acc_line}" >> "$SUMMARY_FILE"
    else
      echo "Model LoRA: ${model_cat}, Val_100 -> [WARN] No Step-level accuracy line found" >> "$SUMMARY_FILE"
    fi
  done
done

echo ""
echo "=========================================="
echo "Category LoRA evaluation on Val_100 completed!"
echo "=========================================="

echo "[INFO] Summary saved to: $SUMMARY_FILE"



