#!/bin/bash

# 在基座模型上做评测：
#   1) 在按类别划分的 5 个测试集上评测（默认模式 categories）
#   2) 在 Val_100.jsonl 上评测（模式 val100）
#
# 用法示例：
#   # 基座模型在 5 个按类别测试集上的结果（使用默认路径 ./test_data_by_category）
#   bash evaluation/test_base_model.sh 0 categories
#
#   # 基座模型在 Val_100.jsonl 上的结果
#   bash evaluation/test_base_model.sh 0 val100 ./Val_100.jsonl
#
# 参数：
#   $1: GPU_ID（默认 0）
#   $2: 模式：categories / val100（默认 categories）
#   $3: 当模式为 categories 时：测试集目录（默认 ./test_data_by_category）
#       当模式为 val100 时：Val_100.jsonl 路径（默认 ./Val_100.jsonl）
#   $4: MODEL_TYPE（默认 qwen2-vl-2b-instruct）
#   $5: MODEL_PATH（默认 /home/hmpiao/hmpiao/Qwen2-VL-2B-Instruct）

GPU_ID=${1:-0}
MODE=${2:-"categories"}

if [ "$MODE" = "val100" ]; then
  VAL_DATASET=${3:-"./Val_100.jsonl"}
  DATASET_DIR=""
else
  DATASET_DIR=${3:-"./test_data_by_category"}
  VAL_DATASET=""
fi

MODEL_TYPE=${4:-"qwen2-vl-2b-instruct"}
MODEL_PATH=${5:-"/home/hmpiao/hmpiao/Qwen2-VL-2B-Instruct"}

CATEGORIES=("Shopping" "Traveling" "Office" "Lives" "Entertainment")

export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAX_PIXELS=200000

echo "[INFO] MODE = $MODE"
echo "[INFO] MODEL_TYPE = $MODEL_TYPE"
echo "[INFO] MODEL_PATH = $MODEL_PATH"

# 所有基座模型测试结果汇总到一个 txt：./output/base_model_eval_summary.txt
SUMMARY_FILE="./output/base_model_eval_summary.txt"
mkdir -p "./output"
{
  echo "Base Model Evaluation Summary"
  echo "============================="
  echo "MODE: $MODE"
  if [ "$MODE" = "val100" ]; then
    echo "Val dataset: $VAL_DATASET"
  else
    echo "Category test data dir: $DATASET_DIR"
  fi
  echo ""
} > "$SUMMARY_FILE"

run_infer_and_eval () {
  local DATA_PATH=$1
  local TAG=$2  # 例如 base_shopping / base_val100
  local RESULT_DIR="./output"

  mkdir -p "$RESULT_DIR"

  echo "[INFO] Running base model inference for $TAG ..."
  # 使用 --result_dir 把推理结果固定到 RESULT_DIR
  swift infer \
    --val_dataset "$DATA_PATH" \
    --model_type "$MODEL_TYPE" \
    --model_id_or_path "$MODEL_PATH" \
    --result_dir "$RESULT_DIR"

  if [ $? -ne 0 ]; then
    echo "[ERROR] Inference failed for $TAG"
    return
  fi

  JSONL_FILE=$(find "$RESULT_DIR" -maxdepth 1 -type f -name "*.jsonl" 2>/dev/null | sort | tail -n 1)
  if [ -z "$JSONL_FILE" ]; then
    echo "[ERROR] No jsonl result found in $RESULT_DIR for $TAG"
    return
  fi

  echo "[INFO] Evaluating $TAG with test_swift.py ..."
  OUT_FILE="$RESULT_DIR/$(basename "$JSONL_FILE" .jsonl)_${TAG}_result.txt"
  python evaluation/test_swift.py --data_path "$JSONL_FILE" > "$OUT_FILE" 2>&1
  echo "[INFO] Result saved to: $OUT_FILE"

  # 从结果中提取 Step-level accuracy，追加到 ./output/base_model_eval_summary.txt
  acc_line=$(grep "Step-level accuracy" "$OUT_FILE" | head -n 1)
  if [ -n "$acc_line" ]; then
    echo "${TAG}: ${acc_line}" >> "$SUMMARY_FILE"
  else
    echo "${TAG}: [WARN] No Step-level accuracy line found" >> "$SUMMARY_FILE"
  fi
}

if [ "$MODE" = "val100" ]; then
  # 基座模型在 Val_100.jsonl 上的综合评分
  if [ ! -f "$VAL_DATASET" ]; then
    echo "[ERROR] Val dataset not found: $VAL_DATASET"
    exit 1
  fi

  run_infer_and_eval "$VAL_DATASET" "base_val100"
else
  # 基座模型在 5 个按类别测试集上的评分
  if [ ! -d "$DATASET_DIR" ]; then
    echo "[ERROR] Test data directory not found: $DATASET_DIR"
    exit 1
  fi

  for CAT in "${CATEGORIES[@]}"; do
    cat_lower=$(echo "$CAT" | tr '[:upper:]' '[:lower:]')
    DATA_FILE="$DATASET_DIR/${cat_lower}_train.jsonl"

    if [ ! -f "$DATA_FILE" ]; then
      echo "[WARNING] Test file not found for $CAT: $DATA_FILE"
      continue
    fi

    run_infer_and_eval "$DATA_FILE" "base_${cat_lower}"
  done
fi

echo "[INFO] Base model evaluation finished (MODE=$MODE)."
echo "[INFO] Summary saved to: $SUMMARY_FILE"



