#!/bin/bash
# 在按category划分的多个测试集上评估所有category的LoRA模型
# 使用方法: bash test_category_loras.sh [GPU_ID] [TEST_DATA_DIR]
# 例如: bash test_category_loras.sh 0 ./test_data_by_category

# 配置参数
GPU_ID=${1:-0}
TEST_DATA_DIR=${2:-"./test_data_by_category"}
BASE_OUTPUT_DIR="./lora_category"
MODEL_TYPE="qwen2-vl-2b-instruct"
MODEL_PATH="/home/hmpiao/hmpiao/Qwen2-VL-2B-Instruct"
CATEGORY_MAPPING_FILE="./episode_category_mapping.json"

# Categories from Table 5（同时作为LoRA类别和测试集类别）
CATEGORIES=("Shopping" "Traveling" "Office" "Lives" "Entertainment")

# 要测试的轮次（checkpoint）
ROUND_LIST=(30)

echo "[INFO] Testing all category LoRAs on category-wise test sets in: $TEST_DATA_DIR"
echo "[INFO] Using GPU: $GPU_ID"
echo "[INFO] Base output directory: $BASE_OUTPUT_DIR"

# 检查测试数据目录是否存在
if [ ! -d "$TEST_DATA_DIR" ]; then
    echo "[ERROR] Test data directory not found: $TEST_DATA_DIR"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAX_PIXELS=200000

# 为每个LoRA category、每个测试集category和round 30进行推理和评估（5x5组合）
for model_cat in "${CATEGORIES[@]}"; do

    for data_cat in "${CATEGORIES[@]}"; do
        data_lower=$(echo "$data_cat" | tr '[:upper:]' '[:lower:]')
        test_file="$TEST_DATA_DIR/${data_lower}_train.jsonl"

        echo ""
        echo "##########################################"
        echo "Model LoRA: $model_cat  |  Test Set: $data_cat"
        echo "##########################################"

        # 检查测试集是否存在
        if [ ! -f "$test_file" ]; then
            echo "[WARNING] Test dataset not found for category $data_cat: $test_file"
            continue
        fi

        # 简单检查是否为推理格式（不包含\"img\"字段）
        first_line=$(head -n 1 "$test_file" 2>/dev/null)
        if [ -n "$first_line" ] && echo "$first_line" | grep -q '"img"'; then
            echo "[ERROR] Detected original episode format in $test_file (contains 'img' or 'imgs')."
            echo "       Please convert to inference format first, for example:"
            echo "       python evaluation/convert_to_inference_format.py --input $test_file --output ${test_file%.jsonl}_infer.jsonl"
            continue
        fi

        for round in "${ROUND_LIST[@]}"; do
            echo ""
            echo "=========================================="
            echo "Model: $model_cat, Test: $data_cat, Round: $round"
            echo "=========================================="

            # 查找对应round的checkpoint目录（兼容嵌套结构，如 qwen2-vl-2b-instruct/v0-xxx/global_lora_$round）
            ckpt_dir_candidates=$(find "$BASE_OUTPUT_DIR/category_lora_${model_cat}" -type d -name "global_lora_$round" 2>/dev/null | sort)
            ckpt_dir=$(echo "$ckpt_dir_candidates" | tail -n 1)

            if [ -z "$ckpt_dir" ] || [ ! -d "$ckpt_dir" ]; then
                echo "[WARNING] No checkpoint directory found for round $round under $BASE_OUTPUT_DIR/category_lora_${model_cat}, skipping..."
                continue
            fi

            echo "[INFO] Using checkpoint directory: $ckpt_dir"

            infer_dir="$ckpt_dir/infer_result"
            combo_dir="$infer_dir/${data_lower}"

            mkdir -p "$infer_dir"

            # 如果该组合已经有推理结果，则跳过推理
            existing_jsonl=$(find "$combo_dir" -type f -name "*.jsonl" 2>/dev/null | head -n 1)
            if [ -z "$existing_jsonl" ]; then
                echo "[INFO] Running inference for model $model_cat on test set $data_cat, round $round..."

                # 清理顶层旧的jsonl（保留子目录）
                find "$infer_dir" -maxdepth 1 -type f -name "*.jsonl" -delete

                # 运行推理
                swift infer \
                  --ckpt_dir "$ckpt_dir" \
                  --val_dataset "$test_file" \
                  --model_type $MODEL_TYPE \
                  --model_id_or_path $MODEL_PATH \
                  --sft_type lora

                if [ $? -ne 0 ]; then
                    echo "[ERROR] Inference failed for model $model_cat on test set $data_cat, round $round"
                    continue
                fi

                mkdir -p "$combo_dir"
                # 将本次生成的jsonl移动到对应子目录
                new_jsonl_files=$(find "$infer_dir" -maxdepth 1 -type f -name "*.jsonl" 2>/dev/null)
                for f in $new_jsonl_files; do
                    mv "$f" "$combo_dir"/
                done
            else
                echo "[INFO] Inference results already exist for model $model_cat on test set $data_cat, round $round, skipping inference."
            fi

            # 计算准确率
            jsonl_files=$(find "$combo_dir" -type f -name "*.jsonl" 2>/dev/null)
            if [ -z "$jsonl_files" ]; then
                echo "[WARNING] No inference results found in $combo_dir"
                continue
            fi

            for jsonl_file in $jsonl_files; do
                echo "[INFO] Processing: $jsonl_file"
                base_name="$(basename "$jsonl_file" .jsonl)"
                output_file="$combo_dir/${base_name}_model-${model_cat}_data-${data_lower}_result.txt"

                # 使用test_swift_cate.py进行评估（按category统计）
                if [ -f "$CATEGORY_MAPPING_FILE" ]; then
                    python evaluation/test_swift_cate.py \
                      --data_path "$jsonl_file" \
                      --category_file "$CATEGORY_MAPPING_FILE" > "$output_file" 2>&1
                else
                    # 如果没有category映射文件，使用普通的test_swift.py
                    python evaluation/test_swift.py --data_path "$jsonl_file" > "$output_file" 2>&1
                fi

                echo "[INFO] Results saved to: $output_file"
            done
        done
    done
done

echo ""
echo "=========================================="
echo "All category LoRA evaluation (5x5 combinations) completed!"
echo "=========================================="

# 汇总结果（使用 Python 脚本，兼容嵌套的 global_lora_* 目录）
echo ""
echo "Generating summary report with Python summarizer..."
python evaluation/summarize_category_lora_results.py \
  --base_output_dir "$BASE_OUTPUT_DIR" \
  --summary_file "$BASE_OUTPUT_DIR/category_lora_summary.txt"




