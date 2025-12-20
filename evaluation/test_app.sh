#!/bin/bash
# 在按app划分的多个测试集上评估所有app的LoRA模型（7x7组合）
# 使用方法: bash evaluation/test_app.sh [GPU_ID] [TEST_DATA_DIR]
# 例如: bash evaluation/test_app.sh 0 ./test_data_by_app

# 配置参数
GPU_ID=${1:-0}
TEST_DATA_DIR=${2:-"./test_data_by_app"}
BASE_OUTPUT_DIR="./lora_app"
MODEL_TYPE="qwen2-vl-2b-instruct"
MODEL_PATH="/home/hmpiao/hmpiao/Qwen2-VL-2B-Instruct"

# Apps from Table 5（同时作为LoRA类别和测试集类别）
APPS=("youtube")
APPS_=("amazon" "ebay" "flipkart" "gmail" "clock" "reminder" "youtube")

# 要测试的轮次（checkpoint）
ROUND_LIST=(30)

echo "[INFO] Testing all app LoRAs on app-wise test sets in: $TEST_DATA_DIR"
echo "[INFO] Using GPU: $GPU_ID"
echo "[INFO] Base output directory: $BASE_OUTPUT_DIR"

# 检查测试数据目录是否存在
if [ ! -d "$TEST_DATA_DIR" ]; then
    echo "[ERROR] Test data directory not found: $TEST_DATA_DIR"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAX_PIXELS=600000

# 为每个LoRA app、每个测试集 app 和 round 30 进行推理和评估（7x7组合）
for model_app in "${APPS[@]}"; do
    model_root="$BASE_OUTPUT_DIR/app_lora_${model_app}"

    if [ ! -d "$model_root" ]; then
        echo "[WARNING] Model directory not found for app $model_app: $model_root"
        continue
    fi

    for data_app in "${APPS_[@]}"; do
        test_file="$TEST_DATA_DIR/${data_app}_train.jsonl"

        echo ""
        echo "##########################################"
        echo "Model LoRA: $model_app  |  Test Set: $data_app"
        echo "##########################################"

        # 检查测试集是否存在
        if [ ! -f "$test_file" ]; then
            echo "[WARNING] Test dataset not found for app $data_app: $test_file"
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
            echo "Model: $model_app, Test: $data_app, Round: $round"
            echo "=========================================="

            # 查找对应round的checkpoint目录（兼容嵌套结构，如 qwen2-vl-2b-instruct/v0-xxx/global_lora_$round）
            ckpt_dir_candidates=$(find "$model_root" -type d -name "global_lora_$round" 2>/dev/null | sort)
            ckpt_dir=$(echo "$ckpt_dir_candidates" | tail -n 1)

            if [ -z "$ckpt_dir" ] || [ ! -d "$ckpt_dir" ]; then
                echo "[WARNING] No checkpoint directory found for app $model_app, round $round under $model_root, skipping..."
                continue
            fi

            echo "[INFO] Using checkpoint directory: $ckpt_dir"

            infer_dir="$ckpt_dir/infer_result"
            combo_dir="$infer_dir/${data_app}"

            mkdir -p "$infer_dir"

            # 如果该组合已经有推理结果，则跳过推理
            existing_jsonl=$(find "$combo_dir" -type f -name "*.jsonl" 2>/dev/null | head -n 1)
            if [ -z "$existing_jsonl" ]; then
                echo "[INFO] Running inference for model $model_app on test set $data_app, round $round..."

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
                    echo "[ERROR] Inference failed for model $model_app on test set $data_app, round $round"
                    continue
                fi

                mkdir -p "$combo_dir"
                # 将本次生成的jsonl移动到对应子目录
                new_jsonl_files=$(find "$infer_dir" -maxdepth 1 -type f -name "*.jsonl" 2>/dev/null)
                for f in $new_jsonl_files; do
                    mv "$f" "$combo_dir"/
                done
            else
                echo "[INFO] Inference results already exist for model $model_app on test set $data_app, round $round, skipping inference."
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
                output_file="$combo_dir/${base_name}_model-${model_app}_data-${data_app}_result.txt"

                # 使用test_swift.py进行评估（step-level / episode-level）
                python evaluation/test_swift.py --data_path "$jsonl_file" > "$output_file" 2>&1

                echo "[INFO] Results saved to: $output_file"
            done
        done
    done
done

echo ""
echo "=========================================="
echo "All app LoRA evaluation (7x7 combinations) completed!"
echo "=========================================="

# 汇总结果到 lora_app 目录下
echo ""
echo "Generating summary report for app LoRA evaluation..."
python evaluation/summarize_app_lora_results.py \
  --base_output_dir "$BASE_OUTPUT_DIR" \
  --summary_file "$BASE_OUTPUT_DIR/app_lora_summary.txt"
