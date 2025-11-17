#!/bin/bash
# 快速测试checkpoint效果的脚本
# 使用方法: bash bash/test_checkpoint.sh [GPU_ID] [TEST_DATA]

GPU_ID=${1:-0}
TEST_DATA=${2:-"test_data.jsonl"}  # 默认测试数据路径
CKPT_DIR="output/v30-20251117-203755/checkpoint-96"
MODEL_TYPE="qwen2-vl-2b-instruct"
MODEL_PATH="/home/hmpiao/hmpiao/Qwen2-VL-2B-Instruct"

echo "=========================================="
echo "Checkpoint 验证脚本"
echo "=========================================="
echo "Checkpoint: $CKPT_DIR"
echo "测试数据: $TEST_DATA"
echo "GPU: $GPU_ID"
echo ""

# 检查checkpoint是否存在
if [ ! -d "$CKPT_DIR" ]; then
    echo "❌ 错误: Checkpoint目录不存在: $CKPT_DIR"
    exit 1
fi

# 检查必需文件
echo "=== 1. 检查Checkpoint文件 ==="
required_files=("adapter_model.safetensors" "adapter_config.json" "trainer_state.json")
for file in "${required_files[@]}"; do
    if [ -f "$CKPT_DIR/$file" ]; then
        echo "✅ $file 存在"
    else
        echo "❌ $file 缺失"
        exit 1
    fi
done

# 显示训练状态
echo ""
echo "=== 2. 训练状态 ==="
if [ -f "$CKPT_DIR/trainer_state.json" ]; then
    echo "训练步数: $(cat $CKPT_DIR/trainer_state.json | grep -o '"global_step": [0-9]*' | grep -o '[0-9]*')"
    echo "Loss: $(cat $CKPT_DIR/trainer_state.json | grep -o '"loss": [0-9.]*' | head -1 | grep -o '[0-9.]*')"
    echo "Accuracy: $(cat $CKPT_DIR/trainer_state.json | grep -o '"acc": [0-9.]*' | head -1 | grep -o '[0-9.]*')"
fi

# 运行推理（如果测试数据存在）
if [ -f "$TEST_DATA" ]; then
    echo ""
    echo "=== 3. 运行推理 ==="
    echo "正在推理，请稍候..."
    
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    MAX_PIXELS=200000 \
    swift infer \
      --ckpt_dir "$CKPT_DIR" \
      --val_dataset "$TEST_DATA" \
      --model_type "$MODEL_TYPE" \
      --model_id_or_path "$MODEL_PATH" \
      --sft_type lora \
      --max_length 1024
    
    if [ $? -eq 0 ]; then
        echo "✅ 推理完成"
        
        # 查找推理结果
        INFER_RESULT=$(find "$CKPT_DIR/infer_result" -name "*.jsonl" 2>/dev/null | head -1)
        if [ -n "$INFER_RESULT" ]; then
            echo ""
            echo "=== 4. 计算准确率 ==="
            python evaluation/test_swift.py --data_path "$INFER_RESULT"
        fi
    else
        echo "❌ 推理失败"
    fi
else
    echo ""
    echo "⚠️  测试数据不存在: $TEST_DATA"
    echo "   跳过推理步骤"
    echo "   使用方法: bash bash/test_checkpoint.sh $GPU_ID /path/to/test_data.jsonl"
fi

echo ""
echo "=========================================="
echo "验证完成"
echo "=========================================="

