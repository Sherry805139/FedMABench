#!/bin/bash

# 配置基础路径 - 根据您的实际路径修改
# 如果脚本在项目根目录运行，使用相对路径；如果在服务器上，使用绝对路径
base_path=./output/qwen2-vl-2b-instruct
# base_path=/data1/hmpiao/xuerong/FedMABench/output  # 服务器上的路径示例

# 模型配置 - 根据您的训练配置修改
model=qwen2-vl-2b-instruct
model_id_or_path=/home/hmpiao/hmpiao/Qwen2-VL-2B-Instruct
# 如果模型路径不同，请修改上面的路径

# 要测试的轮次列表
round_list=(5 10 15 20 25 30)

# 验证数据集路径 - 需要您提供实际的验证数据集路径
# 格式应该是jsonl文件，每行包含images、query、label等字段
# 示例路径（请根据实际情况修改）：
val_dataset=./data/Basic-AC/Val_100.jsonl

peft_list=(
v31-20251117-220840
)

for round in ${round_list[@]}; do  
    for i in ${peft_list[@]}; do
        echo "Testing round $round with $i"
        
        # 检查checkpoint目录是否存在
        ckpt_dir="$base_path/$i/global_lora_$round"
        if [ ! -d "$ckpt_dir" ]; then
            echo "Warning: Checkpoint directory $ckpt_dir does not exist, skipping..."
            continue
        fi
        
        # 检查是否已有推理结果
        jsonl_files=$(find "$ckpt_dir/infer_result" -type f -name "*.jsonl" 2>/dev/null)

        if [ -z "$jsonl_files" ]; then
            echo "Running inference for round $round..."
            # 检查验证数据集是否存在
            if [ ! -f "$val_dataset" ]; then
                echo "Error: Validation dataset $val_dataset not found!"
                echo ""
                echo "提示: 如果您的数据格式是原始episode格式（包含img、instruction、acts_convert等字段），"
                echo "      请先使用转换脚本转换为推理格式："
                echo "      python evaluation/convert_to_inference_format.py --input <原始文件> --output $val_dataset"
                echo ""
                continue
            fi
            
            # 检查数据格式（简单检查）
            first_line=$(head -n 1 "$val_dataset" 2>/dev/null)
            if [ -n "$first_line" ]; then
                if echo "$first_line" | grep -q '"img"'; then
                    echo "警告: 检测到原始格式（包含'img'字段），需要转换为推理格式"
                    echo "      请运行: python evaluation/convert_to_inference_format.py --input $val_dataset --output <输出文件>"
                    continue
                fi
            fi
            
            MAX_PIXELS=602112 CUDA_VISIBLE_DEVICES=$1 swift infer \
              --ckpt_dir "$ckpt_dir" \
              --val_dataset "$val_dataset" \
              --model_type $model \
              --model_id_or_path $model_id_or_path \
              --sft_type lora
        else
            echo "Inference results already exist, skipping inference step."
        fi

        # 计算准确率
        jsonl_files=$(find "$ckpt_dir/infer_result" -type f -name "*.jsonl" 2>/dev/null)
        if [ -z "$jsonl_files" ]; then
            echo "Warning: No inference results found in $ckpt_dir/infer_result"
            continue
        fi
        
        for jsonl_file in $jsonl_files; do
            echo "Processing: $jsonl_file"
            output_file="$ckpt_dir/infer_result/$(basename $jsonl_file .jsonl)_result.txt"
            # 使用test_swift.py进行评估（如果test_swift_fed.py不存在）
            python evaluation/test_swift.py --data_path "$jsonl_file" > "$output_file" 2>&1
            echo "Results saved to: $output_file"
        done
    done
done