# Android Control 数据转换流程

## 概述

将解压后的 Android Control 数据转换为训练所需的对话格式，需要两个步骤：

1. **生成 JSONL 格式**：将解压后的目录结构转换为 JSONL 文件
2. **转换为对话格式**：将 JSONL 转换为训练所需的 conversations 格式

## 步骤 1: 生成 JSONL 格式

运行 `2_gen_jsonl_from_unpack_ac.py` 脚本：

```bash
cd ~/hmpiao/xuerong/FedMABench
python data_process/2_gen_jsonl_from_unpack_ac.py \
    --data_dir ~/hmpiao/xuerong/FedMABench/android_control_unpack
```

**输出文件**：
- `android_control_unpack/step-wise-all.jsonl`：每个步骤一条记录（用于 step-wise 训练）
- `android_control_unpack/episode-wise-all.jsonl`：每个 episode 一条记录（用于 episode-wise 训练，推荐）

**字段说明**：
- `episode_id`: episode ID
- `instruction`: 任务目标（goal）
- `sub_instructions`: 子目标列表（sub_goals）
- `acts_origin`: 原始动作列表（JSON 字符串列表）
- `acts_convert`: 转换后的动作描述列表
- `imgs`: 截图路径列表
- `lightxmls`: XML 文件路径列表

## 步骤 2: 转换为对话格式

使用 `convert_to_conversations.py` 将 JSONL 转换为对话格式：

```bash
python scripts/utils/convert_to_conversations.py \
    --input ~/hmpiao/xuerong/FedMABench/android_control_unpack/episode-wise-all.jsonl \
    --output ~/hmpiao/xuerong/FedMABench/android_control_unpack/episode-wise-conversations.jsonl \
    --image_root ~/hmpiao/xuerong/FedMABench
```

**参数说明**：
- `--input`: 输入的 JSONL 文件路径
- `--output`: 输出的对话格式 JSONL 文件路径
- `--image_root`: 图片路径的根目录（用于将相对路径转换为绝对路径）
- `--drop_if_missing`: （可选）如果图片文件不存在则丢弃该样本

**输出格式**：
```json
{
  "conversations": [
    {"from": "user", "value": "<image>\n<image>\n任务目标"},
    {"from": "assistant", "value": "动作1\n动作2\n..."}
  ],
  "images": ["/path/to/image1.png", "/path/to/image2.png", ...],
  "episode_id": "000123"
}
```

## 步骤 3: 用于训练

在训练命令中使用转换后的数据：

```bash
CUDA_VISIBLE_DEVICES=1 MAX_PIXELS=602112 swift sft \
    --round 30 --fed_alg fedavg --client_num 30 --client_sample 5 \
    --model_type qwen2-vl-2b-instruct \
    --model_id_or_path /home/hmpiao/hmpiao/Qwen2-VL-2B-Instruct \
    --check_model_is_latest False --lazy_tokenize True --preprocess_num_proc 4 \
    --dataset "/home/hmpiao/hmpiao/xuerong/FedMABench/android_control_unpack/episode-wise-conversations.jsonl" \
    --sft_type lora --tuner_backend peft --dtype AUTO --output_dir output \
    --train_dataset_sample -1 --dataset_test_ratio 0 --max_steps -1 --max_length 4096 \
    --check_dataset_strategy warning --lora_rank 8 --lora_alpha 32 --lora_dropout 0.05 \
    --gradient_checkpointing true --batch_size 1 --weight_decay 0.1 --learning_rate 5e-5 \
    --gradient_accumulation_steps 4 --max_grad_norm 0.5 --warmup_ratio 0.03 \
    --eval_strategy no --save_strategy no --logging_steps 100
```

## 注意事项

1. **路径问题**：确保所有路径使用绝对路径或正确的相对路径
2. **图片路径**：`convert_to_conversations.py` 会自动检查图片文件是否存在，如果使用 `--drop_if_missing`，不存在的图片会被丢弃
3. **数据格式**：训练框架会自动识别 `conversations` 字段并使用 `ConversationsPreprocessor` 处理数据
4. **内存占用**：如果数据量很大，可以考虑分批处理或使用流式处理

## 故障排查

### 问题：找不到图片文件
- 检查 `--image_root` 参数是否正确
- 检查图片路径是否为绝对路径
- 使用 `--drop_if_missing` 跳过缺失的图片

### 问题：字段不匹配
- 确保使用 `episode-wise-all.jsonl`（字段名匹配）
- 检查 `convert_to_conversations.py` 的版本是否支持这些字段

### 问题：内存不足
- 减少 `--preprocess_num_proc` 的值
- 分批处理数据文件

