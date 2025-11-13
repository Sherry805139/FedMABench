# 训练错误排查指南

## 错误 1: `cannot import name 'IMAGE_FACTOR' from 'qwen_vl_utils.vision_process'`

### 问题描述
`qwen_vl_utils` 的某些版本不包含 `IMAGE_FACTOR` 常量，导致导入失败。

### 解决方案
**已修复**：已在 `swift/llm/utils/template.py` 中添加了兼容性处理，如果导入失败会使用默认值 `16`。

如果问题仍然存在，可以尝试：
```bash
# 更新 qwen_vl_utils 到最新版本
pip install --upgrade qwen-vl-utils
```

## 错误 2: `ValueError: Please check if the max_length is appropriate.`

### 问题描述
序列长度超过了设置的 `max_length=4096`。这通常发生在：
- 使用 `episode-wise` 数据时，一个 episode 包含多张图片和长文本
- 图片数量较多，导致 token 数超过限制

### 解决方案

#### 方案 1: 增加 max_length（推荐）
在训练命令中增加 `max_length` 的值：

```bash
CUDA_VISIBLE_DEVICES=1 MAX_PIXELS=602112 swift sft \
    --round 30 --fed_alg fedavg --client_num 30 --client_sample 5 \
    --model_type qwen2-vl-2b-instruct \
    --model_id_or_path /home/hmpiao/hmpiao/Qwen2-VL-2B-Instruct \
    --check_model_is_latest False --lazy_tokenize True --preprocess_num_proc 4 \
    --dataset "/home/hmpiao/hmpiao/xuerong/FedMABench/android_control_unpack/episode-wise-conversations.jsonl" \
    --sft_type lora --tuner_backend peft --dtype AUTO --output_dir output \
    --train_dataset_sample -1 --dataset_test_ratio 0 --max_steps -1 --max_length 8192 \
    --check_dataset_strategy warning --lora_rank 8 --lora_alpha 32 --lora_dropout 0.05 \
    --gradient_checkpointing true --batch_size 1 --weight_decay 0.1 --learning_rate 5e-5 \
    --gradient_accumulation_steps 4 --max_grad_norm 0.5 --warmup_ratio 0.03 \
    --eval_strategy no --save_strategy no --logging_steps 100
```

**注意**：将 `--max_length 4096` 改为 `--max_length 8192` 或更大值（如 `16384`）。

#### 方案 2: 使用 step-wise 数据（如果 episode-wise 太长）
如果 episode-wise 数据太长，可以使用 step-wise 数据：

```bash
# 先转换 step-wise 数据
python scripts/utils/convert_to_conversations.py \
    --input ~/hmpiao/xuerong/FedMABench/android_control_unpack/step-wise-all.jsonl \
    --output ~/hmpiao/xuerong/FedMABench/android_control_unpack/step-wise-conversations.jsonl \
    --image_root ~/hmpiao/xuerong/FedMABench

# 然后在训练中使用 step-wise 数据
--dataset "/home/hmpiao/hmpiao/xuerong/FedMABench/android_control_unpack/step-wise-conversations.jsonl"
```

#### 方案 3: 过滤过长的样本
如果某些 episode 特别长，可以在数据转换时过滤掉：

修改 `convert_to_conversations.py` 或在转换后手动过滤 JSONL 文件。

### 如何判断合适的 max_length

1. **检查模型的最大长度限制**：
   - Qwen2-VL-2B-Instruct 通常支持 8192 或更大的序列长度
   - 检查模型配置文件中的 `max_position_embeddings`

2. **估算 token 数**：
   - 每张图片大约占用 256-512 tokens（取决于图片大小和模型）
   - 文本部分：instruction + actions，通常几百到几千 tokens
   - 例如：7 张图片 × 400 tokens + 2000 tokens 文本 ≈ 4800 tokens

3. **测试建议**：
   - 先用 `max_length=8192` 尝试
   - 如果还有错误，逐步增加到 `16384` 或 `32768`
   - 注意：更大的 `max_length` 会占用更多显存

## 其他常见问题

### 问题：显存不足（OOM）
**解决方案**：
- 减小 `batch_size`（已经是 1，可以尝试更小的值）
- 增加 `gradient_accumulation_steps`（已经是 4）
- 减小 `MAX_PIXELS`（当前是 602112）
- 使用更小的 `max_length`

### 问题：数据加载慢
**解决方案**：
- 减少 `preprocess_num_proc`（当前是 4，可以改为 2）
- 确保数据文件在本地或高速存储上
- 使用 SSD 而不是 HDD

### 问题：训练不稳定
**解决方案**：
- 检查学习率（当前是 5e-5，可以尝试 1e-5）
- 增加 `warmup_ratio`（当前是 0.03）
- 检查数据质量，确保没有异常样本

