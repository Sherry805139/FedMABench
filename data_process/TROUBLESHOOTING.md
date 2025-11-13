# 训练错误排查指南

## 错误 1: `cannot import name 'IMAGE_FACTOR'/'MIN_PIXELS'/'MAX_PIXELS' from 'qwen_vl_utils.vision_process'`

### 问题描述
`qwen_vl_utils` 的某些版本不包含这些常量，导致导入失败。

### 解决方案
**已修复**：已在 `swift/llm/utils/template.py` 中添加了兼容性处理：
- `IMAGE_FACTOR`: 默认值 `16`
- `MIN_PIXELS`: 默认值 `576`
- `MAX_PIXELS`: 默认值 `1048576`

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

## 错误 3: `AttributeError: 'Qwen2VLForConditionalGeneration' object has no attribute 'get_rope_index'`

### 问题描述
某些版本的 Qwen2-VL 模型或 transformers 库可能不包含 `get_rope_index` 方法，导致在 `data_collator` 中调用失败。

### 解决方案
**已修复**：已在 `swift/llm/utils/template.py` 的 `_Qwen2VLTemplateMixin.data_collator` 方法中添加了兼容性处理：
- 首先检查 `get_rope_index` 方法是否存在
- 如果存在，使用该方法计算 `position_ids`
- 如果不存在或调用失败，使用默认方式计算 `position_ids`（基于 `attention_mask`）

### 说明
`get_rope_index` 是 Qwen2-VL 用于处理图像和视频 token 的特殊方法，用于计算正确的位置编码。如果方法不存在，使用标准的 `position_ids` 计算方式（`torch.arange`）作为后备方案，这在大多数情况下也能正常工作。

## 错误 4: `AttributeError: 'Qwen2VLModel' object has no attribute 'embed_tokens'`

### 问题描述
在使用 LoRA 训练 Qwen2-VL 模型时，`_post_encode` 方法无法找到 `embed_tokens` 属性。这是因为：
1. **LoRA 包装**：使用 PeftModel 包装后，模型结构变为 `PeftModel -> base_model -> model -> embed_tokens`
2. **模型版本差异**：不同版本的 Qwen2-VL 模型可能有不同的内部结构
3. **路径问题**：代码只尝试了简单的路径，没有处理所有可能的情况

### 解决方案
**已修复**：已在 `swift/llm/utils/template.py` 的 `_Qwen2VLTemplateMixin._post_encode` 方法中添加了完整的路径查找逻辑：
- **PeftModel (LoRA) 情况**：优先检查 `model.base_model.model.embed_tokens`
- **常规模型**：尝试多个路径：
  - `model.model.embed_tokens`
  - `model.model.model.embed_tokens`
  - `model.get_input_embeddings()`（作为后备方案）
- **错误信息**：如果都失败，会提供详细的错误信息帮助调试

### 说明
这不是模型版本的问题，而是代码需要处理不同的模型包装情况（特别是 LoRA）。修复后的代码会自动检测模型类型并选择正确的路径。

## 错误 5: `KeyError: 'response'` (使用 conversations 格式数据时)

### 问题描述
当使用 `conversations` 格式的数据时，数据集检查函数在预处理之前运行，此时数据还没有被转换成 `query` 和 `response` 格式，导致 `KeyError`。

### 解决方案
**已修复**：已在 `swift/llm/utils/dataset.py` 的 `_check_dataset` 函数中添加了对 `conversations` 格式的支持。如果数据包含 `conversations` 字段，会跳过 `response` 字段的检查。

### 验证
确保你的数据格式正确：
```json
{
  "conversations": [
    {"from": "user", "value": "<image>\n任务目标"},
    {"from": "assistant", "value": "动作描述"}
  ],
  "images": ["/path/to/image.png"]
}
```

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

