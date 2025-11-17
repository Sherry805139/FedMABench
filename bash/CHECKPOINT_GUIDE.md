# Checkpoint 结构说明与验证指南

## 📁 Checkpoint 文件结构

你的 checkpoint 位于：`output/v30-20251117-203755/checkpoint-96/`

### 文件列表及说明

```
checkpoint-96/
├── adapter_model.safetensors    # LoRA权重文件（56MB）- 核心文件
├── adapter_config.json          # LoRA配置（rank, alpha, dropout等）
├── trainer_state.json           # 训练状态（loss, step, epoch等）
├── training_args.bin            # 训练参数（二进制）
├── sft_args.json               # SFT训练参数（JSON格式）
├── configuration.json           # 模型配置
├── generation_config.json       # 生成配置
├── additional_config.json       # 额外配置
└── README.md                    # 模型卡片（自动生成）
```

---

## 📊 关键文件详解

### 1. `adapter_model.safetensors` - LoRA权重

**作用**：包含所有可训练的LoRA权重参数

**大小**：56MB（相对较小，因为只保存LoRA参数，不是完整模型）

**内容**：
- LoRA的A矩阵和B矩阵权重
- 只包含训练时更新的参数（rank=8, alpha=32）

**重要性**：⭐⭐⭐⭐⭐ **这是最核心的文件**

---

### 2. `adapter_config.json` - LoRA配置

**内容**：
```json
{
  "base_model_name_or_path": "/home/hmpiao/hmpiao/Qwen2-VL-2B-Instruct",
  "peft_type": "LORA",
  "r": 8,                    // LoRA rank
  "lora_alpha": 32,          // LoRA alpha
  "lora_dropout": 0.05,      // LoRA dropout
  "target_modules": "^(model)(?!.*(lm_head|output|emb|wte|shared)).*",
  "task_type": "CAUSAL_LM"
}
```

**作用**：告诉加载器如何将LoRA权重应用到基础模型

---

### 3. `trainer_state.json` - 训练状态

**关键信息**：
```json
{
  "global_step": 96,              // 总训练步数
  "epoch": 1.0,                   // 训练轮数
  "max_steps": 96,                // 最大步数
  "log_history": [
    {
      "step": 1,
      "loss": 2.12130213,         // 训练损失
      "acc": 0.49849987,           // 准确率（49.85%）
      "learning_rate": 1.67e-05,   // 学习率
      "grad_norm": 2.42,           // 梯度范数
      "memory(GiB)": 14.81,        // 显存占用
      "train_speed(iter/s)": 0.056 // 训练速度
    }
  ]
}
```

**分析**：
- ✅ **Loss**: 2.12（初始损失，需要更多步数才能看到下降）
- ⚠️ **Acc**: 49.85%（接近随机猜测，说明训练刚开始）
- ✅ **显存**: 14.81 GiB（优化成功，没有OOM）
- ✅ **训练速度**: 0.056 iter/s（约18秒/步）

---

### 4. `sft_args.json` - 训练参数

**关键参数**：
```json
{
  "model_type": "qwen2-vl-2b-instruct",
  "sft_type": "lora",
  "lora_rank": 8,
  "lora_alpha": 32,
  "max_length": 1024,
  "learning_rate": 5e-5,
  "batch_size": 1,
  "gradient_accumulation_steps": 16,
  "max_steps": 96,
  "dataset": ["/home/hmpiao/.../episode-wise-conversations.jsonl"]
}
```

---

## 🔍 如何验证训练效果

### 方法 1: 使用 Swift Infer 进行推理测试（推荐）

#### 步骤 1: 准备测试数据

创建测试数据集（JSONL格式），例如 `test_data.jsonl`：

```jsonl
{"images": ["path/to/image1.png"], "query": "What is in this image?", "response": "..."}
{"images": ["path/to/image2.png"], "query": "Click the button", "response": "..."}
```

#### 步骤 2: 运行推理

```bash
# 单GPU推理
CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=200000 \
swift infer \
  --ckpt_dir output/v30-20251117-203755/checkpoint-96 \
  --val_dataset test_data.jsonl \
  --model_type qwen2-vl-2b-instruct \
  --model_id_or_path /home/hmpiao/hmpiao/Qwen2-VL-2B-Instruct \
  --sft_type lora \
  --max_length 1024
```

**输出**：会在 `checkpoint-96/infer_result/` 目录下生成推理结果

---

### 方法 2: 使用 Python 代码加载模型

```python
from swift.llm import get_model_tokenizer
from swift.tuners import Swift
from peft import PeftModel
from transformers import AutoProcessor
import torch

# 1. 加载基础模型
model, tokenizer = get_model_tokenizer(
    model_type='qwen2-vl-2b-instruct',
    model_id_or_path='/home/hmpiao/hmpiao/Qwen2-VL-2B-Instruct',
    torch_dtype=torch.bfloat16,
    device_map='auto'
)

# 2. 加载LoRA权重
model = PeftModel.from_pretrained(
    model,
    'output/v30-20251117-203755/checkpoint-96'
)

# 3. 设置为评估模式
model.eval()

# 4. 测试推理
processor = tokenizer.processor
image = Image.open('test_image.png')
query = "What is in this image?"

# 准备输入
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": query}
        ]
    }
]

# 生成回复
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = processor.process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

print(f"Response: {response}")
```

---

### 方法 3: 使用项目提供的评估脚本

根据你的项目结构，可以使用 `evaluation/` 目录下的脚本：

```bash
# 1. 运行推理
CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=200000 \
swift infer \
  --ckpt_dir output/v30-20251117-203755/checkpoint-96 \
  --val_dataset /path/to/val_dataset.jsonl \
  --model_type qwen2-vl-2b-instruct \
  --model_id_or_path /home/hmpiao/hmpiao/Qwen2-VL-2B-Instruct \
  --sft_type lora

# 2. 计算准确率
python evaluation/test_swift.py \
  --data_path output/v30-20251117-203755/checkpoint-96/infer_result/*.jsonl
```

---

## 📈 评估指标说明

### 1. **Loss（损失）**

**期望趋势**：
- ✅ 训练过程中应该**逐渐下降**
- ⚠️ 当前值：2.12（较高，因为只训练了96步）

**判断标准**：
- Loss < 1.0：训练良好
- Loss < 0.5：训练很好
- Loss < 0.1：可能过拟合

---

### 2. **Accuracy（准确率）**

**当前值**：49.85%

**判断标准**：
- 对于Android Control任务：
  - **随机猜测**: ~25-33%（取决于动作空间）
  - **基线模型**: ~40-50%
  - **良好模型**: >60%
  - **优秀模型**: >75%

**⚠️ 注意**：当前准确率接近随机，因为：
- 只训练了96步（1个epoch）
- 需要更多训练步数才能看到明显提升

---

### 3. **训练稳定性**

检查 `trainer_state.json` 中的：
- **grad_norm**: 2.42（正常，说明梯度没有爆炸）
- **learning_rate**: 1.67e-05（正常衰减）
- **memory**: 14.81 GiB（稳定，没有OOM）

---

## 🎯 验证 Checklist

### ✅ 基础检查

- [x] Checkpoint文件完整（所有必需文件都存在）
- [x] LoRA权重文件大小合理（56MB，符合rank=8）
- [x] 训练完成（global_step = max_steps = 96）
- [x] 没有OOM错误
- [x] 梯度正常（grad_norm < 10）

### ⚠️ 需要改进

- [ ] Loss仍然较高（2.12），需要更多训练
- [ ] Accuracy较低（49.85%），接近随机猜测
- [ ] 只训练了96步，可能需要更多步数

---

## 🔧 如何继续训练

如果需要继续训练，可以从checkpoint恢复：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
MAX_PIXELS=200000 \
swift sft \
  --resume_from_checkpoint output/v30-20251117-203755/checkpoint-96 \
  --model_type qwen2-vl-2b-instruct \
  --model_id_or_path /home/hmpiao/hmpiao/Qwen2-VL-2B-Instruct \
  --dataset /home/hmpiao/.../episode-wise-conversations.jsonl \
  --max_steps 500 \  # 增加训练步数
  --save_steps 100 \
  ...其他参数...
```

---

## 📝 快速测试脚本

创建一个简单的测试脚本 `test_checkpoint.sh`：

```bash
#!/bin/bash
# 测试checkpoint效果

CKPT_DIR="output/v30-20251117-203755/checkpoint-96"
MODEL_TYPE="qwen2-vl-2b-instruct"
MODEL_PATH="/home/hmpiao/hmpiao/Qwen2-VL-2B-Instruct"
TEST_DATA="test_data.jsonl"  # 你的测试数据

echo "=== 1. 运行推理 ==="
CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=200000 \
swift infer \
  --ckpt_dir "$CKPT_DIR" \
  --val_dataset "$TEST_DATA" \
  --model_type "$MODEL_TYPE" \
  --model_id_or_path "$MODEL_PATH" \
  --sft_type lora \
  --max_length 1024

echo "=== 2. 计算准确率 ==="
python evaluation/test_swift.py \
  --data_path "$CKPT_DIR/infer_result/*.jsonl"

echo "=== 3. 查看训练日志 ==="
cat "$CKPT_DIR/trainer_state.json" | python -m json.tool
```

---

## 🎓 总结

### Checkpoint结构
- ✅ **LoRA权重** (`adapter_model.safetensors`) - 核心文件
- ✅ **配置** (`adapter_config.json`) - 加载配置
- ✅ **训练状态** (`trainer_state.json`) - 训练信息

### 当前状态
- ✅ 训练成功完成（96步）
- ✅ 没有OOM问题
- ⚠️ Loss和Accuracy需要更多训练才能改善

### 验证方法
1. **Swift Infer** - 推荐，最简单
2. **Python代码** - 灵活，可自定义
3. **评估脚本** - 项目提供，标准化

### 下一步
1. 运行推理测试，查看实际效果
2. 如果效果不佳，继续训练更多步数
3. 调整超参数（learning_rate, batch_size等）

