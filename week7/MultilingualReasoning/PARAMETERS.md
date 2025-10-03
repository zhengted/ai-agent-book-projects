# 参数对照表

本文档列出所有训练参数，确保与 OpenAI Cookbook Notebook 完全一致。

## 原始 Notebook 参数

来源：https://github.com/openai/openai-cookbook/blob/main/articles/gpt-oss/fine-tune-transfomers.ipynb

### 模型配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `model_name` | `openai/gpt-oss-20b` | 基础模型 |
| `attn_implementation` | `eager` | 注意力实现方式 |
| `torch_dtype` | `bfloat16` | 模型精度 |
| `use_cache` | `False`（训练）/ `True`（推理） | KV 缓存 |
| `device_map` | `auto` | 设备分配 |

### LoRA 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `r` | `16` | LoRA 秩（rank） |
| `lora_alpha` | `16` | LoRA 缩放参数 |
| `lora_dropout` | `0.05` | Dropout 概率 |
| `bias` | `none` | 不训练偏置 |
| `task_type` | `CAUSAL_LM` | 任务类型 |
| `target_modules` | `["q_proj", "v_proj"]` | 目标模块 |

### 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_train_epochs` | `3` | 训练轮数 |
| `per_device_train_batch_size` | `8` | 每设备批次大小 |
| `gradient_accumulation_steps` | `2` | 梯度累积步数 |
| **有效批次大小** | **16** | **8 × 2** |
| `learning_rate` | `2e-5` | 学习率 |
| `lr_scheduler_type` | `cosine` | 学习率调度器 |
| `warmup_ratio` | `0.1` | 预热比例 |
| `weight_decay` | `0.01` | 权重衰减 |
| `max_seq_length` | `2048` | 最大序列长度 |

### 优化器配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `optim` | `adamw_torch_fused` | 优化器（融合版 AdamW） |
| `bf16` | `True` | 使用 bfloat16 混合精度 |
| `tf32` | `True` | 使用 TF32（Ampere GPU） |

### 日志和保存

| 参数 | 值 | 说明 |
|------|-----|------|
| `logging_steps` | `10` | 每 10 步记录一次 |
| `save_strategy` | `steps` | 按步数保存 |
| `save_steps` | `100` | 每 100 步保存一次 |
| `save_total_limit` | `2` | 只保留最新 2 个检查点 |
| `report_to` | `none` | 不使用实验跟踪 |

### 生成参数（推理时）

| 参数 | 值 | 说明 |
|------|-----|------|
| `max_new_tokens` | `512` | 最大生成 token 数 |
| `do_sample` | `True` | 启用采样 |
| `temperature` | `0.6` | 采样温度 |
| `top_p` | `None` | 不使用核采样 |
| `top_k` | `None` | 不使用 top-k 采样 |

## 资源消耗

| 指标 | 值 | 说明 |
|------|-----|------|
| **峰值显存** | **~97GB** | **实测值，超过 80GB！** |
| 推荐 GPU | H200（141GB） | 可以完美运行 |
| 不推荐 GPU | H100/A100（80GB） | 会出现 OOM |
| 训练时间 | ~18 分钟 | 在 H100 上（理论值） |

## 本项目实现

### 命令行参数

```bash
python gpt_oss_20b_sft.py \
  --mode full \
  --model_name openai/gpt-oss-20b \
  --output_dir ./gpt-oss-20b-multilingual-reasoner \
  --batch_size 8 \
  --num_epochs 3 \
  --learning_rate 2e-5 \
  --max_seq_length 2048 \
  --lora_rank 16 \
  --lora_alpha 16
```

### 代码实现位置

1. **数据集加载**: `load_and_prepare_dataset()` - 第 27 行
2. **模型加载**: `load_base_model()` - 第 77 行
3. **LoRA 配置**: `prepare_model_for_lora()` - 第 132 行
4. **训练配置**: `train_model()` - 第 198 行
5. **推理生成**: `generate_response()` - 第 357 行

### 与 Notebook 的差异

✅ **完全一致的参数**:
- 所有训练超参数
- LoRA 配置
- 优化器配置
- 模型加载方式

⚠️ **实际发现的差异**:
- Notebook 声称在 H100 上可运行
- **实际峰值显存 ~97GB，H100（80GB）会 OOM**
- 需要使用 H200 或采用优化策略

## 验证清单

- [x] 模型配置与 notebook 一致
- [x] LoRA 配置与 notebook 一致
- [x] 训练超参数与 notebook 一致
- [x] 优化器配置与 notebook 一致
- [x] 推理生成参数与 notebook 一致
- [x] 数据集使用正确（HuggingFaceH4/Multilingual-Thinking）
- [x] 系统提示格式正确（`reasoning language: {language}`）
- [x] 显存需求已在文档中明确说明

## 调试和故障排除

### 如果遇到 OOM 错误

1. **确认 GPU 显存**:
   ```bash
   nvidia-smi
   ```

2. **查看实际显存使用**:
   在训练过程中监控显存：
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **减小显存占用**（会偏离原始配置）:
   ```bash
   # 方案 1: 减小批次大小
   python gpt_oss_20b_sft.py --batch_size 4 --max_seq_length 1536
   
   # 方案 2: 更激进的减小
   python gpt_oss_20b_sft.py --batch_size 2 --max_seq_length 1024
   ```

4. **使用多 GPU**:
   ```bash
   # DeepSpeed ZeRO-3
   deepspeed --num_gpus=2 gpt_oss_20b_sft.py --mode train
   ```

## 参考

- [OpenAI Cookbook - Fine-tuning Transformers](https://github.com/openai/openai-cookbook/blob/main/articles/gpt-oss/fine-tune-transfomers.ipynb)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Multilingual-Thinking Dataset](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking)

