# 快速开始指南

## ⚠️ 在开始之前

**请确保你有 H200 GPU（141GB 显存）或更高配置！**

峰值显存占用约 97GB，单卡 80GB GPU（H100/A100）会出现 OOM 错误。

## 第一步：安装依赖

```bash
# 进入项目目录
cd projects/week7/MultilingualReasoning

# 安装 PyTorch（CUDA 12.8）
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 安装其他依赖
pip install -r requirements.txt

# 登录 Hugging Face
huggingface-cli login
```

## 第二步：训练模型

### 完整训练流程（推荐）

```bash
python gpt_oss_20b_sft.py --mode full
```

这将：
1. 自动下载数据集
2. 加载 openai/gpt-oss-20b 模型
3. 配置 LoRA
4. 训练 3 轮（约 18 分钟）
5. 保存模型
6. 运行推理示例

### 仅训练（不运行推理）

```bash
python gpt_oss_20b_sft.py --mode train
```

### 自定义参数训练

```bash
python gpt_oss_20b_sft.py \
  --mode train \
  --batch_size 8 \
  --num_epochs 3 \
  --learning_rate 2e-5 \
  --max_seq_length 2048 \
  --output_dir ./my_model
```

## 第三步：推理测试

### 使用内置示例

```bash
python gpt_oss_20b_sft.py --mode inference --output_dir ./gpt-oss-20b-multilingual-reasoner
```

### 使用交互式界面

```bash
python quickstart.py
```

然后按提示输入：
- 推理语言（如：Chinese, English, German）
- 你的问题

## 显存不足？

如果你只有 80GB GPU，可以尝试：

### 方案 1：减小批次和序列长度

```bash
python gpt_oss_20b_sft.py --batch_size 4 --max_seq_length 1536
```

### 方案 2：使用 4-bit 量化

```bash
python gpt_oss_20b_sft.py --use_4bit --batch_size 6
```

### 方案 3：多 GPU 训练

```bash
# 使用 2 张 GPU
deepspeed --num_gpus=2 gpt_oss_20b_sft.py --mode train
```

**注意**：这些方案会偏离原始 notebook 配置，可能影响效果。

## 常见命令

```bash
# 查看 GPU 状态
nvidia-smi

# 监控显存使用（训练时在另一个终端运行）
watch -n 1 nvidia-smi

# 查看所有可用参数
python gpt_oss_20b_sft.py --help

# 推送模型到 Hugging Face Hub
python gpt_oss_20b_sft.py --mode train --push_to_hub --hub_model_id your-username/model-name
```

## 预期输出

### 训练日志示例

```
================================================================================
步骤 1: 加载数据集
================================================================================
数据集加载完成！
训练样本数: 1000

================================================================================
步骤 2: 加载基础模型
================================================================================
加载模型: openai/gpt-oss-20b
模型加载完成！
模型参数量: 20.00B

================================================================================
步骤 3: 配置 LoRA
================================================================================
可训练参数统计:
  - 可训练参数: 16,777,216 (0.08%)
  - 总参数: 20,000,000,000

================================================================================
步骤 4: 开始训练
================================================================================
⚠️  峰值显存占用约 97GB，需要 H200 GPU 或更高配置
⚠️  单卡 80GB GPU（如 H100/A100）可能会出现 OOM 错误
--------------------------------------------------------------------------------
Training... 
Step 10/300 | Loss: 2.45 | LR: 1.8e-05
...
```

### 推理输出示例

```
================================================================================
示例 1: 西班牙语问题 + 德语推理
================================================================================
推理语言: German
用户提问: ¿Cuál es el capital de Australia?
--------------------------------------------------------------------------------
Assistant reasoning:
  Okay, der Benutzer fragt nach der Hauptstadt Australiens...
Assistant response:
  La capital de Australia es **Canberra**...
```

## 故障排除

### 问题 1：OOM 错误

```
RuntimeError: CUDA out of memory
```

**解决方案**：
- 使用 H200 GPU
- 或减小 batch_size 和 max_seq_length
- 或使用多 GPU 训练

### 问题 2：模型下载慢

```
Downloading model...
```

**解决方案**：
- 设置 Hugging Face 镜像（如果在中国）
- 或预先下载模型到本地

### 问题 3：数据集加载失败

```
ConnectionError: Failed to download dataset
```

**解决方案**：
- 检查网络连接
- 检查 Hugging Face Token 是否有效
- 使用代理：`export HF_ENDPOINT=https://hf-mirror.com`

## 下一步

- 查看 [README.md](README.md) 了解详细文档
- 查看 [PARAMETERS.md](PARAMETERS.md) 了解参数配置
- 尝试不同的推理语言组合
- 在你自己的数据集上微调

## 获取帮助

- 查看 [常见问题](README.md#常见问题)
- 查看 [性能优化建议](README.md#性能优化建议)
- 查看原始 [OpenAI Cookbook](https://github.com/openai/openai-cookbook/blob/main/articles/gpt-oss/fine-tune-transfomers.ipynb)

