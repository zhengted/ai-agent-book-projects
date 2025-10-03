# 输出格式说明

本文档说明改进后的输出格式，使用明显的标记来区分对话中的不同角色。

## 为什么需要格式化输出？

在多语言推理任务中，对话包含多个部分：
1. **系统提示**（System）：指定推理语言
2. **用户问题**（User）：用户的提问
3. **助手推理**（Assistant Reasoning）：模型的思考过程
4. **助手响应**（Assistant Response）：最终答案

使用明显的标记（如 `<SYSTEM>`, `<USER>`, `<ASSISTANT>`）可以：
- ✅ 清晰地区分不同角色
- ✅ 便于调试和分析
- ✅ 更好地理解模型的推理过程
- ✅ 方便后续的文本处理

## 输出格式示例

### 示例 1：基本格式

```
================================================================================
<SYSTEM>
reasoning language: German

<USER>
¿Cuál es el capital de Australia?

<ASSISTANT>
Okay, der Benutzer fragt nach der Hauptstadt Australiens. Ich erinnere mich, 
dass Canberra die Hauptstadt ist. Sydney ist die größte Stadt, aber die 
Hauptstadt ist Canberra...

La capital de Australia es **Canberra**. Aunque es la ciudad más pequeña de 
las principales capitales del país, fue elegida en 1908 como la sede del 
gobierno federal para equilibrar la influencia entre las ciudades de Sydney 
y Melbourne.
================================================================================
```

### 示例 2：中文推理

```
================================================================================
<SYSTEM>
reasoning language: Chinese

<USER>
What is the national symbol of Canada?

<ASSISTANT>
好的，用户问加拿大的国家象征是什么。我知道加拿大的国家象征是枫叶。
枫叶图案出现在加拿大国旗上，也是加拿大文化和身份的重要标志...

The national symbol of Canada is the **maple leaf**. It appears prominently on 
the Canadian flag and represents Canadian identity, unity, and natural heritage.
================================================================================
```

### 示例 3：数学推理（中文）

```
================================================================================
<SYSTEM>
reasoning language: Chinese

<USER>
求解 x^2 - 2x + 1 = 0 的根

<ASSISTANT>
好的，用户想要求解方程 x^2 - 2x + 1 = 0 的根。先检查一下方程是否可以因式分解。
x^2 - 2x + 1 看起来像是 (x - 1)^2，因为 (x - 1)^2 = x^2 - 2x + 1。
是的，方程可以写成 (x - 1)^2 = 0。那根就很容易找到，x - 1 = 0，所以 x = 1。
这个根是重复根，或者说重根，意味着它只出现一次，但在方程中出现两次。

方程 x^2 - 2x + 1 = 0 的根是 x = 1（重根）。

因式分解：(x - 1)^2 = 0
所以：x - 1 = 0
解得：x = 1
================================================================================
```

## 代码实现

### 1. 主训练和推理脚本（gpt_oss_20b_sft.py）

```python
def generate_response(model, tokenizer, reasoning_language, user_prompt, 
                     max_new_tokens=512, temperature=0.6, format_output=True):
    # ... 生成逻辑 ...
    
    # 解码输出 - 同时保留两个版本
    response_with_tokens = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]
    response_clean = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    
    if format_output:
        # 格式化输出，使用明显的标记
        formatted_response = format_response_output(response_clean, system_prompt, user_prompt)
        return formatted_response
    else:
        return response_clean
```

### 2. 格式化函数

```python
def format_response_output(response, system_prompt, user_prompt):
    formatted = []
    formatted.append("=" * 80)
    formatted.append("<SYSTEM>")
    formatted.append(system_prompt)
    formatted.append("")
    formatted.append("<USER>")
    formatted.append(user_prompt)
    formatted.append("")
    formatted.append("<ASSISTANT>")
    formatted.append(response.strip())
    formatted.append("=" * 80)
    
    return "\n".join(formatted)
```

### 3. 快速开始脚本（quickstart.py）

```python
def chat(model, tokenizer, reasoning_language, user_message, format_output=True):
    # ... 生成逻辑 ...
    
    if format_output:
        # 格式化输出
        formatted = []
        formatted.append("\n" + "=" * 80)
        formatted.append("<SYSTEM>")
        formatted.append(f"  {system_prompt}")
        formatted.append("")
        formatted.append("<USER>")
        formatted.append(f"  {user_message}")
        formatted.append("")
        formatted.append("<ASSISTANT>")
        formatted.append(f"  {response_clean.strip()}")
        formatted.append("=" * 80)
        return "\n".join(formatted)
    else:
        return response_clean
```

## 使用方法

### 启用格式化输出（默认）

```python
response = generate_response(
    model, tokenizer,
    reasoning_language="Chinese",
    user_prompt="你的问题",
    format_output=True,  # 默认为 True
)
print(response)
```

### 禁用格式化输出（原始格式）

```python
response = generate_response(
    model, tokenizer,
    reasoning_language="Chinese",
    user_prompt="你的问题",
    format_output=False,  # 返回原始文本
)
print(response)
```

## 对比：改进前后

### 改进前

```
Okay, der Benutzer fragt nach der Hauptstadt Australiens... La capital de Australia es **Canberra**...
```

难以区分哪部分是推理，哪部分是最终答案。

### 改进后

```
================================================================================
<SYSTEM>
reasoning language: German

<USER>
¿Cuál es el capital de Australia?

<ASSISTANT>
Okay, der Benutzer fragt nach der Hauptstadt Australiens...
La capital de Australia es **Canberra**...
================================================================================
```

清晰地显示了系统提示、用户问题和助手响应的边界。

## 特殊标记说明

| 标记 | 含义 | 说明 |
|------|------|------|
| `<SYSTEM>` | 系统提示 | 指定推理语言等配置 |
| `<USER>` | 用户输入 | 用户的问题或指令 |
| `<ASSISTANT>` | 助手输出 | 模型的完整响应（包含推理+答案） |
| `<ASSISTANT_REASONING>` | 助手推理 | 仅推理部分（如果能识别） |
| `<ASSISTANT_RESPONSE>` | 助手响应 | 最终答案（如果能识别） |

## 高级用法

### 自定义格式化

如果需要自定义输出格式，可以修改 `format_response_output` 函数：

```python
def format_response_output(response, system_prompt, user_prompt):
    # 自定义格式
    formatted = []
    formatted.append("╔" + "═" * 78 + "╗")
    formatted.append("║ SYSTEM: " + system_prompt.ljust(68) + "║")
    formatted.append("╠" + "═" * 78 + "╣")
    formatted.append("║ USER: " + user_prompt[:68].ljust(70) + "║")
    formatted.append("╠" + "═" * 78 + "╣")
    formatted.append("║ ASSISTANT:")
    for line in response.strip().split('\n'):
        formatted.append("║   " + line[:74].ljust(74) + "║")
    formatted.append("╚" + "═" * 78 + "╝")
    
    return "\n".join(formatted)
```

### 解析特殊标记

如果模型输出包含特殊的分隔符（如 "reasoning:" 或 "response:"），可以进一步解析：

```python
if "reasoning:" in response.lower():
    parts = response.split("reasoning:", 1)
    reasoning_part = parts[1].split("response:", 1)[0] if len(parts) > 1 else ""
    response_part = parts[1].split("response:", 1)[1] if "response:" in parts[1] else ""
    
    formatted.append("<ASSISTANT_REASONING>")
    formatted.append(reasoning_part.strip())
    formatted.append("")
    formatted.append("<ASSISTANT_RESPONSE>")
    formatted.append(response_part.strip())
```

## 调试技巧

### 查看原始 tokens

```python
# 查看带 special tokens 的版本
response_with_tokens = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]
print("带 tokens:", response_with_tokens)

# 查看不带 special tokens 的版本
response_clean = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
print("不带 tokens:", response_clean)
```

### 保存到文件

```python
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(response)
```

## 总结

新的格式化输出功能：
- ✅ 使用 `<SYSTEM>`, `<USER>`, `<ASSISTANT>` 等明显标记
- ✅ 清晰区分对话中的不同角色
- ✅ 便于调试和分析推理过程
- ✅ 可选启用/禁用（默认启用）
- ✅ 支持自定义格式化

这使得多语言推理模型的输出更易读、更易分析！

