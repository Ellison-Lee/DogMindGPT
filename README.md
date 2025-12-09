# NLP-Final: Qwen3-1.7B LoRA 微调项目

本项目基于 Qwen3-1.7B 模型，使用 LoRA（Low-Rank Adaptation）技术在进行微调，并实现了模型量化、性能评估和语音交互等功能，用以部署至Unitree 机器狗上，通过人类语音控制其动作。

## 📋 项目概述

本项目实现了完整的 LLM 微调流程：
1. **数据集生成**：使用 SiliconFlow API 生成机器人控制指令训练数据
2. **模型下载**：下载基础模型
3. **LoRA 微调**：在微调数据集上进行 LoRA 微调
4. **模型合并**：将 LoRA 权重合并到基础模型
5. **模型量化**：使用 4-bit 量化减少模型大小和推理延迟
6. **性能评估**：对比量化前后的准确率和性能指标
7. **语音交互**：支持语音输入进行对话

## 🗂️ 项目结构

```
NLP-Final/
├── 0_download_model.py          # 通过huggingface下载 Qwen3 基础模型
├── 0_generate_dataset.py         # 使用 SiliconFlow API 生成训练数据集
├── 1_qwen3-1.7b-lora-financial-training.ipynb  # LoRA 微调 Notebook
├── 2_merge_lora.py               # 合并 LoRA 权重到基础模型
├── 3_quantize_and_chat.py        # 量化模型并实现交互式对话
├── compare_accuracy.py           # 对比量化前后模型准确率
├── compare_performance.py        # 性能指标对比（延迟、吞吐量等）
├── data/                         # 数据集目录
│   └── generated_dataset.json    # 生成的训练数据集
├── models/                       # 模型目录
│   ├── Qwen3-1.7B/              # 基础模型
│   ├── lora-qwen3-1.7B/         # LoRA 适配器
│   ├── merged-qwen3-1.7B/       # 合并后的模型
│   └── merged-qwen3-1.7B-gguf/  # GGUF 量化模型
├── output/                       # 输出结果目录
│   ├── performance_results.json  # 性能测试结果
│   ├── latency_curves.png       # 延迟曲线图
│   └── throughput.png           # 吞吐量图
├── ollama_test/                  # Ollama 测试相关
│   ├── dog_prompt.txt           # 系统提示词
│   ├── Modelfile                # Ollama 模型配置文件
│   ├── ollama_chat.py           # Ollama 聊天脚本
│   └── local_chat.py            # 本地模型聊天脚本
└── voice_input/                  # 语音输入功能
    ├── get_voice.py             # 语音识别
    └── voice_chat.py            # 语音对话
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- CUDA（可选，用于 GPU 加速）
- Ollama（用于模型下载和管理）

### 安装依赖

```bash
pip install torch transformers peft accelerate bitsandbytes
pip install requests loguru matplotlib seaborn numpy
```

### 使用步骤

#### 1. 下载基础模型

```bash
python 0_download_model.py
```
该脚本会下载 Qwen3 1.7b 模型。

#### 2. 生成训练数据集

```bash
export SILICONFLOW_API_KEY='your-api-key-here'
python 0_generate_dataset.py
```

该脚本会使用 SiliconFlow API 生成机器人控制指令的训练数据集，保存到 `data/generated_dataset.json`。

#### 3. LoRA 微调

打开 `1_qwen3-1.7b-lora-financial-training.ipynb` Notebook，按照步骤进行：
- 加载基础模型
- 配置 LoRA 参数
- 训练 LoRA 适配器
- 保存训练结果

#### 4. 合并 LoRA 权重

```bash
python 2_merge_lora.py
```

将训练好的 LoRA 适配器合并到基础模型中，生成完整模型。

#### 5. 量化并测试对话

```bash
python 3_quantize_and_chat.py
```

加载量化后的模型，进行交互式对话测试。

#### 6. 性能评估

**准确率对比：**
```bash
python compare_accuracy.py
```

**性能指标对比：**
```bash
python compare_performance.py
```


## 注意事项

1. **内存要求**: 全精度模型需要较大内存，建议使用量化版本
2. **CUDA 支持**: 4-bit 量化需要 CUDA 支持，CPU 环境会自动回退到 float16
3. **API 限制**: SiliconFlow API 有调用频率限制，生成大量数据时注意控制速率
4. **模型路径**: 确保所有路径配置正确，否则会报错


