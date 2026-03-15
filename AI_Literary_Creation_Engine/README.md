# AI Literary Creation Engine: Solving Emotional Reasoning Hallucination with RAG and Factual Enhancement
# AI 文学创作引擎：基于 RAG 和事实增强解码解决情感推理幻觉

## 📖 Introduction / 项目简介
This project implements a novel architecture designed to solve **Emotional Reasoning Hallucinations** in AI literary creation. By combining **Specific-Domain RAG (Retrieval-Augmented Generation)** with **Factual Enhancement Decoding**, it enforces emotional consistency and logic during the generation process.
本项目实现了一种用于解决 AI 文学创作中 **情感推理幻觉** 的创新架构。通过结合 **特定领域 RAG（检索增强生成）** 与 **事实增强解码（Factual Enhancement Decoding）**，在推理阶段强制约束生成内容的情感一致性与逻辑性。

**[New] Google Colab User?** 
If you prefer running the training on Google Colab, simply upload `train_colab.ipynb` to your Google Drive and follow the instructions inside.
如果你习惯使用 Google Colab，只需将项目中的 `train_colab.ipynb` 上传至 Google Drive 并按照指引运行即可。

---

## 🚀 Feasibility Verification / 可行性验证 (Core Metrics)
To verify the effectiveness of this framework for your paper, we rely on the following quantitative metrics calculated by `src/evaluate.py`:
为了验证本框架在论文中的有效性，我们依赖 `src/evaluate.py` 计算以下核心量化指标：

1.  **ECFR (Emotional Correctness Fit Rate / 情绪正确性符合率)**: 
    *   The percentage of generated texts where the dominant emotion matches the target sentiment.
    *   生成文本的主导情感与目标情感一致的比例。
    *   *Goal / 目标*: > 85% (Baseline usually < 70%)

2.  **EER (Extreme Emotion Rate / 极端情绪检出率)**:
    *   The rate of hallucinated content that expresses widely inappropriate or pathologically intense emotions.
    *   模型生成出极度夸张、病态或完全相反情感（幻觉）的频率。
    *   *Goal / 目标*: < 5%

3.  **Experimental Method / 实验方法**:
    *   Run `src/main.py` to generate 100 poems with specific sentiments (e.g., "sadness").
    *   Run `src/evaluate.py` to automatically score them against a standard BERT classifier.
    *   运行 `src/main.py` 生成 100 首指定情感（如“悲伤”）的诗歌。
    *   运行 `src/evaluate.py` 使用标准 BERT 分类器进行自动打分。

---

## 🛠️ Model Selection / 模型选择
We have selected the **Meta-Llama-3-8B-Instruct** as the base model.
我们选择了 **Meta-Llama-3-8B-Instruct** 作为基座模型。

*   **Reason 1 (State-of-the-Art)**: It is currently the most capable 8B model, outperforming many larger models in literary nuance and instruction following.
    *   它是目前最强的 8B 模型，在文学细腻度和指令跟随能力上超越许多更大参数的模型。
*   **Reason 2 (Context Window)**: Supports 8K context, perfect for RAG retrieval of literary fragments.
    *   支持 8K 上下文，非常适合 RAG 检索大量文学片段。
*   **Reason 3 (Hardware Friendly)**: Can be fine-tuned or run efficiently on a single consumer GPU (RTX 4090) or cloud GPU (A100).
    *   可以在单张消费级显卡（RTX 4090）或云端显卡（A100）上高效微调或推理。

**Auxiliary Models / 辅助模型**:
*   **Retrieval**: `BAAI/bge-m3` (High precision semantic embedding / 高精度语义向量)
*   **Evaluation**: `bhadresh-savani/bert-base-uncased-emotion` (Unbiased judge / 公正裁判)

---

## 💾 Data Selection Guidelines / 数据选取注意事项
The quality of your Knowledge Base determines the upper limit of the RAG system.
知识库的质量决定了 RAG 系统的上限。

1.  **Corpus Source / 语料来源**: 
    *   Select classical English literature (19th-20th century) known for distinct emotional styles (e.g., *Wuthering Heights* for gloom/passion, *Pride and Prejudice* for wit/restraint).
    *   选取具有鲜明情感风格的经典英文著作（如《呼啸山庄》用于压抑/激情，《傲慢与偏见》用于机智/克制）。
2.  **Text Cleaning / 文本清洗**:
    *   Use `src/data_processing.py`. It automatically removes headers, footers, and chaotic formatting.
    *   使用 `src/data_processing.py`。它会自动去除页眉、页脚和混乱格式。
3.  **Annotation Logic / 标注逻辑**:
    *   **Emotional Intensity (1-5)**: Ensure distinct separation. "Sorrow" (3) vs "Despair" (5).
    *   **Imagery Extraction**: The script extracts nouns automatically. Validate that words like "crow" align with "sadness" in your data.
    *   **情感强度 (1-5)**：确保区分度。“悲伤”(3) vs “绝望”(5)。
    *   **意象提取**：脚本会自动提取名词。需人工抽检确认“乌鸦”等词是否在你的数据中确实对应“悲伤”。

---

## ⚙️ Fine-tuning Methodology / 微调方法 (Optional / 可选)
While the `LogitsProcessor` (Inference-time intervention) works out of the box, **LoRA Fine-tuning** helps the model better understand the nuances of the specific literary style you are targeting.
虽然 `LogitsProcessor`（推理时干预）开箱即用，但 **LoRA 微调** 能帮助模型更好地理解特定文学风格的细微差别。

### 1. Prepare Training Data / 准备训练数据
Your dataset must be in **JSONL format**. Each line should be a JSON object containing `instruction`, `input`, and `output` fields.
训练数据必须是 **JSONL 格式**。每行一个 JSON 对象，包含 `instruction`（指令）、`input`（输入上下文）和 `output`（期望输出）。

**Example (`dataset/fine_tuning_sample.jsonl`)**:
```json
{"instruction": "Write a short poem about despair.", "input": "Keywords: darkness, void", "output": "In the silent void where shadows creep,\nMy weary soul forgets to weep..."}
{"instruction": "Continue this story with a joyful tone.", "input": "The sun rose above the hills.", "output": "Birds sang a melody of pure delight,\nChasing away the remnants of night."}
```

### 2. Run Training Script / 运行训练脚本
We provide a dedicated training script `src/train.py` that utilizes **QLoRA** (4-bit quantization + LoRA) for memory-efficient fine-tuning.
我们提供了 `src/train.py` 脚本，使用 **QLoRA**（4-bit 量化 + LoRA）进行显存高效的微调。

**Command / 命令**:
```bash
# Make sure to activate the conda environment first
conda activate ai_writing_v2

# Run training
python src/train.py \
    --model_name "meta-llama/Meta-Llama-3-8B-Instruct" \
    --dataset_path "dataset/fine_tuning_sample.jsonl" \
    --output_dir "./checkpoints/adapter_v1" \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4
```

### 3. Merge Weights (Optional) / 合并权重（可选）
After training, you will get LoRA adapters in `output_dir`. You can load them dynamically or merge them back to the base model.
训练完成后，你会在 `output_dir` 得到 LoRA 适配器权重。推理时可以动态加载，也可以将其合并回基座模型。

---

## 💻 GPU Memory Requirement / 显存需求
*   **Inference (Testing) / 推理测试**: ~16GB VRAM (4-bit loading)
*   **Fine-tuning (Training) /微调训练**: ~24GB VRAM (QLoRA, batch_size=4, seq_len=2048)
    *   *Compatible with RTX 3090 / 4090 / A10 / A100.*


---

## 🧪 Quick Start / 快速开始

### 1. Environment / 环境搭建

**Option A: Conda (Recommended for GPU) / 方案 A：Conda（推荐，支持 GPU）**

```bash
# 1. Create environment / 创建环境
conda create -n ai_writing_v2 python=3.10 -y
conda activate ai_writing_v2

# 2. Install PyTorch & Faiss-GPU / 安装 PyTorch 和 Faiss-GPU
# (Note: These files are large, download may take time / 注意：文件较大，下载可能较慢)
conda install -y -c pytorch -c nvidia -c conda-forge pytorch torchvision torchaudio pytorch-cuda=12.1 faiss-gpu

# 3. Install remaining dependencies / 安装剩余依赖
pip install -r requirements_conda.txt
python -m spacy download en_core_web_sm
```

**Option B: Pip (CPU Only / Fallback) / 方案 B：Pip（仅 CPU / 备用）**

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Data Processing / 数据处理
Place your raw `.txt` books in `dataset/`.
将原始 `.txt` 书籍放入 `dataset/`。
```python
# Run via python shell or script
from src.data_processing import DatasetAnnotator
annotator = DatasetAnnotator()
annotator.process_file("dataset/wuthering_heights.txt", "processed_data/output.jsonl")
```

### 3. Run RAG Generation / 运行生成
```bash
# Generate a poem about 'moonlight' with 'sadness' tone
python src/main.py --prompt "The moonlight reflects on the lake" --target_sentiment "sadness"
```

### 4. Evaluate / 运行评测
```bash
python src/evaluate.py
```

---

## 💡 Expert Tips for Your Paper / 论文专家建议

*   **Terminology / 术语包装**: 
    *   Don't just say "we changed the code". Say "we implemented a **Dynamic Sentiment-Constraint Decoding Strategy**".
    *   不要只说“修改了代码”。要说“实现了 **动态情感约束解码策略**”。
*   **A/B Testing / 对比实验**: 
    *   In your paper, you MUST compare "Base Model" vs "Ours". The huge gap in ECFR score is your key contribution.
    *   在论文中，你必须展示“基座模型”与“我们的方法”的对比。ECFR 分数的巨大差距是你的核心贡献。
*   **Failure Analysis / 失败分析**:
    *   Include a section on *when* your model fails (e.g., with very abstract concepts). Honesty increases academic credibility.
    *   包含一个章节讨论你的模型 *何时* 会失败（例如面对极其抽象的概念时）。诚实能增加学术可信度。

---

### Project Structure / 项目结构
```
AI_Literary_Creation_Engine/
├── dataset/                # Raw text files (Put your books here)
├── knowledge_base/         # FAISS Vector Index (Generated automatically)
├── processed_data/         # JSONL files with annotations
├── src/
│   ├── config.py           # Configuration (Paths, Model names)
│   ├── data_processing.py  # Cleaning & Annotation Script
│   ├── decoding.py         # THE INNOVATION: Logits Processor
│   ├── evaluate.py         # ECFR/EER Metrics Script
│   ├── main.py             # Entry point
│   ├── model_loader.py     # LoRA/PEFT Loader
│   ├── rag_engine.py       # FAISS/BGE-M3 Retrieval
│   ├── prepare_training_data.py # Training Data Formatter
│   └── train.py            # LoRA Fine-Tuning Script
└── requirements.txt        # Dependencies
```

## 🔥 Fine-Tuning Instructions / 模型微调指南

If you want to train the model on your custom literary dataset to improve its style and instruction following capabilities, follow these steps.
如果你希望在自定义文学数据集上训练模型以提升其风格和指令跟随能力，请按照以下步骤操作。

### 1. Prepare Environment / 准备环境
Install the necessary dependencies for fine-tuning (in addition to the base requirements):
安装微调所需的依赖项（除了基础依赖外）：
```bash
pip install trl datasets bitsandbytes scipy
```

### 2. Prepare Data / 准备数据
Ensure you have run the annotation process first. Then, convert the annotated data into a training-ready format:
确保你已经完成了数据标注。然后，将标注数据转换为训练格式：
```bash
python src/prepare_training_data.py
```
This will create `processed_data/training/fine_tuning_data.jsonl`.
这将生成 `processed_data/training/fine_tuning_data.jsonl` 文件。

### 3. Run Training / 运行训练
Use the `src/train.py` script to fine-tune the model. You can run this on a local GPU or a rented server (e.g., AutoDL, vast.ai).
使用 `src/train.py` 脚本微调模型。你可以在本地 GPU 或租赁的服务器（如 AutoDL, vast.ai）上运行。

**Example Command (Single GPU) / 单卡训练示例:**
```bash
python src/train.py \
    --model_name "meta-llama/Meta-Llama-3-8B-Instruct" \
    --dataset_path "processed_data/training/fine_tuning_data.jsonl" \
    --output_dir "./fine_tuned_model" \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4
```

**Key Arguments / 关键参数:**
*   `--model_name`: The HF model ID or local path (default: `meta-llama/Meta-Llama-3-8B-Instruct`).
*   `--dataset_path`: Path to your formatted JSONL data.
*   `--output_dir`: Where to save the LoRA adapters.
*   `--epochs`: Number of training passes.
*   `--batch_size`: Batch size per device (Adjust based on VRAM. 4 fits in ~12GB VRAM with 4-bit quantization).

### 4. Merge & Export / 合并与导出 (Optional)
After training, the `output_dir` will contain LoRA adapters. To use them, you can load them on top of the base model using `PeftModel` (as shown in `src/model_loader.py`).
训练后，`output_dir` 将包含 LoRA 适配器。要使用它们，可以通过 `src/model_loader.py` 中的 `PeftModel` 加载到基座模型之上。
