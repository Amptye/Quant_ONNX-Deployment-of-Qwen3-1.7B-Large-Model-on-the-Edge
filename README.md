# 实验手册：Qwen3-1.7B 大模型端侧部署实战
**课程名称**：大模型系统与应用  
**实验环境**：Linux Server (CPU Only) / Python 3.10  
**建议时长**：2-3 小时
---

## 参考文档：
1. ONNX:https://onnx.ai/onnx/#
2. ONNX Runtime: https://onnxruntime.ai/

## 1. 实验背景与目的
随着大语言模型（LLM）的发展，将模型部署在手机、PC 等端侧设备上（On-Device AI）成为热门趋势。本实验旨在让学生掌握 LLM 从 PyTorch 训练框架迁移到 ONNX 通用推理框架的完整流程。
**实验核心目标**：
1.  **环境搭建**：掌握无 GPU 环境下的大模型开发环境配置。
2.  **模型导出 (Export)**：理解 PyTorch 动态图与 ONNX 静态图的区别，掌握通过“算子补丁 (Monkey Patching)”和“模型包装 (Wrapper)”解决导出兼容性问题的方法。
3.  **模型量化 (Quantization)**：实现 INT8 静态量化，理解量化对模型体积和推理速度的影响。
4.  **推理实战**：编写 Python 脚本，加载量化后的模型实现一个简单的对话机器人。
---

## 2. 实验准备

### 2.1 硬件要求
* **CPU**: 建议 4 核以上 (Intel/AMD，指令集支持 AVX2/AVX512 体验更佳)
* **RAM**: 建议 16GB 以上 (FP32 模型导出过程瞬间内存占用较大)
* **Disk**: 至少 20GB 可用空间

### 2.2 软件依赖安装
请在终端中执行以下命令，创建 Conda 环境并安装 CPU 版本的 PyTorch 及相关工具。

```bash
# 1. 创建环境
conda create -n qwen_onnx python=3.10 -y
conda activate qwen_onnx

# 2. 安装 PyTorch (CPU版，减小下载体积)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)

# 3. 安装 HuggingFace 生态与 ONNX 工具链
# onnxscript 是 PyTorch 2.x 导出器的新依赖
pip install transformers accelerate onnx onnxruntime onnxscript

# 4. (可选) 设置 HuggingFace 国内镜像，加速模型下载
export HF_ENDPOINT=[https://hf-mirror.com](https://hf-mirror.com)
pip install -U huggingface_hub
```

### 2.3 下载模型权重
使用命令行工具将 Qwen3-1.7B 模型下载到本地 Qwen3-1.7B 文件夹，避免实验中途网络中断。
```bash
huggingface-cli download --resume-download Qwen/Qwen3-1.7B --local-dir Qwen3-1.7B
```

## 3. 实验步骤

### 3.1 步骤一：编写导出脚本 (Export)注意：由于 Qwen3 使用了较新的注意力机制实现（如 vmap）,需要特殊处理机制
运行以下脚本会产生导出的模型文件
```bash
python export_stdu.py
```
### 3.2 INT8 静态量化 (Quantization)
```bash
python quant_stu.py
```
为了在 CPU 上流畅运行，我们需要将 FP32 模型量化为 INT8。这需要准备校准数据集，让模型“预演”推理过程以确定量化参数。

### 3.3：推理对话 Demo (Inference)
任务：创建文件 chat.py, 编写一个简单的对话脚本，验证量化后的模型是否能正常“说人话”。
```bash
python chatbot_stu.py
```

## 实验观察
文件体积对比： 记录导出模型和量化后的具体文件大小，并计算压缩比（例如：2.76GB / 7.56GB ≈ 36.5%）。
效果观察：
1. 量化后的模型在 CPU 上生成文字的速度如何？（主观感受即可，如：每秒大概几个字？）  
   **结论：速度大约是每秒 10 个字**

2. 量化后的模型是否依然能生成通顺的中文？请在报告中附上 2-3 轮对话的截图。  
   **结论：能够生成通顺的中文，这是对话的截图：**![](chat_figures/chat_length_32.png)

## 思考题：
在任务三中，目前的推理逻辑每生成一个字都要重新计算整个序列。如果想要加速，应该引入什么机制？（提示：KV Cache）  
**答：** 要加速文本生成过程，应该引入KV Cache（键值缓存）机制。在Transformer的计算中，每个token都会生成对应的Key和Value，在传统的生成方式中，每次生成新token时都需要重新计算整个序列，导致大量重复计算。而KV Cache通过缓存历史token的Key和Value，每次只需要计算新token的内容，再并把新产生的 Key、Value 放入缓存，显著提高了生成速度，特别在长对话时可以带来较大的的性能提升。

# 提交方式：
填写登记表：https://table.nju.edu.cn/dtable/forms/bc0d4e0e-3ec8-452a-8c99-9999cf70d932/
