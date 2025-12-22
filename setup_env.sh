#!/bin/bash
# 1. 定义环境名称
ENV_NAME="qwen3_onnx_cpu"

echo "--- 正在开始创建 Conda 环境: $ENV_NAME ---"

# 2. 创建环境并指定 Python 版本
# -y 表示自动确认
conda create -n $ENV_NAME python=3.10 -y

# 3. 激活环境 (在脚本中使用 source 激活)
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "--- 环境已激活，正在安装依赖 (CPU 优化版) ---"

# 4. 设置国内镜像源 
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 5. 安装 PyTorch CPU 版本 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 6. 安装模型导出、量化及推理所需的库
pip install "optimum[onnxruntime]" transformers datasets accelerate onnxscript

echo "--- 正在验证安装 ---"
python -c "import onnxruntime; import optimum; print('ONNX Runtime 版本:', onnxruntime.__version__)"

echo "------------------------------------------------"
echo "环境配置完成！"
echo "请运行以下命令进入环境："
echo "conda activate $ENV_NAME"
echo "------------------------------------------------"