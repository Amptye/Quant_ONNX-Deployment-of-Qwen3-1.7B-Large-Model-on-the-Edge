import onnxruntime
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
from transformers import AutoTokenizer
import numpy as np
import os

# ================= TODO 4: 实现校准数据读取器 =================
class SmartCalibrationDataReader(CalibrationDataReader):
    def __init__(self, tokenizer, model_path):
        self.tokenizer = tokenizer
        # 自动获取模型输入名 (防止 input name mismatch)
        session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_names = [inp.name for inp in session.get_inputs()]
        
        self.data = iter([
            "人工智能是计算机科学的一个分支。",
            "Deep learning requires a lot of computing power.",
            "今天天气真不错。",
            "Python is popular."
        ])

    def get_next(self):
        text = next(self.data, None)
        if text is None: return None
        
        # [YOUR CODE HERE] 
        # 1. 使用 tokenizer 处理 text，return_tensors="np"
        # 2. 将数据转换为 int64 类型
        # 3. 返回一个字典，键名必须与 self.input_names 匹配
        #    (提示：检查 input_ids 和 attention_mask 是否都在 input_names 里)
        return {} 

# 主程序
model_fp32 = "qwen3_fp32.onnx"
model_int8 = "qwen3_int8.onnx"

if not os.path.exists(model_fp32):
    print("未找到 FP32 模型，请先完成任务一。")
    exit(1)

tokenizer = AutoTokenizer.from_pretrained("./Qwen3-1.7B", trust_remote_code=True)
dr = SmartCalibrationDataReader(tokenizer, model_fp32)

print("--- Starting Quantization ---")

# ================= TODO 5: 执行静态量化 =================
# 提示：由于模型大于 2GB，直接量化会报错 Protobuf parsing failed。
# 你需要设置哪个参数来启用外部数据存储？
quantize_static(
    model_input=model_fp32,
    model_output=model_int8,
    calibration_data_reader=dr,
    quant_format=onnxruntime.quantization.QuantFormat.QDQ,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    
    # [YOUR CODE HERE] 填入解决大模型存储限制的关键参数
    
)

print(f"✅ Quantization Complete!")