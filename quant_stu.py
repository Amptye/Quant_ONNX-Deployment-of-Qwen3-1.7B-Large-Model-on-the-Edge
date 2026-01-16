import onnxruntime
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
from transformers import AutoTokenizer
import numpy as np
import os
import config_stu

# outputs_dir = "outputs"

# ================= TODO 4: 实现校准数据读取器 =================
class SmartCalibrationDataReader(CalibrationDataReader):
    def __init__(self, tokenizer, model_path):
        self.tokenizer = tokenizer
        self.fixed_length = config_stu.FIXED_SEQ_LEN
        # 自动获取模型输入名 (防止 input name mismatch)
        session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_names = [inp.name for inp in session.get_inputs()]
        
        self.data = iter(config_stu.CALIBRATION_DATA)

    def get_next(self):
        text = next(self.data, None)
        if text is None: return None
        
        # [YOUR CODE HERE] 
        # 1. 使用 tokenizer 处理 text，return_tensors="np"
        # 2. 将数据转换为 int64 类型
        # 3. 返回一个字典，键名必须与 self.input_names 匹配
        #    (提示：检查 input_ids 和 attention_mask 是否都在 input_names 里)
        if self.fixed_length:
            # 固定长度模式
            inputs = tokenizer(
                text,
                return_tensors="np",
                padding="max_length",   # 填充到相同长度
                max_length=self.fixed_length,   # 统一最大长度
                truncation=True,    # 截断过长的文本
            )
        else:
            # 动态模式
            inputs = tokenizer(
                text,
                return_tensors="np",
                padding=True,  # 自动填充
                truncation=True,
            )
        input_dict = {}
        if "input_ids" in self.input_names and "input_ids" in inputs:
            input_dict["input_ids"] = inputs["input_ids"].astype(np.int64)
        if "attention_mask" in self.input_names and "attention_mask" in inputs:
            input_dict["attention_mask"] = inputs["attention_mask"].astype(np.int64)

        if not input_dict:
            for name in self.input_names:
                if name in inputs:
                    input_dict[name] = inputs[name].astype(np.int64)

        return input_dict

# 主程序
model_fp32 = config_stu.MODEL_FP32
model_int8 = config_stu.MODEL_INT8

if not os.path.exists(model_fp32):
    print("未找到 FP32 模型，请先完成任务一。")
    exit(1)

tokenizer = AutoTokenizer.from_pretrained(config_stu.QWEN3_PATH, trust_remote_code=True)
dr = SmartCalibrationDataReader(tokenizer, model_fp32)

print("--- Starting Quantization ---")

# ================= TODO 5: 执行静态量化 =================
# 提示：由于模型大于 2GB，直接量化会报错 Protobuf parsing failed。
# 你需要设置哪个参数来启用外部数据存储？
quantize_static(
    model_input=model_fp32,
    model_output=model_int8,
    calibration_data_reader=dr,
    quant_format=onnxruntime.quantization.QuantFormat.QOperator,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul", "Gemm"],
    per_channel=True,
    reduce_range=True,
    extra_options={
        "MatMulConstBOnly": True,
        "ActivationSymmetric": False,
        "WeightSymmetric": True,
    },

    
    # [YOUR CODE HERE] 填入解决大模型存储限制的关键参数
    use_external_data_format=True
)

print(f"✅ Quantization Complete!")