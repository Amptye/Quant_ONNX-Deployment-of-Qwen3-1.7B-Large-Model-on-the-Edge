import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import os

model_path = "qwen3_int8.onnx"
tokenizer_path = "./Qwen3-1.7B"

sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

# ================= TODO 6: 实现自回归生成循环 =================
def generate(prompt, max_tokens=50):
    # 1. 预处理 Prompt
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(text, return_tensors="np")
    input_ids = inputs["input_ids"].astype(np.int64)
    
    print(f"Qwen: ", end="", flush=True)
    
    for _ in range(max_tokens):
        # [YOUR CODE HERE]
        # 1. 构造推理输入字典 ort_inputs
        
        # 2. 执行推理 sess.run
        # outputs = ...
        
        # 3. 获取下一个 token 的 ID (提示：取 logits 的最后一个位置，做 argmax)
        # next_token = ...
        
        # 4. 结束条件判断 (EOS token)
        # if next_token == tokenizer.eos_token_id: break
        
        # 5. 打印当前生成的字
        word = tokenizer.decode([next_token])
        print(word, end="", flush=True)
        
        # 6. 更新 input_ids (将新 token 拼接到末尾)
        # input_ids = np.append(...)
        pass # 删除此行
        
    print("\n")

while True:
    q = input("\nUser: ")
    if q == "exit": break
    generate(q)