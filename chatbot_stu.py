import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import os
import config_stu

model_path = config_stu.MODEL_INT8
tokenizer_path = config_stu.QWEN3_PATH
# outputs_dir = "outputs"

sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
# 获取模型输入名称
input_names = [inp.name for inp in sess.get_inputs()]

# ================= TODO 6: 实现自回归生成循环 =================
def generate(prompt, max_tokens=50):
    # 1. 预处理 Prompt
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    fixed_length = config_stu.FIXED_SEQ_LEN
    if fixed_length:
        # 固定长度模式
        inputs = tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=fixed_length,
        )
    else:
        # 动态模式
        inputs = tokenizer(
            text,
            return_tensors="np",
            padding=True,  # 自动填充
            truncation=True,
        )
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.astype(np.int64)
    else:
        attention_mask = (input_ids != tokenizer.pad_token_id).astype(np.int64)
    token_num = int(attention_mask[0].sum())
    # if token_num <= 0:
    #     token_num = 1
    
    print(f"Qwen: ", end="", flush=True)
    
    for _ in range(max_tokens):
        # [YOUR CODE HERE]
        # 1. 构造推理输入字典 ort_inputs
        if attention_mask is None:
            attention_mask = np.ones_like(input_ids, dtype=np.int64)
        ort_inputs = {}
        for name in input_names:
            if "input_ids" in name:
                ort_inputs[name] = input_ids.astype(np.int64)
            elif "attention_mask" in name:
                ort_inputs[name] = attention_mask.astype(np.int64)
        
        # 2. 执行推理 sess.run
        # outputs = ...
        outputs = sess.run(None, ort_inputs)
        
        # 3. 获取下一个 token 的 ID (提示：取 logits 的最后一个位置，做 argmax)
        # next_token = ...
        # pos = max(token_num - 1, 0)
        logits = outputs[0]
        next_token = np.argmax(logits[0, token_num-1, :])
        
        # 4. 结束条件判断 (EOS token)
        # if next_token == tokenizer.eos_token_id: break
        if next_token == tokenizer.eos_token_id:
            break
        
        # 5. 打印当前生成的字
        word = tokenizer.decode([next_token], skip_special_tokens=True)
        print(word, end="", flush=True)
        
        # 6. 更新 input_ids (将新 token 拼接到末尾)
        # input_ids = np.append(...)
        # next_token_array = np.array([[next_token]], dtype=np.int64)
        # input_ids = np.concatenate([input_ids, next_token_array], axis=1)
        if fixed_length:
            if token_num < fixed_length:
                # 填充到空位
                input_ids[0, token_num] = next_token
                attention_mask[0, token_num] = 1
                token_num += 1
            else:
                # 滑动窗口
                input_ids = np.concatenate([
                    input_ids[:, 1:],  # 移除第一个
                    np.array([[next_token]], dtype=np.int64)  # 添加新的
                ], axis=1)
                assert input_ids.shape[1] == fixed_length
                attention_mask = np.ones_like(input_ids, dtype=np.int64)
                # token_num = fixed_length
        else:
            # next_token_array = np.array([[next_token]], dtype=np.int64)
            # input_ids = np.concatenate([input_ids, next_token_array], axis=1)
            # attention_mask = np.concatenate(
            #     [attention_mask, np.ones((attention_mask.shape[0], 1), dtype=np.int64)],
            #     axis=1,
            # )
            input_ids = np.append(input_ids, [[next_token]], axis=1)
            attention_mask = np.append(attention_mask, [[1]], axis=1)
            token_num += 1
        
    print("\n")

if __name__ == "__main__":
    while True:
        q = input("\nUser: ")
        if q == "exit": break
        generate(q)