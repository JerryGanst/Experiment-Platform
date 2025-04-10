import torch
from transformers import AutoTokenizer
import requests
from streaming_llm.modeling_streaming_llama import StreamingLlama
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def download_url(url, save_path):
    # 如果 save_path 是一个目录，抛出异常
    if os.path.isdir(save_path):
        raise IsADirectoryError(f"Provided save_path '{save_path}' is a directory. Please provide a file path.")

    # 确保保存路径的目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 下载文件
    response = requests.get(url, stream=True)
    response.raise_for_status()  # 检查请求是否成功
    with open(save_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

def load(model_name_or_path, use_streaming=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if use_streaming:
        # 使用自定义 StreamingLlama 模型结构
        model = StreamingLlama(
            num_layers=12,
            hidden_size=768,
            num_heads=12,
            intermediate_size=3072,
            vocab_size=tokenizer.vocab_size
        )
    else:
        # 使用标准 Hugging Face 模型
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    model.to("cpu")  # 如需 GPU 可改为 "cuda"
    return model, tokenizer

def load_jsonl(file_path):
    import json
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data
