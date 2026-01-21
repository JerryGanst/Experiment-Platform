#!/usr/bin/env python3
"""测试HACE pref_mode是否生效的小规模实验"""
import os
import sys
from pathlib import Path

# 设置路径
ROOT = Path(__file__).resolve().parents[1]
CAKE_ROOT = ROOT / "vendor" / "cake"
sys.path.insert(0, str(ROOT / "vendor"))

import torch
import json
from datasets import load_dataset
from cake.longbench.pred_cake import load_model_and_tokenizer, build_chat
from cake.compress_config import CompressConfig


def test_pref_mode(pref_mode_value):
    """测试特定pref_mode下的模型行为"""
    print(f"\n{'='*80}")
    print(f"测试 pref_mode = '{pref_mode_value}'")
    print(f"{'='*80}\n")

    # 设置环境变量
    os.environ["HACE_PREF_MODE"] = pref_mode_value
    print(f"✓ 设置环境变量: HACE_PREF_MODE={os.environ['HACE_PREF_MODE']}")

    # 配置
    model_name = "qwen2.5-7b-instruct"
    model_path = "/model/ModelScope/Qwen/Qwen2.5-7B-Instruct"
    cache_size = 128
    window_size = 32

    compress_config = CompressConfig(True, False)
    compress_config.cache_size = cache_size
    compress_config.window_size = window_size
    compress_config.hyper = [1.0, 1.0, 200.0]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"✓ 加载模型: {model_path}")
    model, tokenizer = load_model_and_tokenizer(model_path, model_name, device, compress_config)

    # 简单测试prompt
    prompt = "Summarize this meeting: The team discussed the new product launch."
    prompt = build_chat(tokenizer, prompt, model_name)

    print(f"✓ 生成测试...")
    print(f"  (如果pref_mode生效，应该看到 '[HACE] Using pref_mode: {pref_mode_value}' 消息)")
    print()

    inputs = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=20,  # 只生成20个token做测试
            num_beams=1,
            do_sample=False,
            temperature=1.0,
        )[0]

    result = tokenizer.decode(output[inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    print(f"\n生成结果: {result}")

    # 清理
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return result


if __name__ == "__main__":
    print("\n" + "="*80)
    print("HACE Pref Mode 功能验证测试")
    print("="*80)

    # 测试两种模式
    result1 = test_pref_mode("normal")
    result2 = test_pref_mode("reverse_disp")

    print(f"\n{'='*80}")
    print("对比结果")
    print(f"{'='*80}\n")

    print(f"normal 模式输出: {result1}")
    print(f"reverse_disp 模式输出: {result2}")

    if result1 == result2:
        print("\n⚠️  警告：两种模式的输出完全相同！")
        print("这可能意味着：")
        print("  1. cache_size=128 太大，两种策略都能保留足够信息")
        print("  2. 这个简单prompt太短，看不出差异")
        print("  3. pref_mode可能没有生效")
    else:
        print("\n✓ 两种模式的输出不同，pref_mode正常工作！")
