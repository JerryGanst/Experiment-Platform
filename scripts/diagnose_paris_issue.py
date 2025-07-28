#!/usr/bin/env python3
"""
诊断"Paris"问题的脚本

这个脚本会逐步测试各个组件，找出问题所在
"""

import os
import sys
import torch
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))


def get_model_path():
    """动态获取模型路径"""
    # 1. 先检查命令行参数
    if len(sys.argv) > 1:
        return sys.argv[1]
    
    # 2. 检查环境变量
    if "MODEL_PATH" in os.environ:
        return os.environ["MODEL_PATH"]
    
    # 3. 检查test_corecode_hotpotqa_fixed.py中的默认路径
    try:
        test_script = Path(__file__).parent.parent / "test_corecode_hotpotqa_fixed.py"
        if test_script.exists():
            with open(test_script, 'r', encoding='utf-8') as f:
                content = f.read()
                import re
                match = re.search(r'def __init__\(self, model_name="([^"]+)"', content)
                if match:
                    return match.group(1)
    except:
        pass
    
    # 4. 默认值
    return "mistralai/Mistral-7B-Instruct-v0.3"


def test_basic_generation():
    """测试基本的模型生成功能"""
    print("="*60)
    print("1. 测试基本生成功能（不使用CAKE和CoreCode）")
    print("="*60)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # 动态获取模型路径
        model_path = get_model_path()
        
        print(f"加载模型: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 测试简单生成
        test_prompts = [
            "What is the capital of France?",
            "What is 2 + 2?",
            "The sun is a",
        ]
        
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    temperature=1.0
                )
            
            generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            print(f"\n输入: {prompt}")
            print(f"输出: {generated}")
        
        return True
        
    except Exception as e:
        print(f"❌ 基本生成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_instruction_template():
    """测试使用指令模板的生成"""
    print("\n" + "="*60)
    print("2. 测试Mistral指令模板")
    print("="*60)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_path = get_model_path()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 测试不同的模板格式
        templates = [
            # 格式1：简单格式
            lambda q: f"[INST] {q} [/INST]",
            # 格式2：带<s>标签
            lambda q: f"<s>[INST] {q} [/INST]",
            # 格式3：带系统提示
            lambda q: f'<s>[INST] You are a helpful assistant. {q} [/INST]',
        ]
        
        question = "What is the capital of France?"
        
        for i, template in enumerate(templates):
            prompt = template(question)
            inputs = tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    temperature=0.1
                )
            
            generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            print(f"\n模板{i+1}: {prompt[:50]}...")
            print(f"输出: {generated}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模板测试失败: {e}")
        return False


def test_cake_conversion():
    """测试CAKE转换是否影响生成"""
    print("\n" + "="*60)
    print("3. 测试CAKE转换的影响")
    print("="*60)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from hace_core.models.cake_converter import apply_cake_to_model, is_cake_available
        
        if not is_cake_available():
            print("⚠️ CAKE不可用，跳过测试")
            return False
        
        model_path = get_model_path()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 测试转换前
        prompt = "<s>[INST] What is the capital of France? [/INST]"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        print("转换前生成:")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        generated_before = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"输出: {generated_before}")
        
        # 应用CAKE转换
        print("\n应用CAKE转换...")
        model_config_hf = model.config.to_dict()
        cake_exp_config = {
            "layer_allocation_strategies": ["dynamic"],
            "cache_budgets": [0.7]
        }
        cake_model_config = {
            "window_size": 32,
            "gamma": 0.8,
            "tau1": 1.0,
            "tau2": 1.0
        }
        
        model = apply_cake_to_model(model, model_config_hf, cake_exp_config, cake_model_config)
        print(f"CAKE已启用: {getattr(model, 'is_cake_enabled', False)}")
        
        # 测试转换后
        print("\n转换后生成:")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        generated_after = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"输出: {generated_after}")
        
        if generated_after == "Paris" and generated_before != "Paris":
            print("\n❌ 发现问题：CAKE转换导致输出变成'Paris'！")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ CAKE测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generation_parameters():
    """测试不同的生成参数"""
    print("\n" + "="*60)
    print("4. 测试生成参数的影响")
    print("="*60)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_path = get_model_path()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        prompt = "<s>[INST] What is the capital of France? [/INST]"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # 测试不同的参数组合
        param_sets = [
            {"do_sample": False, "temperature": 0.1},
            {"do_sample": False, "temperature": 1.0},
            {"do_sample": True, "temperature": 0.7, "top_p": 0.95},
            {"do_sample": False, "num_beams": 2},
        ]
        
        for params in param_sets:
            print(f"\n参数: {params}")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    **params
                )
            generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            print(f"输出: {generated}")
        
        return True
        
    except Exception as e:
        print(f"❌ 参数测试失败: {e}")
        return False


def main():
    """运行所有诊断测试"""
    print("🔍 开始诊断'Paris'问题")
    print("\n使用方法:")
    print("1. python scripts/diagnose_paris_issue.py [模型路径]")
    print("2. export MODEL_PATH=模型路径 && python scripts/diagnose_paris_issue.py")
    print("3. 或者先运行 auto_detect_model_path.py 设置默认路径\n")
    
    model_path = get_model_path()
    print(f"📌 当前使用模型: {model_path}\n")
    
    tests = [
        test_basic_generation,
        test_with_instruction_template,
        test_cake_conversion,
        test_generation_parameters,
    ]
    
    for test in tests:
        if not test():
            print("\n⚠️ 发现问题，请检查上面的输出")
            break
    
    print("\n诊断完成！")


if __name__ == "__main__":
    main() 