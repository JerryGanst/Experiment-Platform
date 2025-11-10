#!/usr/bin/env python3
"""
修复Hugging Face模型缓存问题

这个脚本会清理损坏的缓存文件并重新下载模型
"""

import os
import shutil
import subprocess
from pathlib import Path


def clear_model_cache(model_name="mistralai/Mistral-7B-Instruct-v0.3"):
    """清理指定模型的缓存"""
    print(f"🧹 清理模型缓存: {model_name}")
    
    # Hugging Face缓存目录
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    
    # 转换模型名为缓存目录名
    cache_model_name = model_name.replace("/", "--")
    model_cache_dir = cache_dir / f"models--{cache_model_name}"
    
    if model_cache_dir.exists():
        print(f"📁 找到缓存目录: {model_cache_dir}")
        try:
            shutil.rmtree(model_cache_dir)
            print("✅ 缓存目录已清理")
        except Exception as e:
            print(f"❌ 清理失败: {e}")
            return False
    else:
        print("⚠️ 未找到缓存目录")
    
    return True


def setup_mirror_config():
    """设置Hugging Face镜像站配置"""
    print("🌐 配置Hugging Face镜像站")
    print("请选择镜像站:")
    print("1. 官方站点 (huggingface.co)")
    print("2. 魔搭社区 (modelscope.cn)")
    print("3. HF-Mirror (hf-mirror.com)")
    print("4. 自定义镜像站")
    print("5. 跳过下载，只清理缓存")
    
    while True:
        choice = input("\n请输入选择 (1-5): ").strip()
        if choice == "1":
            return None, False  # 使用默认官方站点
        elif choice == "2":
            print("⚠️ ModelScope需要专门的SDK，请手动安装: pip install modelscope")
            return "modelscope", False
        elif choice == "3":
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            return "hf-mirror", True
        elif choice == "4":
            mirror_url = input("请输入自定义镜像站URL: ").strip()
            if mirror_url:
                os.environ["HF_ENDPOINT"] = mirror_url
                return "custom", True
            else:
                print("❌ URL不能为空")
        elif choice == "5":
            return None, False  # 跳过下载
        else:
            print("❌ 无效选择，请重新输入")


def download_model(model_name="mistralai/Mistral-7B-Instruct-v0.3"):
    """配置模型下载（不实际下载）"""
    mirror_type, should_download = setup_mirror_config()
    
    if not should_download:
        if mirror_type == "modelscope":
            print("\n📋 ModelScope下载命令:")
            print(f"from modelscope import snapshot_download")
            print(f"snapshot_download('{model_name}')")
        else:
            print("\n⚠️ 跳过下载，请手动下载模型")
            print("💡 建议使用以下方法之一:")
            print("1. 设置镜像站环境变量: export HF_ENDPOINT=https://hf-mirror.com")
            print("2. 使用魔搭社区: pip install modelscope")
            print("3. 手动下载模型文件")
        return False
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"📥 从镜像站下载模型: {model_name}")
        print(f"🌐 当前端点: {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}")
        
        print("🔄 下载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✅ 分词器下载完成")
        
        print("🔄 下载模型（这可能需要几分钟）...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            low_cpu_mem_usage=True
        )
        print("✅ 模型下载完成")
        
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("💡 建议:")
        print("1. 检查网络连接")
        print("2. 尝试其他镜像站")
        print("3. 手动下载模型")
        return False


def test_model_loading(model_name="mistralai/Mistral-7B-Instruct-v0.3"):
    """测试模型是否可以正常加载"""
    print(f"🧪 测试模型加载: {model_name}")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        
        # 简单测试
        inputs = tokenizer("Hello", return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=5)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"✅ 模型加载成功！测试输出: {generated}")
        return True
        
    except Exception as e:
        print(f"❌ 模型加载测试失败: {e}")
        return False


def main():
    """主函数"""
    print("🔧 修复Hugging Face模型缓存问题")
    
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    
    print(f"\n🎯 目标模型: {model_name}")
    
    # 步骤1: 清理损坏的缓存
    print("\n📋 步骤1: 清理损坏的缓存")
    if not clear_model_cache(model_name):
        print("❌ 缓存清理失败")
        return
    
    # 步骤2: 配置模型下载
    print("\n📋 步骤2: 配置模型下载")
    download_success = download_model(model_name)
    
    if download_success:
        # 步骤3: 测试模型加载
        print("\n📋 步骤3: 测试模型加载")
        if test_model_loading(model_name):
            print("\n🎉 模型修复成功！")
            print("现在可以正常运行: python test_corecode_hotpotqa_fixed.py")
        else:
            print("\n❌ 模型加载测试失败，请检查下载是否完整")
    else:
        print("\n✅ 缓存已清理，请按提示手动下载模型")
        print("\n📋 下载完成后的测试步骤:")
        print("1. python test_corecode_hotpotqa_fixed.py")
        print("2. 或运行: python scripts/fix_model_cache.py (选择测试模式)")


if __name__ == "__main__":
    main() 