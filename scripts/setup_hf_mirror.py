#!/usr/bin/env python3
"""
设置Hugging Face镜像站

这个脚本帮助配置Hugging Face镜像站环境变量和下载命令
"""

import os


def print_mirror_commands():
    """打印各种镜像站的配置命令"""
    print("🌐 Hugging Face镜像站配置指南")
    print("="*60)
    
    print("\n1️⃣ HF-Mirror (推荐)")
    print("   官网: https://hf-mirror.com")
    print("   配置命令:")
    print("   export HF_ENDPOINT=https://hf-mirror.com")
    print("   # 或者在~/.bashrc中添加上述命令")
    
    print("\n2️⃣ 魔搭社区 ModelScope")
    print("   官网: https://modelscope.cn")
    print("   安装命令:")
    print("   pip install modelscope")
    print("   使用示例:")
    print("   from modelscope import snapshot_download")
    print("   snapshot_download('mistralai/Mistral-7B-Instruct-v0.3')")
    
    print("\n3️⃣ 其他镜像站")
    print("   可以设置环境变量: export HF_ENDPOINT=<镜像站URL>")
    
    print("\n🔧 当前配置检查:")
    hf_endpoint = os.environ.get('HF_ENDPOINT', '未设置')
    print(f"   HF_ENDPOINT: {hf_endpoint}")
    
    if hf_endpoint == '未设置':
        print("   ✅ 使用官方站点 huggingface.co")
    else:
        print(f"   ✅ 使用镜像站: {hf_endpoint}")


def set_hf_mirror():
    """交互式设置镜像站"""
    print("\n🛠️ 设置镜像站 (仅对当前会话有效)")
    print("请选择:")
    print("1. HF-Mirror (hf-mirror.com)")
    print("2. 自定义镜像站")
    print("3. 重置为官方站点")
    print("4. 退出")
    
    choice = input("\n请输入选择 (1-4): ").strip()
    
    if choice == "1":
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("✅ 已设置为 HF-Mirror")
    elif choice == "2":
        url = input("请输入镜像站URL: ").strip()
        if url:
            os.environ["HF_ENDPOINT"] = url
            print(f"✅ 已设置为: {url}")
        else:
            print("❌ URL不能为空")
    elif choice == "3":
        if "HF_ENDPOINT" in os.environ:
            del os.environ["HF_ENDPOINT"]
        print("✅ 已重置为官方站点")
    elif choice == "4":
        return
    else:
        print("❌ 无效选择")


def generate_download_commands():
    """生成下载命令"""
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    
    print(f"\n📥 下载 {model_name} 的命令:")
    print("="*60)
    
    print("\n🐍 Python代码 (Transformers):")
    print("```python")
    print("from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"tokenizer = AutoTokenizer.from_pretrained('{model_name}')")
    print(f"model = AutoModelForCausalLM.from_pretrained('{model_name}')")
    print("```")
    
    print("\n🐍 Python代码 (ModelScope):")
    print("```python")
    print("from modelscope import snapshot_download")
    print(f"snapshot_download('{model_name}')")
    print("```")
    
    print("\n💻 命令行 (huggingface-hub):")
    print("```bash")
    print("pip install huggingface-hub")
    print(f"huggingface-cli download {model_name}")
    print("```")


def main():
    """主函数"""
    print("🔧 Hugging Face镜像站配置工具")
    
    while True:
        print("\n" + "="*60)
        print("请选择操作:")
        print("1. 查看镜像站配置指南")
        print("2. 设置镜像站")
        print("3. 生成下载命令")
        print("4. 退出")
        
        choice = input("\n请输入选择 (1-4): ").strip()
        
        if choice == "1":
            print_mirror_commands()
        elif choice == "2":
            set_hf_mirror()
        elif choice == "3":
            generate_download_commands()
        elif choice == "4":
            print("👋 再见！")
            break
        else:
            print("❌ 无效选择，请重新输入")


if __name__ == "__main__":
    main() 