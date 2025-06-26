#!/bin/bash
# Experiment Platform GPU环境安装脚本
# 支持 macOS 和 Linux

echo "🚀 开始安装 Experiment Platform GPU环境..."

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1-2)
echo "📋 Python版本: $python_version"

if [[ $(echo "$python_version >= 3.8" | bc -l) -eq 0 ]]; then
    echo "❌ 需要 Python 3.8 或更高版本"
    exit 1
fi

# 检查操作系统
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 检测到 macOS 系统"
    # macOS 使用 MPS 后端
    echo "💡 macOS 将使用 MPS (Metal Performance Shaders) 加速"
    pip3 install torch torchvision torchaudio
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "🐧 检测到 Linux 系统"
    # 检查 CUDA
    if command -v nvidia-smi &> /dev/null; then
        cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        echo "🎮 检测到 CUDA 版本: $cuda_version"
        # 安装 CUDA 版本的 PyTorch
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        echo "⚠️ 未检测到 CUDA，安装 CPU 版本"
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
else
    echo "❓ 未知操作系统，使用默认安装"
    pip3 install torch torchvision torchaudio
fi

# 安装其余依赖
echo "📦 安装其余依赖包..."
pip3 install -r requirements.txt

# 验证安装
echo "🧪 验证 PyTorch GPU 安装..."
python3 -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 版本: {torch.version.cuda}')
    print(f'GPU 数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
elif torch.backends.mps.is_available():
    print('MPS (Metal) 可用: True')
else:
    print('使用 CPU 模式')
"

echo "✅ 安装完成！"
echo "💡 运行测试: python test_scoring_fix.py" 