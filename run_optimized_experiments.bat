@echo off
echo ================================================
echo 优化版FullKVCache实验启动脚本
echo 按照用户建议修复的版本
echo ================================================

echo.
echo 🔧 步骤1: 检查数据集文件...
if exist "data\hotpotqa.jsonl" (
    echo ✅ hotpotqa.jsonl 存在
) else (
    echo ❌ hotpotqa.jsonl 不存在，正在下载...
    python download_datasets.py
    if errorlevel 1 (
        echo 数据集下载失败，退出
        pause
        exit /b 1
    )
)

echo.
echo 🚀 步骤2: 运行优化版实验...
echo 特点：
echo - 模型只加载一次，循环复用
echo - 修复了HotpotQA数据集加载问题
echo - 优化的GPU内存管理
echo.

python fullkvcache_main_optimized.py --enable_scoring --is_baseline_run

echo.
echo 🎉 实验完成！
echo.
pause 