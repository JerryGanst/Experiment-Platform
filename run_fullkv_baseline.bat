@echo off
setlocal enabledelayedexpansion

echo ===============================================
echo        建立Full KV基线 - 100分基准
echo ===============================================

:: 设置环境变量以改善错误报告和调试
set CUDA_LAUNCH_BLOCKING=1
set PYTORCH_NO_CUDA_MEMORY_CACHING=1
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

:: 基础配置
set MODEL_NAME=NousResearch/Llama-2-7b-hf
set DATASETS=hotpotqa,multi_news
set BATCH_SIZE=1
set MAX_NEW_TOKENS=100
set REPETITIONS=1

:: 设置工作目录和输出
set WORK_DIR=%cd%
set OUTPUT_DIR=results\fullkv_baseline
set TIMESTAMP=%date:~10,4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%

echo 开始时间: %date% %time%
echo 模型: %MODEL_NAME%
echo 数据集: %DATASETS%
echo 批处理大小: %BATCH_SIZE%
echo 最大新令牌: %MAX_NEW_TOKENS%
echo 重复次数: %REPETITIONS%
echo 输出目录: %OUTPUT_DIR%
echo.

:: 创建输出目录
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

:: 激活虚拟环境（如果存在）
if exist "venv\Scripts\activate.bat" (
    echo 激活虚拟环境...
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    echo 激活虚拟环境...
    call .venv\Scripts\activate.bat
) else (
    echo 未找到虚拟环境，使用系统Python
)

:: 切换到正确的工作目录
cd /d "%WORK_DIR%"

echo.
echo ===============================================
echo      开始运行Full KV基线实验
echo ===============================================

:: 运行Full KV实验建立基线
echo 运行Full KV基线实验...
python hace-kv-optimization\baselines\fullkvcache_main.py ^
    --model_name "%MODEL_NAME%" ^
    --datasets "%DATASETS%" ^
    --batch_sizes "%BATCH_SIZE%" ^
    --max_new_tokens %MAX_NEW_TOKENS% ^
    --repetitions %REPETITIONS% ^
    --output_dir "%OUTPUT_DIR%" ^
    --log_level INFO ^
    --seed 42 ^
    --run_name "fullkv_baseline_%TIMESTAMP%" ^
    --enable_scoring ^
    --is_baseline_run

set FULLKV_EXIT_CODE=%errorlevel%

echo.
echo ===============================================
echo             基线建立结果
echo ===============================================

if %FULLKV_EXIT_CODE% EQU 0 (
    echo ✅ Full KV基线实验成功完成！
    echo 基线分数已保存到: hace-kv-optimization\baseline_fullkv.json
    echo 实验结果保存在: %OUTPUT_DIR%
    
    :: 显示基线分数
    if exist "hace-kv-optimization\baseline_fullkv.json" (
        echo.
        echo 📊 基线分数预览:
        type "hace-kv-optimization\baseline_fullkv.json"
    )
) else (
    echo ❌ Full KV基线实验失败，退出码: %FULLKV_EXIT_CODE%
    echo 请检查日志文件并解决问题后重试
)

echo.
echo 基线建立完成时间: %date% %time%
echo.

:: 清理GPU内存
echo 清理GPU内存...
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None; print('GPU内存已清理')"

echo.
echo ===============================================
echo           下一步操作指南
echo ===============================================
echo 1. 基线建立成功后，可以运行其他策略实验
echo 2. 运行CAKE实验: run_cake_experiments.bat
echo 3. 所有实验将自动使用该基线计算相对分数
echo 4. 相对分数以百分制显示，Full KV = 100分
echo ===============================================

pause 