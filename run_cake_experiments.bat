@echo off
setlocal enabledelayedexpansion

echo ===============================================
echo             CAKE KV Cache 实验运行器
echo ===============================================

:: 设置环境变量以改善错误报告和调试
set CUDA_LAUNCH_BLOCKING=1
set PYTORCH_NO_CUDA_MEMORY_CACHING=1
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

:: 基础配置 - 和baseline保持一致
set MODEL_NAME=NousResearch/Llama-2-7b-hf
set DATASETS=hotpotqa,multi_news
set KV_CACHE_LENGTHS=128,256,512,1024,2048
set BATCH_SIZE=1
set MAX_NEW_TOKENS=100
set REPETITIONS=1

:: CAKE特定配置
set ALLOCATION_STRATEGIES=uniform,adaptive
set CACHE_BUDGETS=0.7,0.8,0.9

:: 设置工作目录和输出
set WORK_DIR=%cd%
set OUTPUT_DIR=results\cake_experiments
set TIMESTAMP=%date:~10,4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%

echo 开始时间: %date% %time%
echo 模型: %MODEL_NAME%
echo 数据集: %DATASETS%
echo KV Cache长度: %KV_CACHE_LENGTHS%
echo 批处理大小: %BATCH_SIZE%
echo 最大新令牌: %MAX_NEW_TOKENS%
echo 分配策略: %ALLOCATION_STRATEGIES%
echo 缓存预算: %CACHE_BUDGETS%
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
echo            开始运行CAKE实验
echo ===============================================

:: 运行CAKE实验
echo 运行CAKE实验...
python hace-kv-optimization\baselines\cake_main.py ^
    --model_name "%MODEL_NAME%" ^
    --datasets "%DATASETS%" ^
    --kv_cache_lengths "%KV_CACHE_LENGTHS%" ^
    --batch_sizes "%BATCH_SIZE%" ^
    --max_new_tokens %MAX_NEW_TOKENS% ^
    --allocation_strategies "%ALLOCATION_STRATEGIES%" ^
    --cache_budgets "%CACHE_BUDGETS%" ^
    --repetitions %REPETITIONS% ^
    --output_dir "%OUTPUT_DIR%" ^
    --log_level INFO ^
    --seed 42 ^
    --run_name "cake_baseline_comparison_%TIMESTAMP%" ^
    --enable_scoring

set CAKE_EXIT_CODE=%errorlevel%

echo.
echo ===============================================
echo             实验结果汇总
echo ===============================================

if %CAKE_EXIT_CODE% EQU 0 (
    echo ✅ CAKE实验成功完成！
    echo 结果保存在: %OUTPUT_DIR%
) else (
    echo ❌ CAKE实验失败，退出码: %CAKE_EXIT_CODE%
)

echo.
echo 实验完成时间: %date% %time%
echo.

:: 清理GPU内存
echo 清理GPU内存...
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None; print('GPU内存已清理')"

:: 显示结果目录内容
if exist "%OUTPUT_DIR%" (
    echo.
    echo 生成的文件:
    dir /b "%OUTPUT_DIR%"
)

pause 