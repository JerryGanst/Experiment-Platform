 @echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

REM ================================================================
REM HotpotQA Document QA Baseline Experiment
REM Uses same max_new_tokens as CAKE (32)
REM ================================================================

REM Prerequisites check
if not defined HACE_MODEL_PATH (
    echo [ERROR] HACE_MODEL_PATH is not set. Please set it first.
    echo Example: set HACE_MODEL_PATH=microsoft/DialoGPT-medium
    pause
    exit /b 1
)

REM Move to project root
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

REM Set PYTHONPATH - 确保路径正确
set PYTHONPATH=%CD%
echo [INFO] PYTHONPATH 设置为: %PYTHONPATH%

REM 验证环境变量
echo [INFO] 验证环境变量:
echo   HACE_MODEL_PATH = %HACE_MODEL_PATH%
echo   PYTHONPATH = %PYTHONPATH%
echo.

REM Configuration parameters - matching CAKE settings
set DATASET=hotpotqa
set KV_CACHE_LENGTHS=128,1024
set BATCH_SIZES=1
set REPETITIONS=1
set MAX_NEW_TOKENS=32

echo ==============================================================
echo Running HotpotQA Document QA Baseline Experiment (完整数据集)
echo Dataset: %DATASET% (~11.5MB, 预计15-30分钟)
echo Model: %HACE_MODEL_PATH%
echo Max New Tokens: %MAX_NEW_TOKENS% (matching CAKE settings)
echo KV Cache Lengths: %KV_CACHE_LENGTHS%
echo 警告: HotpotQA是最大的数据集，处理时间较长！
echo ==============================================================

set /p continue="确认开始完整HotpotQA基线实验? (y/N): "
if /i not "%continue%"=="y" (
    echo 实验已取消
    pause
    exit /b 0
)

echo.
echo [开始时间] %time%
echo 正在处理完整HotpotQA数据集，请耐心等待...

python evaluation/baselines/fullkvcache_main.py ^
    --model_name "%HACE_MODEL_PATH%" ^
    --enable_scoring ^
    --is_baseline_run ^
    --datasets %DATASET% ^
    --kv_cache_lengths %KV_CACHE_LENGTHS% ^
    --batch_sizes %BATCH_SIZES% ^
    --repetitions %REPETITIONS% ^
    --output_dir ./runs/%DATASET%_baseline_fullkv

REM 注意: 不再指定 --max_new_tokens，将自动使用CAKE配置中的32 tokens

echo.
echo [完成时间] %time%
echo ✅ HotpotQA baseline experiment completed!
echo Results saved to: ./runs/%DATASET%_baseline_fullkv
echo.
pause