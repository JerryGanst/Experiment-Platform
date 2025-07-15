@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

REM ================================================================
REM Document QA Baseline Experiments - Updated for CAKE Token Matching
REM Runs Qasper, MultifieldQA English, and HotpotQA baselines
REM Uses same max_new_tokens as CAKE for each dataset
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

REM Set PYTHONPATH
set PYTHONPATH=%CD%

echo ==============================================================
echo Starting Document QA Baseline Experiments
echo Model: %HACE_MODEL_PATH%
echo Datasets: Qasper, MultifieldQA-EN, HotpotQA
echo Token Settings: Matching CAKE configuration
echo ==============================================================

REM Dataset 1: Qasper (128 tokens - matching CAKE)
echo.
echo [1/3] Running Qasper baseline (max_new_tokens=128)...
python evaluation/baselines/fullkvcache_main.py ^
    --model_name "%HACE_MODEL_PATH%" ^
    --enable_scoring ^
    --is_baseline_run ^
    --datasets qasper ^
    --kv_cache_lengths 128,1024 ^
    --batch_sizes 1 ^
    --max_new_tokens 128 ^
    --repetitions 1 ^
    --output_dir ./runs/qasper_baseline_fullkv

if errorlevel 1 (
    echo [ERROR] Qasper baseline failed!
    pause
    exit /b 1
)

REM Dataset 2: MultifieldQA English (64 tokens - matching CAKE)
echo.
echo [2/3] Running MultifieldQA English baseline (max_new_tokens=64)...
python evaluation/baselines/fullkvcache_main.py ^
    --model_name "%HACE_MODEL_PATH%" ^
    --enable_scoring ^
    --is_baseline_run ^
    --datasets multifieldqa_en ^
    --kv_cache_lengths 128,1024 ^
    --batch_sizes 1 ^
    --max_new_tokens 64 ^
    --repetitions 1 ^
    --output_dir ./runs/multifieldqa_en_baseline_fullkv

if errorlevel 1 (
    echo [ERROR] MultifieldQA English baseline failed!
    pause
    exit /b 1
)

REM Dataset 3: HotpotQA (32 tokens - matching CAKE)
echo.
echo [3/3] Running HotpotQA baseline (max_new_tokens=32)...
python evaluation/baselines/fullkvcache_main.py ^
    --model_name "%HACE_MODEL_PATH%" ^
    --enable_scoring ^
    --is_baseline_run ^
    --datasets hotpotqa ^
    --kv_cache_lengths 128,1024 ^
    --batch_sizes 1 ^
    --max_new_tokens 32 ^
    --repetitions 1 ^
    --output_dir ./runs/hotpotqa_baseline_fullkv

if errorlevel 1 (
    echo [ERROR] HotpotQA baseline failed!
    pause
    exit /b 1
)

echo.
echo ==============================================================
echo All Document QA baseline experiments completed successfully!
echo.
echo Results saved to:
echo   - ./runs/qasper_baseline_fullkv
echo   - ./runs/multifieldqa_en_baseline_fullkv  
echo   - ./runs/hotpotqa_baseline_fullkv
echo ==============================================================
echo.
pause