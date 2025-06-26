@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

echo 开始运行CAKE原论文标准设置的单文档问答组实验...
echo.

REM 设置环境变量
set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%
set PYTHONPATH=%SCRIPT_DIR%
set HACE_MODEL_PATH=C:/Users/Administrator/mistral_models/7B-Instruct-v0.3

echo 环境设置:
echo PYTHONPATH=%PYTHONPATH%
echo HACE_MODEL_PATH=%HACE_MODEL_PATH%
echo.

REM 创建输出目录
if not exist "runs\results_cake_original_settings" mkdir "runs\results_cake_original_settings"

echo ============================================================
echo 实验1: narrativeqa + qasper (128 tokens - CAKE原论文设置)
echo ============================================================
echo 运行实验参数:
echo - 数据集: narrativeqa, qasper
echo - KV缓存长度: 128, 1024
echo - 批次大小: 1
echo - 最大新token数: 128 (CAKE原论文设置)
echo - 重复次数: 3
echo - 启用评分: 是
echo - 基线运行: 是
echo.

echo 开始实验1...
python evaluation/baselines/fullkvcache_main.py ^
    --model_name "%HACE_MODEL_PATH%" ^
    --enable_scoring ^
    --is_baseline_run ^
    --datasets narrativeqa,qasper ^
    --kv_cache_lengths 128,1024 ^
    --batch_sizes 1 ^
    --max_new_tokens 128 ^
    --repetitions 3 ^
    --output_dir ./runs/results_cake_original_settings/narrativeqa_qasper_128tokens

echo.
echo ============================================================
echo 实验2: multifieldqa_en (64 tokens - CAKE原论文设置)
echo ============================================================
echo 运行实验参数:
echo - 数据集: multifieldqa_en
echo - KV缓存长度: 128, 1024
echo - 批次大小: 1
echo - 最大新token数: 64 (CAKE原论文设置)
echo - 重复次数: 3
echo - 启用评分: 是
echo - 基线运行: 是
echo.

echo 开始实验2...
python evaluation/baselines/fullkvcache_main.py ^
    --model_name "%HACE_MODEL_PATH%" ^
    --enable_scoring ^
    --is_baseline_run ^
    --datasets multifieldqa_en ^
    --kv_cache_lengths 128,1024 ^
    --batch_sizes 1 ^
    --max_new_tokens 64 ^
    --repetitions 3 ^
    --output_dir ./runs/results_cake_original_settings/multifieldqa_en_64tokens

echo.
echo ============================================================
echo 所有实验完成！
echo ============================================================
echo 结果保存在以下目录：
echo - narrativeqa + qasper (128 tokens): runs\results_cake_original_settings\narrativeqa_qasper_128tokens
echo - multifieldqa_en (64 tokens): runs\results_cake_original_settings\multifieldqa_en_64tokens
echo.
pause