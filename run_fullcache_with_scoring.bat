@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul


echo Setting environment variables...

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%
set PYTHONPATH=%SCRIPT_DIR%

REM Check if HACE_MODEL_PATH is set
if not defined HACE_MODEL_PATH (
    echo ERROR: HACE_MODEL_PATH environment variable is not set
    echo.
    echo Please set HACE_MODEL_PATH before running this script.
    echo Examples:
    echo   set HACE_MODEL_PATH=microsoft/DialoGPT-medium
    echo   set HACE_MODEL_PATH=C:\path\to\local\model
    echo.
    echo Or run this script with the model path as parameter:
    echo   %0 "microsoft/DialoGPT-medium"
    echo.
    
    REM Check if first parameter is provided
    if "%~1"=="" (
        echo No model path provided. Exiting.
        pause
        exit /b 1
    ) else (
        echo Using model path from parameter: %~1
        set HACE_MODEL_PATH=%~1
    )
)

echo Verifying settings...

echo PYTHONPATH=%PYTHONPATH%
echo HACE_MODEL_PATH=%HACE_MODEL_PATH%
echo.

echo Running fullcache experiment with scoring...
python evaluation/baselines/fullkvcache_main.py --model_name "%HACE_MODEL_PATH%" --datasets trec --kv_cache_lengths 128 --batch_sizes 1 --repetitions 1 --max_new_tokens 50 --enable_scoring

echo.
echo Experiment completed.
pause 