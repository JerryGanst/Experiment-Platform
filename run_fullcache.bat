@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
echo Setting environment variables...

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0
REM Remove trailing backslash
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%
set PYTHONPATH=%SCRIPT_DIR%

REM Check if running in non-interactive environment
set INTERACTIVE=1
echo %CMDCMDLINE% | findstr /C:"/c" >nul && set INTERACTIVE=0

:check_model_path
if not defined HACE_MODEL_PATH (
    echo ERROR: HACE_MODEL_PATH environment variable is not set
    echo Please set HACE_MODEL_PATH to point to your model path, for example:
    echo set HACE_MODEL_PATH=C:\path\to\your\model
    echo.
    
    if %INTERACTIVE%==0 (
        echo ERROR: Running in non-interactive mode, cannot prompt for input
        echo Please set the HACE_MODEL_PATH environment variable before running this script
        echo Example: set HACE_MODEL_PATH=microsoft/DialoGPT-medium
        exit /b 1
    )
    
    set /p HACE_MODEL_PATH=Please enter model path: 
    
    REM Validate the input
    if "!HACE_MODEL_PATH!"=="" (
        echo ERROR: Model path cannot be empty!
        echo.
        goto check_model_path
    )
    
    REM Check if the path exists (for local paths)
    if exist "!HACE_MODEL_PATH!" (
        echo Model path validated: !HACE_MODEL_PATH!
    ) else (
        echo WARNING: The specified path does not exist locally: !HACE_MODEL_PATH!
        echo This might be a model identifier from Hugging Face Hub.
        echo.
        
        if %INTERACTIVE%==1 (
            set /p CONTINUE=Continue anyway? (y/N): 
            if /i not "!CONTINUE!"=="y" (
                echo.
                goto check_model_path
            )
        ) else (
            echo Assuming this is a Hugging Face model identifier and continuing...
        )
    )
)

echo Verifying path settings...
echo PYTHONPATH=%PYTHONPATH%
echo HACE_MODEL_PATH=%HACE_MODEL_PATH%

echo Running fullcache experiment...
python evaluation/baselines/fullkvcache_main.py --model_name "%HACE_MODEL_PATH%" --datasets trec --kv_cache_lengths 128 --batch_sizes 1 --repetitions 1 --max_new_tokens 50

pause 