@echo off
chcp 65001 >nul
echo Setting environment variables...
set PYTHONPATH=%CD%

if not defined HACE_MODEL_PATH (
    echo WARNING: HACE_MODEL_PATH environment variable is not set
    echo Please set HACE_MODEL_PATH to point to your model path, for example:
    echo set HACE_MODEL_PATH=C:\path\to\your\model
    echo.
    set /p HACE_MODEL_PATH=Please enter model path: 
)

echo Verifying path settings...
echo PYTHONPATH=%PYTHONPATH%
echo HACE_MODEL_PATH=%HACE_MODEL_PATH%

echo Running fullcache experiment...
python evaluation/baselines/fullkvcache_main.py --model_name "%HACE_MODEL_PATH%" --datasets trec --kv_cache_lengths 128 --batch_sizes 1 --repetitions 1 --max_new_tokens 50

pause 