@echo off
chcp 65001 >nul

echo ====================================================================
echo 环境变量设置脚本
echo ====================================================================

REM 设置模型路径（如果未设置）
if not defined HACE_MODEL_PATH (
    echo [INFO] 设置 HACE_MODEL_PATH...
    set HACE_MODEL_PATH=C:\Users\Administrator\mistral_models\7B-Instruct-v0.3
)

REM 设置Python路径
echo [INFO] 设置 PYTHONPATH...
set PYTHONPATH=%CD%

REM 显示当前设置
echo.
echo ✅ 环境变量设置完成:
echo    HACE_MODEL_PATH = %HACE_MODEL_PATH%
echo    PYTHONPATH = %PYTHONPATH%
echo.

REM 验证模型路径
if exist "%HACE_MODEL_PATH%" (
    echo ✅ 模型路径验证: 存在
) else (
    echo ❌ 模型路径验证: 不存在
    echo    请检查模型路径是否正确
)

echo.
echo 💡 使用方法:
echo    1. 运行此脚本: setup_env.bat
echo    2. 在同一命令窗口中运行基线实验
echo.
echo 🚀 准备就绪！现在可以运行基线实验了。
echo ==================================================================== 