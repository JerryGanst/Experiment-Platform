@echo off
chcp 65001 > nul
title AIword 原生悬浮窗启动器

echo.
echo ====================================
echo     AIword 原生智能提示词补全
echo ====================================
echo      基于tkinter，无需浏览器
echo ====================================
echo.

:: 检查Python是否安装
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 错误: 未找到Python！
    echo 请先安装Python 3.7+
    pause
    exit /b 1
)

echo ✅ Python环境正常

:: 检查核心依赖
echo 🔍 检查依赖包...
python -c "import tkinter; import yaml" > nul 2>&1
if %errorlevel% neq 0 (
    echo 📦 安装必要依赖包...
    pip install PyYAML
    if %errorlevel% neq 0 (
        echo ❌ 依赖安装失败！
        pause
        exit /b 1
    )
)

echo ✅ 依赖检查完成

:: 启动原生悬浮窗应用
echo 🚀 启动AIword原生悬浮窗...
echo.
python native_floating_app.py

if %errorlevel% neq 0 (
    echo.
    echo ❌ 应用启动失败！
    echo 💡 可能的解决方案：
    echo    1. 检查completion_config.yaml文件是否存在
    echo    2. 检查completion_engine.py文件是否存在
    echo    3. 重新安装依赖：pip install PyYAML
    pause
)

echo.
echo 👋 感谢使用AIword！
pause 