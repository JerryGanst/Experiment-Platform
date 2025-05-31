@echo off
chcp 65001 > nul
title AIword 悬浮窗启动器

echo.
echo ====================================
echo    AIword 智能提示词补全悬浮窗
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

:: 检查依赖是否安装
echo 🔍 检查依赖包...
pip show webview > nul 2>&1
if %errorlevel% neq 0 (
    echo 📦 安装依赖包...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ❌ 依赖安装失败！
        pause
        exit /b 1
    )
)

:: 启动悬浮窗应用
echo 🚀 启动AIword悬浮窗...
echo.
python floating_app.py

if %errorlevel% neq 0 (
    echo.
    echo ❌ 应用启动失败！
    echo 💡 可能的解决方案：
    echo    1. 检查端口5000是否被占用
    echo    2. 重新安装依赖：pip install -r requirements.txt
    echo    3. 检查防火墙设置
    pause
)

echo.
echo 👋 感谢使用AIword！
pause 