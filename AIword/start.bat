@echo off
echo ================================
echo   AIword 垂直化提示词补全系统
echo ================================
echo.
echo 正在启动系统...
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

REM 检查依赖是否安装
echo 检查依赖包...
pip show Flask >nul 2>&1
if errorlevel 1 (
    echo [提示] 正在安装依赖包...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [错误] 依赖安装失败
        pause
        exit /b 1
    )
)

echo [信息] 依赖检查完成
echo.

REM 启动Flask应用
echo 启动Web服务器...
echo 访问地址: http://localhost:5000
echo.
echo 按 Ctrl+C 停止服务
echo.

python web_interface.py

pause 