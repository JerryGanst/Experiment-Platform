@echo off
chcp 65001 >nul
echo.
echo 🎯 Windows环境验证脚本
echo 验证硬编码路径修复是否成功
echo ======================================
echo.

echo 📁 当前工作目录: %CD%
echo.

echo 🔍 检查项目文件结构...
if exist "hace_core\config.py" (
    echo ✅ 找到配置文件: hace_core\config.py
) else (
    echo ❌ 未找到配置文件: hace_core\config.py
    echo    请确保在项目根目录运行此脚本
    pause
    exit /b 1
)

if exist "check_model_config.py" (
    echo ✅ 找到检查工具: check_model_config.py
) else (
    echo ❌ 未找到检查工具: check_model_config.py
    pause
    exit /b 1
)

echo.
echo 🐍 检查Python环境...
python --version >nul 2>&1
if %ERRORLEVEL% == 0 (
    python --version
) else (
    echo ❌ Python未安装或未添加到PATH
    echo    请先安装Python并添加到环境变量
    pause
    exit /b 1
)

echo.
echo 🔧 运行配置检查工具...
echo ======================================
python check_model_config.py
set CHECK_RESULT=%ERRORLEVEL%

echo.
echo ======================================
if %CHECK_RESULT% == 0 (
    echo ✅ 配置检查通过！
) else (
    echo ⚠️  配置需要设置（这是预期的）
)

echo.
echo 📋 Windows环境下的模型路径配置方案:
echo ======================================
echo.
echo 方案1: 使用环境变量（推荐）
echo   临时设置 ^(当前命令行会话^):
echo     set HACE_MODEL_PATH=D:\AI\models\mistral-7b-instruct-v0.3
echo.
echo   永久设置 ^(系统环境变量^):
echo     1. Win+R 运行 sysdm.cpl
echo     2. 高级 ^> 环境变量
echo     3. 新建系统变量:
echo        变量名: HACE_MODEL_PATH
echo        变量值: D:\AI\models\mistral-7b-instruct-v0.3
echo.
echo 方案2: 项目本地models目录
echo   mkdir models
echo   mkdir models\mistral-7b-instruct-v0.3
echo   ^(将模型文件放入该目录^)
echo.
echo 方案3: 用户主目录models
echo   mkdir %%USERPROFILE%%\models
echo   mkdir %%USERPROFILE%%\models\mistral-7b-instruct-v0.3
echo   ^(将模型文件放入该目录^)
echo.

echo 🧪 验证步骤:
echo ======================================
echo 1. 选择上述配置方案之一设置模型路径
echo 2. 重新运行: python check_model_config.py
echo 3. 确认输出显示"✅ 模型文件存在"
echo 4. 运行你的实验程序验证功能
echo.

echo 💡 提示:
echo   - 确保模型路径中不包含空格或特殊字符
echo   - 使用绝对路径避免相对路径问题
echo   - 模型文件夹应包含config.json、pytorch_model.bin等文件
echo.

pause 