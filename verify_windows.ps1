# Windows PowerShell 验证脚本
# 验证硬编码路径修复是否成功

Write-Host ""
Write-Host "🎯 Windows环境验证脚本 (PowerShell版本)" -ForegroundColor Green
Write-Host "验证硬编码路径修复是否成功" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Yellow
Write-Host ""

# 显示当前目录
Write-Host "📁 当前工作目录: $(Get-Location)" -ForegroundColor Cyan
Write-Host ""

# 检查项目文件结构
Write-Host "🔍 检查项目文件结构..." -ForegroundColor Cyan

if (Test-Path "hace_core\config.py") {
    Write-Host "✅ 找到配置文件: hace_core\config.py" -ForegroundColor Green
} else {
    Write-Host "❌ 未找到配置文件: hace_core\config.py" -ForegroundColor Red
    Write-Host "   请确保在项目根目录运行此脚本" -ForegroundColor Red
    Read-Host "按任意键退出"
    exit 1
}

if (Test-Path "check_model_config.py") {
    Write-Host "✅ 找到检查工具: check_model_config.py" -ForegroundColor Green
} else {
    Write-Host "❌ 未找到检查工具: check_model_config.py" -ForegroundColor Red
    Read-Host "按任意键退出"
    exit 1
}

Write-Host ""

# 检查Python环境
Write-Host "🐍 检查Python环境..." -ForegroundColor Cyan
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ $pythonVersion" -ForegroundColor Green
    } else {
        throw "Python command failed"
    }
} catch {
    Write-Host "❌ Python未安装或未添加到PATH" -ForegroundColor Red
    Write-Host "   请先安装Python并添加到环境变量" -ForegroundColor Red
    Read-Host "按任意键退出"
    exit 1
}

Write-Host ""

# 运行配置检查工具
Write-Host "🔧 运行配置检查工具..." -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Yellow

$checkResult = $null
try {
    python check_model_config.py
    $checkResult = $LASTEXITCODE
} catch {
    $checkResult = 1
}

Write-Host ""
Write-Host "======================================" -ForegroundColor Yellow

if ($checkResult -eq 0) {
    Write-Host "✅ 配置检查通过！" -ForegroundColor Green
} else {
    Write-Host "⚠️  配置需要设置（这是预期的）" -ForegroundColor Yellow
}

Write-Host ""

# 显示配置方案
Write-Host "📋 Windows环境下的模型路径配置方案:" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Yellow
Write-Host ""

Write-Host "方案1: 使用环境变量（推荐）" -ForegroundColor White
Write-Host "  临时设置 (当前PowerShell会话):" -ForegroundColor Gray
Write-Host "    `$env:HACE_MODEL_PATH='D:\AI\models\mistral-7b-instruct-v0.3'" -ForegroundColor Yellow
Write-Host ""
Write-Host "  永久设置 (用户环境变量):" -ForegroundColor Gray
Write-Host "    [Environment]::SetEnvironmentVariable('HACE_MODEL_PATH', 'D:\AI\models\mistral-7b-instruct-v0.3', 'User')" -ForegroundColor Yellow
Write-Host ""
Write-Host "  永久设置 (系统环境变量，需管理员权限):" -ForegroundColor Gray
Write-Host "    [Environment]::SetEnvironmentVariable('HACE_MODEL_PATH', 'D:\AI\models\mistral-7b-instruct-v0.3', 'Machine')" -ForegroundColor Yellow
Write-Host ""

Write-Host "方案2: 项目本地models目录" -ForegroundColor White
Write-Host "  New-Item -ItemType Directory -Path 'models\mistral-7b-instruct-v0.3' -Force" -ForegroundColor Yellow
Write-Host "  (将模型文件放入该目录)" -ForegroundColor Gray
Write-Host ""

Write-Host "方案3: 用户主目录models" -ForegroundColor White
Write-Host "  New-Item -ItemType Directory -Path '`$env:USERPROFILE\models\mistral-7b-instruct-v0.3' -Force" -ForegroundColor Yellow
Write-Host "  (将模型文件放入该目录)" -ForegroundColor Gray
Write-Host ""

# 显示验证步骤
Write-Host "🧪 验证步骤:" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Yellow
Write-Host "1. 选择上述配置方案之一设置模型路径" -ForegroundColor White
Write-Host "2. 重新运行: python check_model_config.py" -ForegroundColor White
Write-Host "3. 确认输出显示 '✅ 模型文件存在'" -ForegroundColor White
Write-Host "4. 运行你的实验程序验证功能" -ForegroundColor White
Write-Host ""

# 显示提示信息
Write-Host "💡 提示:" -ForegroundColor Cyan
Write-Host "  - 确保模型路径中不包含空格或特殊字符" -ForegroundColor Gray
Write-Host "  - 使用绝对路径避免相对路径问题" -ForegroundColor Gray
Write-Host "  - 模型文件夹应包含config.json、pytorch_model.bin等文件" -ForegroundColor Gray
Write-Host "  - 如果模型路径包含空格，请用引号包围" -ForegroundColor Gray
Write-Host ""

# 提供快速配置选项
Write-Host "🚀 快速配置助手:" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Yellow
$choice = Read-Host "是否要现在设置环境变量？(y/N)"

if ($choice -eq 'y' -or $choice -eq 'Y') {
    $modelPath = Read-Host "请输入你的模型完整路径 (例: D:\AI\models\mistral-7b-instruct-v0.3)"
    
    if ($modelPath -and (Test-Path $modelPath)) {
        # 设置用户环境变量
        [Environment]::SetEnvironmentVariable('HACE_MODEL_PATH', $modelPath, 'User')
        $env:HACE_MODEL_PATH = $modelPath  # 同时设置当前会话
        
        Write-Host ""
        Write-Host "✅ 环境变量设置成功！" -ForegroundColor Green
        Write-Host "   HACE_MODEL_PATH = $modelPath" -ForegroundColor Green
        Write-Host ""
        Write-Host "🔄 重新验证配置..." -ForegroundColor Cyan
        python check_model_config.py
    } elseif ($modelPath) {
        Write-Host ""
        Write-Host "❌ 指定的路径不存在: $modelPath" -ForegroundColor Red
        Write-Host "   请确认路径正确并重新运行脚本" -ForegroundColor Red
    }
} else {
    Write-Host ""
    Write-Host "ℹ️  你可以稍后手动配置环境变量" -ForegroundColor Blue
}

Write-Host ""
Read-Host "按任意键退出" 