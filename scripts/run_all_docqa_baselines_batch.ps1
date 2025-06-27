# ================================================================
# Document QA Baseline Experiments - PowerShell Batch Runner
# 运行三个Document QA基线实验，每个重复3次
# 数据集: Qasper (128 tokens), MultifieldQA-EN (64 tokens), HotpotQA (32 tokens)
# ================================================================

param(
    [string]$ModelPath = $env:HACE_MODEL_PATH,
    [int]$Repetitions = 3,
    [switch]$SkipReset
)

# 设置控制台编码
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [console]::InputEncoding = [console]::OutputEncoding = New-Object System.Text.UTF8Encoding

Write-Host "====================================================================" -ForegroundColor Green
Write-Host "Document QA Baseline Experiments - PowerShell Batch Runner" -ForegroundColor Green
Write-Host "将运行三个基线实验，每个重复 $Repetitions 次，总共 $($Repetitions * 3) 个实验" -ForegroundColor Green
Write-Host "====================================================================" -ForegroundColor Green

# 检查前提条件
if (-not $ModelPath) {
    Write-Host "[ERROR] HACE_MODEL_PATH 未设置。请先设置环境变量。" -ForegroundColor Red
    Write-Host "示例: `$env:HACE_MODEL_PATH = 'microsoft/DialoGPT-medium'" -ForegroundColor Yellow
    Read-Host "按回车键退出"
    exit 1
}

# 移动到项目根目录
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

# 设置 PYTHONPATH
$env:PYTHONPATH = $ProjectRoot

# 实验配置
$Experiments = @(
    @{
        Dataset = "qasper"
        MaxTokens = 128
        OutputDir = "./runs/qasper_baseline_fullkv"
    },
    @{
        Dataset = "multifieldqa_en"
        MaxTokens = 64
        OutputDir = "./runs/multifieldqa_en_baseline_fullkv"
    },
    @{
        Dataset = "hotpotqa"
        MaxTokens = 32
        OutputDir = "./runs/hotpotqa_baseline_fullkv"
    }
)

Write-Host "模型: $ModelPath" -ForegroundColor Cyan
Write-Host "开始时间: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host ""

# 重置基线目录
if (-not $SkipReset) {
    Write-Host "====================================================================" -ForegroundColor Yellow
    Write-Host "步骤 1/2: 重置基线目录" -ForegroundColor Yellow
    Write-Host "====================================================================" -ForegroundColor Yellow

    foreach ($exp in $Experiments) {
        Write-Host "正在重置 $($exp.Dataset) 基线目录: $($exp.OutputDir)" -ForegroundColor White
        
        if (Test-Path $exp.OutputDir) {
            Write-Host "  删除现有目录: $($exp.OutputDir)" -ForegroundColor Gray
            Remove-Item -Path $exp.OutputDir -Recurse -Force -ErrorAction SilentlyContinue
        }
        
        Write-Host "  创建新目录: $($exp.OutputDir)" -ForegroundColor Gray
        New-Item -Path $exp.OutputDir -ItemType Directory -Force | Out-Null
        Write-Host "  ✅ $($exp.Dataset) 基线目录已重置" -ForegroundColor Green
        Write-Host ""
    }
} else {
    Write-Host "跳过目录重置（使用 -SkipReset 参数）" -ForegroundColor Yellow
    Write-Host ""
}

# 运行实验
Write-Host "====================================================================" -ForegroundColor Yellow
Write-Host "步骤 2/2: 运行基线实验 (每个数据集重复 $Repetitions 次)" -ForegroundColor Yellow
Write-Host "====================================================================" -ForegroundColor Yellow

$TotalExperiments = $Experiments.Count * $Repetitions
$CurrentExperiment = 1
$SuccessCount = 0
$FailCount = 0

for ($round = 1; $round -le $Repetitions; $round++) {
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Magenta
    Write-Host "第 $round 轮实验开始" -ForegroundColor Magenta
    Write-Host "================================================================" -ForegroundColor Magenta
    
    foreach ($exp in $Experiments) {
        Write-Host ""
        Write-Host "[$CurrentExperiment/$TotalExperiments] 运行 $($exp.Dataset) 基线实验 (第$round次)" -ForegroundColor White
        Write-Host "  数据集: $($exp.Dataset)" -ForegroundColor Gray
        Write-Host "  Max New Tokens: $($exp.MaxTokens)" -ForegroundColor Gray
        Write-Host "  输出目录: $($exp.OutputDir)" -ForegroundColor Gray
        Write-Host "  开始时间: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Gray
        Write-Host ""
        
        $StartTime = Get-Date
        
        try {
            # 运行Python脚本
            $arguments = @(
                "evaluation/baselines/fullkvcache_main.py",
                "--model_name", "`"$ModelPath`"",
                "--enable_scoring",
                "--is_baseline_run",
                "--datasets", $exp.Dataset,
                "--kv_cache_lengths", "128,1024",
                "--batch_sizes", "1",
                "--max_new_tokens", $exp.MaxTokens,
                "--repetitions", "1",
                "--output_dir", $exp.OutputDir
            )
            
            $process = Start-Process -FilePath "python" -ArgumentList $arguments -Wait -PassThru -NoNewWindow
            
            if ($process.ExitCode -eq 0) {
                $SuccessCount++
                $Duration = (Get-Date) - $StartTime
                Write-Host "✅ $($exp.Dataset) 基线实验完成 (第$round次)" -ForegroundColor Green
                Write-Host "  结束时间: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Gray
                Write-Host "  耗时: $($Duration.ToString('mm\:ss'))" -ForegroundColor Gray
            } else {
                $FailCount++
                Write-Host "❌ $($exp.Dataset) 基线实验失败！" -ForegroundColor Red
                Write-Host "  退出代码: $($process.ExitCode)" -ForegroundColor Red
                Write-Host "  失败的实验: $($exp.Dataset) (第$round次运行)" -ForegroundColor Red
                
                $continue = Read-Host "是否继续其他实验？ (y/N)"
                if ($continue -ne "y" -and $continue -ne "Y") {
                    Write-Host "实验被用户中止" -ForegroundColor Red
                    exit 1
                }
            }
        } catch {
            $FailCount++
            Write-Host "❌ 运行 $($exp.Dataset) 实验时发生异常:" -ForegroundColor Red
            Write-Host "  错误: $($_.Exception.Message)" -ForegroundColor Red
            
            $continue = Read-Host "是否继续其他实验？ (y/N)"
            if ($continue -ne "y" -and $continue -ne "Y") {
                Write-Host "实验被用户中止" -ForegroundColor Red
                exit 1
            }
        }
        
        $CurrentExperiment++
    }
    
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Magenta
    Write-Host "第 $round 轮实验完成" -ForegroundColor Magenta
    Write-Host "================================================================" -ForegroundColor Magenta
}

# 最终结果
Write-Host ""
Write-Host "====================================================================" -ForegroundColor Green
if ($FailCount -eq 0) {
    Write-Host "🎉 所有基线实验完成！" -ForegroundColor Green
} else {
    Write-Host "⚠️  基线实验完成（包含失败）" -ForegroundColor Yellow
}
Write-Host "====================================================================" -ForegroundColor Green
Write-Host "实验统计:" -ForegroundColor White
Write-Host "  总实验数: $TotalExperiments" -ForegroundColor Gray
Write-Host "  成功: $SuccessCount" -ForegroundColor Green
Write-Host "  失败: $FailCount" -ForegroundColor Red
Write-Host "完成时间: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host ""
Write-Host "结果保存位置:" -ForegroundColor White
Write-Host "  Qasper:        ./runs/qasper_baseline_fullkv" -ForegroundColor Gray
Write-Host "  MultifieldQA:  ./runs/multifieldqa_en_baseline_fullkv" -ForegroundColor Gray
Write-Host "  HotpotQA:      ./runs/hotpotqa_baseline_fullkv" -ForegroundColor Gray
Write-Host ""
Write-Host "下一步:" -ForegroundColor White
Write-Host "  1. 查看实验结果和日志文件" -ForegroundColor Gray
Write-Host "  2. 运行CAKE实验进行对比" -ForegroundColor Gray
Write-Host "  3. 生成性能对比报告" -ForegroundColor Gray
Write-Host "====================================================================" -ForegroundColor Green

Read-Host "按回车键退出" 