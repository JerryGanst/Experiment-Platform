# HACE vs CAKE 对比实验脚本 (Windows PowerShell版本)
# 基于baseline论文的实验设置设计
# 匹配: 多种LLM架构 + LongBench + NeedleBench + 多种上下文长度

param(
    [Parameter(HelpMessage="设置实验阶段: 1=MHA核心实验, 2=GQA对比, 3=LongBench全面评估, 4=NeedleBench检索测试, all=全部阶段")]
    [string]$Stage = "all",
    
    [Parameter(HelpMessage="设置输出目录")]
    [string]$OutputDir = "hace_vs_cake_results",
    
    [Parameter(HelpMessage="启用详细输出模式")]
    [switch]$VerboseMode,
    
    [Parameter(HelpMessage="跳过GPU缓存清理")]
    [switch]$SkipCleanup
)

Write-Host "开始HACE vs CAKE对比实验 (匹配baseline论文设置)" -ForegroundColor Green
Write-Host "开始时间: $(Get-Date)" -ForegroundColor Yellow

# 设置错误处理
$ErrorActionPreference = "Continue"

# 创建实验结果目录
if (!(Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}
Set-Location $OutputDir

# =============================================================================
# 实验参数设置 (基于baseline论文)
# =============================================================================

# 模型选择: 使用与baseline一致的模型
$MHA_MODELS = @(
    "NousResearch/Llama-2-7b-hf",
    "facebook/opt-1.3b"
)

$GQA_MODELS = @(
    "NousResearch/Llama-2-7b-hf",
    "facebook/opt-2.7b",
    "microsoft/DialoGPT-large"
)

# 上下文长度: 从实际可行的范围开始，逐步扩展
$CONTEXTS = @(512, 1024, 2048, 4096)
# $CONTEXTS_EXTENDED = @(8192, 16384, 32768)  # 云服务器扩展范围，按需取消注释

# 缓存预算: 匹配baseline的memory budget范围
$BUDGETS = @(0.1, 0.2, 0.3, 0.5, 0.7, 1.0)

# LongBench任务分类 (基于baseline论文的6大类)
$LONGBENCH_SINGLE_DOC_QA = @("narrativeqa", "qasper", "multifieldqa_en")
$LONGBENCH_MULTI_DOC_QA = @("hotpotqa", "2wikimqa", "musique")
$LONGBENCH_SUMMARIZATION = @("gov_report", "qmsum", "multi_news")
$LONGBENCH_FEWSHOT = @("trec", "triviaqa", "samsum")
$LONGBENCH_SYNTHETIC = @("passage_count", "passage_retrieval_en")
$LONGBENCH_CODE = @("lcc", "repobench-p")

# NeedleBench任务 (基于baseline论文的3个子任务)
$NEEDLEBENCH_TASKS = @("single_needle", "multi_needle_retrieval", "multi_needle_reasoning")

# 实验统计变量
$script:TotalExperiments = 0
$script:CompletedExperiments = 0
$script:FailedExperiments = 0

# 辅助函数: GPU缓存清理
function Clear-GPUCache {
    if (-not $SkipCleanup) {
        try {
            python -c "import torch; torch.cuda.empty_cache(); print('GPU缓存已清理')" 2>$null
            Start-Sleep -Seconds 1
        } catch {
            Write-Warning "GPU缓存清理失败: $_"
        }
    }
}

# 辅助函数: 运行单个实验
function Run-Experiment {
    param(
        [string]$Model,
        [string]$Dataset,
        [int]$Context,
        [double]$Budget,
        [string]$OutputPath,
        [string]$ExperimentType = "CAKE"
    )
    
    $script:TotalExperiments++
    
    Write-Host "运行实验 $script:CompletedExperiments/$script:TotalExperiments`: $Model - $Dataset - ${Context}tokens - ${Budget}budget" -ForegroundColor Cyan
    
    # 确保输出目录存在
    $OutputFullPath = Join-Path (Get-Location) $OutputPath
    if (!(Test-Path (Split-Path $OutputFullPath))) {
        New-Item -ItemType Directory -Path (Split-Path $OutputFullPath) -Force | Out-Null
    }
    
    try {
        $pythonArgs = @(
            "../baselines/cake_main.py",
            "--model_name", $Model,
            "--datasets", $Dataset,
            "--kv_cache_lengths", $Context,
            "--batch_sizes", "1",
            "--max_new_tokens", "256",
            "--allocation_strategies", "adaptive",
            "--cache_budgets", $Budget,
            "--repetitions", "1",
            "--seed", "42",
            "--output_dir", $OutputPath
        )
        
        $process = Start-Process -FilePath "python" -ArgumentList $pythonArgs -Wait -PassThru -NoNewWindow
        
        if ($process.ExitCode -eq 0) {
            $script:CompletedExperiments++
            Write-Host "✓ 实验成功完成" -ForegroundColor Green
        } else {
            $script:FailedExperiments++
            Write-Warning "✗ 实验失败，退出代码: $($process.ExitCode)"
        }
    } catch {
        $script:FailedExperiments++
        Write-Error "实验执行错误: $_"
    }
    
    Clear-GPUCache
}

# =============================================================================
# 阶段1: 核心对比实验 (Multi-head Attention模型)
# =============================================================================

function Run-Stage1 {
    Write-Host "`n=== 阶段1: Multi-head Attention模型核心实验 ===" -ForegroundColor Magenta
    
    foreach ($model in $MHA_MODELS) {
        Write-Host "测试模型: $model" -ForegroundColor Yellow
        
        # LongBench单文档QA任务 (核心能力测试)
        foreach ($task in $LONGBENCH_SINGLE_DOC_QA) {
            foreach ($context in $CONTEXTS) {
                foreach ($budget in $BUDGETS) {
                    $modelSafe = $model -replace '[/\\:]', '_'
                    $outputPath = "mha_results/$modelSafe/${task}_ctx${context}_bud${budget}"
                    Run-Experiment -Model $model -Dataset $task -Context $context -Budget $budget -OutputPath $outputPath
                }
            }
        }
    }
}

# =============================================================================
# 阶段2: Grouped-query Attention模型对比
# =============================================================================

function Run-Stage2 {
    Write-Host "`n=== 阶段2: Grouped-query Attention模型对比实验 ===" -ForegroundColor Magenta
    
    foreach ($model in $GQA_MODELS) {
        Write-Host "测试模型: $model" -ForegroundColor Yellow
        
        # 重点测试NarrativeQA (代表性长文档QA任务)
        foreach ($context in $CONTEXTS) {
            foreach ($budget in $BUDGETS) {
                $modelSafe = $model -replace '[/\\:]', '_'
                $outputPath = "gqa_results/$modelSafe/narrativeqa_ctx${context}_bud${budget}"
                Run-Experiment -Model $model -Dataset "narrativeqa" -Context $context -Budget $budget -OutputPath $outputPath
            }
        }
    }
}

# =============================================================================
# 阶段3: LongBench全面评估 (使用最佳模型)
# =============================================================================

function Run-Stage3 {
    Write-Host "`n=== 阶段3: LongBench全面评估 ===" -ForegroundColor Magenta
    
    # 选择OPT-1.3b作为主要测试模型 (开放访问，平衡性能和可用性)
    $MAIN_MODEL = "facebook/opt-1.3b"
    $MAIN_CONTEXT = 2048  # 平衡长度
    $MAIN_BUDGETS = @(0.2, 0.5, 1.0)  # 关键预算点对比
    
    Write-Host "使用主要模型进行全面评估: $MAIN_MODEL" -ForegroundColor Yellow
    
    # 测试所有LongBench任务类别
    $ALL_LONGBENCH_TASKS = $LONGBENCH_SINGLE_DOC_QA + $LONGBENCH_MULTI_DOC_QA + $LONGBENCH_SUMMARIZATION + $LONGBENCH_FEWSHOT + $LONGBENCH_SYNTHETIC + $LONGBENCH_CODE
    
    foreach ($task in $ALL_LONGBENCH_TASKS) {
        foreach ($budget in $MAIN_BUDGETS) {
            $outputPath = "longbench_full/${task}_ctx${MAIN_CONTEXT}_bud${budget}"
            Run-Experiment -Model $MAIN_MODEL -Dataset $task -Context $MAIN_CONTEXT -Budget $budget -OutputPath $outputPath
        }
    }
}

# =============================================================================
# 阶段4: NeedleBench检索推理测试
# =============================================================================

function Run-Stage4 {
    Write-Host "`n=== 阶段4: NeedleBench检索推理测试 ===" -ForegroundColor Magenta
    
    $MAIN_MODEL = "NousResearch/Llama-2-7b-hf"
    # NeedleBench需要更长的上下文来体现检索能力
    $NEEDLE_CONTEXTS = @(1024, 2048, 4096)
    $NEEDLE_BUDGETS = @(0.1, 0.2, 0.3, 0.5)  # 重点测试压缩对检索的影响
    
    foreach ($task in $NEEDLEBENCH_TASKS) {
        foreach ($context in $NEEDLE_CONTEXTS) {
            foreach ($budget in $NEEDLE_BUDGETS) {
                $outputPath = "needlebench/${task}_ctx${context}_bud${budget}"
                Run-Experiment -Model $MAIN_MODEL -Dataset $task -Context $context -Budget $budget -OutputPath $outputPath
            }
        }
    }
}

# =============================================================================
# 主执行逻辑
# =============================================================================

# 预计算总实验数量
function Calculate-TotalExperiments {
    $total = 0
    
    if ($Stage -eq "all" -or $Stage -eq "1") {
        # 阶段1: 2个MHA模型 * 3个任务 * 4个上下文 * 6个预算
        $total += $MHA_MODELS.Count * $LONGBENCH_SINGLE_DOC_QA.Count * $CONTEXTS.Count * $BUDGETS.Count
    }
    
    if ($Stage -eq "all" -or $Stage -eq "2") {
        # 阶段2: 3个GQA模型 * 1个任务 * 4个上下文 * 6个预算  
        $total += $GQA_MODELS.Count * 1 * $CONTEXTS.Count * $BUDGETS.Count
    }
    
    if ($Stage -eq "all" -or $Stage -eq "3") {
        # 阶段3: 1个模型 * 18个任务 * 1个上下文 * 3个预算
        $ALL_LONGBENCH_TASKS = $LONGBENCH_SINGLE_DOC_QA + $LONGBENCH_MULTI_DOC_QA + $LONGBENCH_SUMMARIZATION + $LONGBENCH_FEWSHOT + $LONGBENCH_SYNTHETIC + $LONGBENCH_CODE
        $total += 1 * $ALL_LONGBENCH_TASKS.Count * 1 * 3
    }
    
    if ($Stage -eq "all" -or $Stage -eq "4") {
        # 阶段4: 1个模型 * 3个任务 * 3个上下文 * 4个预算
        $total += 1 * $NEEDLEBENCH_TASKS.Count * 3 * 4
    }
    
    return $total
}

$script:TotalExperiments = Calculate-TotalExperiments
Write-Host "预计总实验数量: $($script:TotalExperiments)" -ForegroundColor Yellow

# 根据参数运行相应阶段
switch ($Stage) {
    "1" { Run-Stage1 }
    "2" { Run-Stage2 }
    "3" { Run-Stage3 }
    "4" { Run-Stage4 }
    "all" { 
        Run-Stage1
        Run-Stage2
        Run-Stage3
        Run-Stage4
    }
    default { 
        Write-Error "无效的阶段参数: $Stage. 请使用 1, 2, 3, 4, 或 all"
        exit 1
    }
}

# =============================================================================
# 实验完成统计
# =============================================================================

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "所有实验完成!" -ForegroundColor Green
Write-Host "完成时间: $(Get-Date)" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Green

# 统计实验结果
Write-Host "`n实验结果统计:" -ForegroundColor Cyan
Write-Host "- 总实验数: $($script:TotalExperiments)" -ForegroundColor White
Write-Host "- 成功完成: $($script:CompletedExperiments)" -ForegroundColor Green
Write-Host "- 失败实验: $($script:FailedExperiments)" -ForegroundColor Red

# 统计结果文件
try {
    $mhaResults = (Get-ChildItem -Path "mha_results" -Filter "*.json" -Recurse -ErrorAction SilentlyContinue).Count
    $gqaResults = (Get-ChildItem -Path "gqa_results" -Filter "*.json" -Recurse -ErrorAction SilentlyContinue).Count
    $longbenchResults = (Get-ChildItem -Path "longbench_full" -Filter "*.json" -Recurse -ErrorAction SilentlyContinue).Count
    $needlebenchResults = (Get-ChildItem -Path "needlebench" -Filter "*.json" -Recurse -ErrorAction SilentlyContinue).Count
    
    Write-Host "- MHA模型结果文件: $mhaResults 个" -ForegroundColor White
    Write-Host "- GQA模型结果文件: $gqaResults 个" -ForegroundColor White
    Write-Host "- LongBench全面评估文件: $longbenchResults 个" -ForegroundColor White
    Write-Host "- NeedleBench检索测试文件: $needlebenchResults 个" -ForegroundColor White
} catch {
    Write-Warning "无法统计结果文件数量: $_"
}

Write-Host "`n结果保存路径: $((Get-Location).Path)" -ForegroundColor Yellow
Write-Host "可以开始分析实验结果，对比HACE与baseline性能差异" -ForegroundColor Green

# 返回到原始目录
Set-Location .. 