@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

echo ====================================================================
echo 完整CAKE数据集基线实验 - 支持所有22个数据集
echo ====================================================================

REM 检查环境变量
if not defined HACE_MODEL_PATH (
    echo [ERROR] HACE_MODEL_PATH 未设置！
    echo 请先运行: setup_env.bat
    pause
    exit /b 1
)

if not defined PYTHONPATH (
    echo [ERROR] PYTHONPATH 未设置！
    echo 请先运行: setup_env.bat
    pause
    exit /b 1
)

echo [INFO] 环境变量检查通过
echo   HACE_MODEL_PATH = %HACE_MODEL_PATH%
echo   PYTHONPATH = %PYTHONPATH%
echo.

echo ====================================================================
echo CAKE数据集完整列表 (共22个):
echo.
echo 单文档问答:
echo   1. narrativeqa (max_tokens: 128)
echo   2. qasper (max_tokens: 128) 
echo   3. multifieldqa_en (max_tokens: 64)
echo   4. multifieldqa_zh (max_tokens: 64)
echo.
echo 多文档问答:
echo   5. hotpotqa (max_tokens: 32)
echo   6. 2wikimqa (max_tokens: 32)
echo   7. musique (max_tokens: 32)
echo   8. dureader (max_tokens: 128)
echo.
echo 摘要生成:
echo   9. gov_report (max_tokens: 512)
echo   10. qmsum (max_tokens: 512)
echo   11. multi_news (max_tokens: 512)
echo   12. vcsum (max_tokens: 512)
echo.
echo 少样本学习:
echo   13. trec (max_tokens: 64)
echo   14. triviaqa (max_tokens: 32)
echo   15. samsum (max_tokens: 128)
echo   16. lsht (max_tokens: 64)
echo.
echo 合成任务:
echo   17. passage_count (max_tokens: 32)
echo   18. passage_retrieval_en (max_tokens: 32)
echo   19. passage_retrieval_zh (max_tokens: 32)
echo.
echo 代码任务:
echo   20. lcc (max_tokens: 64)
echo   21. repobench-p (max_tokens: 64)
echo ====================================================================

echo 注意: 
echo - 每个数据集将使用CAKE官方配置的max_new_tokens
echo - 将运行KV缓存长度: 128, 1024
echo - 预计总时间: 2-4小时 (取决于数据集大小)
echo ====================================================================

set /p continue="确认开始完整CAKE数据集基线实验? (y/N): "
if /i not "%continue%"=="y" (
    echo 实验已取消
    pause
    exit /b 0
)

echo.
echo 开始批量实验...
echo 开始时间: %time%
echo ====================================================================

REM 定义数据集数组
set datasets=narrativeqa qasper multifieldqa_en multifieldqa_zh hotpotqa 2wikimqa musique dureader gov_report qmsum multi_news vcsum trec triviaqa samsum lsht passage_count passage_retrieval_en passage_retrieval_zh lcc repobench-p

set count=0
for %%d in (%datasets%) do (
    set /a count+=1
    echo.
    echo [!count!/21] 开始 %%d 基线实验...
    echo 时间: %time%
    echo ====================================================================
    
    python evaluation/baselines/fullkvcache_main.py ^
        --model_name "%HACE_MODEL_PATH%" ^
        --enable_scoring ^
        --is_baseline_run ^
        --datasets %%d ^
        --kv_cache_lengths 128,1024 ^
        --batch_sizes 1 ^
        --repetitions 1 ^
        --output_dir ./runs/%%d_baseline_fullkv
    
    if errorlevel 1 (
        echo [ERROR] %%d 实验失败！
        echo 继续下一个数据集...
        echo.
    ) else (
        echo [SUCCESS] %%d 完成！时间: %time%
        echo.
    )
)

echo ====================================================================
echo 🎉 所有CAKE数据集基线实验完成！
echo ====================================================================
echo 结果保存在 ./runs/ 目录下的各个数据集文件夹中
echo.
echo 完成时间: %time%
echo ====================================================================

REM 生成结果汇总
echo.
echo 正在生成结果汇总...
echo ====================================================================

echo 实验结果汇总: > results_summary.txt
echo 执行时间: %date% %time% >> results_summary.txt
echo. >> results_summary.txt

for %%d in (%datasets%) do (
    if exist "./runs/%%d_baseline_fullkv/" (
        echo [✓] %%d - 成功完成 >> results_summary.txt
    ) else (
        echo [✗] %%d - 失败或跳过 >> results_summary.txt
    )
)

echo. >> results_summary.txt
echo 详细结果请查看各数据集目录下的日志文件 >> results_summary.txt

echo 结果汇总已保存到: results_summary.txt
type results_summary.txt

pause 