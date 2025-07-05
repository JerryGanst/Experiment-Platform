@echo off
chcp 65001 >nul

echo ====================================================================
echo 重置基线实验结果
echo ====================================================================

echo [INFO] 删除之前的基线实验结果...

REM 删除基线结果目录
if exist "runs\qasper_baseline_fullkv" (
    echo [INFO] 删除 Qasper 基线结果...
    rmdir /s /q "runs\qasper_baseline_fullkv"
)

if exist "runs\multifieldqa_en_baseline_fullkv" (
    echo [INFO] 删除 MultifieldQA EN 基线结果...
    rmdir /s /q "runs\multifieldqa_en_baseline_fullkv"
)

if exist "runs\hotpotqa_baseline_fullkv" (
    echo [INFO] 删除 HotpotQA 基线结果...
    rmdir /s /q "runs\hotpotqa_baseline_fullkv"
)

REM 删除全局基线结果文件
if exist "evaluation\baseline_fullkv.json" (
    echo [INFO] 删除全局基线配置文件...
    del "evaluation\baseline_fullkv.json"
)

echo.
echo ✅ 基线重置完成！
echo ====================================================================
echo 现在可以运行完整的基线实验了：
echo.
echo 1. 设置环境变量: setup_env.bat
echo 2. 运行实验:
echo    - cd scripts
echo    - run_qasper_baseline.bat
echo    - run_multifieldqa_en_baseline.bat  
echo    - run_hotpotqa_baseline.bat
echo.
echo 注意：完整数据集会需要更长时间！
echo ====================================================================
pause 