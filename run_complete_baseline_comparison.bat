@echo off
setlocal enabledelayedexpansion

echo ===============================================
echo      完整基线对比实验流程 - Full KV vs CAKE
echo ===============================================

set TIMESTAMP=%date:~10,4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%

echo 开始时间: %date% %time%
echo 实验流程:
echo 1. 建立Full KV基线 (100分基准)
echo 2. 运行CAKE实验并计算相对分数
echo 3. 生成对比报告
echo.

echo ===============================================
echo          步骤1: 建立Full KV基线
echo ===============================================

echo 运行Full KV基线实验...
call run_fullkv_baseline.bat

if %errorlevel% NEQ 0 (
    echo ❌ Full KV基线建立失败，终止实验流程
    echo 请检查错误信息并重试
    pause
    exit /b 1
)

echo.
echo ✅ Full KV基线建立成功！
echo.

:: 等待一段时间确保基线文件完全写入
timeout /t 3 /nobreak >nul

echo ===============================================
echo          步骤2: 运行CAKE实验
echo ===============================================

echo 运行CAKE实验并计算相对分数...
call run_cake_experiments.bat

if %errorlevel% NEQ 0 (
    echo ❌ CAKE实验失败
    echo 请检查错误信息
    pause
    exit /b 1
)

echo.
echo ✅ CAKE实验完成！
echo.

echo ===============================================
echo          步骤3: 生成对比报告
echo ===============================================

:: 检查基线文件
if exist "hace-kv-optimization\baseline_fullkv.json" (
    echo 📊 当前基线分数:
    type "hace-kv-optimization\baseline_fullkv.json"
    echo.
) else (
    echo ⚠️ 未找到基线分数文件
)

:: 显示结果目录
echo 📁 实验结果位置:
if exist "results\fullkv_baseline" (
    echo Full KV基线结果: results\fullkv_baseline\
)
if exist "results\cake_experiments" (
    echo CAKE实验结果: results\cake_experiments\
)

:: 查找并显示评分报告
echo.
echo 📈 评分报告:
for /r "results" %%f in (*scoring_report.txt) do (
    echo 发现报告: %%f
    echo 内容预览:
    echo ----------------------------------------
    type "%%f"
    echo ----------------------------------------
    echo.
)

echo ===============================================
echo              实验流程完成
echo ===============================================

echo 结束时间: %date% %time%
echo.
echo 📋 总结:
echo - Full KV基线已建立 (100分基准)
echo - CAKE实验已完成并计算相对分数
echo - 可在结果目录中查看详细数据和报告
echo - 相对分数越接近100，表示CAKE越接近Full KV性能
echo.

pause 