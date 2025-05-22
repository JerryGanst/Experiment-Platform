# 基线测试执行脚本 - Windows PowerShell版本
Write-Host "开始H2O缓存优化基线测试..." -ForegroundColor Green

# 创建必要的目录
New-Item -Path "./results/baseline" -ItemType Directory -Force | Out-Null
New-Item -Path "./logs" -ItemType Directory -Force | Out-Null

# 设置基础配置参数
$KV_LENGTHS = "128,256,512,1024,2048"  # KV缓存长度
$BATCH_SIZES = "1,4,8,16"  # 批处理大小

# 记录开始时间
$startTime = Get-Date

# 英文数据集基线测试
Write-Host "========== 运行英文数据集测试 ==========" -ForegroundColor Cyan

# 运行MMLU数据集测试
Write-Host "1. 开始测试MMLU数据集..." -ForegroundColor Yellow
python main.py --output_dir "./results/baseline/english/mmlu" --dataset mmlu --language english --kv_cache_lengths $KV_LENGTHS --batch_sizes $BATCH_SIZES
Write-Host "MMLU数据集测试完成" -ForegroundColor Green
Write-Host "------------------------------------------------------------"

# 运行HellaSwag数据集测试
Write-Host "2. 开始测试HellaSwag数据集..." -ForegroundColor Yellow
python main.py --output_dir "./results/baseline/english/hellaswag" --dataset hellaswag --language english --kv_cache_lengths $KV_LENGTHS --batch_sizes $BATCH_SIZES
Write-Host "HellaSwag数据集测试完成" -ForegroundColor Green
Write-Host "------------------------------------------------------------"

# 运行LongBench数据集测试
Write-Host "3. 开始测试LongBench数据集..." -ForegroundColor Yellow
python main.py --output_dir "./results/baseline/english/longbench" --dataset longbench --language english --kv_cache_lengths $KV_LENGTHS --batch_sizes $BATCH_SIZES
Write-Host "LongBench数据集测试完成" -ForegroundColor Green
Write-Host "------------------------------------------------------------"

# 运行SQuAD 2.0数据集测试
Write-Host "4. 开始测试SQuAD 2.0数据集..." -ForegroundColor Yellow
python main.py --output_dir "./results/baseline/english/squad2" --dataset squad2 --language english --kv_cache_lengths $KV_LENGTHS --batch_sizes $BATCH_SIZES
Write-Host "SQuAD 2.0数据集测试完成" -ForegroundColor Green
Write-Host "------------------------------------------------------------"

# 中文数据集基线测试
Write-Host "========== 运行中文数据集测试 ==========" -ForegroundColor Cyan

# 运行CLUEWSC2020数据集测试
Write-Host "5. 开始测试CLUEWSC2020数据集..." -ForegroundColor Yellow
python main.py --output_dir "./results/baseline/chinese/cluewsc2020" --dataset cluewsc2020 --language chinese --kv_cache_lengths $KV_LENGTHS --batch_sizes $BATCH_SIZES
Write-Host "CLUEWSC2020数据集测试完成" -ForegroundColor Green
Write-Host "------------------------------------------------------------"

# 运行C-Eval数据集测试
Write-Host "6. 开始测试C-Eval数据集..." -ForegroundColor Yellow
python main.py --output_dir "./results/baseline/chinese/ceval" --dataset ceval --language chinese --kv_cache_lengths $KV_LENGTHS --batch_sizes $BATCH_SIZES
Write-Host "C-Eval数据集测试完成" -ForegroundColor Green
Write-Host "------------------------------------------------------------"

# 运行RACE-中文数据集测试
Write-Host "7. 开始测试RACE-中文数据集..." -ForegroundColor Yellow
python main.py --output_dir "./results/baseline/chinese/race" --dataset race --language chinese --kv_cache_lengths $KV_LENGTHS --batch_sizes $BATCH_SIZES
Write-Host "RACE-中文数据集测试完成" -ForegroundColor Green
Write-Host "------------------------------------------------------------"

# 运行OpenEval数据集测试
Write-Host "8. 开始测试OpenEval数据集..." -ForegroundColor Yellow
python main.py --output_dir "./results/baseline/chinese/openeval" --dataset openeval --language chinese --kv_cache_lengths $KV_LENGTHS --batch_sizes $BATCH_SIZES
Write-Host "OpenEval数据集测试完成" -ForegroundColor Green
Write-Host "------------------------------------------------------------"

# 计算并显示总运行时间
$endTime = Get-Date
$elapsedTime = $endTime - $startTime
$hours = [math]::Floor($elapsedTime.TotalHours)
$minutes = [math]::Floor($elapsedTime.TotalMinutes) % 60
$seconds = [math]::Floor($elapsedTime.TotalSeconds) % 60

Write-Host "所有基线测试完成!" -ForegroundColor Green
Write-Host "总运行时间: $hours 小时 $minutes 分钟 $seconds 秒"
Write-Host "结果保存在 ./results/baseline 目录"