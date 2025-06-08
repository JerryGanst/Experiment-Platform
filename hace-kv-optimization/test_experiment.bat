@echo off
chcp 65001
echo Testing single CAKE experiment...

python baselines/cake_main.py ^
    --model_name "NousResearch/Llama-2-7b-hf" ^
    --datasets "narrativeqa" ^
    --kv_cache_lengths "512" ^
    --batch_sizes "1" ^
    --max_new_tokens "64" ^
    --allocation_strategies "adaptive" ^
    --cache_budgets "0.5" ^
    --repetitions "1" ^
    --seed "42" ^
    --output_dir "test_single"

echo Experiment completed.
pause 