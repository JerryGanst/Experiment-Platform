@echo off
echo 设置环境变量...
set PYTHONPATH=C:\Users\Administrator\PycharmProjects3
set HACE_MODEL_PATH=C:\Users\Administrator\mistral_models\7B-Instruct-v0.3

echo 验证路径设置...
echo PYTHONPATH=%PYTHONPATH%
echo HACE_MODEL_PATH=%HACE_MODEL_PATH%

echo 运行fullcache实验...
python evaluation/baselines/fullkvcache_main.py --model_name "C:\Users\Administrator\mistral_models\7B-Instruct-v0.3" --datasets trec --kv_cache_lengths 128 --batch_sizes 1 --repetitions 1 --max_new_tokens 50

pause 