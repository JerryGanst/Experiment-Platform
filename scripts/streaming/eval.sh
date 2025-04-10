method=$1
if [[ ${method} == 'h2o' ]]; then
    CUDA_VISIBLE_DEVICES=0 python run_streaming.py \
        --model_name gpt2 \
        --enable_streaming_with_H2O \
        --heavy_hitter_size 48 \
        --recent_size 2000
elif [[ ${method} == 'full' ]]; then
    CUDA_VISIBLE_DEVICES=0 python run_streaming.py \
        --model_name gpt2
else
    echo 'unknown argument for method'
fi
