# HACE实验运行指南

## 前置条件
1. 确保你在服务器上：`/cloud/cloud-ssd1/Experiment-Platform`
2. 确保GPU可用：`nvidia-smi`
3. 确保已同步以下修改后的文件：
   - `scripts/run_hace_qmsum.py` (已添加日志和清空旧文件)
   - `src/third_party/cakekv-main/cakekv-main/cake/model/modify_qwen2.py` (已添加pref_mode验证日志)
   - `run_hace_experiments.sh` (完整实验脚本)

## 快速验证（单个样本测试）

```bash
cd /cloud/cloud-ssd1/Experiment-Platform

# 测试1: normal模式（5个样本）
CUDA_VISIBLE_DEVICES=0 python -c "
from datasets import load_dataset
data = load_dataset('json', data_files={'test': 'data/qmsum.jsonl'}, split='test')
print(f'Total samples: {len(data)}')
print('First sample:', data[0]['input'][:100])
"
```

## 完整实验运行

### 选项A: 使用自动化脚本（推荐）

```bash
cd /cloud/cloud-ssd1/Experiment-Platform

# 确保脚本可执行
chmod +x run_hace_experiments.sh

# 运行完整对比实验（3个模式）
./run_hace_experiments.sh 2>&1 | tee hace_experiments.log
```

这会依次运行：
1. Normal模式 (高熵→高预算)
2. Reverse_disp模式 (低熵→高预算，导师建议)
3. Reverse模式 (低熵+低方差→高预算)

### 选项B: 单独运行每个实验

```bash
cd /cloud/cloud-ssd1/Experiment-Platform

# 实验1: Normal模式
CUDA_VISIBLE_DEVICES=0 python scripts/run_hace_qmsum.py \
  --model qwen2.5-7b-instruct \
  --cache_size 128 \
  --window_size 32 \
  --pref_mode normal \
  --pred_name hace_normal_128 \
  --device 0

# 评估
python src/third_party/cakekv-main/cakekv-main/experiments/LongBench/eval.py \
  --model qwen2.5-7b-instruct \
  --dir_path results_clean/hace_normal_128

# 实验2: Reverse_disp模式（导师建议）
CUDA_VISIBLE_DEVICES=0 python scripts/run_hace_qmsum.py \
  --model qwen2.5-7b-instruct \
  --cache_size 128 \
  --window_size 32 \
  --pref_mode reverse_disp \
  --pred_name hace_reverse_disp_128 \
  --device 0

# 评估
python src/third_party/cakekv-main/cakekv-main/experiments/LongBench/eval.py \
  --model qwen2.5-7b-instruct \
  --dir_path results_clean/hace_reverse_disp_128
```

## 验证实验正在正确运行

运行时应该看到：
```
[HACE] Using pref_mode: reverse_disp  # 在模型第一次前向传播时打印
============================================================
HACE Experiment Configuration:
  Model: qwen2.5-7b-instruct
  Dataset: qmsum
  Pref Mode: reverse_disp
  Cache Size: 128
  Window Size: 32
  Output: results_clean/hace_reverse_disp_128/qmsum.jsonl
============================================================
```

## 查看结果

```bash
# 查看生成的预测
head -3 results_clean/hace_reverse_disp_128/qmsum.jsonl | jq '.'

# 查看评估分数
cat results_clean/hace_reverse_disp_128/result.json

# 对比所有模式
for mode in normal reverse_disp reverse; do
  echo "=== ${mode} ==="
  cat results_clean/hace_${mode}_128/result.json 2>/dev/null || echo "Not found"
  echo ""
done
```

## 预期输出格式

result.json示例：
```json
{
  "qmsum": {
    "rouge-1": 0.XX,
    "rouge-2": 0.XX,
    "rouge-l": 0.XX
  }
}
```

## 常见问题

### Q: 模型路径找不到
A: 检查 `src/third_party/cakekv-main/cakekv-main/experiments/LongBench/config/model2path.json`，确保路径正确

### Q: CUDA Out of Memory
A: 减少batch size或使用更小的window_size

### Q: pref_mode没生效
A: 检查环境变量：`echo $HACE_PREF_MODE`，应该看到你设置的值

### Q: 结果被污染（重复运行）
A: 现在脚本会自动清空旧文件，不用担心

## 实验完成后

对比三个模式的ROUGE-L分数：
- 如果 reverse_disp > normal：导师是对的
- 如果 normal > reverse_disp：原设计是对的
- 如果差距<1%：熵的方向影响不大
