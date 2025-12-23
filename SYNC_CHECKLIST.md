# 同步到服务器检查清单

## 需要同步的新文件（3个）
✅ `scripts/run_hace_qmsum.py` - 主运行脚本（已添加日志和文件清理）
✅ `run_hace_experiments.sh` - 自动化实验脚本
✅ `RUNNING_INSTRUCTIONS.md` - 运行指南

## 需要同步的修改文件（1个关键）
✅ `src/third_party/cakekv-main/cakekv-main/cake/model/modify_qwen2.py`
   - 已添加pref_mode验证日志（第135-138行）
   - 已实现三种模式：normal, reverse_disp, reverse

## 同步命令（如果使用rsync）

```bash
# 从Mac本地同步到服务器
rsync -avz --include='scripts/run_hace_qmsum.py' \
           --include='run_hace_experiments.sh' \
           --include='RUNNING_INSTRUCTIONS.md' \
           --include='src/third_party/cakekv-main/cakekv-main/cake/model/modify_qwen2.py' \
           /Users/jerryganst/PycharmProjects/Experiment-Platform/ \
           your-username@your-server:/cloud/cloud-ssd1/Experiment-Platform/
```

或者简单粗暴（同步整个目录）：

```bash
rsync -avz --exclude='.git' --exclude='results_clean' \
  /Users/jerryganst/PycharmProjects/Experiment-Platform/ \
  your-username@your-server:/cloud/cloud-ssd1/Experiment-Platform/
```

## 验证同步成功

在服务器上运行：

```bash
ssh your-server
cd /cloud/cloud-ssd1/Experiment-Platform

# 检查关键文件存在
ls -lh scripts/run_hace_qmsum.py
ls -lh run_hace_experiments.sh
ls -lh RUNNING_INSTRUCTIONS.md

# 检查modify_qwen2.py的修改
grep -A 5 "HACE_PREF_MODE" src/third_party/cakekv-main/cakekv-main/cake/model/modify_qwen2.py
# 应该能看到第133-149行的pref_mode逻辑

# 检查run_hace_qmsum.py的修改
grep -A 3 "Clear old results" scripts/run_hace_qmsum.py
# 应该能看到第117-120行的文件清理逻辑
```

## 快速测试（在服务器上）

```bash
# 测试脚本是否能找到模型
python scripts/run_hace_qmsum.py --help

# 应该看到所有参数说明

# 测试数据文件
head -1 data/qmsum.jsonl | python -m json.tool
```

## 准备运行

```bash
# 给脚本执行权限
chmod +x run_hace_experiments.sh

# 检查GPU
nvidia-smi

# 开始运行（选择其一）

# 选项1: 完整自动化实验
./run_hace_experiments.sh 2>&1 | tee hace_experiments.log

# 选项2: 单独运行一个模式测试
CUDA_VISIBLE_DEVICES=0 python scripts/run_hace_qmsum.py \
  --pref_mode reverse_disp \
  --pred_name test_reverse_disp \
  --device 0
```
