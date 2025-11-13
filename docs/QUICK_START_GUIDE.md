# 快速开始指南 - Linux / SSH 环境

本指南假设你通过 SSH 登录远端 GPU 服务器，不再提供任何 Windows `.bat` 脚本。所有命令均可直接粘贴到 Bash / Zsh。

---

## 🚨 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `ModuleNotFoundError: hace_core` | `PYTHONPATH` 未包含项目根目录 | 在 shell 中 `export PYTHONPATH=$PWD:$PYTHONPATH` |
| 找不到模型/数据 | 没有设置 `CORECODE_MODEL_PATH` / `DATASETS_ROOT` | 参考下文环境变量配置 |
| 评分脚本读取失败 | 数据集路径写死为 `./data` | 通过 `DATASETS_ROOT` 统一指向远端共享目录 |

---

## 🛠️ 环境准备

### 1. SSH 登录并载入虚拟环境
```bash
ssh gpu-server
conda activate corecode  # 或你的虚拟环境
cd ~/projects/Experiment-Platform
```

### 2. 设置环境变量
```bash
export CORECODE_MODEL_PATH="/mnt/models/mistral-7b-instruct-v0.3"
export HACE_MODEL_PATH="$CORECODE_MODEL_PATH"
export DATASETS_ROOT="/mnt/datasets/corecode"
export HF_HOME="/mnt/hf-cache"
export PYTHONPATH="$PWD:$PYTHONPATH"
```
> 提示：`scripts/maintenance/auto_detect_model_path.py` 会生成 `.env.corecode`，SSH 登录后 `source .env.corecode` 即可。

### 3. 安装依赖
```bash
pip install -r requirements.txt
pip install -r evaluation/requirements.txt
```

---

## 🚀 运行实验

### A. 干运行（确认命令）
```bash
python run_experiments.py --experiment full_evaluation --dry-run
```

### B. 运行基线
```bash
python run_experiments.py     --experiment baseline     --datasets hotpotqa,multi_news     --kv_lengths 128,1024     --tag baseline_remote
```

### C. 运行 CAKE / H2O
```bash
python run_experiments.py     --experiment cake     --allocation_strategies adaptive     --cache_budgets 0.5,0.7,0.9     --tag cake_remote
```

### D. 运行对比分析
```bash
python run_experiments.py --experiment comparison --tag prod_vs_baseline
```

---

## 📂 数据 & 结果

- 输入：`DATASETS_ROOT`（例如 `/mnt/datasets/corecode`）
- 输出：`runs/<tag>/...`，可用 `rsync` 回传本地

```bash
ls -1 runs | tail
```

---

## 🔧 故障排除

| 症状 | 处理方式 |
|------|----------|
| 缺少数据集 | `python scripts/download_data.py --dataset hotpotqa --output $DATASETS_ROOT` |
| `trec.jsonl` 未找到 | 设置 `DATASETS_ROOT` 或软链接到 `data/` |
| 模型路径错误 | 重新运行 `scripts/maintenance/auto_detect_model_path.py` 并 `source .env.corecode` |

```bash
python -m compileall tests/ -q
pytest tests/test_scoring_fix.py -q
```

---

## ✅ 一键流程示例
```bash
ssh gpu-server <<'REMOTE'
cd ~/projects/Experiment-Platform
source .env.corecode 2>/dev/null || true
export DATASETS_ROOT="/mnt/datasets/corecode"
export PYTHONPATH="$PWD:$PYTHONPATH"
conda activate corecode
python run_experiments.py --experiment full_evaluation --tag nightly_$(date +%F)
REMOTE
```

按照以上步骤即可在纯 Linux / SSH 环境完成基线与优化实验。🚀
