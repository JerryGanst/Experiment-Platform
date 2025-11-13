# 远端 GPU / SSH 部署指南

本指南帮助你在远端 Linux 服务器上配置 Experiment-Platform，重点解决模型、数据与环境变量问题。

---

## 1. 目录结构约定

| 类型 | 建议路径 | 说明 |
|------|----------|------|
| 项目代码 | `~/projects/Experiment-Platform` | `git clone` 后的仓库根目录 |
| 模型权重 | `/mnt/models` 或 `/opt/models` | 由 `CORECODE_MODEL_PATH` 指向 |
| 数据集 | `/mnt/datasets/corecode` | 由 `DATASETS_ROOT` 指向 |
| HF 缓存 | `/mnt/hf-cache` | 可通过 `HF_HOME` 控制 |

---

## 2. 环境变量模板

```bash
# ~/.bashrc 或 .env.corecode
export CORECODE_MODEL_PATH="/mnt/models/mistral-7b-instruct-v0.3"
export HACE_MODEL_PATH="$CORECODE_MODEL_PATH"
export DATASETS_ROOT="/mnt/datasets/corecode"
export HF_HOME="/mnt/hf-cache"
export PYTHONPATH="$HOME/projects/Experiment-Platform:$PYTHONPATH"
```

> `scripts/maintenance/auto_detect_model_path.py` 会生成 `.env.corecode`，SSH 登录后 `source .env.corecode` 即可。

---

## 3. SSH 登录流程示例

```bash
ssh gpu-server <<'REMOTE'
cd ~/projects/Experiment-Platform
source .env.corecode 2>/dev/null || true
export DATASETS_ROOT="/mnt/datasets/corecode"
export PYTHONPATH="$PWD:$PYTHONPATH"
conda activate corecode
python run_experiments.py --experiment full_evaluation --tag nightly_remote
REMOTE
```

---

## 4. 数据同步

1. **远端下载**：`python scripts/download_data.py --dataset hotpotqa --output $DATASETS_ROOT`
2. **本地→远端**：`rsync -av data/*.jsonl server:/mnt/datasets/corecode/`
3. **远端→本地**：`rsync -av server:~/projects/Experiment-Platform/runs ./runs_remote`

---

## 5. 验证脚本

```bash
python -m compileall tests/ -q
pytest tests/test_scoring_fix.py -q
python tests/baselines/test_hotpotqa_extraction.py
```

若以上脚本全部通过，说明模型/数据路径以及依赖均配置正确。

---

## 6. 常见问题

| 问题 | 解决方案 |
|------|----------|
| `hotpotqa.jsonl` 不存在 | 检查 `DATASETS_ROOT`，运行数据下载脚本 |
| GPU 无法加载模型 | 确认 `CORECODE_MODEL_PATH` 指向本地磁盘而非 Windows 路径 |
| Hugging Face 速度慢 | 设置国内镜像或运行 `python scripts/setup_hf_mirror.py` |

---

完成以上步骤后，即可在远端 GPU 上稳定运行所有实验流程。🚀
