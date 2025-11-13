# FullCache 实验运行指南（Linux / SSH）

## 🎯 修复概览
- 移除全部 Windows `.bat` 脚本，统一使用 Python 命令
- 评分逻辑（Ground Truth 提取、分类/QA 指标）已全量修复
- `DATASETS_ROOT`、`CORECODE_MODEL_PATH` 控制数据与模型位置，方便在远端 GPU 上执行

---

## ⚙️ 环境准备
```bash
export CORECODE_MODEL_PATH="/mnt/models/mistral-7b-instruct-v0.3"
export HACE_MODEL_PATH="$CORECODE_MODEL_PATH"
export DATASETS_ROOT="/mnt/datasets/corecode"
export PYTHONPATH="$PWD:$PYTHONPATH"
```
可执行 `python scripts/maintenance/auto_detect_model_path.py` 生成 `.env.corecode`。

依赖：`pip install -r evaluation/requirements.txt`

---

## 🚀 运行命令

### 1. 带评分的 FullCache 基线
```bash
python evaluation/baselines/fullkvcache_main.py     --model_name "$CORECODE_MODEL_PATH"     --datasets trec     --kv_cache_lengths 128     --batch_sizes 1     --repetitions 1     --max_new_tokens 50     --enable_scoring
```

### 2. 指定数据集 / KV 长度
```bash
python evaluation/baselines/fullkvcache_main.py     --model_name "$CORECODE_MODEL_PATH"     --datasets hotpotqa     --kv_cache_lengths 64,128,256     --batch_sizes 1     --enable_scoring
```

### 3. 快速调试（无评分）
```bash
python evaluation/baselines/fullkvcache_main.py     --model_name mistralai/Mistral-7B-Instruct-v0.3     --datasets multi_news     --max_new_tokens 128
```

---

## 📁 结果结构
```
fullkvcache_run_20250113_101530/
├── all_fullkvcache_experiments_summary.csv
├── ds_trec_kv128_bs1_rep0/
│   ├── evaluation_results_fullkvcache_trec_kv128_bs1_rep0.json
│   └── fullkvcache_metrics_fullkvcache_trec_kv128_bs1_rep0.json
└── logs/
```

- `evaluation_results_*.json`：生成文本、标准答案、评分详情
- `fullkvcache_metrics_*.json`：吞吐量、延迟、显存

---

## 🔧 参数速查
| 参数 | 说明 | 示例 |
|------|------|------|
| `--model_name` | 本地或 HF Hub 模型 | `$CORECODE_MODEL_PATH` |
| `--datasets` | 逗号分隔列表 | `trec,hotpotqa` |
| `--kv_cache_lengths` | KV 长度 | `64,128,256` |
| `--batch_sizes` | Batch size | `1,2,4` |
| `--max_new_tokens` | 生成长度 | `50` |
| `--enable_scoring` | 启用评分 | 加上即可 |

---

## 🐛 故障排除

| 症状 | 处理方式 |
|------|----------|
| 找不到 `trec.jsonl` | 确保 `DATASETS_ROOT` 包含该文件或运行下载脚本 |
| `ModuleNotFoundError` | 检查 `PYTHONPATH` 是否已包含项目根目录 |
| 评分为 0 | 运行 `tests/test_scoring_fix.py` 验证评分逻辑 |

```bash
pytest tests/test_scoring_fix.py -q
```

---

## ✅ 成功判定
```
✅ 评分模块加载成功
✅ 使用处理后的 reference: Other location
✅ 评分详情: 函数=classification_score, 分数=0.0
✅ Evaluation score: 0.0000
```

现在即可在纯 Linux/SSH 环境下稳定运行 fullcache 实验。🚀
