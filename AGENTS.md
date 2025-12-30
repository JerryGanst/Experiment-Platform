# AGENTS.md

These instructions describe how Codex should reason about file paths, project layout, 
the remote execution environment, and model locations for this repository.

Codex must follow these rules for all future interactions within this project.

---

## 1. Workspace & Execution Environment
 
- This project runs **on a remote GPU server** accessed via SSH:
  `root@117.50.34.209` (port 23)
- The **remote interpreter** is Python 3.10+ inside an environment labeled `py312`.
- **All code generated, executed, or inspected by Codex must assume it runs on the remote machine**,
  not on the local macOS host.

---

## 2. Remote Project Root (IMPORTANT)

The canonical remote workspace root is:

```
/cloud/cloud-ssd1/Experiment-Platform
```

Codex must treat this as the base directory for:
- locating project files
- creating new files
- writing logs or outputs
- resolving relative paths

Codex must **NOT** use or reference the previous temporary PyCharm sync directory:
`/tmp/pycharm_project_*`

---

## 3. Repository Structure (Verified)

The repository layout under the project root is:

```
/cloud/cloud-ssd1/Experiment-Platform/
├─ configs/
├─ data/
├─ docs/
├─ download.py
├─ evaluation/
├─ hace_core/
├─ models/
├─ requirements.txt
├─ runs/
├─ scripts/
├─ src/
├─ tests/
└─ AGENTS.md  (this file)
```

Notes:

- `run_experiments.py` (in root or under `scripts/`) is the main experiment entrypoint.
- `models/` is the **project-local cache**.  
  Large models live on the global model disk (`/model`) and are referenced via symlink.

---

## 4. Global Model Storage (Stable Disk)

Large pretrained models are stored on the **separate global model disk** at:

```
/model
```

This includes:
- `/model/HuggingFace`
- `/model/ModelScope`
- `/model/llm`
- `/model/comfyui`
- `/model/ollama`
- ...

To unify access, a symlink exists (or should be created) in the project:

```bash
ln -s /model /cloud/cloud-ssd1/Experiment-Platform/model
```

Codex must:
- reference model paths using **absolute paths**
- prefer the symlinked `model/` folder inside the project root
- avoid writing anything to `/tmp`

---

## 5. Path Rules (STRICT)

Codex must apply the following:

- **Always use absolute paths** when referring to files or directories outside the project root.
- For project files, paths must resolve under:
  `/cloud/cloud-ssd1/Experiment-Platform`
- For models, prefer:
  `/cloud/cloud-ssd1/Experiment-Platform/model/`
  or the underlying: `/model/`
- Codex must *never* assume local macOS paths such as:
  `/Users/jerryganst/…`
- Codex must *not* reference or infer paths under:
  `/tmp/pycharm_project_*`

---

## 6. Running Code

Codex must assume execution happens via the remote interpreter.

SSH connection:
```bash
ssh -p 23 root@117.50.34.209
```

All commands should reflect remote Linux execution.

Example:
```bash
python run_experiments.py --help
pytest tests
bash scripts/*.sh
```

---

## 7. Data & Output Conventions

- Project reads datasets from `data/` unless overridden via environment variables or configuration.
- Model weights default to `model/` (symlink to `/model`)
- Experiment outputs are automatically written under `runs/<timestamp_or_tag>/`
- Codex-generated files should follow the same structure unless explicitly instructed otherwise.

---


## 7.1 HACE 注意力权重与 Glue

- HACE 方法的注意力权重由 `hace_core/models/attention_collector.py` 中的 `AttentionData.save` 保存
- 缓存路径由 `hace_core/config.py` 的 `VLLM_CONFIG["attention_collection"]["cache_attention_file"]` 指定；若为 None 则不会落盘
- 推荐格式：`.pkl/.pickle`（`.json` 体积大且慢），字段包括 `attention_weights_list/num_layers/num_heads/seq_length/model_name/collection_mode`

HACE Glue 预算计算示例：
```bash
python - <<'PY'
from hace_core.models.attention_collector import AttentionData
from hace_core.core import CakeAdaKVIntegration, IntegrationConfig

data = AttentionData.load("/cloud/cloud-ssd1/Experiment-Platform/runs/attn_cache.pkl")
integration = CakeAdaKVIntegration(IntegrationConfig(total_cache_size=4096))
layer_budgets, head_budgets = integration.optimize_cache(data.attention_weights_list)
print(layer_budgets[:4])
PY
```

注意力熵方向验证方案（简版）：
- 固定预算（如 60%），对比“高熵保留” vs “低熵保留”
- 评价：ROUGE-L/PPL + 耗时/预算 + 相关性 `corr(H_attn, delta_LL)`
- 方向判断：`corr(H_attn, delta_LL) > 0` → 高熵更应保留，反之低熵更应保留


## 8. Editing & File Creation Rules

Codex may:
- create new scripts under `scripts/`  
- create experiment configs under `configs/`  
- write analysis / notebooks under `evaluation/analysis/`  
- write doc files under `docs/`

Codex must NOT:
- modify the system directories (`/root`, `/usr`, `/opt`, `/model`)
- write files outside the project root unless explicitly instructed

---

## 9. Sum   mary (What Codex Must Remember)

1. **Remote workspace root = `/cloud/cloud-ssd1/Experiment-Platform`**
2. **Global model root = `/model` → linked into project as `model/`**
3. **Absolute paths only; never use macOS or `/tmp/pycharm_project_*` paths**
4. **Execution happens on the remote GPU interpreter**
5. **All file operations must be scoped to the project root unless otherwise instructed**  

---

## Security (No Secrets In Repo)

- Do not store passwords, API keys, tokens, private keys, or other secrets in this repository (including `AGENTS.md`).
- Use SSH keys for server access; if a secret was ever committed or shared, rotate it immediately.

## 10. HACE 实验架构与计划

### 10.1 CAKE 工作流程理解

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CAKE 完整流程                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  【Prefill阶段】 modify_qwen2.py:128-186                             │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ 1. 计算注意力权重 tmp_attn_weights                           │    │
│  │ 2. 计算层级指标：                                            │    │
│  │    - disp (H) = calculate_entropy(attn[:, :, -W:, :-W])     │    │
│  │    - var (V)  = torch.var(attn).sum()                       │    │
│  │ 3. 计算层级偏好分数：                                         │    │
│  │    pref_score = H^(1/τ1) × V^(1/τ2)                         │    │
│  │ 4. 计算头级evict分数 (hh_score):                             │    │
│  │    attn_mean = attention_score.mean(dim=-2)  # 按query平均   │    │
│  │    attn_var  = attention_score.var(dim=-2)                  │    │
│  │    attn_cache = attn_mean + γ × attn_var                    │    │
│  │    hh_score = attn_cache.mean(dim=-2)  # shape:[bsz,heads,seq]│   │
│  │ 5. 保存分数：                                                 │    │
│  │    cache.update_score(pref_score, hh_score)                 │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ↓                                      │
│  【KV压缩阶段】 cake_cache.py:218-275                                │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ CakeprefillKVCache.__call__():                              │    │
│  │ 1. 层级预算分配：                                            │    │
│  │    layer_budget[i] = pref_score[i] / Σpref_scores × total   │    │
│  │                                                              │    │
│  │ 2. 层内token选择 (evict_layer_kvcache):                      │    │
│  │    # 当前：所有head用相同的topk indices                       │    │
│  │    indices = hh_score.topk(budget, dim=-1).indices          │    │
│  │    # 所有head一起压缩到budget个token                          │    │
│  │    k_compress = key_cache.gather(index=indices)             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  🔴 问题：hh_score是按head分开的，但topk是跨head共享的！              │
│     即：所有head保留相同的token位置                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 10.2 Ada-KV 头级优化核心思想

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Ada-KV 头级差异化                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  CAKE现在的做法：                                                    │
│  ┌───────────────────────────────────────────────────────────┐      │
│  │ layer_budget = 128                                        │      │
│  │ head[0]: 保留token位置 [5,12,45,78,...]  (128个)           │      │
│  │ head[1]: 保留token位置 [5,12,45,78,...]  (128个)  ← 一样！  │      │
│  │ head[2]: 保留token位置 [5,12,45,78,...]  (128个)           │      │
│  │ ...                                                        │      │
│  └───────────────────────────────────────────────────────────┘      │
│                                                                     │
│  Ada-KV应该做的：                                                    │
│  ┌───────────────────────────────────────────────────────────┐      │
│  │ layer_budget = 128                                        │      │
│  │ head[0]: 集中度高(0.9) → 分配80个token  [topk=80]          │      │
│  │ head[1]: 集中度中(0.5) → 分配128个token [topk=128]         │      │
│  │ head[2]: 集中度低(0.2) → 分配176个token [topk=176]         │      │
│  │ ...                                                        │      │
│  │ 约束：Σhead_budget[i] = layer_budget × num_heads           │      │
│  └───────────────────────────────────────────────────────────┘      │
│                                                                     │
│  核心问题：高集中度头应该多分还是少分？                                 │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ 假设A (直觉)：                                               │    │
│  │   集中度高 → 只关注少数token → 可以用更小的cache             │    │
│  │   head_budget ∝ (1 - concentration)                        │    │
│  │                                                              │    │
│  │ 假设B (导师观点)：                                            │    │
│  │   集中度高 → 这些token很重要 → 要确保不丢失，给更大cache       │    │
│  │   head_budget ∝ concentration                               │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### 10.3 实验矩阵

| 实验ID | 层级策略 | 头级策略 | 描述 | 预期 |
|--------|----------|----------|------|------|
| L0 | ❌ Full KV | - | 基线 | 23.64 |
| L1 | H*V→高预算 | ❌均匀 | CAKE原版 | 20.95 |
| L2 | 1/H→高预算 | ❌均匀 | reverse_disp | ≈L1 |
| LH1 | H*V→高预算 | 低集中度→大cache | HACE假设A | ? |
| LH2 | H*V→高预算 | 高集中度→大cache | HACE假设B（导师） | ? |
| LH3 | 1/H→高预算 | 低集中度→大cache | 混合测试 | ? |

### 10.4 改动点识别

```python
# 文件：cake_cache.py 的 evcit_layer_kvcache 方法（第446-527行）

# 当前代码（所有head共享topk）：
def evcit_layer_kvcache(self, past_key_values, layer_idx, budget):
    hh_score = past_key_values.evict_scores[layer_idx]  # [bsz, heads, seq]
    indices = hh_score.topk(budget, dim=-1).indices     # 所有head用同一个budget
    # ... 所有head保留相同位置

# 需要改成（每个head独立topk）：
def evcit_layer_kvcache(self, past_key_values, layer_idx, budget, head_budgets=None):
    hh_score = past_key_values.evict_scores[layer_idx]  # [bsz, heads, seq]

    if head_budgets is None:
        # 原来的均匀分配
        indices = hh_score.topk(budget, dim=-1).indices
    else:
        # 新的头级差异化分配
        for h, h_budget in enumerate(head_budgets):
            indices_h = hh_score[:, h, :].topk(h_budget, dim=-1).indices
            # 每个head独立压缩
```

### 10.5 实验流程

```
Phase 1: 验证头级差异化机制能跑通
├── 1.1 修改 cake_cache.py 支持 head_budgets 参数 ✅ (已有框架)
├── 1.2 修改 modify_qwen2.py 计算 head_concentration
├── 1.3 添加环境变量 HACE_HEAD_MODE={uniform|low_conc|high_conc}
└── 1.4 跑一个小测试确认不报错

Phase 2: 跑完整对比实验
├── 2.1 HACE-LH1: HACE_HEAD_MODE=low_conc (低集中度头→大cache)
├── 2.2 HACE-LH2: HACE_HEAD_MODE=high_conc (高集中度头→大cache)
└── 2.3 对比 L1 vs LH1 vs LH2

Phase 3: 分析结果
├── 3.1 如果 LH1 > L1 → 假设A正确（低集中度需要更多cache）
├── 3.2 如果 LH2 > L1 → 假设B正确（高集中度的token更重要）
└── 3.3 如果都差不多 → 头级差异化可能没用

Phase 4: 接入Glue
└── 4.1 用 hace_core/core/ 的代码替换硬编码逻辑
```

### 10.6 环境配置

当前 GCP 环境配置：
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120, 98GB)
- Python: 3.10
- PyTorch: 2.9.0+cu128
- Transformers: 4.44.2
- CUDA: 12.8

运行命令示例：
```bash
source /mnt/disks/experiment/my_env/bin/activate
export LD_LIBRARY_PATH=/mnt/disks/experiment/my_env/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

# 运行实验
HACE_PREF_MODE=normal HACE_HEAD_MODE=low_entropy python scripts/run_cake_experiment.py
```

## 不要乱改东西