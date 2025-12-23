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

## 不要乱改东西
# AGENTS.md
####