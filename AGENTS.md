# AGENTS.md

These instructions describe how Codex should reason about file paths, project layout, 
the remote execution environment, and model locations for this repository.

Codex must follow these rules for all future interactions within this project.

---

## 1. Workspace & Execution Environment

- This project runs **on a remote GPU server** accessed via SSH:
  `root@117.50.175.133` (port 23)
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
ssh -p 23 root@117.50.175.133
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
