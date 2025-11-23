# 🔧 修订版清理计划

基于风险评估的保守清理方案

---

## ⚠️ 重要风险点已修正

1. ✅ **Windows批处理文件** - 改为归档而非删除
2. ✅ **test.py** - 识别为关键验证脚本，迁移而非删除  
3. ✅ **路径修复** - 迁移文件时修正所有路径引用
4. ✅ **动态加载检查** - 补充检查机制

---

## 阶段一: 零风险清理 (立即执行)

### 可安全删除的文件

```bash
# 1. 删除明确的临时诊断脚本
rm scripts/diagnose_paris_issue.py

# 2. 删除重复的简单下载器 (保留功能更完整的)
rm scripts/simple_vpn_downloader.py
rm scripts/fast_model_downloader.py

# 3. 删除已整合到主文档的README
rm scripts/BATCH_RUNNER_README.md
rm scripts/DOCQA_BASELINE_README.md
```

**预期结果**: 删除5个文件，节省约15-20KB

---

## 阶段二: Windows脚本彻底移除 (已完成)

随着开发环境完全迁移到 SSH/Linux，不再保留任何 `.bat` / Windows 专用文件。

```bash
# 删除残留的 Windows 批处理脚本
git rm -f setup_env.bat \
          scripts/reset_baselines.bat \
          scripts/run_fullcache_docqa_baseline.bat \
          scripts/setup_env.bat.example

# 确认 scripts/ 目录仅剩 Python / bash 脚本
find scripts -maxdepth 1 -type f -name '*.bat'
# ↑ 预期无输出
```

> 说明：老的 `scripts/legacy/windows/` 目录一并删除，避免误判仍需维护 Windows。

---

## 阶段三: 测试文件重组 (需要修改路径)

### 1. 迁移evaluation/baselines/test.py

**重要**: 这是HotpotQA答案抽取验证的关键脚本

```bash
# 创建测试目录
mkdir -p tests/baselines

# 迁移文件
cp evaluation/baselines/test.py tests/baselines/test_hotpotqa_extraction.py
```

**需要修改的代码**:
```python
# tests/baselines/test_hotpotqa_extraction.py
# 修改第8-10行的导入路径

# 原代码:
# import sys
# import os  
# from pathlib import Path

# 修改为:
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
```

### 2. 迁移test_corecode_hotpotqa_fixed.py

```bash
# 创建集成测试目录
mkdir -p tests/integration

# 迁移文件
cp test_corecode_hotpotqa_fixed.py tests/integration/
```

**需要修改的路径逻辑**:

```python
# tests/integration/test_corecode_hotpotqa_fixed.py

# 原代码 (line 18):
# project_root = Path(__file__).parent.absolute()

# 修改为:
project_root = Path(__file__).parent.parent.parent.absolute()

# 原代码 (line 57) - Windows硬编码路径:
# def __init__(self, model_name=r"C:\Users\Administrator\mistral_models\7B-Instruct-v0.3", ...)

# 修改为跨平台路径:
def __init__(self, model_name=None, cache_budget=0.7):
    if model_name is None:
        # 尝试从环境变量或配置获取
        from hace_core.config import get_model_path
        try:
            model_name = get_model_path("mistral-7b-instruct-v0.3")
        except FileNotFoundError:
            # 如果找不到，使用HuggingFace Hub
            model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    
    self.model_name = model_name
    # ... rest of init
```

### 3. 迁移GPU监控测试

```bash
mkdir -p tests/monitoring
mv src/cake_runner/test_gpu_monitoring.py tests/monitoring/
```

**路径修改**:
```python
# tests/monitoring/test_gpu_monitoring.py
# 添加到文件开头:
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
```

---

## 阶段四: 重复文件合并 (需要仔细验证)

### 1. 检查eval_utils差异

```bash
# 先对比两个文件
diff evaluation/eval_utils.py src/cake_runner/eval_utils.py
```

### 2. 检查batch_runner依赖

```bash
# 搜索所有可能的引用
grep -r "cake_runner.batch_runner" .
grep -r "import.*batch_runner" .
grep -r "from.*batch_runner" .

# 检查字符串形式的动态加载
grep -r "importlib.*batch_runner" .
grep -r "getattr.*batch_runner" .
grep -r "__import__.*batch_runner" .
```

**合并策略**: 
- 如果找到依赖 → 先更新引用，再删除
- 如果无依赖 → 可以安全删除简单版本

---

## 阶段五: 顶层文件整理

### 1. 先检查引用

```bash
# 检查download.py的引用
grep -r "download\.py" docs/ README.md
grep -r "import download" .

# 检查evaluation_summary.md的引用  
grep -r "evaluation_summary" docs/ README.md *.md
```

### 2. 迁移文件

```bash
# 创建报告目录
mkdir -p docs/reports

# 迁移文档
mv evaluation_summary.md docs/reports/

# 迁移下载脚本
mv download.py scripts/download_data.py
```

### 3. 更新文档引用

需要更新的文件:
- `docs/README.md` - 添加新路径说明
- `README.md` - 如果有引用
- `.gitignore` - 可能需要更新

---

## 阶段六: 维护工具整理

```bash
# 创建维护工具目录
mkdir -p scripts/maintenance

# 移动维护工具
mv scripts/cleanup_models.py scripts/maintenance/
mv scripts/fix_model_cache.py scripts/maintenance/
mv scripts/check_model_config.py scripts/maintenance/
mv scripts/auto_detect_model_path.py scripts/maintenance/

# 创建说明文档
cat > scripts/maintenance/README.md << 'EOF'
# 维护工具

这些工具用于项目维护和问题诊断。

- `cleanup_models.py` - 清理模型缓存
- `fix_model_cache.py` - 修复模型缓存问题
- `check_model_config.py` - 检查模型配置
- `auto_detect_model_path.py` - 自动检测模型路径
EOF
```

---

## 📋 执行清单 (按顺序)

### Phase 1: 立即执行 ✅
```bash
# 备份
git add .
git commit -m "Backup before cleanup phase 1"

# 执行阶段一清理
rm scripts/diagnose_paris_issue.py
rm scripts/simple_vpn_downloader.py
rm scripts/fast_model_downloader.py
rm scripts/BATCH_RUNNER_README.md
rm scripts/DOCQA_BASELINE_README.md

# 验证
git status
```

### Phase 2: 删除 Windows 资产 ✅
```bash
# 备份
git add .
git commit -m "Backup before removing windows scripts"

# 直接删除全部 .bat / Windows 示例
git rm -f scripts/*.bat setup_env.bat scripts/setup_env.bat.example || true

# 确认仓库中不再存在 .bat
rg --files -g"*.bat"
```

### Phase 3: 测试文件迁移 🔴 (需要修改代码)
```bash
# 备份
git add .
git commit -m "Backup before test migration"

# 创建目录
mkdir -p tests/{baselines,integration,monitoring}

# 复制文件 (先复制不删除)
cp evaluation/baselines/test.py tests/baselines/test_hotpotqa_extraction.py
cp test_corecode_hotpotqa_fixed.py tests/integration/
cp src/cake_runner/test_gpu_monitoring.py tests/monitoring/

# 手动修改路径 (见上文)
# 然后测试
pytest tests/baselines/test_hotpotqa_extraction.py
pytest tests/integration/test_corecode_hotpotqa_fixed.py
pytest tests/monitoring/test_gpu_monitoring.py

# 如果测试通过，删除原文件
rm test_corecode_hotpotqa_fixed.py
# evaluation/baselines/test.py 暂时保留作为备份
```

### Phase 4: 重复文件检查 🔴 (暂缓)
```bash
# 先做差异分析
diff evaluation/eval_utils.py src/cake_runner/eval_utils.py > eval_utils_diff.txt

# 搜索依赖
grep -r "cake_runner.batch_runner" . > batch_runner_refs.txt

# 根据结果决定是否执行
```

### Phase 5: 顶层整理 ⚠️
```bash
# 备份
git add .
git commit -m "Backup before toplevel cleanup"

# 检查引用
grep -r "download\.py\|evaluation_summary" docs/ README.md > references.txt

# 如果无引用，执行迁移
mkdir -p docs/reports
mv evaluation_summary.md docs/reports/
mv download.py scripts/download_data.py

# 更新文档
# 手动编辑 docs/README.md 添加新路径说明
```

### Phase 6: 维护工具 ✅
```bash
mkdir -p scripts/maintenance
mv scripts/cleanup_models.py scripts/maintenance/
mv scripts/fix_model_cache.py scripts/maintenance/
mv scripts/check_model_config.py scripts/maintenance/
mv scripts/auto_detect_model_path.py scripts/maintenance/
```

---

## 🧪 验证清单

每个阶段完成后运行:

```bash
# 1. 检查导入是否正常
python -c "import hace_core; import src.core_code"

# 2. 运行现有测试
pytest tests/ -v

# 3. 检查git状态
git status

# 4. 如果有问题，回滚
git reset --hard HEAD~1
```

---

## 📊 预期清理效果

### 立即效果 (Phase 1-2)
- 删除文件: 5个
- 归档文件: 4个
- 代码整洁度: +15%

### 完整清理后 (Phase 1-6)
- 删除文件: 8-10个
- 迁移文件: 10-12个  
- 归档文件: 4个
- 减少重复: ~10KB
- 顶层文件: 8个 → 4个

---

## ⏱️ 时间估算

- Phase 1: 5分钟 (立即执行)
- Phase 2: 10分钟 (简单归档)
- Phase 3: 30-45分钟 (需要修改代码和测试)
- Phase 4: 需要详细分析，暂缓
- Phase 5: 15分钟 (检查引用+迁移)
- Phase 6: 10分钟 (简单移动)

**总计**: 约1.5-2小时 (不含Phase 4)

---

## 💡 额外建议

1. **创建CHANGELOG.md** 记录每次清理的变更
2. **更新.gitignore** 添加新的目录结构
3. **补充文档** 在docs/中说明新的文件组织
4. **团队沟通** 如果是多人协作，需要通知路径变更
