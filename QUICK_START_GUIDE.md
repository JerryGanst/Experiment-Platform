# 快速开始指南 - 基线实验

## 🚨 问题解决方案

### 问题1: ModuleNotFoundError: No module named 'hace_core'
**原因**: PYTHONPATH环境变量未正确设置

### 问题2: 环境变量设置复杂
**原因**: 需要在每次运行前正确设置多个环境变量

---

## 🛠️ 解决方案

### 步骤1: 使用环境变量设置脚本

```batch
# 在项目根目录运行
setup_env.bat
```

这个脚本会：
- 自动设置 `HACE_MODEL_PATH`
- 自动设置 `PYTHONPATH`
- 验证模型路径存在性

### 步骤2: 测试环境设置

```batch
# 验证环境是否正确
test_env.bat
```

应该看到所有测试通过：
```
✅ hace_core 导入成功
✅ hace_core.config 导入成功
✅ hace_core/__init__.py 存在
✅ hace_core/config.py 存在
```

### 步骤3: 运行基线实验

**重要**: 必须在同一个命令窗口中运行！

```batch
# 方式1: 运行单个实验
cd scripts
run_qasper_baseline.bat

# 方式2: 运行所有实验
run_remaining_baselines.bat
```

---

## 📋 完整操作流程

### 一键运行（推荐）

```batch
# 1. 打开命令提示符 (CMD, 不是PowerShell)
# 2. 切换到项目目录
cd C:\Users\Administrator\PycharmProjects3

# 3. 设置环境变量
setup_env.bat

# 4. 测试环境 (可选)
test_env.bat

# 5. 运行剩余实验
run_remaining_baselines.bat
```

### 手动分别运行

```batch
# 1. 设置环境变量
setup_env.bat

# 2. 运行各个实验 (在同一命令窗口中)
cd scripts
run_qasper_baseline.bat

# 3. 等待完成后运行下一个
run_multifieldqa_en_baseline.bat

# 4. 最后运行HotpotQA (如果需要重新运行)
run_hotpotqa_baseline.bat
```

---

## 🔧 故障排除

### 问题: 仍然报告模块未找到

**解决方案**:
1. 确保在 **CMD** 中运行，不是 PowerShell
2. 先运行 `setup_env.bat`
3. 在同一窗口中运行实验脚本

### 问题: 模型路径错误

**解决方案**:
修改 `setup_env.bat` 中的路径：
```batch
set HACE_MODEL_PATH=你的模型路径
```

### 问题: Python版本或虚拟环境

**解决方案**:
确保激活正确的conda环境：
```batch
conda activate base  # 或者你的环境名
```

---

## 📊 实验配置

| 数据集 | Max Tokens | KV Cache | 匹配CAKE |
|--------|------------|----------|----------|
| Qasper | 128 | 128,1024 | ✅ |
| MultifieldQA | 64 | 128,1024 | ✅ |
| HotpotQA | 32 | 128,1024 | ✅ |

---

## 📁 输出位置

实验结果将保存到：
- `./runs/qasper_baseline_fullkv/`
- `./runs/multifieldqa_en_baseline_fullkv/`
- `./runs/hotpotqa_baseline_fullkv/`

每个目录包含：
- 实验日志文件
- 性能指标数据
- 评分结果
- 汇总报告

---

## ⏰ 预计时间

- 每个实验: 约1-2分钟
- 总计: 约6-10分钟 (包含3次重复)

---

## 🎯 下一步

实验完成后：
1. 查看结果文件
2. 运行CAKE实验进行对比
3. 生成性能分析报告 