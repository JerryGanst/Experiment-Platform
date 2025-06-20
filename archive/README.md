# Archive 文件夹说明

这个文件夹包含了在项目开发过程中使用过的调试、修复和测试脚本。这些文件已经完成了它们的使命，为了保持主目录的整洁而移动到这里。

## 📁 文件清单

### 🔧 调试和修复脚本
- **`debug_eval_utils.py`** - 调试eval_utils.py路径问题的脚本
- **`debug_scoring.py`** - 调试评分系统的测试脚本
- **`fix_scoring_integration.py`** - 修复fullkvcache_main.py中评分系统集成的脚本
- **`repairy.py`** - 修复基线文件数据结构的脚本

### 🌐 数据处理脚本
- **`longbench_official_fix.py`** - LongBench官方数据修复脚本，解决预处理问题
- **`download_correct_hotpotqa.py`** - 下载正确HotpotQA验证集的脚本

### 🧪 测试脚本
- **`test_data_paths.py`** - 数据路径修复验证脚本

### 🚀 批处理文件
- **`run_optimized_experiments.bat`** - 优化版实验启动脚本（引用了不存在的文件）

### 📄 其他文件
- **`shuoming.md`** - 空的说明文件

## 🗂️ 保留原因

这些文件虽然不再直接使用，但保留的原因包括：

1. **历史参考** - 记录了问题解决的过程和方法
2. **调试参考** - 如果将来遇到类似问题，可以参考这些解决方案
3. **代码复用** - 某些功能可能在将来的开发中有用
4. **学习价值** - 展示了从问题发现到解决的完整过程

## 🚮 清理建议

如果确认不再需要这些文件，可以安全删除：
- 所有调试脚本都已完成使命
- 数据处理脚本的功能已集成到主脚本中
- 测试脚本已验证完成

---
*移动时间: 2025-06-18*  
*移动原因: 主目录清理，保持项目结构整洁* 