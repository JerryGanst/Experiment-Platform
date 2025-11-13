# 维护工具目录

这些工具用于项目维护和问题诊断。

## 工具说明

### cleanup_models.py
清理模型缓存，释放磁盘空间。

### fix_model_cache.py  
修复模型缓存损坏问题，重建索引。

### check_model_config.py
检查模型配置是否正确，验证路径和环境变量。

### auto_detect_model_path.py
- 扫描 SSH/Linux 环境常见的模型目录
- 生成 `CORECODE_MODEL_PATH` 的设置建议
- 自动写入 `.env.corecode`，登录远端后可 `source` 加载

## 使用时机
- 模型加载失败时运行 `check_model_config.py`
- 缓存出现问题时运行 `fix_model_cache.py`
- 磁盘空间不足时运行 `cleanup_models.py`
- 首次配置环境时运行 `auto_detect_model_path.py`

## 开发状态
这些工具主要在开发阶段使用，生产环境应该有稳定的配置。
