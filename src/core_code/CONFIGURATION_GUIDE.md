# CAKE-AdaKV 配置指南

## 概述

CAKE-AdaKV统一集成框架提供了灵活的配置系统，支持多种配置方式以满足不同场景的需求。

## 配置优先级

配置按以下优先级生效（从高到低）：

1. **命令行参数** - 直接在命令行指定的参数
2. **配置文件** - YAML配置文件中的设置
3. **环境预设** - dev/prod环境的预设值
4. **默认值** - 代码中定义的默认值

## 配置方式

### 1. 交互式配置（推荐新手）

```bash
python src/core_code/run_cake_adakv.py
```

通过交互式菜单选择运行模式，系统会引导您完成配置。

### 2. 配置文件方式

创建YAML配置文件（参考 `config_template.yaml`）：

```yaml
cache:
  total_size: 8192  # 或使用 bl: 1024

performance:
  enable_monitoring: true
  enable_auto_tuning: true

thresholds:
  high_dispersion_threshold: 0.8
  high_dynamics_threshold: 0.6
```

使用配置文件运行：

```bash
python src/core_code/launcher.py --config my_config.yaml
```

### 3. 命令行参数

直接通过命令行参数配置：

```bash
python src/core_code/launcher.py --bl 512 --monitor --env dev
```

### 4. 环境预设

使用预定义的环境配置：

- **开发环境** (`--env dev`)：启用所有监控和日志
- **生产环境** (`--env prod`)：优化性能，关闭调试
- **自定义** (`--env custom`)：使用其他参数指定的值

## 完整配置选项

### 基础配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `total_cache_size` | int | 4096 | 总缓存大小（tokens） |
| `bl` | int | - | Budget Limit，会覆盖total_cache_size |

### 性能配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_monitoring` | bool | True | 启用性能监控 |
| `enable_auto_tuning` | bool | True | 启用自动调优 |
| `warmup_samples` | int | 5 | 预热样本数 |
| `performance_tracking` | bool | True | 性能跟踪 |
| `enable_fallback` | bool | True | 启用回退机制 |
| `memory_efficient` | bool | False | 内存高效模式 |

### 实验配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `experiment_mode` | bool | False | 实验模式 |
| `detailed_logging` | bool | False | 详细日志 |
| `v_metric` | str | "var" | V指标计算方式 |

### 算法配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `tau1` | float | 0.1 | H指标温度参数 |
| `tau2` | float | 0.1 | V指标温度参数 |
| `window_size` | int | 32 | 窗口大小 |
| `base_capacity` | int | 512 | 基础容量 |
| `kernel_size` | int | 7 | 卷积核大小 |
| `floor_alpha` | float | 0.5 | 下限系数 |
| `beta` | int | 20 | AdaKV的beta参数 |

### 策略阈值

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `high_dispersion_threshold` | float | 0.7 | H指标高分散阈值 |
| `high_dynamics_threshold` | float | 0.5 | V指标高动态阈值 |
| `key_head_ratio` | float | 0.2 | 关键头比例 |
| `min_budget_ratio` | float | 0.02 | 最小预算比例 |

## 使用示例

### 示例1：快速测试

```bash
# 使用默认配置快速测试
python src/core_code/launcher.py --synthetic --bl 256 --detailed
```

### 示例2：生产环境部署

```bash
# 创建生产配置文件 prod_config.yaml
cat > prod_config.yaml << EOF
cache:
  bl: 4096

performance:
  enable_monitoring: false
  memory_efficient: true
  enable_fallback: true

experiment:
  experiment_mode: false
  detailed_logging: false
EOF

# 使用生产配置运行
python src/core_code/launcher.py --config prod_config.yaml -i attention_weights.npy
```

### 示例3：性能调优

```bash
# 启用自动调优找到最佳缓存大小
python src/core_code/launcher.py --synthetic --auto-tune --warmup-samples 10
```

### 示例4：研究实验

```bash
# 开发环境下进行详细实验
python src/core_code/launcher.py \
  --env dev \
  --bl 512 \
  --experiment-mode \
  --detailed-logging \
  --v-metric entropy \
  --high-dispersion-threshold 0.8 \
  -o experiment_results.json
```

## 配置验证

系统会自动验证配置的合理性：

1. **缓存大小验证**：确保缓存大小大于0
2. **阈值范围验证**：确保阈值在0-1之间
3. **V指标验证**：确保V指标计算方式有效
4. **文件路径验证**：检查输入文件是否存在

## 故障排除

### 常见问题

1. **配置文件加载失败**
   - 检查YAML语法是否正确
   - 确保已安装PyYAML：`pip install pyyaml`

2. **参数冲突**
   - BL参数会覆盖cache-size
   - 环境预设会覆盖某些单独参数

3. **内存不足**
   - 使用`--memory-efficient`模式
   - 减小BL值

## 最佳实践

1. **开发阶段**：使用`--env dev`获取详细信息
2. **测试阶段**：使用配置文件保证可重复性
3. **生产部署**：使用`--env prod`优化性能
4. **基准测试**：使用`run_cake_adakv.py`的基准测试模式

## 扩展配置

如需添加自定义配置：

1. 修改`IntegrationConfig`数据类
2. 更新`launcher.py`的参数解析
3. 在配置模板中添加新选项

```python
# 示例：添加自定义配置
@dataclass
class IntegrationConfig:
    # ... 现有配置 ...
    
    # 自定义配置
    my_custom_option: str = "default_value"
``` 