# H2O实验监控系统使用指南

本文档提供了H2O实验框架中监控系统的详细说明和使用方法。

## 1. 监控系统概述

H2O实验框架的监控系统由以下组件组成：

- **监控管理器 (MonitoringManager)**: 统一管理所有监控组件的核心类
- **GPU监控器 (GPUMonitor)**: 监控GPU内存、利用率和温度
- **系统监控器 (SystemMonitor)**: 监控CPU、系统内存和磁盘使用情况

监控系统设计遵循以下原则：

1. **简单易用**: 通过单一入口启动所有监控功能
2. **可配置性**: 丰富的配置选项适应不同实验需求
3. **统计信息丰富**: 收集多种关键性能指标
4. **容错性**: 各组件独立工作，一个组件失败不影响其他组件

## 2. 快速入门

### 基本使用方法

```python
from h2o_experiment.utils import MonitoringManager

# 初始化监控管理器
monitoring_manager = MonitoringManager(experiment_id="my_experiment")

# 启动所有监控组件
monitoring_manager.start_monitoring()

# 执行实验...

# 停止监控并获取指标
monitoring_manager.stop_monitoring()
metrics = monitoring_manager.get_metrics()

# 保存监控数据到文件
monitoring_manager.save_metrics(output_dir="./results/monitoring")
```

### 配置监控选项

可以通过配置文件控制监控行为：

```python
# 从配置文件导入监控配置
from h2o_experiment.config import MONITORING_CONFIG

# 自定义配置
custom_config = MONITORING_CONFIG.copy()
custom_config["gpu_monitoring_interval"] = 0.5  # 修改GPU监控间隔
custom_config["system_monitoring"] = False  # 禁用系统监控

# 使用自定义配置初始化监控管理器
monitoring_manager = MonitoringManager(config=custom_config)
```

## 3. 监控组件详解

### 3.1 监控管理器 (MonitoringManager)

监控管理器是统一管理所有监控组件的中心类。它负责初始化、启动和停止各个监控组件，以及收集和整合监控数据。

**主要方法**:
- `start_monitoring()`: 启动所有配置的监控组件
- `stop_monitoring()`: 停止所有监控组件并收集数据
- `get_metrics()`: 获取所有监控指标
- `save_metrics()`: 保存监控数据到文件

### 3.2 GPU监控器 (GPUMonitor)

GPU监控器专注于收集GPU相关的性能指标。它使用PyTorch和NVIDIA Management Library (NVML)获取详细信息。

**收集的指标**:
- 内存使用 (分配和保留)
- GPU利用率
- GPU温度
- 各指标的峰值、平均值和标准差

### 3.3 系统监控器 (SystemMonitor)

系统监控器负责收集主机系统的资源使用情况，包括CPU、内存和磁盘。

**收集的指标**:
- CPU使用率
- 内存使用率和可用量
- 磁盘使用情况
- 系统基本信息

## 4. 配置选项

监控系统可通过`config.py`中的`MONITORING_CONFIG`字典进行配置，主要配置选项包括：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| gpu_monitoring | 是否启用GPU监控 | True |
| gpu_monitoring_interval | GPU监控采样间隔(秒) | 0.1 |
| system_monitoring | 是否启用系统监控 | True |
| system_monitoring_interval | 系统监控采样间隔(秒) | 1.0 |
| temperature_monitoring | 是否启用温度监控 | False |
| auto_save | 是否自动保存监控数据 | True |
| monitoring_dir | 监控数据保存目录 | "./monitoring" |

## 5. 实用示例

### 5.1 仅使用GPU监控

```python
from h2o_experiment.utils import GPUMonitor

# 初始化GPU监控器
gpu_monitor = GPUMonitor()

# 启动监控
gpu_monitor.start_monitoring()

# 执行实验...

# 停止监控
gpu_monitor.stop_monitoring()
gpu_stats = gpu_monitor.get_statistics()
print(f"峰值GPU内存: {gpu_stats['device_0']['peak_memory_mb']} MB")
```

### 5.2 运行示例脚本

项目包含一个监控组件示例脚本，可以用来测试监控功能：

```bash
python -m h2o_experiment.utils.monitor_example --model facebook/opt-125m --duration 60
```

参数说明：
- `--model`: 要加载的模型
- `--duration`: 模拟运行时间(秒)
- `--gpu_only`: 仅使用GPU监控
- `--system_only`: 仅使用系统监控
- `--output_dir`: 输出目录

## 6. 排错指南

### 常见问题

1. **找不到GPU设备**:
   - 确保CUDA正确安装并可被PyTorch检测到
   - 检查是否已安装pynvml库

2. **无法启动系统监控**:
   - 确保已安装psutil库: `pip install psutil`

3. **监控数据不完整**:
   - 检查实验持续时间是否足够长
   - 调整监控间隔为更小的值获取更详细数据

### 依赖库问题

监控系统依赖以下库：
- torch (GPU监控)
- pynvml (GPU监控)
- psutil (系统监控)

如果遇到依赖问题，请运行：
```bash
pip install -r h2o_experiment/requirements.txt
```

## 7. 扩展监控系统

如需添加新的监控组件，只需：

1. 创建新的监控类，实现以下方法：
   - `start_monitoring()`
   - `stop_monitoring()`
   - `get_statistics()`
   - `reset_statistics()`

2. 在`MonitoringManager`中注册新组件

监控组件设计遵循统一接口原则，方便扩展。 