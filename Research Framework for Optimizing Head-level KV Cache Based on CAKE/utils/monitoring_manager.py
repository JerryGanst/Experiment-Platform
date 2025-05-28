"""
监控管理器模块

提供统一的接口管理所有监控组件，包括GPU监控、系统状态监控和温度监控
"""
import os
import logging
import time
import threading
import json
from datetime import datetime

# 假设GPU监控器已在其他模块中实现
try:
    from .gpu_monitor import GPUMonitor
    gpu_monitor_available = True
except ImportError:
    gpu_monitor_available = False

# 假设有系统监控模块
try:
    from .system_monitor import SystemMonitor
    system_monitor_available = True
except ImportError:
    system_monitor_available = False

# 假设有温度监控模块
try:
    from .temperature_monitor import TemperatureMonitor
    temperature_monitor_available = True
except ImportError:
    temperature_monitor_available = False

logger = logging.getLogger(__name__)

class MonitoringManager:
    """监控管理器类，统一管理各种监控组件"""
    
    def __init__(self, config=None, experiment_id=None):
        """
        初始化监控管理器
        
        Args:
            config: 监控配置字典，如果为None则使用默认配置
            experiment_id: 实验ID，用于标识此次实验的监控数据
        """
        from config import MONITORING_CONFIG
        
        self.config = config or MONITORING_CONFIG
        self.experiment_id = experiment_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.monitors = {}
        self.active = False
        self.metrics = {}
        self.start_time = None
        
        logger.info(f"初始化监控管理器，实验ID: {self.experiment_id}")
        
        # 根据配置初始化各种监控组件
        self._init_monitors()
    
    def _init_monitors(self):
        """初始化各种监控组件"""
        # 初始化GPU监控
        if self.config.get("gpu_monitoring", True) and gpu_monitor_available:
            try:
                interval = self.config.get("gpu_monitoring_interval", 0.1)
                self.monitors["gpu"] = GPUMonitor(
                    monitoring_interval=interval
                )
                logger.info(f"GPU监控器初始化成功，监控间隔: {interval}秒")
            except Exception as e:
                logger.warning(f"GPU监控器初始化失败: {e}")
        
        # 初始化系统状态监控
        # TEMPORARILY DISABLE SYSTEM MONITORING DUE TO psutil ISSUES
        if False and self.config.get("system_monitoring", False) and system_monitor_available: # MODIFIED TO False
        # if self.config.get("system_monitoring", False) and system_monitor_available: # ORIGINAL LINE
            try:
                interval = self.config.get("system_monitoring_interval", 1.0)
                self.monitors["system"] = SystemMonitor(
                    monitoring_interval=interval
                )
                logger.info(f"系统监控器初始化成功，监控间隔: {interval}秒")
            except Exception as e:
                logger.warning(f"系统监控器初始化失败: {e}")
        else:
            logger.warning("System monitoring is temporarily disabled or psutil is not available.") # ADDED WARNING
        
        # 初始化温度监控
        if self.config.get("temperature_monitoring", False) and temperature_monitor_available:
            try:
                interval = self.config.get("temperature_monitoring_interval", 1.0)
                self.monitors["temperature"] = TemperatureMonitor(
                    monitoring_interval=interval
                )
                logger.info(f"温度监控器初始化成功，监控间隔: {interval}秒")
            except Exception as e:
                logger.warning(f"温度监控器初始化失败: {e}")
    
    def start_monitoring(self):
        """启动所有监控组件"""
        if self.active:
            logger.warning("监控已经处于活动状态")
            return
        
        logger.info("启动所有监控组件")
        self.start_time = time.time()
        self.active = True
        
        # 启动各个监控组件
        for name, monitor in self.monitors.items():
            try:
                monitor.start_monitoring()
                logger.info(f"{name}监控器启动成功")
            except Exception as e:
                logger.error(f"{name}监控器启动失败: {e}")
    
    def stop_monitoring(self):
        """停止所有监控组件并收集指标"""
        if not self.active:
            logger.warning("监控未处于活动状态")
            return
        
        logger.info("停止所有监控组件")
        self.active = False
        
        # 停止各个监控组件并收集指标
        for name, monitor in self.monitors.items():
            try:
                monitor.stop_monitoring()
                # 收集各监控器的统计数据
                self.metrics[name] = monitor.get_statistics()
                logger.info(f"{name}监控器停止成功，收集指标完成")
            except Exception as e:
                logger.error(f"{name}监控器停止失败: {e}")
        
        # 记录总监控时间
        if self.start_time:
            self.metrics["total_monitoring_time"] = time.time() - self.start_time
    
    def get_metrics(self):
        """
        获取所有监控指标
        
        Returns:
            dict: 包含所有监控指标的字典
        """
        return {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": self.metrics
        }
    
    def save_metrics(self, output_dir="./monitoring", filename=None):
        """
        保存监控指标到文件
        
        Args:
            output_dir: 输出目录
            filename: 文件名，默认使用实验ID
            
        Returns:
            filepath: 保存的文件路径
        """
        metrics = self.get_metrics()
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if filename is None:
            filename = f"monitoring_{self.experiment_id}.json"
        
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"监控指标已保存到 {filepath}")
        return filepath
    
    def is_active(self):
        """
        检查监控是否处于活动状态
        
        Returns:
            bool: 是否活动
        """
        return self.active 