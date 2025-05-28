"""
系统监控模块

监控CPU使用率、内存使用情况和其他系统资源
"""
import os
import time
import logging
import threading
import platform
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import psutil
    psutil_available = True
except ImportError:
    psutil_available = False
    logger.warning("无法导入psutil，系统监控将受限")

class SystemMonitor:
    """系统资源监控类"""
    
    def __init__(self, monitoring_interval=1.0):
        """
        初始化系统监控器
        
        Args:
            monitoring_interval: 监控间隔时间（秒）
        """
        self.monitoring_interval = monitoring_interval
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        self.stats = {
            "cpu_percent": [],
            "memory_percent": [],
            "memory_used_gb": [],
            "memory_available_gb": [],
            "disk_usage_percent": [],
            "timestamps": []
        }
        
        # 检查psutil可用性
        if not psutil_available:
            logger.error("psutil不可用，系统监控将无法工作")
        
        logger.info(f"系统监控器初始化完成，监控间隔: {monitoring_interval}秒，"
                   f"当前系统: {platform.system()} {platform.release()}")
    
    def _monitor_loop(self):
        """监控循环，收集系统资源统计信息"""
        logger.info("系统监控循环已启动")
        
        while not self.stop_event.is_set():
            current_time = time.time()
            self.stats["timestamps"].append(current_time)
            
            if psutil_available:
                try:
                    # CPU使用率
                    self.stats["cpu_percent"].append(psutil.cpu_percent(interval=0.1))
                    
                    # 内存使用情况
                    memory = psutil.virtual_memory()
                    self.stats["memory_percent"].append(memory.percent)
                    self.stats["memory_used_gb"].append(memory.used / (1024**3))  # 转换为GB
                    self.stats["memory_available_gb"].append(memory.available / (1024**3))  # 转换为GB
                    
                    # 磁盘使用情况
                    disk = psutil.disk_usage('/')
                    self.stats["disk_usage_percent"].append(disk.percent)
                except Exception as e:
                    logger.error(f"获取系统资源信息失败: {e}")
                    # 添加默认值
                    self.stats["cpu_percent"].append(0)
                    self.stats["memory_percent"].append(0)
                    self.stats["memory_used_gb"].append(0)
                    self.stats["memory_available_gb"].append(0)
                    self.stats["disk_usage_percent"].append(0)
            else:
                # 如果psutil不可用，添加默认值
                self.stats["cpu_percent"].append(0)
                self.stats["memory_percent"].append(0)
                self.stats["memory_used_gb"].append(0)
                self.stats["memory_available_gb"].append(0)
                self.stats["disk_usage_percent"].append(0)
            
            # 等待下一个监控周期
            time.sleep(self.monitoring_interval)
    
    def start_monitoring(self):
        """开始系统监控"""
        if not psutil_available:
            logger.warning("psutil不可用，无法启动系统监控")
            return
            
        if self.monitoring_thread is not None and self.monitoring_thread.is_alive():
            logger.warning("系统监控器已经在运行")
            return
        
        logger.info("启动系统监控")
        self.stop_event.clear()
        self.monitoring_thread = threading.Thread(target=self._monitor_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """停止系统监控"""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            logger.warning("系统监控器未运行")
            return
        
        logger.info("停止系统监控")
        self.stop_event.set()
        self.monitoring_thread.join(timeout=5.0)
    
    def get_statistics(self):
        """
        获取收集的系统资源统计信息
        
        Returns:
            dict: 系统资源统计信息字典
        """
        # 检查是否有收集到的数据
        if not self.stats["timestamps"]:
            return {
                "system_info": {
                    "system": platform.system(),
                    "version": platform.version(),
                    "processor": platform.processor()
                },
                "cpu": {
                    "count": psutil.cpu_count() if psutil_available else 0
                },
                "memory": {},
                "disk": {},
                "sample_count": 0
            }
        
        results = {
            # 系统信息
            "system_info": {
                "system": platform.system(),
                "version": platform.version(),
                "processor": platform.processor()
            },
            
            # CPU统计信息
            "cpu": {
                "count": psutil.cpu_count() if psutil_available else 0,
                "peak_percent": max(self.stats["cpu_percent"]),
                "average_percent": np.mean(self.stats["cpu_percent"]),
                "std_dev_percent": np.std(self.stats["cpu_percent"])
            },
            
            # 内存统计信息
            "memory": {
                "peak_percent": max(self.stats["memory_percent"]),
                "average_percent": np.mean(self.stats["memory_percent"]),
                "peak_used_gb": max(self.stats["memory_used_gb"]),
                "average_used_gb": np.mean(self.stats["memory_used_gb"]),
                "min_available_gb": min(self.stats["memory_available_gb"])
            },
            
            # 磁盘统计信息
            "disk": {
                "peak_usage_percent": max(self.stats["disk_usage_percent"]),
                "average_usage_percent": np.mean(self.stats["disk_usage_percent"])
            },
            
            # 样本信息
            "sample_count": len(self.stats["timestamps"]),
            "timestamp": datetime.now().isoformat()
        }
        
        # 计算监控持续时间
        if len(self.stats["timestamps"]) >= 2:
            results["monitoring_duration_sec"] = self.stats["timestamps"][-1] - self.stats["timestamps"][0]
        
        return results
    
    def reset_statistics(self):
        """重置收集的统计信息"""
        self.stats = {
            "cpu_percent": [],
            "memory_percent": [],
            "memory_used_gb": [],
            "memory_available_gb": [],
            "disk_usage_percent": [],
            "timestamps": []
        }
        logger.info("已重置系统监控统计信息")
    
    def is_monitoring(self):
        """
        检查监控是否在进行中
        
        Returns:
            bool: 是否正在监控
        """
        return self.monitoring_thread is not None and self.monitoring_thread.is_alive() 