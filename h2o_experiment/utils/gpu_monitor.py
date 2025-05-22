"""
GPU监控模块

监控GPU内存使用、利用率和温度等信息
"""
import os
import time
import logging
import threading
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False
    logger.warning("无法导入PyTorch，GPU监控将受限")

# pynvml_available 和相关导入已移除

class GPUMonitor:
    """GPU监控类"""
    
    def __init__(self, monitoring_interval=0.1, device_indices=None):
        """初始化GPU监控器"""
        self.monitoring_interval = monitoring_interval
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        self.stats = {}
        
        # 确定要监控的设备
        self.device_indices = device_indices
        if self.device_indices is None and torch_available:
            if torch.cuda.is_available():
                self.device_indices = list(range(torch.cuda.device_count()))
                logger.info(f"使用PyTorch监控GPU，找到{len(self.device_indices)}个设备")
            else:
                self.device_indices = []
                logger.warning("没有可用的CUDA设备")
        elif self.device_indices is None:
            self.device_indices = []
            logger.warning("未指定监控设备且PyTorch不可用或CUDA不可用，不监控GPU")

        # 为每个设备初始化统计信息
        for device_idx in self.device_indices:
            self.stats[device_idx] = {
                "memory_allocated": [],
                "memory_reserved": [],
                "peak_memory_allocated": [],
                "peak_memory_reserved": [],
                "timestamps": []
            }
        
        logger.info(f"GPU监控器初始化完成，监控间隔: {monitoring_interval}秒")
    
    def _monitor_loop(self):
        """监控循环，收集GPU统计信息"""
        logger.info("GPU监控循环已启动")
        
        while not self.stop_event.is_set():
            current_time = time.time()
            
            for device_idx in self.device_indices:
                # 记录时间戳
                self.stats[device_idx]["timestamps"].append(current_time)
                
                # 使用PyTorch收集内存信息
                if torch_available and torch.cuda.is_available():
                    try:
                        # 获取内存信息（以MB为单位）
                        allocated = torch.cuda.memory_allocated(device_idx) / (1024 * 1024)
                        reserved = torch.cuda.memory_reserved(device_idx) / (1024 * 1024)
                        peak_allocated = torch.cuda.max_memory_allocated(device_idx) / (1024 * 1024)
                        peak_reserved = torch.cuda.max_memory_reserved(device_idx) / (1024 * 1024)
                        
                        self.stats[device_idx]["memory_allocated"].append(allocated)
                        self.stats[device_idx]["memory_reserved"].append(reserved)
                        self.stats[device_idx]["peak_memory_allocated"].append(peak_allocated)
                        self.stats[device_idx]["peak_memory_reserved"].append(peak_reserved)
                    except Exception as e:
                        logger.error(f"通过PyTorch获取GPU {device_idx}的内存信息失败: {e}")
                        self.stats[device_idx]["memory_allocated"].append(0)
                        self.stats[device_idx]["memory_reserved"].append(0)
                        self.stats[device_idx]["peak_memory_allocated"].append(0)
                        self.stats[device_idx]["peak_memory_reserved"].append(0)
                else:
                    self.stats[device_idx]["memory_allocated"].append(0)
                    self.stats[device_idx]["memory_reserved"].append(0)
                    self.stats[device_idx]["peak_memory_allocated"].append(0)
                    self.stats[device_idx]["peak_memory_reserved"].append(0)
            
            # 等待下一个监控周期
            time.sleep(self.monitoring_interval)
    
    def start_monitoring(self):
        """开始GPU监控"""
        if not self.device_indices:
            logger.info("没有配置GPU设备进行监控，跳过启动GPU监控器。")
            return

        if self.monitoring_thread is not None and self.monitoring_thread.is_alive():
            logger.warning("GPU监控器已经在运行")
            return
        
        logger.info("启动GPU监控")
        self.stop_event.clear()
        # 重置统计数据以防多次启动/停止
        self.reset_statistics()
        # 重置PyTorch内置的峰值内存统计
        if torch_available and torch.cuda.is_available():
            for device_idx in self.device_indices:
                try:
                    torch.cuda.reset_peak_memory_stats(device_idx)
                    logger.debug(f"重置GPU {device_idx} 的峰值内存统计。")
                except Exception as e:
                    logger.error(f"重置GPU {device_idx} 的峰值内存统计失败: {e}")

        self.monitoring_thread = threading.Thread(target=self._monitor_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """停止GPU监控"""
        if not self.device_indices or self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            # logger.warning("GPU监控器未运行或未配置设备") # 减少不必要的警告，如果从未启动则不警告
            if self.monitoring_thread is not None:
                 logger.info("GPU监控器已停止或未配置设备。")
            return
        
        logger.info("停止GPU监控")
        self.stop_event.set()
        self.monitoring_thread.join(timeout=max(1.0, self.monitoring_interval * 5)) # 给足够的时间Join
        logger.info("GPU监控线程已加入")
    
    def get_statistics(self):
        """获取收集的GPU统计信息"""
        results = {}
        if not self.device_indices:
            logger.info("没有监控GPU设备，无统计信息。")
            return {"total_devices": 0, "timestamp": datetime.now().isoformat(), "message": "No devices monitored"}

        for device_idx in self.device_indices:
            device_stats = self.stats.get(device_idx, {})
            
            # 只有在有数据的情况下才计算统计信息
            if device_stats.get("memory_allocated"): # 检查列表是否为空
                # 使用收集到的 peak_memory_allocated 列表中的最大值作为真正的峰值
                # 如果 peak_memory_allocated 列表为空（不太可能发生如果循环正常运行），则回退到 memory_allocated
                peak_alloc_list = device_stats.get("peak_memory_allocated", [])
                peak_reserv_list = device_stats.get("peak_memory_reserved", [])

                device_results = {
                    # 内存统计（MB）
                    "peak_memory_mb": max(peak_alloc_list) if peak_alloc_list else (max(device_stats["memory_allocated"]) if device_stats["memory_allocated"] else 0),
                    "average_memory_mb": np.mean(device_stats["memory_allocated"]) if device_stats["memory_allocated"] else 0,
                    "std_dev_memory_mb": np.std(device_stats["memory_allocated"]) if device_stats["memory_allocated"] else 0,
                    "peak_reserved_mb": max(peak_reserv_list) if peak_reserv_list else (max(device_stats["memory_reserved"]) if device_stats["memory_reserved"] else 0),
                    "average_reserved_mb": np.mean(device_stats["memory_reserved"]) if device_stats["memory_reserved"] else 0,
                    
                    # 样本数
                    "sample_count": len(device_stats.get("timestamps", []))
                }
                
                # 计算监控持续时间
                if len(device_stats.get("timestamps", [])) >= 2:
                    device_results["monitoring_duration_sec"] = device_stats["timestamps"][-1] - device_stats["timestamps"][0]
                else:
                    device_results["monitoring_duration_sec"] = 0

                results[f"device_{device_idx}"] = device_results
            else:
                # 如果没有数据，也返回一个占位符，以便聚合逻辑不会出错
                 results[f"device_{device_idx}"] = {
                    "peak_memory_mb": 0,
                    "average_memory_mb": 0,
                    "std_dev_memory_mb": 0,
                    "peak_reserved_mb": 0,
                    "average_reserved_mb": 0,
                    "sample_count": 0,
                    "monitoring_duration_sec": 0,
                    "message": "No data collected for this device"
                 }

        # 添加聚合统计信息
        results["total_devices"] = len(self.device_indices)
        results["timestamp"] = datetime.now().isoformat()
        
        # 计算所有设备的总峰值内存
        total_peak_memory_sum = 0
        for idx in self.device_indices:
            device_key = f"device_{idx}"
            if device_key in results and results[device_key].get("peak_memory_mb") is not None:
                 total_peak_memory_sum += results[device_key]["peak_memory_mb"]
        results["total_peak_memory_mb"] = total_peak_memory_sum
        
        return results

    def reset_statistics(self):
        """重置收集的统计信息"""
        for device_idx in self.device_indices:
            self.stats[device_idx] = {
                "memory_allocated": [],
                "memory_reserved": [],
                "peak_memory_allocated": [],
                "peak_memory_reserved": [],
                "timestamps": []
            }
        logger.info("已重置GPU统计信息")
    
    def is_monitoring(self):
        """
        检查监控是否在进行中
        
        Returns:
            bool: 是否正在监控
        """
        return self.monitoring_thread is not None and self.monitoring_thread.is_alive() 