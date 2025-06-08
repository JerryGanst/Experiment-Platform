

"""
统一监控模块

整合GPU监控、系统监控和性能指标收集功能
"""
import os
import time
import logging
import threading
import numpy as np
import json
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# 尝试导入依赖
try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False
    logger.warning("PyTorch不可用，GPU监控将受限")

try:
    import psutil
    psutil_available = True
except ImportError:
    psutil_available = False
    logger.warning("psutil不可用，系统监控将受限")


class UnifiedMonitor:
    """统一监控器，整合GPU、系统和性能指标监控"""
    
    def __init__(self, config=None, experiment_id=None):
        """
        初始化统一监控器
        
        Args:
            config: 监控配置字典
            experiment_id: 实验ID
        """
        from .. import config as global_config
        
        self.config = config or global_config.MONITORING_CONFIG
        self.experiment_id = experiment_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 监控状态
        self.active = False
        self.start_time = None
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        
        # 监控间隔
        self.gpu_interval = self.config.get("gpu_monitoring_interval", 0.1)
        self.system_interval = self.config.get("system_monitoring_interval", 1.0)
        
        # 数据存储
        self.gpu_stats = {}
        self.system_stats = {
            "cpu_percent": [],
            "memory_percent": [],
            "memory_used_gb": [],
            "timestamps": []
        }
        
        # 性能指标
        self.performance_metrics = {
            "start_time": None,
            "first_token_time": None,
            "token_times": [],
            "tokens_generated": 0,
            "total_time": 0,
            "success": True,
            "config": {}
        }
        
        # 初始化GPU监控
        self._init_gpu_monitoring()
        
        logger.info(f"统一监控器初始化完成，实验ID: {self.experiment_id}")
    
    def _init_gpu_monitoring(self):
        """初始化GPU监控"""
        self.device_indices = []
        if torch_available and torch.cuda.is_available():
            self.device_indices = list(range(torch.cuda.device_count()))
            logger.info(f"发现{len(self.device_indices)}个GPU设备")
            
            # 为每个设备初始化统计信息
            for device_idx in self.device_indices:
                self.gpu_stats[device_idx] = {
                    "memory_allocated": [],
                    "memory_reserved": [],
                    "peak_memory_allocated": [],
                    "peak_memory_reserved": [],
                    "timestamps": []
                }
        else:
            logger.warning("没有可用的GPU设备")
    
    def _monitoring_loop(self):
        """监控循环"""
        logger.info("统一监控循环已启动")
        
        last_gpu_time = 0
        last_system_time = 0
        
        while not self.stop_event.is_set():
            current_time = time.time()
            
            # GPU监控
            if current_time - last_gpu_time >= self.gpu_interval:
                self._collect_gpu_stats(current_time)
                last_gpu_time = current_time
            
            # 系统监控
            if current_time - last_system_time >= self.system_interval:
                self._collect_system_stats(current_time)
                last_system_time = current_time
            
            # 短暂休眠
            time.sleep(min(self.gpu_interval, self.system_interval) / 2)
    
    def _collect_gpu_stats(self, current_time):
        """收集GPU统计信息"""
        for device_idx in self.device_indices:
            try:
                self.gpu_stats[device_idx]["timestamps"].append(current_time)
                
                if torch_available and torch.cuda.is_available():
                    # 获取内存信息（以MB为单位）
                    allocated = torch.cuda.memory_allocated(device_idx) / (1024 * 1024)
                    reserved = torch.cuda.memory_reserved(device_idx) / (1024 * 1024)
                    peak_allocated = torch.cuda.max_memory_allocated(device_idx) / (1024 * 1024)
                    peak_reserved = torch.cuda.max_memory_reserved(device_idx) / (1024 * 1024)
                    
                    self.gpu_stats[device_idx]["memory_allocated"].append(allocated)
                    self.gpu_stats[device_idx]["memory_reserved"].append(reserved)
                    self.gpu_stats[device_idx]["peak_memory_allocated"].append(peak_allocated)
                    self.gpu_stats[device_idx]["peak_memory_reserved"].append(peak_reserved)
                else:
                    # 填充默认值
                    for key in ["memory_allocated", "memory_reserved", "peak_memory_allocated", "peak_memory_reserved"]:
                        self.gpu_stats[device_idx][key].append(0)
            except Exception as e:
                logger.error(f"收集GPU {device_idx}统计信息失败: {e}")
    
    def _collect_system_stats(self, current_time):
        """收集系统统计信息"""
        try:
            self.system_stats["timestamps"].append(current_time)
            
            if psutil_available:
                # CPU和内存使用率
                self.system_stats["cpu_percent"].append(psutil.cpu_percent(interval=0.1))
                memory = psutil.virtual_memory()
                self.system_stats["memory_percent"].append(memory.percent)
                self.system_stats["memory_used_gb"].append(memory.used / (1024**3))
            else:
                # 填充默认值
                self.system_stats["cpu_percent"].append(0)
                self.system_stats["memory_percent"].append(0)
                self.system_stats["memory_used_gb"].append(0)
        except Exception as e:
            logger.error(f"收集系统统计信息失败: {e}")
    
    def start_monitoring(self):
        """开始监控"""
        if self.active:
            logger.warning("监控已经在运行")
            return
        
        logger.info("启动统一监控")
        self.start_time = time.time()
        self.active = True
        self.stop_event.clear()
        
        # 重置统计数据
        self._reset_stats()
        
        # 重置PyTorch峰值内存统计
        if torch_available and torch.cuda.is_available():
            for device_idx in self.device_indices:
                try:
                    torch.cuda.reset_peak_memory_stats(device_idx)
                except Exception as e:
                    logger.error(f"重置GPU {device_idx}峰值内存统计失败: {e}")
        
        # 启动监控线程
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        if not self.active:
            logger.warning("监控未在运行")
            return
        
        logger.info("停止统一监控")
        self.active = False
        self.stop_event.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
    
    def _reset_stats(self):
        """重置统计数据"""
        # 重置GPU统计
        for device_idx in self.device_indices:
            self.gpu_stats[device_idx] = {
                "memory_allocated": [],
                "memory_reserved": [],
                "peak_memory_allocated": [],
                "peak_memory_reserved": [],
                "timestamps": []
            }
        
        # 重置系统统计
        self.system_stats = {
            "cpu_percent": [],
            "memory_percent": [],
            "memory_used_gb": [],
            "timestamps": []
        }
    
    # 性能指标收集方法
    def record_config(self, config):
        """记录实验配置"""
        self.performance_metrics["config"] = config
        logger.info("记录实验配置")
    
    def start_generation(self):
        """开始生成计时"""
        self.performance_metrics["start_time"] = time.time()
        self.performance_metrics["token_times"] = []
        self.performance_metrics["tokens_generated"] = 0
        logger.info("开始生成计时")
    
    def record_first_token(self):
        """记录第一个令牌生成时间"""
        if self.performance_metrics["start_time"] is None:
            logger.warning("在start_generation之前调用record_first_token")
            return
        
        self.performance_metrics["first_token_time"] = time.time()
        self.performance_metrics["token_times"].append(self.performance_metrics["first_token_time"])
        self.performance_metrics["tokens_generated"] += 1
        logger.info("记录第一个令牌生成")
    
    def record_token(self):
        """记录后续令牌生成时间"""
        if self.performance_metrics["start_time"] is None:
            logger.warning("在start_generation之前调用record_token")
            return
        
        self.performance_metrics["token_times"].append(time.time())
        self.performance_metrics["tokens_generated"] += 1
    
    def end_generation(self):
        """结束生成计时"""
        if self.performance_metrics["start_time"] is None:
            logger.warning("在start_generation之前调用end_generation")
            return
        
        end_time = time.time()
        self.performance_metrics["total_time"] = end_time - self.performance_metrics["start_time"]
        logger.info(f"结束生成计时，总时间: {self.performance_metrics['total_time']:.2f}秒")
    
    def mark_failure(self, error_message):
        """标记实验失败"""
        self.performance_metrics["success"] = False
        self.performance_metrics["error_message"] = error_message
        logger.error(f"实验标记为失败: {error_message}")
    
    def get_comprehensive_metrics(self):
        """获取综合指标"""
        # 计算性能指标
        performance = self._compute_performance_metrics()
        
        # 获取GPU统计
        gpu_summary = self._get_gpu_summary()
        
        # 获取系统统计
        system_summary = self._get_system_summary()
        
        return {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now().isoformat(),
            "performance": performance,
            "gpu": gpu_summary,
            "system": system_summary,
            "monitoring_duration": time.time() - self.start_time if self.start_time else 0
        }
    
    def _compute_performance_metrics(self):
        """计算性能指标"""
        if not self.performance_metrics["success"]:
            return {
                "success": False,
                "error_message": self.performance_metrics.get("error_message", "Unknown error")
            }
        
        if not self.performance_metrics["token_times"]:
            return {
                "success": False,
                "error_message": "No timing data collected"
            }
        
        # 计算TTFT和TPOT
        start_time = self.performance_metrics["start_time"]
        first_token_time = self.performance_metrics["first_token_time"]
        token_times = self.performance_metrics["token_times"]
        total_time = self.performance_metrics["total_time"]
        tokens_generated = self.performance_metrics["tokens_generated"]
        
        ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
        
        tpot = 0
        if len(token_times) > 1:
            intervals = [token_times[i+1] - token_times[i] for i in range(len(token_times)-1)]
            tpot = np.mean(intervals) * 1000
        
        throughput = tokens_generated / total_time if total_time > 0 else 0
        
        return {
            "success": True,
            "ttft_ms": ttft,
            "tpot_ms": tpot,
            "throughput_tokens_per_sec": throughput,
            "total_time_sec": total_time,
            "tokens_generated": tokens_generated,
            **self.performance_metrics["config"]
        }
    
    def _get_gpu_summary(self):
        """获取GPU统计摘要"""
        if not self.device_indices:
            return {"total_devices": 0, "message": "No GPU devices"}
        
        summary = {"total_devices": len(self.device_indices)}
        total_peak_memory = 0
        
        for device_idx in self.device_indices:
            device_stats = self.gpu_stats.get(device_idx, {})
            
            if device_stats.get("peak_memory_allocated"):
                peak_memory = max(device_stats["peak_memory_allocated"])
                total_peak_memory += peak_memory
                
                summary[f"device_{device_idx}"] = {
                    "peak_memory_mb": peak_memory,
                    "average_memory_mb": np.mean(device_stats["memory_allocated"]),
                    "sample_count": len(device_stats["timestamps"])
                }
        
        summary["total_peak_memory_mb"] = total_peak_memory
        return summary
    
    def _get_system_summary(self):
        """获取系统统计摘要"""
        if not self.system_stats["timestamps"]:
            return {"message": "No system data collected"}
        
        return {
            "peak_cpu_percent": max(self.system_stats["cpu_percent"]) if self.system_stats["cpu_percent"] else 0,
            "average_cpu_percent": np.mean(self.system_stats["cpu_percent"]) if self.system_stats["cpu_percent"] else 0,
            "peak_memory_percent": max(self.system_stats["memory_percent"]) if self.system_stats["memory_percent"] else 0,
            "peak_memory_used_gb": max(self.system_stats["memory_used_gb"]) if self.system_stats["memory_used_gb"] else 0,
            "sample_count": len(self.system_stats["timestamps"])
        }
    
    def save_metrics(self, output_dir="./results", filename=None):
        """保存指标到文件"""
        metrics = self.get_comprehensive_metrics()
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if filename is None:
            filename = f"unified_metrics_{self.experiment_id}.json"
        
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"统一监控指标已保存到: {filepath}")
        return filepath 