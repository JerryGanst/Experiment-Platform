"""
性能指标收集模块
"""
import time
import logging
import numpy as np
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class PerformanceMetricsCollector:
    """性能指标收集类"""
    
    def __init__(self, experiment_id=None):
        """
        初始化性能指标收集器
        
        Args:
            experiment_id: 实验ID，用于标识此次实验
        """
        self.experiment_id = experiment_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = None
        self.first_token_time = None
        self.token_times = []
        self.tokens_generated = 0
        self.total_time = 0
        self.gpu_stats = None
        self.config = {}
        self.success = True
        
        logger.info(f"Performance metrics collector initialized with ID: {self.experiment_id}")
    
    def record_config(self, config):
        """
        记录实验配置
        
        Args:
            config: 配置字典
        """
        self.config = config
        logger.info(f"Recorded experiment config")
    
    def start_generation(self):
        """开始生成计时"""
        self.start_time = time.time()
        self.token_times = []
        self.tokens_generated = 0
        logger.info("Started generation timing")
    
    def record_first_token(self):
        """记录第一个令牌生成时间"""
        if self.start_time is None:
            logger.warning("Called record_first_token before start_generation")
            return
        
        self.first_token_time = time.time()
        self.token_times.append(self.first_token_time)
        self.tokens_generated += 1
        logger.info("Recorded first token generation")
    
    def record_token(self):
        """记录后续令牌生成时间"""
        if self.start_time is None:
            logger.warning("Called record_token before start_generation")
            return
        
        self.token_times.append(time.time())
        self.tokens_generated += 1
    
    def end_generation(self):
        """结束生成计时"""
        if self.start_time is None:
            logger.warning("Called end_generation before start_generation")
            return
        
        end_time = time.time()
        self.total_time = end_time - self.start_time
        logger.info(f"Ended generation timing. Total time: {self.total_time:.2f}s")
    
    def record_gpu_stats(self, gpu_stats):
        """
        记录GPU统计信息
        
        Args:
            gpu_stats: GPU统计信息字典
        """
        self.gpu_stats = gpu_stats
        logger.info("Recorded GPU statistics")
    
    def mark_failure(self, error_message):
        """
        标记实验失败
        
        Args:
            error_message: 错误信息
        """
        self.success = False
        self.error_message = error_message
        logger.error(f"Experiment marked as failed: {error_message}")
    
    def record_error(self, error_message):
        """
        记录错误信息
        
        Args:
            error_message: 错误信息
        """
        self.mark_failure(error_message)
    
    def finalize_metrics_on_error(self):
        """
        当实验失败时完成指标收集
        
        Returns:
            metrics: 包含错误信息的指标字典
        """
        return self.compute_metrics()
    
    def compute_metrics(self):
        """
        计算性能指标
        
        Returns:
            metrics: 计算得到的性能指标字典
        """
        # 如果实验失败，返回基本信息和错误消息
        if not self.success:
            return {
                "experiment_id": self.experiment_id,
                "success": False,
                "error_message": getattr(self, "error_message", "Unknown error"),
                "timestamp": datetime.now().isoformat()
            }
        
        # 如果没有计时数据，返回警告
        if self.start_time is None or not self.token_times:
            logger.warning("Cannot compute metrics: no timing data available")
            return {
                "experiment_id": self.experiment_id,
                "success": False,
                "error_message": "No timing data collected",
                "timestamp": datetime.now().isoformat()
            }
        
        # 首个令牌生成时间(TTFT)
        ttft = (self.first_token_time - self.start_time) * 1000 if self.first_token_time else 0
        
        # 每令牌生成时间(TPOT)
        tpot = 0
        if len(self.token_times) > 1:
            intervals = [self.token_times[i+1] - self.token_times[i] for i in range(len(self.token_times)-1)]
            tpot = np.mean(intervals) * 1000
        
        # 吞吐量（每秒令牌数）
        throughput = self.tokens_generated / self.total_time if self.total_time > 0 else 0
        
        metrics = {
            "experiment_id": self.experiment_id,
            "success": True,
            "ttft_ms": ttft,
            "tpot_ms": tpot,
            "throughput_tokens_per_sec": throughput,
            "total_time_sec": self.total_time,
            "tokens_generated": self.tokens_generated,
        }
        
        # 添加GPU统计信息（如果有）
        if self.gpu_stats:
            reported_peak_memory = 0
            if "total_peak_memory_mb" in self.gpu_stats:
                reported_peak_memory = self.gpu_stats["total_peak_memory_mb"] 
            elif "peak_memory_mb" in self.gpu_stats: # Fallback for direct structure or single device summary by MonitoringManager
                 reported_peak_memory = self.gpu_stats["peak_memory_mb"]
            # If GPUMonitor returns stats for device_0 and it's the only device, 
            # total_peak_memory_mb should reflect that. Otherwise, if monitoring manager 
            # directly passes device_0 stats as self.gpu_stats, this could be a fallback.
            elif "device_0" in self.gpu_stats and "peak_memory_mb" in self.gpu_stats["device_0"] and self.gpu_stats.get("total_devices") == 1:
                reported_peak_memory = self.gpu_stats["device_0"]["peak_memory_mb"]

            metrics.update({
                "peak_gpu_memory_mb": reported_peak_memory,
            })
        
        # 添加配置信息
        metrics.update({
            "model_name": self.config.get("model_name", ""),
            "precision": self.config.get("precision", ""),
            "batch_size": self.config.get("batch_size", 0),
            "kv_cache_length": self.config.get("kv_cache_length", 0),
            "max_new_tokens": self.config.get("max_new_tokens", 0),
            "use_h2o": self.config.get("use_h2o", False),
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info("Computed performance metrics")
        return metrics
    
    def save_metrics(self, output_dir="./results", filename=None):
        """
        保存性能指标到文件
        
        Args:
            output_dir: 输出目录
            filename: 文件名，默认使用实验ID
            
        Returns:
            filepath: 保存的文件路径
        """
        metrics = self.compute_metrics()
        if not metrics:
            logger.warning("No metrics to save")
            return None
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if filename is None:
            filename = f"metrics_{self.experiment_id}.json"
        
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved metrics to {filepath}")
        return filepath
    
    def compute_and_save_metrics(self, output_dir="./results", filename_prefix="metrics"):
        """
        计算并保存性能指标到文件
        
        Args:
            output_dir: 输出目录
            filename_prefix: 文件名前缀
            
        Returns:
            metrics: 计算得到的性能指标字典
        """
        metrics = self.compute_metrics()
        if not metrics:
            logger.warning("No metrics to compute and save")
            return None
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        filename = f"{filename_prefix}_{self.experiment_id}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Computed and saved metrics to {filepath}")
        return metrics 