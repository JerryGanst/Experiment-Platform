"""
统一监控包装器 - 监控GPU内存、时间和性能指标
"""
import time
import json
import os
import logging
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available, GPU monitoring disabled")

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logging.warning("pynvml not available, advanced GPU monitoring disabled")

logger = logging.getLogger(__name__)

class UnifiedMonitor:
    """统一的实验监控器"""
    
    def __init__(self, exp_id: str):
        self.exp_id = exp_id
        self.start_time = None
        self.end_time = None
        self.metrics = {}
        
        # GPU监控初始化
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.gpu_available = True
        else:
            self.gpu_available = False
            
    def start(self):
        """开始监控"""
        self.start_time = time.time()
        
        if self.gpu_available:
            # 重置GPU内存统计
            torch.cuda.reset_peak_memory_stats()
            self.metrics['gpu_init_memory_mb'] = torch.cuda.memory_allocated() / (1024 ** 2)
            
        logger.info(f"[{self.exp_id}] Monitoring started")
        
    def stop(self):
        """停止监控"""
        self.end_time = time.time()
        
        if self.gpu_available:
            self.metrics['gpu_peak_memory_mb'] = torch.cuda.max_memory_allocated() / (1024 ** 2)
            self.metrics['gpu_current_memory_mb'] = torch.cuda.memory_allocated() / (1024 ** 2)
            
        self.metrics['duration_seconds'] = self.end_time - self.start_time
        
        logger.info(f"[{self.exp_id}] Monitoring stopped - Duration: {self.metrics['duration_seconds']:.1f}s")
        
    def add_metric(self, key: str, value):
        """添加自定义指标"""
        self.metrics[key] = value
        
    def save_metrics(self, output_path: str):
        """保存监控指标到文件"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 添加实验基本信息
        final_metrics = {
            "exp_id": self.exp_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **self.metrics
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_metrics, f, indent=2, ensure_ascii=False)
            
        logger.info(f"[{self.exp_id}] Metrics saved to {output_path}")
        
    def summary(self) -> str:
        """生成监控摘要"""
        summary_lines = [
            f"Experiment: {self.exp_id}",
            f"Duration: {self.metrics.get('duration_seconds', 0):.1f}s"
        ]
        
        if self.gpu_available:
            peak_mem = self.metrics.get('gpu_peak_memory_mb', 0)
            summary_lines.append(f"GPU Peak Memory: {peak_mem:.1f}MB")
            
        return "\n".join(summary_lines)
        
    def __enter__(self):
        """支持上下文管理器"""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持上下文管理器"""
        self.stop() 