#!/usr/bin/env python
"""
监控组件使用示例脚本

演示如何使用监控管理器监控实验执行过程
"""
import os
import time
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from monitoring_manager import MonitoringManager
from gpu_monitor import GPUMonitor
from system_monitor import SystemMonitor

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="监控组件使用示例")
    parser.add_argument("--model", type=str, default="facebook/opt-125m", help="要加载的模型")
    parser.add_argument("--output_dir", type=str, default="./monitor_results", help="输出目录")
    parser.add_argument("--duration", type=int, default=30, help="模拟运行时间（秒）")
    parser.add_argument("--gpu_only", action="store_true", help="仅使用GPU监控器")
    parser.add_argument("--system_only", action="store_true", help="仅使用系统监控器")
    return parser.parse_args()

def simulate_model_workload(model, duration=30):
    """
    模拟模型工作负载
    
    Args:
        model: 模型
        duration: 持续时间（秒）
    """
    print(f"模拟模型工作负载，持续{duration}秒...")
    
    # 创建一些随机输入
    batch_size = 4
    seq_length = 128
    device = model.device
    
    start_time = time.time()
    elapsed = 0
    
    while elapsed < duration:
        # 生成随机输入
        input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_length), device=device)
        attention_mask = torch.ones_like(input_ids)
        
        # 运行前向传播
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # 更新已过时间
        elapsed = time.time() - start_time
        
        # 每5秒打印一次进度
        if int(elapsed) % 5 == 0 and int(elapsed) != int(elapsed - 0.1):
            print(f"已运行 {int(elapsed)}/{duration} 秒...")
    
    print("模型工作负载模拟完成")

def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    print(f"加载模型: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # 选择要使用的监控方式
    if args.gpu_only:
        print("仅使用GPU监控器")
        monitor = GPUMonitor()
        monitor.start_monitoring()
        simulate_model_workload(model, args.duration)
        monitor.stop_monitoring()
        stats = monitor.get_statistics()
        print(f"GPU统计信息: {stats}")
        
    elif args.system_only:
        print("仅使用系统监控器")
        monitor = SystemMonitor()
        monitor.start_monitoring()
        simulate_model_workload(model, args.duration)
        monitor.stop_monitoring()
        stats = monitor.get_statistics()
        print(f"系统统计信息: {stats}")
        
    else:
        print("使用监控管理器（整合所有监控组件）")
        # 使用监控管理器
        monitoring_manager = MonitoringManager()
        monitoring_manager.start_monitoring()
        
        # 模拟模型工作
        simulate_model_workload(model, args.duration)
        
        # 停止监控并获取结果
        monitoring_manager.stop_monitoring()
        monitoring_metrics = monitoring_manager.get_metrics()
        
        # 保存监控结果
        monitoring_manager.save_metrics(output_dir=args.output_dir)
        
        print(f"监控指标已保存到 {args.output_dir}")
        
        # 打印GPU指标摘要（如果有）
        if "gpu" in monitoring_metrics["metrics"]:
            gpu_metrics = monitoring_metrics["metrics"]["gpu"]
            print("\nGPU指标摘要:")
            for device_id, stats in [(k, v) for k, v in gpu_metrics.items() if k.startswith("device_")]:
                print(f"  {device_id}:")
                print(f"    峰值内存: {stats.get('peak_memory_mb', 0):.2f} MB")
                print(f"    平均内存: {stats.get('average_memory_mb', 0):.2f} MB")
                print(f"    峰值利用率: {stats.get('peak_utilization', 0):.2f}%")
                print(f"    平均利用率: {stats.get('average_utilization', 0):.2f}%")
                print(f"    峰值温度: {stats.get('peak_temperature', 0):.2f}°C")
        
        # 打印系统指标摘要（如果有）
        if "system" in monitoring_metrics["metrics"]:
            system_metrics = monitoring_metrics["metrics"]["system"]
            print("\n系统指标摘要:")
            if "cpu" in system_metrics:
                cpu_stats = system_metrics["cpu"]
                print(f"  CPU:")
                print(f"    峰值使用率: {cpu_stats.get('peak_percent', 0):.2f}%")
                print(f"    平均使用率: {cpu_stats.get('average_percent', 0):.2f}%")
            
            if "memory" in system_metrics:
                mem_stats = system_metrics["memory"]
                print(f"  内存:")
                print(f"    峰值使用率: {mem_stats.get('peak_percent', 0):.2f}%")
                print(f"    平均使用GB: {mem_stats.get('average_used_gb', 0):.2f} GB")
                print(f"    最小可用GB: {mem_stats.get('min_available_gb', 0):.2f} GB")

if __name__ == "__main__":
    main() 