#!/usr/bin/env python3
"""
GPU监控诊断测试脚本
"""

import torch
import time
import json

def test_gpu_monitoring():
    """测试GPU监控功能"""
    print("🔍 开始GPU监控诊断...")
    
    # 检查CUDA可用性
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，无法进行GPU监控")
        return
    
    print(f"GPU设备数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}, 总显存: {props.total_memory / 1024**3:.1f}GB")
    
    # 重置内存统计
    torch.cuda.reset_peak_memory_stats(0)
    print("\n重置GPU内存统计...")
    
    # 初始内存状态
    print(f"初始GPU内存状态:")
    print(f"  已分配: {torch.cuda.memory_allocated(0) / 1024**2:.1f}MB")
    print(f"  已保留: {torch.cuda.memory_reserved(0) / 1024**2:.1f}MB")
    print(f"  峰值已分配: {torch.cuda.max_memory_allocated(0) / 1024**2:.1f}MB")
    print(f"  峰值已保留: {torch.cuda.max_memory_reserved(0) / 1024**2:.1f}MB")
    
    # 加载一个简单的模型来使用GPU内存
    print("\n🚀 加载简单模型...")
    try:
        # 创建一些张量来占用GPU内存
        device = torch.device("cuda:0")
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        print(f"创建张量后GPU内存状态:")
        print(f"  已分配: {torch.cuda.memory_allocated(0) / 1024**2:.1f}MB")
        print(f"  已保留: {torch.cuda.memory_reserved(0) / 1024**2:.1f}MB")
        print(f"  峰值已分配: {torch.cuda.max_memory_allocated(0) / 1024**2:.1f}MB")
        print(f"  峰值已保留: {torch.cuda.max_memory_reserved(0) / 1024**2:.1f}MB")
        
        # 执行一些操作
        z = torch.matmul(x, y)
        
        print(f"执行矩阵乘法后GPU内存状态:")
        print(f"  已分配: {torch.cuda.memory_allocated(0) / 1024**2:.1f}MB")
        print(f"  已保留: {torch.cuda.memory_reserved(0) / 1024**2:.1f}MB")
        print(f"  峰值已分配: {torch.cuda.max_memory_allocated(0) / 1024**2:.1f}MB")
        print(f"  峰值已保留: {torch.cuda.max_memory_reserved(0) / 1024**2:.1f}MB")
        
        # 清理内存
        del x, y, z
        torch.cuda.empty_cache()
        
        print(f"清理后GPU内存状态:")
        print(f"  已分配: {torch.cuda.memory_allocated(0) / 1024**2:.1f}MB")
        print(f"  已保留: {torch.cuda.memory_reserved(0) / 1024**2:.1f}MB")
        print(f"  峰值已分配: {torch.cuda.max_memory_allocated(0) / 1024**2:.1f}MB")
        print(f"  峰值已保留: {torch.cuda.max_memory_reserved(0) / 1024**2:.1f}MB")
        
        # 测试监控期间的内存跟踪
        print("\n📊 测试持续监控...")
        
        memory_samples = []
        for i in range(10):
            # 创建和释放内存
            temp = torch.randn(500, 500, device=device)
            
            allocated = torch.cuda.memory_allocated(0) / 1024**2
            reserved = torch.cuda.memory_reserved(0) / 1024**2
            peak_allocated = torch.cuda.max_memory_allocated(0) / 1024**2
            peak_reserved = torch.cuda.max_memory_reserved(0) / 1024**2
            
            memory_samples.append({
                "iteration": i,
                "allocated_mb": allocated,
                "reserved_mb": reserved,
                "peak_allocated_mb": peak_allocated,
                "peak_reserved_mb": peak_reserved
            })
            
            del temp
            if i % 2 == 0:
                torch.cuda.empty_cache()
            
            time.sleep(0.1)
        
        print("内存采样结果:")
        for sample in memory_samples:
            print(f"  迭代{sample['iteration']}: 分配={sample['allocated_mb']:.1f}MB, 峰值={sample['peak_allocated_mb']:.1f}MB")
        
        # 保存测试结果
        with open("hace-kv-optimization/gpu_monitoring_test_results.json", "w") as f:
            json.dump({
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count(),
                "memory_samples": memory_samples,
                "final_peak_allocated": torch.cuda.max_memory_allocated(0) / 1024**2,
                "final_peak_reserved": torch.cuda.max_memory_reserved(0) / 1024**2
            }, f, indent=2)
        
        print("\n✅ GPU监控测试完成，结果保存到 gpu_monitoring_test_results.json")
        
    except Exception as e:
        print(f"❌ GPU监控测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gpu_monitoring() 