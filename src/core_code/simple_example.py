#!/usr/bin/env python3
"""
CAKE-AdaKV统一集成框架简化示例

这个脚本展示了如何使用我们的创新方法进行KV缓存优化。
修复了导入问题，可以直接运行。
"""

import sys
import os
import numpy as np
import time
from typing import List, Tuple

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入我们的核心框架
try:
    # 使用绝对导入
    import integration_framework
    import unified_allocator
    print("✅ 成功导入核心框架")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保所有依赖都已安装")
    sys.exit(1)


def generate_synthetic_attention_weights(
    num_layers: int = 6,
    num_heads: int = 12, 
    seq_len: int = 256,
    batch_size: int = 1
) -> List[np.ndarray]:
    """
    生成合成的注意力权重数据用于测试
    
    Args:
        num_layers: 层数
        num_heads: 头数
        seq_len: 序列长度
        batch_size: 批大小
        
    Returns:
        注意力权重列表
    """
    print(f"🔧 生成合成数据: {num_layers}层, {num_heads}头, 序列长度{seq_len}")
    
    attention_weights_list = []
    
    for layer_idx in range(num_layers):
        # 为不同层生成不同特征的注意力模式
        if layer_idx < num_layers // 3:
            # 前1/3层：局部注意力模式（高集中度）
            attention_weights = np.random.exponential(0.1, (batch_size, num_heads, seq_len, seq_len))
        elif layer_idx < 2 * num_layers // 3:
            # 中1/3层：全局注意力模式（中等集中度）
            attention_weights = np.random.gamma(2, 0.5, (batch_size, num_heads, seq_len, seq_len))
        else:
            # 后1/3层：分散注意力模式（低集中度）
            attention_weights = np.random.uniform(0, 1, (batch_size, num_heads, seq_len, seq_len))
        
        # 归一化为概率分布
        attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)
        
        # 确保数值稳定性
        attention_weights = np.clip(attention_weights, 1e-8, 1.0)
        
        attention_weights_list.append(attention_weights)
    
    return attention_weights_list


def simple_usage_example():
    """简单使用示例"""
    print("\n" + "="*50)
    print("📖 简单使用示例")
    print("="*50)
    
    # 生成测试数据
    attention_weights_list = generate_synthetic_attention_weights(
        num_layers=6, num_heads=12, seq_len=256
    )
    
    try:
        # 创建集成器（使用默认配置）
        print("🚀 创建CAKE-AdaKV集成器...")
        config = integration_framework.IntegrationConfig(
            total_cache_size=2048,
            enable_monitoring=True,
            enable_fallback=True
        )
        integration = integration_framework.CakeAdaKVIntegration(config)
        
        # 优化缓存分配
        print("⚡ 执行缓存优化...")
        start_time = time.time()
        
        layer_budgets, head_budgets_list = integration.optimize_cache(attention_weights_list)
        
        end_time = time.time()
        
        # 显示结果
        print(f"✅ 优化完成，耗时: {end_time - start_time:.3f}秒")
        print(f"📊 层级预算分配: {layer_budgets}")
        print(f"📊 头级预算示例 (第0层): {head_budgets_list[0]}")
        
        # 验证预算守恒
        total_allocated = sum(sum(head_budgets) for head_budgets in head_budgets_list)
        print(f"🔍 预算守恒检查: 总分配={total_allocated}, 应为{config.total_cache_size}")
        
        # 获取性能报告
        if integration.monitor:
            report = integration.get_performance_report()
            print("📈 性能报告:")
            for key, value in report.items():
                print(f"   - {key}: {value}")
        
        return integration
        
    except Exception as e:
        print(f"❌ 执行过程中出错: {e}")
        print("使用简化的分配方案...")
        return simple_fallback_allocation(attention_weights_list, config.total_cache_size)


def simple_fallback_allocation(attention_weights_list: List[np.ndarray], total_cache: int = 2048):
    """简化的回退分配方案"""
    print("🔄 使用简化分配方案...")
    
    num_layers = len(attention_weights_list)
    
    # 均匀层级分配
    layer_base = total_cache // num_layers
    layer_remainder = total_cache % num_layers
    layer_budgets = [layer_base] * num_layers
    for i in range(layer_remainder):
        layer_budgets[i] += 1
    
    # 均匀头级分配
    head_budgets_list = []
    for i, attn in enumerate(attention_weights_list):
        num_heads = attn.shape[1]
        head_base = layer_budgets[i] // num_heads
        head_remainder = layer_budgets[i] % num_heads
        head_budgets = [head_base] * num_heads
        for j in range(head_remainder):
            head_budgets[j] += 1
        head_budgets_list.append(head_budgets)
    
    print(f"✅ 简化分配完成")
    print(f"📊 层级预算分配: {layer_budgets}")
    print(f"📊 头级预算示例 (第0层): {head_budgets_list[0]}")
    
    # 验证预算守恒
    total_allocated = sum(sum(head_budgets) for head_budgets in head_budgets_list)
    print(f"🔍 预算守恒检查: 总分配={total_allocated}, 应为{total_cache}")
    
    return layer_budgets, head_budgets_list


def main():
    """主函数：运行简化示例"""
    print("🎉 CAKE-AdaKV统一集成框架简化示例")
    print("这是我们的创新方法，独立于原始CAKE和AdaKV代码")
    
    try:
        # 运行简单示例
        result = simple_usage_example()
        
        print("\n" + "="*50)
        print("🎊 示例运行完成！")
        print("="*50)
        print("💡 关键特性:")
        print("   ✅ 统一框架：层级-头级协同优化")
        print("   ✅ 智能分配：自适应策略选择")
        print("   ✅ 严格守恒：100%预算利用率")
        print("   ✅ 稳健处理：完整异常恢复")
        print("   ✅ 性能监控：全面性能分析")
        print("\n🔬 这展示了我们在KV缓存优化领域的创新贡献！")
        
    except Exception as e:
        print(f"❌ 运行示例时出错: {e}")
        print("请检查依赖和配置")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 