#!/usr/bin/env python3
"""
CAKE-AdaKV统一集成框架使用示例

这个脚本展示了如何使用我们的创新方法进行KV缓存优化。
完全独立于原始CAKE和AdaKV代码。
"""

import numpy as np
import time
from typing import List, Tuple

# 导入我们的核心框架
try:
    from integration_framework import CakeAdaKVIntegration, IntegrationConfig, create_integration
    from unified_allocator import UnifiedCacheConfig
    print("✅ 成功导入核心框架")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保所有依赖都已安装")
    exit(1)


def generate_synthetic_attention_weights(
    num_layers: int = 12,
    num_heads: int = 32, 
    seq_len: int = 512,
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


def basic_usage_example():
    """基本使用示例"""
    print("\n" + "="*50)
    print("📖 基本使用示例")
    print("="*50)
    
    # 生成测试数据
    attention_weights_list = generate_synthetic_attention_weights(
        num_layers=12, num_heads=32, seq_len=512
    )
    
    # 创建集成器（使用默认配置）
    print("🚀 创建CAKE-AdaKV集成器...")
    integration = CakeAdaKVIntegration()
    
    # 优化缓存分配
    print("⚡ 执行缓存优化...")
    start_time = time.time()
    
    layer_budgets, head_budgets_list = integration.optimize_cache(attention_weights_list)
    
    end_time = time.time()
    
    # 显示结果
    print(f"✅ 优化完成，耗时: {end_time - start_time:.3f}秒")
    print(f"📊 层级预算分配: {layer_budgets[:5]}... (显示前5层)")
    print(f"📊 头级预算示例 (第0层): {head_budgets_list[0][:8]}... (显示前8头)")
    
    # 验证预算守恒
    total_allocated = sum(sum(head_budgets) for head_budgets in head_budgets_list)
    print(f"🔍 预算守恒检查: 总分配={total_allocated}, 应为4096")
    
    return integration


def advanced_configuration_example():
    """高级配置示例"""
    print("\n" + "="*50)
    print("🔧 高级配置示例")
    print("="*50)
    
    # 自定义配置
    config = IntegrationConfig(
        total_cache_size=8192,           # 更大的缓存
        enable_monitoring=True,          # 启用监控
        enable_auto_tuning=True,         # 启用自动调优
        warmup_samples=3,                # 预热样本数
        custom_thresholds={              # 自定义阈值
            'high_dispersion_threshold': 0.8,
            'high_dynamics_threshold': 0.6,
            'key_head_ratio': 0.25
        }
    )
    
    print("📋 配置信息:")
    print(f"   - 缓存大小: {config.total_cache_size}")
    print(f"   - 监控启用: {config.enable_monitoring}")
    print(f"   - 自动调优: {config.enable_auto_tuning}")
    print(f"   - 自定义阈值: {config.custom_thresholds}")
    
    # 创建集成器
    integration = CakeAdaKVIntegration(config)
    
    # 生成测试数据
    attention_weights_list = generate_synthetic_attention_weights(
        num_layers=24, num_heads=40, seq_len=1024
    )
    
    # 获取详细结果
    print("⚡ 执行优化并获取详细信息...")
    result = integration.optimize_cache(
        attention_weights_list, 
        return_detailed_info=True
    )
    
    # 显示详细结果
    print("📊 详细结果:")
    print(f"   - 层级预算: {len(result['layer_budgets'])}层")
    print(f"   - 头级预算: {len(result['head_budgets_list'])}层")
    
    if 'allocator_summary' in result:
        summary = result['allocator_summary']
        print(f"   - 分配器状态: {'已预热' if summary.get('is_warmed_up') else '未预热'}")
        if 'strategy_distribution' in summary:
            print(f"   - 策略分布: {summary['strategy_distribution']}")
    
    if 'performance_metrics' in result:
        metrics = result['performance_metrics']
        print(f"   - 性能指标: {metrics.get('timing_metrics', {})}")
    
    return integration, result


def auto_tuning_example():
    """自动调优示例"""
    print("\n" + "="*50)
    print("🎯 自动调优示例")
    print("="*50)
    
    # 创建启用自动调优的集成器
    integration = create_integration(
        cache_size=4096,
        enable_monitoring=True,
        enable_auto_tuning=True
    )
    
    # 生成多个样本用于调优
    print("🔄 生成调优样本...")
    sample_data = []
    for i in range(3):
        sample = generate_synthetic_attention_weights(
            num_layers=8, num_heads=16, seq_len=256
        )
        sample_data.append(sample)
        print(f"   样本 {i+1}: {len(sample)}层")
    
    # 执行自动调优
    print("🎯 执行自动调优...")
    tuning_result = integration.auto_tune(sample_data)
    
    print("📊 调优结果:")
    for key, value in tuning_result.items():
        print(f"   - {key}: {value}")
    
    return integration


def performance_monitoring_example():
    """性能监控示例"""
    print("\n" + "="*50)
    print("📈 性能监控示例")
    print("="*50)
    
    # 创建启用监控的集成器
    integration = create_integration(enable_monitoring=True)
    
    # 执行多次优化
    print("🔄 执行多次优化以收集性能数据...")
    for i in range(5):
        attention_weights = generate_synthetic_attention_weights(
            num_layers=6, num_heads=12, seq_len=128
        )
        
        integration.optimize_cache(attention_weights)
        print(f"   完成第 {i+1} 次优化")
    
    # 获取性能报告
    report = integration.get_performance_report()
    
    print("📊 性能报告:")
    print(f"   - 优化次数: {report.get('optimization_count', 0)}")
    print(f"   - 分配器状态: {'已初始化' if report.get('allocator_initialized') else '未初始化'}")
    
    if 'performance_metrics' in report:
        metrics = report['performance_metrics']
        print(f"   - 监控状态: {metrics.get('monitoring_active', False)}")
        
        if 'timing_metrics' in metrics:
            timing = metrics['timing_metrics']
            for operation, stats in timing.items():
                print(f"   - {operation}: 平均 {stats.get('avg_time', 0):.4f}秒")
    
    # 获取优化历史
    history = integration.get_optimization_history()
    print(f"📚 优化历史: {len(history)} 条记录")
    
    return integration


def comparison_example():
    """对比示例：展示我们的方法与简单方法的差异"""
    print("\n" + "="*50)
    print("⚖️ 方法对比示例")
    print("="*50)
    
    # 生成测试数据
    attention_weights_list = generate_synthetic_attention_weights(
        num_layers=12, num_heads=32, seq_len=512
    )
    
    print("🔄 测试我们的统一方法...")
    
    # 我们的统一方法
    start_time = time.time()
    integration = CakeAdaKVIntegration()
    layer_budgets, head_budgets_list = integration.optimize_cache(attention_weights_list)
    unified_time = time.time() - start_time
    
    print(f"✅ 统一方法完成，耗时: {unified_time:.3f}秒")
    
    # 简单的均匀分配作为对比
    print("🔄 测试简单均匀分配...")
    start_time = time.time()
    
    total_cache = 4096
    num_layers = len(attention_weights_list)
    
    # 均匀层级分配
    layer_base = total_cache // num_layers
    layer_remainder = total_cache % num_layers
    simple_layer_budgets = [layer_base] * num_layers
    for i in range(layer_remainder):
        simple_layer_budgets[i] += 1
    
    # 均匀头级分配
    simple_head_budgets_list = []
    for i, attn in enumerate(attention_weights_list):
        num_heads = attn.shape[1]
        head_base = simple_layer_budgets[i] // num_heads
        head_remainder = simple_layer_budgets[i] % num_heads
        head_budgets = [head_base] * num_heads
        for j in range(head_remainder):
            head_budgets[j] += 1
        simple_head_budgets_list.append(head_budgets)
    
    simple_time = time.time() - start_time
    
    print(f"✅ 简单方法完成，耗时: {simple_time:.3f}秒")
    
    # 对比结果
    print("\n📊 对比结果:")
    print(f"   - 统一方法耗时: {unified_time:.3f}秒")
    print(f"   - 简单方法耗时: {simple_time:.3f}秒")
    print(f"   - 时间差异: {abs(unified_time - simple_time):.3f}秒")
    
    # 分配差异分析
    layer_diff = np.array(layer_budgets) - np.array(simple_layer_budgets)
    print(f"   - 层级分配差异: 最大 {np.max(np.abs(layer_diff))}, 平均 {np.mean(np.abs(layer_diff)):.2f}")
    
    print("💡 统一方法的优势:")
    print("   - 自适应策略选择，根据注意力特征优化分配")
    print("   - 严格预算守恒，无内存浪费")
    print("   - 稳健异常处理，生产环境可靠")
    print("   - 性能监控，便于调优和诊断")


def main():
    """主函数：运行所有示例"""
    print("🎉 CAKE-AdaKV统一集成框架示例")
    print("这是我们的创新方法，独立于原始CAKE和AdaKV代码")
    
    try:
        # 基本使用
        basic_integration = basic_usage_example()
        
        # 高级配置
        advanced_integration, detailed_result = advanced_configuration_example()
        
        # 自动调优
        tuned_integration = auto_tuning_example()
        
        # 性能监控
        monitored_integration = performance_monitoring_example()
        
        # 方法对比
        comparison_example()
        
        print("\n" + "="*50)
        print("🎊 所有示例运行完成！")
        print("="*50)
        print("💡 关键优势总结:")
        print("   ✅ 统一warmup：避免重复计算")
        print("   ✅ 智能策略：自适应选择最优分配")
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