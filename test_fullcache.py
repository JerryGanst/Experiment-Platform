#!/usr/bin/env python3
"""
简单的FullKV Cache测试脚本
清理版本，避免路径混乱
"""

import os
import sys

# 确保在正确的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"当前工作目录: {current_dir}")

# 添加路径
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def test_fullcache():
    """测试fullcache基本功能"""
    print("🚀 开始测试FullKV Cache...")
    
    try:
        # 导入baselines中的fullkvcache_main
        from baselines.fullkvcache_main import main
        print("✅ 成功导入fullkvcache_main")
        
        # 设置测试参数
        sys.argv = [
            'test_fullcache.py',
            '--datasets', 'hotpotqa',
            '--kv_cache_lengths', '128',
            '--batch_size', '1',
            '--max_new_tokens', '50',
            '--repetitions', '1',
            '--max_samples', '5'  # 只测试5个样本
        ]
        
        print("📊 测试参数:")
        print(f"  - 数据集: hotpotqa")
        print(f"  - KV长度: 128") 
        print(f"  - 批大小: 1")
        print(f"  - 最大新令牌: 50")
        print(f"  - 样本数: 5")
        
        # 运行测试
        main()
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fullcache() 