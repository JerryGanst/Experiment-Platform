#!/usr/bin/env python3
"""
增强版CAKE子进程适配器测试脚本

测试改进功能：
1. CAKE_ROOT环境变量支持
2. 增强的Hydra参数映射  
3. 统一返回接口格式
4. 流式日志处理
5. 跨平台兼容性
"""

import os
import sys
import logging
import tempfile
from pathlib import Path

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from models.cake_subprocess_adapter import CAKESubprocessAdapter, CAKESubprocessConfig

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_config_param_mapping():
    """测试配置参数映射"""
    print("\n" + "="*60)
    print("测试 1: 配置参数映射")
    print("="*60)
    
    config = CAKESubprocessConfig(
        experiment_name="test_mapping",
        cache_size=512,
        window_size=64,
        gamma=0.9,
        devices=2,
        accelerator="gpu",
        # 测试额外参数
        max_epochs=5,
        limit_train_batches=100
    )
    
    args = config.to_hydra_args()
    print("生成的Hydra参数:")
    for arg in args:
        print(f"  {arg}")
    
    # 验证关键映射
    expected_mappings = [
        "cake.cache_size=512",
        "cake.window_size=64", 
        "cake.gamma=0.9",
        "trainer.devices=2",
        "trainer.accelerator=gpu"
    ]
    
    args_str = " ".join(args)
    passed = all(mapping in args_str for mapping in expected_mappings)
    
    print(f"\n参数映射测试: {'✅ 通过' if passed else '❌ 失败'}")
    return passed

def test_environment_variable_support():
    """测试环境变量支持"""
    print("\n" + "="*60)
    print("测试 2: CAKE_ROOT环境变量支持")
    print("="*60)
    
    # 备份原环境变量
    original_cake_root = os.environ.get("CAKE_ROOT")
    
    # 测试无效路径 - 新逻辑会警告但继续查找其他位置
    invalid_path = "/invalid/cake/path"
    os.environ["CAKE_ROOT"] = invalid_path
    
    try:
        adapter = CAKESubprocessAdapter()
        print("✅ 适配器正确处理无效CAKE_ROOT，继续查找其他位置")
        result = True
    except FileNotFoundError as e:
        print(f"✅ 在所有位置都无效时正确抛出异常: {e}")
        result = True
    
    # 测试有效路径设置（如果我们知道实际路径）
    current_dir = Path(__file__).parent
    valid_cake_path = current_dir / "cakekv-main" / "cakekv-main"
    
    if valid_cake_path.exists():
        os.environ["CAKE_ROOT"] = str(valid_cake_path)
        try:
            adapter2 = CAKESubprocessAdapter()
            print(f"✅ 有效CAKE_ROOT环境变量工作正常: {adapter2.cake_root}")
        except Exception as e:
            print(f"❌ 有效CAKE_ROOT设置失败: {e}")
            result = False
    
    # 恢复环境变量
    if original_cake_root:
        os.environ["CAKE_ROOT"] = original_cake_root
    else:
        os.environ.pop("CAKE_ROOT", None)
    
    print(f"\n环境变量支持测试: {'✅ 通过' if result else '❌ 失败'}")
    return result

def test_value_formatting():
    """测试值格式化"""
    print("\n" + "="*60)
    print("测试 3: 参数值格式化")
    print("="*60)
    
    config = CAKESubprocessConfig()
    
    # 测试不同类型的值
    test_cases = [
        (True, "true"),
        (False, "false"),
        ([1, 2, 3], "1,2,3"),
        (["a", "b"], "a,b"),
        (42, "42"),
        (3.14, "3.14"),
        ("text", "text")
    ]
    
    passed = True
    for value, expected in test_cases:
        result = config._format_value(value)
        if result == expected:
            print(f"✅ {value} -> {result}")
        else:
            print(f"❌ {value} -> {result} (期望: {expected})")
            passed = False
    
    print(f"\n值格式化测试: {'✅ 通过' if passed else '❌ 失败'}")
    return passed

def test_legacy_config_conversion():
    """测试从旧配置格式转换"""
    print("\n" + "="*60)
    print("测试 4: 旧配置格式转换")
    print("="*60)
    
    # 模拟旧配置格式
    cake_experiment_config = {
        "layer_allocation_strategies": ["adaptive", "uniform"],
        "cache_budgets": [0.6, 0.8]
    }
    
    cake_model_config = {
        "window_size": 128,
        "gamma": 0.85,
        "tau1": 1.2,
        "tau2": 0.8
    }
    
    config = CAKESubprocessConfig.from_legacy_config(
        cake_experiment_config,
        cake_model_config
    )
    
    # 验证转换结果
    checks = [
        (config.cache_size, int(2048 * 0.6)),  # 第一个预算
        (config.window_size, 128),
        (config.gamma, 0.85),
        (config.tau1, 1.2),
        (config.tau2, 0.8),
        (config.experiment_name, "cake_adaptive")  # 第一个策略
    ]
    
    passed = True
    for actual, expected in checks:
        if actual == expected:
            print(f"✅ {actual} == {expected}")
        else:
            print(f"❌ {actual} != {expected}")
            passed = False
    
    print(f"\n旧配置转换测试: {'✅ 通过' if passed else '❌ 失败'}")
    return passed

def test_adapter_initialization():
    """测试适配器初始化"""
    print("\n" + "="*60)
    print("测试 5: 适配器初始化")
    print("="*60)
    
    try:
        adapter = CAKESubprocessAdapter()
        print(f"✅ 适配器初始化成功")
        print(f"CAKE根目录: {adapter.cake_root}")
        print(f"主脚本: {adapter.main_script}")
        
        # 检查路径有效性
        path_valid = adapter._is_valid_cake_root(adapter.cake_root)
        print(f"路径验证: {'✅ 有效' if path_valid else '❌ 无效'}")
        
        return True
        
    except Exception as e:
        print(f"❌ 适配器初始化失败: {e}")
        print("提示: 请确保CAKE代码在正确位置或设置CAKE_ROOT环境变量")
        return False

def test_error_result_format():
    """测试错误结果格式"""
    print("\n" + "="*60)
    print("测试 6: 错误结果格式统一性")
    print("="*60)
    
    try:
        adapter = CAKESubprocessAdapter()
        
        # 创建测试配置和临时目录
        config = CAKESubprocessConfig(experiment_name="error_test")
        temp_dir = Path(tempfile.mkdtemp())
        
        # 测试错误结果创建
        error_result = adapter._create_error_result(
            "test_error", 
            "这是一个测试错误",
            config,
            temp_dir
        )
        
        # 验证结果格式
        required_keys = ["status", "error", "metrics", "config", "artifacts"]
        passed = all(key in error_result for key in required_keys)
        
        if passed:
            print("✅ 错误结果包含所有必需字段:")
            for key in required_keys:
                print(f"  {key}: {type(error_result[key])}")
        else:
            print("❌ 错误结果缺少必需字段")
        
        print(f"\n错误结果格式测试: {'✅ 通过' if passed else '❌ 失败'}")
        return passed
        
    except Exception as e:
        print(f"❌ 错误结果格式测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("增强版CAKE子进程适配器测试开始")
    print("="*80)
    
    tests = [
        ("配置参数映射", test_config_param_mapping),
        ("环境变量支持", test_environment_variable_support), 
        ("值格式化", test_value_formatting),
        ("旧配置转换", test_legacy_config_conversion),
        ("适配器初始化", test_adapter_initialization),
        ("错误结果格式", test_error_result_format),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ 测试 '{test_name}' 异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name:20} : {status}")
    
    print(f"\n总体结果: {passed_count}/{total_count} 通过")
    
    if passed_count == total_count:
        print("🎉 所有测试通过！增强版适配器工作正常")
        return 0
    else:
        print("⚠️  部分测试失败，需要检查实现")
        return 1

if __name__ == "__main__":
    exit(main()) 