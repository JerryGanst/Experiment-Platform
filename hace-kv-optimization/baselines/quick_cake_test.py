#!/usr/bin/env python3
"""
CAKE快速验证测试脚本

这个脚本用于快速验证CAKE与实验平台的集成是否正常工作，
使用 cakekv-main 中的核心CAKE功能。
"""

import os
import sys
import torch
import logging
from datetime import datetime

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.dirname(current_dir)
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

# 导入配置和模块
from hace_core import config
from hace_core.models.model_loader import load_model_and_tokenizer, prepare_model_for_cake, is_model_type_supported_by_cake
from hace_core.models.cake_converter import (
    is_cake_available, 
    get_supported_model_types, 
    CAKEExperimentConfig, 
    CAKEModelAdapter,
    apply_cake_to_model
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_cake_availability():
    """测试CAKE核心功能是否可用"""
    logger.info("=== 测试CAKE核心功能可用性 ===")
    
    if is_cake_available():
        logger.info("✅ CAKE核心功能可用")
        supported_models = get_supported_model_types()
        logger.info(f"支持的模型类型: {supported_models}")
        
        if not supported_models:
            logger.warning("支持的模型类型列表为空")
            return False
        
        return True
    else:
        logger.error("❌ CAKE核心功能不可用")
        logger.error("请检查 cakekv-main 目录是否存在且正确")
        return False

def test_experiment_config():
    """测试CAKE实验配置"""
    logger.info("=== 测试CAKE实验配置 ===")
    
    try:
        # 测试不同的配置
        configs = [
            {
                "allocation_strategy": "uniform",
                "cache_budget": 0.8,
                "window_size": 32
            },
            {
                "allocation_strategy": "adaptive", 
                "cache_budget": 0.6,
                "window_size": 64,
                "gamma": 0.9,
                "tau1": 1.2,
                "tau2": 0.8
            },
            {
                "allocation_strategy": "attention_based",
                "cache_budget": 0.7,
                "window_size": 16
            }
        ]
        
        for i, config_dict in enumerate(configs):
            logger.info(f"测试配置 {i+1}: {config_dict['allocation_strategy']}")
            
            experiment_config = CAKEExperimentConfig(**config_dict)
            
            # 验证配置属性
            assert experiment_config.allocation_strategy == config_dict["allocation_strategy"]
            assert experiment_config.cache_budget == config_dict["cache_budget"]
            assert experiment_config.window_size == config_dict["window_size"]
            
            logger.info(f"配置 {i+1} 验证通过")
        
        logger.info("CAKE实验配置测试通过！\n")
        return True
        
    except Exception as e:
        logger.error(f"CAKE实验配置测试失败: {e}")
        return False

def test_model_adapter():
    """测试CAKE模型适配器的基本功能"""
    logger.info("=== 测试CAKE模型适配器 ===")
    
    try:
        # 创建测试配置
        test_config = CAKEExperimentConfig(
            allocation_strategy="adaptive",
            cache_budget=0.8,
            window_size=32
        )
        
        # 创建适配器
        adapter = CAKEModelAdapter(test_config)
        
        # 测试支持的模型检查
        supported_models = get_supported_model_types()
        if supported_models:
            logger.info(f"适配器支持的模型: {adapter.supported_models.keys()}")
            
            # 验证支持的模型类型一致
            for model_type in supported_models:
                assert model_type in adapter.supported_models, f"模型类型 {model_type} 不在适配器中"
            
            logger.info("模型适配器验证通过")
        else:
            logger.warning("无法获取支持的模型类型，跳过适配器验证")
        
        logger.info("CAKE模型适配器测试通过！\n")
        return True
        
    except Exception as e:
        logger.error(f"CAKE模型适配器测试失败: {e}")
        return False

def test_layer_budget_calculation():
    """测试层级预算计算"""
    logger.info("=== 测试层级预算计算 ===")
    
    try:
        test_config = CAKEExperimentConfig(
            allocation_strategy="adaptive",
            cache_budget=0.8,
            window_size=32
        )
        
        adapter = CAKEModelAdapter(test_config)
        
        # 模拟模型配置
        mock_model_configs = [
            {"num_hidden_layers": 12, "max_position_embeddings": 2048},
            {"num_hidden_layers": 24, "max_position_embeddings": 4096},
            {"num_hidden_layers": 6, "max_position_embeddings": 1024}
        ]
        
        for config in mock_model_configs:
            num_layers = config["num_hidden_layers"]
            max_seq_len = config["max_position_embeddings"]
            
            budgets = adapter._calculate_layer_budgets(num_layers, config)
            
            # 验证预算计算
            assert len(budgets) == num_layers, f"预算数量不匹配: {len(budgets)} != {num_layers}"
            assert all(b > 0 for b in budgets), "所有层的预算都应该大于0"
            
            total_budget = sum(budgets)
            expected_total = int(max_seq_len * test_config.cache_budget)
            
            # 允许一定的误差（由于调整过程）
            budget_diff = abs(total_budget - expected_total)
            assert budget_diff <= num_layers, f"总预算偏差过大: {budget_diff}"
            
            logger.info(f"层数={num_layers}, 总预算={total_budget}, 期望={expected_total}")
        
        logger.info("层级预算计算测试通过！\n")
        return True
        
    except Exception as e:
        logger.error(f"层级预算计算测试失败: {e}")
        return False

def test_integration_with_config():
    """测试与配置系统的集成"""
    logger.info("=== 测试与配置系统集成 ===")
    
    try:
        # 测试从配置中提取CAKE参数
        test_cake_exp_config = {
            "layer_allocation_strategies": ["adaptive", "uniform"],
            "cache_budgets": [0.8, 0.6],
            "layer_analysis_configs": {
                "attention_pattern_analysis": False,
                "layer_importance_scoring": True
            }
        }
        
        test_cake_model_config = {
            "window_size": 32,
            "gamma": 0.8,
            "tau1": 1.0,
            "tau2": 1.0
        }
        
        # 模拟HuggingFace配置
        mock_hf_config = {
            "model_type": "llama",
            "num_hidden_layers": 12,
            "max_position_embeddings": 2048
        }
        
        # 测试配置解析（模拟apply_cake_to_model的配置解析部分）
        allocation_strategy = test_cake_exp_config["layer_allocation_strategies"][0]
        cache_budget = test_cake_exp_config["cache_budgets"][0]
        window_size = test_cake_model_config["window_size"]
        
        # 创建实验配置
        experiment_config = CAKEExperimentConfig(
            allocation_strategy=allocation_strategy,
            cache_budget=cache_budget,
            window_size=window_size,
            gamma=test_cake_model_config["gamma"],
            tau1=test_cake_model_config["tau1"],
            tau2=test_cake_model_config["tau2"]
        )
        
        # 验证配置正确解析
        assert experiment_config.allocation_strategy == "adaptive"
        assert experiment_config.cache_budget == 0.8
        assert experiment_config.window_size == 32
        
        logger.info("配置系统集成测试通过！\n")
        return True
        
    except Exception as e:
        logger.error(f"配置系统集成测试失败: {e}")
        return False

def test_model_integration():
    """测试CAKE与模型的集成（可选，需要实际模型）"""
    logger.info("=== 测试CAKE模型集成 ===")
    
    try:
        if not is_cake_available():
            logger.warning("CAKE不可用，跳过模型集成测试")
            return True
        
        # 使用小型模型进行测试
        test_model_config = {
            "model_name_or_path": "facebook/opt-125m",  # 小型模型，快速加载
            "precision": "fp16"
        }
        
        logger.info(f"加载测试模型: {test_model_config['model_name_or_path']}")
        model, tokenizer = load_model_and_tokenizer(test_model_config)
        
        # 检查模型是否支持CAKE
        is_supported = is_model_type_supported_by_cake(model, config.CAKE_MODEL_CONFIG)
        logger.info(f"模型是否支持CAKE: {is_supported}")
        
        if is_supported:
            # 准备CAKE配置
            cake_exp_config = {
                "layer_allocation_strategies": ["uniform"],
                "cache_budgets": [0.7],
                "layer_analysis_configs": {
                    "attention_pattern_analysis": False,
                    "layer_importance_scoring": True,
                    "dynamic_allocation": True
                }
            }
            
            # 应用CAKE
            logger.info("应用CAKE到模型...")
            cake_model = prepare_model_for_cake(model, cake_exp_config, config.CAKE_MODEL_CONFIG)
            
            # 验证模型是否已应用CAKE
            assert hasattr(cake_model, 'is_cake_enabled'), "模型应该有is_cake_enabled属性"
            assert cake_model.is_cake_enabled, "模型应该已启用CAKE"
            
            # 测试简单推理
            test_input = "Hello, how are you?"
            inputs = tokenizer(test_input, return_tensors="pt")
            
            logger.info("测试CAKE模型推理...")
            with torch.no_grad():
                outputs = cake_model.generate(
                    **inputs,
                    max_new_tokens=10,
                    use_cache=True,
                    do_sample=False
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"模型输出: {response}")
            
            logger.info("CAKE模型集成测试通过！")
        else:
            logger.warning("模型不支持CAKE，跳过集成测试")
            
    except Exception as e:
        logger.error(f"模型集成测试失败: {e}")
        logger.warning("这可能是由于模型下载、硬件限制或CAKE核心功能问题导致的")
        return False
    
    logger.info("")
    return True

def main():
    """主测试函数"""
    logger.info(f"开始CAKE实验平台集成验证测试 - {datetime.now()}")
    logger.info("=" * 70)
    
    test_results = []
    
    # 按顺序执行测试
    tests = [
        ("CAKE核心功能可用性", test_cake_availability),
        ("CAKE实验配置", test_experiment_config),
        ("CAKE模型适配器", test_model_adapter),
        ("层级预算计算", test_layer_budget_calculation),
        ("配置系统集成", test_integration_with_config),
    ]
    
    # 如果不跳过模型测试，添加模型集成测试
    if "--skip-model" not in sys.argv:
        tests.append(("CAKE模型集成", test_model_integration))
    else:
        logger.info("跳过模型集成测试（使用 --skip-model 参数）")
    
    for test_name, test_func in tests:
        try:
            logger.info(f"开始测试: {test_name}")
            result = test_func()
            test_results.append((test_name, result))
            
            if result:
                logger.info(f"✅ {test_name} - 通过")
            else:
                logger.error(f"❌ {test_name} - 失败")
                
        except Exception as e:
            logger.error(f"❌ {test_name} - 异常: {e}")
            test_results.append((test_name, False))
        
        logger.info("-" * 50)
    
    # 汇总结果
    logger.info("=" * 70)
    logger.info("测试结果汇总:")
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"总体结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！您的CAKE实验平台集成已准备就绪。")
        logger.info("现在可以开始运行完整的CAKE实验了。")
        return 0
    else:
        logger.error(f"有 {total - passed} 项测试失败，请检查上述错误信息并修复相关问题。")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 