#!/usr/bin/env python3
"""
YAML配置支持模块

基于新设计方案的建议，增加对YAML配置文件的支持，
同时保持与现有字典配置的兼容性。
"""

import os
import yaml
import logging
from typing import Dict, List, Any, Union, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    model_type: str
    context_length: int
    precision: str = "fp16"
    device: str = "auto"

@dataclass 
class CakeConfig:
    """CAKE方法配置"""
    cache_size: int
    window_size: int = 32
    allocation_strategy: str = "adaptive"
    gamma: float = 0.8
    tau1: float = 1.0
    tau2: float = 1.0
    cascading: bool = True

@dataclass
class ExperimentConfig:
    """单个实验配置"""
    name: str
    model: ModelConfig
    method: str  # "CAKE", "baseline", "HeadKV", "HACE"
    dataset: str
    cake_config: Optional[CakeConfig] = None
    
    def to_legacy_format(self) -> tuple:
        """转换为当前系统使用的格式"""
        model_config = {
            "model_name_or_path": self.model.name,
            "precision": self.model.precision
        }
        
        if self.cake_config:
            cake_exp_config = {
                "layer_allocation_strategies": [self.cake_config.allocation_strategy],
                "cache_budgets": [self.cake_config.cache_size / self.model.context_length],
                "layer_analysis_configs": {
                    "attention_pattern_analysis": False,
                    "layer_importance_scoring": True,
                    "dynamic_allocation": True
                }
            }
            
            cake_model_config = {
                "window_size": self.cake_config.window_size,
                "gamma": self.cake_config.gamma,
                "tau1": self.cake_config.tau1,
                "tau2": self.cake_config.tau2
            }
        else:
            cake_exp_config = {}
            cake_model_config = {}
            
        return model_config, cake_exp_config, cake_model_config

@dataclass
class BatchExperimentConfig:
    """批量实验配置"""
    experiments: List[ExperimentConfig]
    output_dir: str = "results"
    parallel: bool = False
    max_workers: int = 4

class ConfigManager:
    """配置管理器：支持YAML和字典两种格式"""
    
    @staticmethod
    def load_from_yaml(config_path: Union[str, Path]) -> BatchExperimentConfig:
        """从YAML文件加载配置"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        experiments = []
        for exp_data in data.get("experiments", []):
            # 解析模型配置
            model_data = exp_data["model"]
            model_config = ModelConfig(**model_data)
            
            # 解析CAKE配置（如果存在）
            cake_config = None
            if "cake_config" in exp_data:
                cake_config = CakeConfig(**exp_data["cake_config"])
            
            # 创建实验配置
            experiment = ExperimentConfig(
                name=exp_data["name"],
                model=model_config,
                method=exp_data["method"],
                dataset=exp_data["dataset"],
                cake_config=cake_config
            )
            experiments.append(experiment)
        
        return BatchExperimentConfig(
            experiments=experiments,
            output_dir=data.get("output_dir", "results"),
            parallel=data.get("parallel", False),
            max_workers=data.get("max_workers", 4)
        )
    
    @staticmethod
    def save_to_yaml(config: BatchExperimentConfig, config_path: Union[str, Path]):
        """保存配置到YAML文件"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "experiments": [asdict(exp) for exp in config.experiments],
            "output_dir": config.output_dir,
            "parallel": config.parallel,
            "max_workers": config.max_workers
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    @staticmethod
    def create_example_config() -> str:
        """创建示例配置"""
        return """
# CAKE实验平台配置示例
experiments:
  - name: "cake_llama_4k"
    model:
      name: "meta-llama/Llama-2-7b-hf"
      model_type: "llama"
      context_length: 4096
      precision: "fp16"
      device: "cuda:0"
    method: "CAKE"
    dataset: "longbench_qa"
    cake_config:
      cache_size: 1024
      window_size: 32
      allocation_strategy: "adaptive"
      gamma: 0.8
      tau1: 1.0
      tau2: 1.0
      cascading: true
  
  - name: "baseline_llama_4k"
    model:
      name: "meta-llama/Llama-2-7b-hf"
      model_type: "llama"
      context_length: 4096
      precision: "fp16"
      device: "cuda:0"
    method: "baseline"
    dataset: "longbench_qa"
  
  - name: "cake_llama_16k"
    model:
      name: "meta-llama/Llama-2-7b-hf"
      model_type: "llama"
      context_length: 16384
      precision: "fp16"
      device: "cuda:0"
    method: "CAKE"
    dataset: "longbench_qa"
    cake_config:
      cache_size: 2048
      window_size: 64
      allocation_strategy: "attention_based"
      gamma: 0.9
      tau1: 1.2
      tau2: 0.8
      cascading: true

# 批量执行配置
output_dir: "results/longbench_comparison"
parallel: false
max_workers: 2
"""

def validate_config(config: ExperimentConfig) -> List[str]:
    """验证配置有效性"""
    errors = []
    
    # 验证方法
    if config.method not in ["CAKE", "baseline", "HeadKV", "HACE"]:
        errors.append(f"不支持的方法: {config.method}")
    
    # 验证CAKE配置
    if config.method == "CAKE" and not config.cake_config:
        errors.append("CAKE方法需要cake_config配置")
    
    if config.cake_config:
        if config.cake_config.cache_size <= 0:
            errors.append("cache_size必须大于0")
        if config.cake_config.window_size <= 0:
            errors.append("window_size必须大于0")
        if not (0 < config.cake_config.gamma <= 1):
            errors.append("gamma必须在(0,1]范围内")
    
    return errors

if __name__ == "__main__":
    # 创建示例配置文件
    example_config = ConfigManager.create_example_config()
    
    config_path = Path("experiments_example.yaml")
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(example_config)
    
    print(f"✅ 示例配置文件已创建: {config_path}")
    
    # 测试加载
    try:
        batch_config = ConfigManager.load_from_yaml(config_path)
        print(f"✅ 成功加载 {len(batch_config.experiments)} 个实验配置")
        
        # 验证配置
        for exp in batch_config.experiments:
            errors = validate_config(exp)
            if errors:
                print(f"❌ 配置 {exp.name} 有误: {errors}")
            else:
                print(f"✅ 配置 {exp.name} 验证通过")
                
    except Exception as e:
        print(f"❌ 配置加载失败: {e}") 