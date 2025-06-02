#!/usr/bin/env python3
"""
批量实验执行器

基于新设计方案实现的批量实验执行功能，支持YAML配置文件和并行执行。
结合了原有的桥接层架构优势和新方案的批量执行优势。
"""

import os
import sys
import json
import logging
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from config_yaml_example import ConfigManager, BatchExperimentConfig, ExperimentConfig, validate_config
from models.model_loader import load_model_and_tokenizer, prepare_model_for_cake, is_model_type_supported_by_cake
from models.cake_converter import is_cake_available, get_supported_model_types, apply_cake_to_model
import config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExperimentResult:
    """实验结果类"""
    
    def __init__(self, experiment_name: str):
        self.name = experiment_name
        self.status = "pending"  # pending, running, completed, failed
        self.start_time = None
        self.end_time = None
        self.metrics = {}
        self.error = None
        
    def start(self):
        self.status = "running"
        self.start_time = datetime.now()
        
    def complete(self, metrics: Dict[str, Any]):
        self.status = "completed"
        self.end_time = datetime.now()
        self.metrics = metrics
        
    def fail(self, error: str):
        self.status = "failed"
        self.end_time = datetime.now()
        self.error = error
        
    def duration(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration(),
            "metrics": self.metrics,
            "error": self.error
        }

class BatchExperimentRunner:
    """批量实验执行器"""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
    def run_single_experiment(self, exp_config: ExperimentConfig) -> ExperimentResult:
        """运行单个实验"""
        result = ExperimentResult(exp_config.name)
        result.start()
        
        try:
            logger.info(f"开始实验: {exp_config.name}")
            
            # 验证配置
            config_errors = validate_config(exp_config)
            if config_errors:
                raise ValueError(f"配置错误: {config_errors}")
            
            # 转换为旧格式配置
            model_config, cake_exp_config, cake_model_config = exp_config.to_legacy_format()
            
            # 加载模型
            logger.info(f"加载模型: {model_config['model_name_or_path']}")
            model, tokenizer = load_model_and_tokenizer(model_config)
            
            # 应用方法
            if exp_config.method == "CAKE":
                if not is_cake_available():
                    raise RuntimeError("CAKE核心功能不可用")
                
                # 检查模型支持
                if not is_model_type_supported_by_cake(model, config.CAKE_MODEL_CONFIG):
                    raise ValueError(f"模型类型不支持CAKE: {model.config.model_type}")
                
                # 应用CAKE
                logger.info("应用CAKE优化...")
                model = prepare_model_for_cake(model, cake_exp_config, cake_model_config)
                
            elif exp_config.method == "baseline":
                logger.info("使用基线方法（无压缩）")
                # 基线方法不需要特殊处理
                
            else:
                raise ValueError(f"不支持的方法: {exp_config.method}")
            
            # 模拟推理和评估（实际项目中这里会调用具体的推理和评估代码）
            metrics = self._simulate_inference_and_evaluation(
                model, tokenizer, exp_config
            )
            
            result.complete(metrics)
            logger.info(f"实验完成: {exp_config.name}, 指标: {metrics}")
            
        except Exception as e:
            error_msg = f"实验失败: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            result.fail(str(e))
        
        return result
    
    def _simulate_inference_and_evaluation(self, model, tokenizer, exp_config: ExperimentConfig) -> Dict[str, Any]:
        """模拟推理和评估过程"""
        # 这里是一个简化的模拟，实际实现中应该调用真实的推理和评估代码
        
        # 简单的推理测试
        test_input = "这是一个测试输入，用于验证模型功能。"
        inputs = tokenizer(test_input, return_tensors="pt", max_length=512, truncation=True)
        
        import torch
        with torch.no_grad():
            # 简单的forward pass测试
            outputs = model(**inputs)
            
        # 模拟指标
        metrics = {
            "method": exp_config.method,
            "model_type": exp_config.model.model_type,
            "context_length": exp_config.model.context_length,
            "test_passed": True,
            "memory_usage_mb": torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        }
        
        if exp_config.cake_config:
            metrics.update({
                "cache_size": exp_config.cake_config.cache_size,
                "window_size": exp_config.cake_config.window_size,
                "allocation_strategy": exp_config.cake_config.allocation_strategy,
                "compression_ratio": exp_config.cake_config.cache_size / exp_config.model.context_length
            })
        
        return metrics
    
    def run_batch(self, batch_config: BatchExperimentConfig):
        """运行批量实验"""
        logger.info(f"开始批量实验，共 {len(batch_config.experiments)} 个实验")
        
        if batch_config.parallel:
            logger.info(f"使用并行执行，最大工作进程: {batch_config.max_workers}")
            self._run_parallel(batch_config.experiments, batch_config.max_workers)
        else:
            logger.info("使用串行执行")
            self._run_sequential(batch_config.experiments)
        
        # 保存结果
        self._save_results(batch_config.output_dir)
        self._generate_report()
    
    def _run_sequential(self, experiments: List[ExperimentConfig]):
        """串行执行实验"""
        for exp_config in experiments:
            result = self.run_single_experiment(exp_config)
            self.results.append(result)
    
    def _run_parallel(self, experiments: List[ExperimentConfig], max_workers: int):
        """并行执行实验"""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_config = {
                executor.submit(self.run_single_experiment, exp_config): exp_config
                for exp_config in experiments
            }
            
            # 收集结果
            for future in as_completed(future_to_config):
                result = future.result()
                self.results.append(result)
    
    def _save_results(self, output_dir: str):
        """保存实验结果"""
        results_file = self.output_dir / "experiment_results.json"
        
        results_data = {
            "summary": {
                "total_experiments": len(self.results),
                "completed": sum(1 for r in self.results if r.status == "completed"),
                "failed": sum(1 for r in self.results if r.status == "failed"),
                "timestamp": datetime.now().isoformat()
            },
            "experiments": [result.to_dict() for result in self.results]
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"实验结果已保存到: {results_file}")
    
    def _generate_report(self):
        """生成实验报告"""
        report_file = self.output_dir / "experiment_report.md"
        
        completed_results = [r for r in self.results if r.status == "completed"]
        failed_results = [r for r in self.results if r.status == "failed"]
        
        report_lines = [
            "# 批量实验报告",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 概览",
            f"- 总实验数: {len(self.results)}",
            f"- 成功完成: {len(completed_results)}",
            f"- 失败: {len(failed_results)}",
            f"- 成功率: {len(completed_results)/len(self.results)*100:.1f}%",
            "",
        ]
        
        if completed_results:
            report_lines.extend([
                "## 成功实验结果",
                "",
                "| 实验名称 | 方法 | 模型类型 | 上下文长度 | 耗时(秒) | 内存使用(MB) |",
                "|----------|------|----------|------------|----------|--------------|",
            ])
            
            for result in completed_results:
                metrics = result.metrics
                report_lines.append(
                    f"| {result.name} | {metrics.get('method', 'N/A')} | "
                    f"{metrics.get('model_type', 'N/A')} | {metrics.get('context_length', 'N/A')} | "
                    f"{result.duration():.1f} | {metrics.get('memory_usage_mb', 0):.1f} |"
                )
            
            report_lines.append("")
        
        if failed_results:
            report_lines.extend([
                "## 失败实验",
                "",
            ])
            
            for result in failed_results:
                report_lines.extend([
                    f"### {result.name}",
                    f"**错误**: {result.error}",
                    ""
                ])
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"实验报告已生成: {report_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="CAKE实验平台批量执行器")
    parser.add_argument("--config", type=str, required=True, help="YAML配置文件路径")
    parser.add_argument("--output-dir", type=str, help="结果输出目录")
    parser.add_argument("--dry-run", action="store_true", help="只验证配置，不执行实验")
    
    args = parser.parse_args()
    
    try:
        # 加载配置
        logger.info(f"加载配置文件: {args.config}")
        batch_config = ConfigManager.load_from_yaml(args.config)
        
        if args.output_dir:
            batch_config.output_dir = args.output_dir
        
        logger.info(f"发现 {len(batch_config.experiments)} 个实验配置")
        
        # 验证所有配置
        all_valid = True
        for exp in batch_config.experiments:
            errors = validate_config(exp)
            if errors:
                logger.error(f"配置 {exp.name} 有误: {errors}")
                all_valid = False
            else:
                logger.info(f"配置 {exp.name} 验证通过")
        
        if not all_valid:
            logger.error("存在无效配置，请修复后重试")
            return 1
        
        if args.dry_run:
            logger.info("--dry-run 模式，配置验证完成，不执行实验")
            return 0
        
        # 检查依赖
        if not is_cake_available():
            logger.error("CAKE核心功能不可用，请检查安装")
            return 1
        
        # 执行批量实验
        runner = BatchExperimentRunner(batch_config.output_dir)
        runner.run_batch(batch_config)
        
        # 输出总结
        completed = sum(1 for r in runner.results if r.status == "completed")
        failed = sum(1 for r in runner.results if r.status == "failed")
        
        logger.info("=" * 60)
        logger.info("批量实验执行完成")
        logger.info(f"成功: {completed}/{len(runner.results)}")
        logger.info(f"失败: {failed}/{len(runner.results)}")
        logger.info(f"结果保存在: {batch_config.output_dir}")
        
        return 0 if failed == 0 else 1
        
    except Exception as e:
        logger.error(f"批量执行失败: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main()) 