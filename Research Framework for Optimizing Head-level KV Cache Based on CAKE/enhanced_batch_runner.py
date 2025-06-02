#!/usr/bin/env python3
"""
增强版批量实验执行器

结合新设计方案和现有桥接层的优势，提供：
1. 多种执行模式：直接导入 vs 子进程调用
2. YAML配置支持
3. 并行批量执行
4. 完善的错误处理和进度监控
"""

import os
import sys
import json
import logging
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from config_yaml_example import (
    ConfigManager, BatchExperimentConfig, ExperimentConfig, 
    validate_config, ModelConfig, CakeConfig
)

# 导入两种执行模式
from models.cake_converter import (
    is_cake_available, apply_cake_to_model, 
    CAKEExperimentConfig, CAKEModelAdapter
)
from models.cake_subprocess_adapter import (
    CAKESubprocessAdapter, CAKESubprocessConfig, run_cake_via_subprocess
)
from models.model_loader import (
    load_model_and_tokenizer, prepare_model_for_cake, 
    is_model_type_supported_by_cake
)
import config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExecutionMode(Enum):
    """执行模式枚举"""
    DIRECT_IMPORT = "direct_import"    # 直接导入模式（当前方案）
    SUBPROCESS = "subprocess"          # 子进程模式（新方案）
    AUTO = "auto"                     # 自动选择

class ExperimentStatus(Enum):
    """实验状态枚举"""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class EnhancedExperimentResult:
    """增强版实验结果类"""
    
    def __init__(self, experiment_name: str, execution_mode: ExecutionMode):
        self.name = experiment_name
        self.execution_mode = execution_mode
        self.status = ExperimentStatus.PENDING
        self.start_time = None
        self.end_time = None
        self.metrics = {}
        self.error = None
        self.warnings = []
        self.model_info = {}
        
    def start(self):
        """开始实验"""
        self.status = ExperimentStatus.RUNNING
        self.start_time = datetime.now()
        
    def complete(self, metrics: Dict[str, Any], model_info: Dict[str, Any] = None):
        """完成实验"""
        self.status = ExperimentStatus.COMPLETED
        self.end_time = datetime.now()
        self.metrics = metrics
        self.model_info = model_info or {}
        
    def fail(self, error: str):
        """实验失败"""
        self.status = ExperimentStatus.FAILED
        self.end_time = datetime.now()
        self.error = error
        
    def skip(self, reason: str):
        """跳过实验"""
        self.status = ExperimentStatus.SKIPPED
        self.error = reason
        
    def add_warning(self, warning: str):
        """添加警告"""
        self.warnings.append(warning)
        logger.warning(f"[{self.name}] {warning}")
        
    def duration(self) -> float:
        """获取执行时长"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "name": self.name,
            "execution_mode": self.execution_mode.value,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration(),
            "metrics": self.metrics,
            "model_info": self.model_info,
            "error": self.error,
            "warnings": self.warnings
        }

class EnhancedBatchRunner:
    """增强版批量实验执行器"""
    
    def __init__(self, 
                 output_dir: str = "results",
                 default_execution_mode: ExecutionMode = ExecutionMode.AUTO):
        """
        初始化批量执行器
        
        Args:
            output_dir: 输出目录
            default_execution_mode: 默认执行模式
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.default_execution_mode = default_execution_mode
        self.results = []
        
        # 检查可用的执行模式
        self.available_modes = self._check_available_modes()
        logger.info(f"可用执行模式: {[mode.value for mode in self.available_modes]}")
        
    def _check_available_modes(self) -> List[ExecutionMode]:
        """检查可用的执行模式"""
        available = []
        
        # 检查直接导入模式
        if is_cake_available():
            available.append(ExecutionMode.DIRECT_IMPORT)
            logger.info("✅ 直接导入模式可用")
        else:
            logger.warning("❌ 直接导入模式不可用")
        
        # 检查子进程模式
        try:
            subprocess_adapter = CAKESubprocessAdapter()
            if subprocess_adapter.test_installation():
                available.append(ExecutionMode.SUBPROCESS)
                logger.info("✅ 子进程模式可用")
            else:
                logger.warning("❌ 子进程模式测试失败")
        except Exception as e:
            logger.warning(f"❌ 子进程模式不可用: {e}")
        
        return available
    
    def _select_execution_mode(self, 
                              preferred_mode: ExecutionMode) -> ExecutionMode:
        """选择执行模式"""
        if preferred_mode == ExecutionMode.AUTO:
            # 自动选择：优先使用直接导入（性能更好），备选子进程（更稳定）
            if ExecutionMode.DIRECT_IMPORT in self.available_modes:
                return ExecutionMode.DIRECT_IMPORT
            elif ExecutionMode.SUBPROCESS in self.available_modes:
                return ExecutionMode.SUBPROCESS
            else:
                raise RuntimeError("没有可用的执行模式")
        
        if preferred_mode not in self.available_modes:
            logger.warning(f"首选模式 {preferred_mode.value} 不可用，切换到自动选择")
            return self._select_execution_mode(ExecutionMode.AUTO)
        
        return preferred_mode
    
    def run_single_experiment(self, 
                            exp_config: ExperimentConfig,
                            execution_mode: Optional[ExecutionMode] = None) -> EnhancedExperimentResult:
        """运行单个实验"""
        # 选择执行模式
        if execution_mode is None:
            execution_mode = self.default_execution_mode
        execution_mode = self._select_execution_mode(execution_mode)
        
        result = EnhancedExperimentResult(exp_config.name, execution_mode)
        result.start()
        
        try:
            logger.info(f"开始实验: {exp_config.name} (模式: {execution_mode.value})")
            
            # 验证配置
            config_errors = validate_config(exp_config)
            if config_errors:
                raise ValueError(f"配置错误: {config_errors}")
            
            # 根据执行模式运行实验
            if execution_mode == ExecutionMode.DIRECT_IMPORT:
                metrics, model_info = self._run_direct_import(exp_config, result)
            elif execution_mode == ExecutionMode.SUBPROCESS:
                metrics, model_info = self._run_subprocess(exp_config, result)
            else:
                raise ValueError(f"不支持的执行模式: {execution_mode}")
            
            result.complete(metrics, model_info)
            logger.info(f"实验完成: {exp_config.name}")
            
        except Exception as e:
            error_msg = f"实验失败: {str(e)}"
            logger.error(f"[{exp_config.name}] {error_msg}")
            logger.debug(traceback.format_exc())
            result.fail(error_msg)
        
        return result
    
    def _run_direct_import(self, 
                          exp_config: ExperimentConfig,
                          result: EnhancedExperimentResult) -> tuple:
        """使用直接导入模式运行实验"""
        # 转换为旧格式配置
        model_config, cake_exp_config, cake_model_config = exp_config.to_legacy_format()
        
        # 加载模型
        logger.info(f"加载模型: {model_config['model_name_or_path']}")
        model, tokenizer = load_model_and_tokenizer(model_config)
        
        model_info = {
            "model_name": model_config['model_name_or_path'],
            "model_type": model.config.model_type,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "device": str(next(model.parameters()).device)
        }
        
        # 应用方法
        if exp_config.method == "CAKE":
            if not is_model_type_supported_by_cake(model, config.CAKE_MODEL_CONFIG):
                result.add_warning(f"模型类型可能不完全支持CAKE: {model.config.model_type}")
            
            # 应用CAKE
            logger.info("应用CAKE优化...")
            model = prepare_model_for_cake(model, cake_exp_config, cake_model_config)
            
            # 验证CAKE应用
            if hasattr(model, 'is_cake_enabled') and model.is_cake_enabled:
                logger.info("CAKE应用成功")
            else:
                result.add_warning("CAKE应用状态未知")
                
        elif exp_config.method == "baseline":
            logger.info("使用基线方法（无压缩）")
        else:
            raise ValueError(f"不支持的方法: {exp_config.method}")
        
        # 执行推理测试
        metrics = self._run_inference_test(model, tokenizer, exp_config)
        
        return metrics, model_info
    
    def _run_subprocess(self, 
                       exp_config: ExperimentConfig,
                       result: EnhancedExperimentResult) -> tuple:
        """使用子进程模式运行实验"""
        if exp_config.method != "CAKE":
            # 子进程模式目前只支持CAKE
            raise ValueError(f"子进程模式不支持方法: {exp_config.method}")
        
        # 转换配置
        model_config, cake_exp_config, cake_model_config = exp_config.to_legacy_format()
        
        # 创建子进程适配器
        adapter = CAKESubprocessAdapter()
        
        # 转换为子进程配置
        subprocess_config = CAKESubprocessConfig.from_legacy_config(
            cake_exp_config, cake_model_config
        )
        subprocess_config.model_name = exp_config.model.model_type
        subprocess_config.output_dir = str(self.output_dir / exp_config.name)
        
        # 运行子进程实验
        logger.info("启动CAKE子进程实验...")
        subprocess_result = adapter.run_experiment(subprocess_config, timeout=3600)
        
        if not subprocess_result["success"]:
            raise RuntimeError(f"CAKE子进程失败: {subprocess_result.get('error', '未知错误')}")
        
        metrics = subprocess_result["metrics"]
        model_info = {
            "model_name": exp_config.model.name,
            "model_type": exp_config.model.model_type,
            "execution_mode": "subprocess",
            "subprocess_returncode": subprocess_result["returncode"]
        }
        
        return metrics, model_info
    
    def _run_inference_test(self, model, tokenizer, exp_config: ExperimentConfig) -> Dict[str, Any]:
        """运行推理测试"""
        # 简化的推理测试
        test_input = "这是一个测试输入，用于验证模型功能。请简要回应。"
        inputs = tokenizer(test_input, return_tensors="pt", max_length=512, truncation=True)
        
        import torch
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # 简单的forward pass测试
            outputs = model(**inputs)
            
            # 尝试生成测试
            try:
                generated = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    use_cache=True
                )
                response = tokenizer.decode(generated[0], skip_special_tokens=True)
                generation_success = True
            except Exception as e:
                logger.warning(f"生成测试失败: {e}")
                response = ""
                generation_success = False
        
        # 构建指标
        metrics = {
            "method": exp_config.method,
            "model_type": exp_config.model.model_type,
            "context_length": exp_config.model.context_length,
            "test_passed": True,
            "generation_success": generation_success,
            "response_length": len(response),
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
    
    def run_batch(self, 
                  batch_config: BatchExperimentConfig,
                  execution_mode: Optional[ExecutionMode] = None):
        """运行批量实验"""
        logger.info(f"开始批量实验，共 {len(batch_config.experiments)} 个实验")
        
        # 选择执行模式
        if execution_mode is None:
            execution_mode = self.default_execution_mode
        execution_mode = self._select_execution_mode(execution_mode)
        logger.info(f"使用执行模式: {execution_mode.value}")
        
        # 执行实验
        if batch_config.parallel and len(batch_config.experiments) > 1:
            logger.info(f"使用并行执行，最大工作进程: {batch_config.max_workers}")
            self._run_parallel(batch_config.experiments, batch_config.max_workers, execution_mode)
        else:
            logger.info("使用串行执行")
            self._run_sequential(batch_config.experiments, execution_mode)
        
        # 保存结果
        self._save_results()
        self._generate_report()
        
        # 输出总结
        self._print_summary()
    
    def _run_sequential(self, 
                       experiments: List[ExperimentConfig],
                       execution_mode: ExecutionMode):
        """串行执行实验"""
        for i, exp_config in enumerate(experiments, 1):
            logger.info(f"执行实验 {i}/{len(experiments)}: {exp_config.name}")
            result = self.run_single_experiment(exp_config, execution_mode)
            self.results.append(result)
    
    def _run_parallel(self, 
                     experiments: List[ExperimentConfig],
                     max_workers: int,
                     execution_mode: ExecutionMode):
        """并行执行实验"""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_config = {
                executor.submit(self.run_single_experiment, exp_config, execution_mode): exp_config
                for exp_config in experiments
            }
            
            # 收集结果
            for future in as_completed(future_to_config):
                result = future.result()
                self.results.append(result)
                
                # 实时进度更新
                completed = len([r for r in self.results if r.status != ExperimentStatus.RUNNING])
                logger.info(f"进度: {completed}/{len(experiments)} 完成")
    
    def _save_results(self):
        """保存实验结果"""
        results_file = self.output_dir / "enhanced_experiment_results.json"
        
        summary = {
            "total_experiments": len(self.results),
            "completed": sum(1 for r in self.results if r.status == ExperimentStatus.COMPLETED),
            "failed": sum(1 for r in self.results if r.status == ExperimentStatus.FAILED),
            "skipped": sum(1 for r in self.results if r.status == ExperimentStatus.SKIPPED),
            "execution_modes": list(set(r.execution_mode.value for r in self.results)),
            "timestamp": datetime.now().isoformat()
        }
        
        results_data = {
            "summary": summary,
            "experiments": [result.to_dict() for result in self.results]
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"实验结果已保存到: {results_file}")
    
    def _generate_report(self):
        """生成实验报告"""
        report_file = self.output_dir / "enhanced_experiment_report.md"
        
        completed_results = [r for r in self.results if r.status == ExperimentStatus.COMPLETED]
        failed_results = [r for r in self.results if r.status == ExperimentStatus.FAILED]
        
        # 按执行模式分组
        mode_groups = {}
        for result in completed_results:
            mode = result.execution_mode.value
            if mode not in mode_groups:
                mode_groups[mode] = []
            mode_groups[mode].append(result)
        
        report_lines = [
            "# 增强版批量实验报告",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 概览",
            f"- 总实验数: {len(self.results)}",
            f"- 成功完成: {len(completed_results)}",
            f"- 失败: {len(failed_results)}",
            f"- 成功率: {len(completed_results)/len(self.results)*100:.1f}%",
            f"- 使用的执行模式: {', '.join(mode_groups.keys())}",
            "",
        ]
        
        # 按执行模式展示结果
        for mode, results in mode_groups.items():
            report_lines.extend([
                f"## {mode.title()} 模式结果",
                "",
                "| 实验名称 | 方法 | 模型类型 | 耗时(秒) | 内存使用(MB) | 压缩比 |",
                "|----------|------|----------|----------|--------------|---------|",
            ])
            
            for result in results:
                metrics = result.metrics
                compression_ratio = metrics.get('compression_ratio', 'N/A')
                if isinstance(compression_ratio, (int, float)):
                    compression_ratio = f"{compression_ratio:.3f}"
                
                report_lines.append(
                    f"| {result.name} | {metrics.get('method', 'N/A')} | "
                    f"{metrics.get('model_type', 'N/A')} | {result.duration():.1f} | "
                    f"{metrics.get('memory_usage_mb', 0):.1f} | {compression_ratio} |"
                )
            
            report_lines.append("")
        
        # 失败实验
        if failed_results:
            report_lines.extend([
                "## 失败实验详情",
                "",
            ])
            
            for result in failed_results:
                report_lines.extend([
                    f"### {result.name} ({result.execution_mode.value})",
                    f"**错误**: {result.error}",
                    ""
                ])
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"实验报告已生成: {report_file}")
    
    def _print_summary(self):
        """打印执行总结"""
        completed = sum(1 for r in self.results if r.status == ExperimentStatus.COMPLETED)
        failed = sum(1 for r in self.results if r.status == ExperimentStatus.FAILED)
        
        logger.info("=" * 60)
        logger.info("增强版批量实验执行完成")
        logger.info(f"成功: {completed}/{len(self.results)}")
        logger.info(f"失败: {failed}/{len(self.results)}")
        logger.info(f"结果保存在: {self.output_dir}")
        
        # 显示执行模式统计
        mode_stats = {}
        for result in self.results:
            mode = result.execution_mode.value
            status = result.status.value
            if mode not in mode_stats:
                mode_stats[mode] = {}
            mode_stats[mode][status] = mode_stats[mode].get(status, 0) + 1
        
        logger.info("\n执行模式统计:")
        for mode, stats in mode_stats.items():
            logger.info(f"  {mode}: {stats}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="增强版CAKE实验平台批量执行器")
    parser.add_argument("--config", type=str, required=True, help="YAML配置文件路径")
    parser.add_argument("--output-dir", type=str, help="结果输出目录")
    parser.add_argument("--execution-mode", 
                       choices=["direct_import", "subprocess", "auto"],
                       default="auto",
                       help="执行模式")
    parser.add_argument("--dry-run", action="store_true", help="只验证配置，不执行实验")
    
    args = parser.parse_args()
    
    try:
        # 解析执行模式
        execution_mode = ExecutionMode(args.execution_mode)
        
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
        
        # 创建并运行批量执行器
        runner = EnhancedBatchRunner(
            batch_config.output_dir,
            execution_mode
        )
        runner.run_batch(batch_config, execution_mode)
        
        return 0
        
    except Exception as e:
        logger.error(f"批量执行失败: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main()) 