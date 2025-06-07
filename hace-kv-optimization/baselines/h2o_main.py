import sys
import os

# 获取当前文件 (e.g., .../h2o_hf/Research Framework for Optimizing Head-level KV Cache Based on CAKE/h2o_main.py) 的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取 hace-kv-optimization 目录的路径
pkg_dir = os.path.dirname(current_file_path)
# 获取项目根目录
project_root_dir = os.path.dirname(pkg_dir)

# 如果项目根目录不在 sys.path 中，则添加它
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

"""
H2O实验执行脚本 - 使用H2O KV缓存优化
"""
import time
import logging
import argparse
import json
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from transformers import LogitsProcessor, LogitsProcessorList

# 导入项目模块
from hace_core import config
MODEL_CONFIG = config.MODEL_CONFIG
EXPERIMENT_CONFIG = config.EXPERIMENT_CONFIG
DATASET_CONFIG = config.DATASET_CONFIG
OUTPUT_CONFIG = config.OUTPUT_CONFIG
MONITORING_CONFIG = config.MONITORING_CONFIG

# 修改导入语句以相对于h2o_experiment包
from hace_core.models.model_loader import (
    load_model_and_tokenizer,
    configure_model_for_kv_cache_length,
    prepare_model_for_h2o,
    is_model_type_supported_by_h2o
)
from hace_core.data.dataset_loader import load_dataset_split, prepare_samples_for_evaluation, prepare_batch
from hace_core.metrics.metrics_collector import PerformanceMetricsCollector
from hace_core.utils.monitoring_manager import MonitoringManager


# 设置日志
def setup_logging(log_file=None, level=logging.INFO):
    """设置日志记录"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_dir = None
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

    handlers = []
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers,
        force=True  # 强制重新配置日志
    )
    # 减少一些库的日志输出
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)

    return logging.getLogger(__name__)


def set_seed(seed):
    """设置随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def run_h2o_experiment(model_config, dataset_name, dataset_config,
                       kv_cache_length, batch_size, max_new_tokens,
                       heavy_ratio, recent_ratio, output_dir, repeat_index=0):
    """
    运行单次H2O实验

    Args:
        model_config: 模型配置
        dataset_name: 数据集名称
        dataset_config: 数据集配置
        kv_cache_length: KV缓存长度
        batch_size: 批处理大小
        max_new_tokens: 最大生成令牌数
        heavy_ratio: Heavy Hitter令牌的比例
        recent_ratio: Recent令牌的比例
        output_dir: 输出目录
        repeat_index: 重复实验的索引

    Returns:
        metrics: 性能指标
    """
    experiment_id = f"h2o_{dataset_name}_kv{kv_cache_length}_bs{batch_size}_h{heavy_ratio}_r{recent_ratio}_rep{repeat_index}_{datetime.now().strftime('%H%M%S')}"
    logger.info(f"Starting H2O experiment: {experiment_id}")

    # 初始化性能指标收集器
    metrics_collector = PerformanceMetricsCollector(experiment_id)
    metrics_collector.record_config({
        "model_name": model_config["model_name_or_path"],
        "precision": model_config["precision"],
        "batch_size": batch_size,
        "kv_cache_length": kv_cache_length,
        "max_new_tokens": max_new_tokens,
        "use_h2o": True,
        "heavy_ratio": heavy_ratio,
        "recent_ratio": recent_ratio,
        "dataset": dataset_name
    })

    try:
        # 加载模型和分词器
        logger.info("Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer(model_config)

        # 检查模型是否支持H2O优化
        if not is_model_type_supported_by_h2o(model):
            raise ValueError(f"模型 {model_config['model_name_or_path']} 类型不支持H2O优化")

        # 配置模型的KV缓存长度
        model = configure_model_for_kv_cache_length(model, kv_cache_length)

        # 应用H2O优化
        model = prepare_model_for_h2o(model, heavy_ratio, recent_ratio)

        # 加载数据集
        logger.info(f"Loading dataset {dataset_name}...")
        dataset = load_dataset_split(dataset_config)

        # 准备评估样本
        samples = prepare_samples_for_evaluation(
            dataset,
            dataset_name,
            num_samples=batch_size * 2,  # 准备足够的样本
            random_seed=EXPERIMENT_CONFIG.get("random_seed", 42)
        )

        # 准备批处理
        logger.info(f"Preparing batch with size {batch_size}...")
        batch = prepare_batch(
            samples,
            tokenizer,
            batch_size,
            max_length=kv_cache_length
        )

        # 将批处理数据移至设备
        inputs = {
            "input_ids": batch["input_ids"].to(model.device),
            "attention_mask": batch["attention_mask"].to(model.device)
        }

        # 预热（可选）
        logger.info("Warming up model...")
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False
            )

        # 清理GPU缓存
        torch.cuda.empty_cache()

        # 初始化监控管理器并启动监控
        monitoring_manager = MonitoringManager(experiment_id=experiment_id)
        monitoring_manager.start_monitoring()

        # 开始性能测量
        logger.info("Starting performance measurement...")
        metrics_collector.start_generation()

        # 定义自定义 LogitsProcessor 来记录令牌生成时间
        class TokenTimeLogitsProcessor(LogitsProcessor):
            def __init__(self, collector):
                self.collector = collector
                self.first_token_recorded = False

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
                if not self.first_token_recorded:
                    self.collector.record_first_token()
                    self.first_token_recorded = True
                else:
                    self.collector.record_token()
                return scores

        # 创建 LogitsProcessor 实例
        token_time_processor = TokenTimeLogitsProcessor(metrics_collector)
        logits_processor_list = LogitsProcessorList([token_time_processor])

        # 生成文本
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                logits_processor=logits_processor_list,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # 结束性能测量
        metrics_collector.end_generation()

        # 停止监控并收集指标
        monitoring_manager.stop_monitoring()
        monitoring_metrics = monitoring_manager.get_metrics()

        # 记录GPU指标
        if "gpu" in monitoring_metrics["metrics"]:
            metrics_collector.record_gpu_stats(monitoring_metrics["metrics"]["gpu"])

        # 保存监控数据（可选）
        monitoring_dir = os.path.join(output_dir, "monitoring")
        os.makedirs(monitoring_dir, exist_ok=True)
        monitoring_manager.save_metrics(
            output_dir=monitoring_dir,
            filename=f"monitoring_{experiment_id}.json"
        )

        # 解码输出（可选，用于质量评估）
        generated_texts = tokenizer.batch_decode(
            outputs[:, batch["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        # 计算和保存指标
        metrics = metrics_collector.compute_metrics()
        metrics_filepath = metrics_collector.save_metrics(output_dir)

        # 保存生成的文本（可选）
        if OUTPUT_CONFIG.get("save_model_outputs", False):
            outputs_dir = os.path.join(output_dir, "model_outputs")
            os.makedirs(outputs_dir, exist_ok=True)
            with open(os.path.join(outputs_dir, f"{experiment_id}_outputs.txt"), "w") as f:
                for i, text in enumerate(generated_texts):
                    f.write(f"Sample {i}:\n")
                    f.write(f"Input: {batch['samples'][i]['prompt']}\n")
                    f.write(f"Output: {text}\n")
                    f.write("-" * 50 + "\n")

        logger.info(f"Experiment {experiment_id} completed successfully")
        logger.info(f"TTFT: {metrics['ttft_ms']:.2f}ms, TPOT: {metrics['tpot_ms']:.2f}ms")
        logger.info(f"Peak GPU Memory: {metrics['peak_gpu_memory_mb']:.2f}MB")

        return metrics

    except Exception as e:
        # 处理实验过程中的错误
        logger.error(f"Error in experiment {experiment_id}: {str(e)}", exc_info=True)
        metrics_collector.mark_failure(str(e))
        metrics = metrics_collector.compute_metrics()
        metrics_collector.save_metrics(output_dir)
        return metrics
    finally:
        # 确保GPU内存被清理
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Run H2O KV Cache Experiments")
    # 基本参数
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results")
    parser.add_argument("--dataset", type=str, default="mmlu",
                        help="Dataset to use for evaluation")
    parser.add_argument("--language", type=str, default="english",
                        choices=["english", "chinese"],
                        help="Language of the dataset")
    
    # 模型参数
    parser.add_argument("--model_name", type=str, 
                        default=MODEL_CONFIG.get("model_name_or_path", "facebook/opt-125m"),
                        help="Name of the model to test")
    parser.add_argument("--precision", type=str, 
                        default=MODEL_CONFIG.get("precision", "fp16"),
                        choices=["fp16", "fp32", "bf16", "int8"],
                        help="Model precision to use")
    parser.add_argument("--device", type=str, 
                        default=MODEL_CONFIG.get("device", "cuda"),
                        choices=["cuda", "cpu"],
                        help="Device to run model on")
    
    # 实验参数
    parser.add_argument("--kv_cache_lengths", type=str, 
                        default=",".join(map(str, EXPERIMENT_CONFIG.get("kv_cache_lengths", [512, 1024]))),
                        help="Comma-separated list of KV cache lengths to test")
    parser.add_argument("--batch_sizes", type=str, 
                        default=",".join(map(str, EXPERIMENT_CONFIG.get("batch_sizes", [1, 4]))),
                        help="Comma-separated list of batch sizes to test")
    parser.add_argument("--h2o_ratios", type=str, 
                        default=",".join(map(str, EXPERIMENT_CONFIG.get("h2o_ratios", [0.1, 0.2]))),
                        help="Comma-separated list of H2O ratios to test")
    parser.add_argument("--max_new_tokens", type=int, 
                        default=EXPERIMENT_CONFIG.get("max_new_tokens", 128),
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--num_repeats", type=int, 
                        default=EXPERIMENT_CONFIG.get("repeat_times", 1),
                        help="Number of times to repeat each experiment configuration")
    parser.add_argument("--random_seed", type=int, 
                        default=EXPERIMENT_CONFIG.get("random_seed", 42),
                        help="Random seed for reproducibility")
    
    # 日志参数
    parser.add_argument("--log_level", type=str, 
                        default=MONITORING_CONFIG.get("logging_level", "INFO"),
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--logs_dir", type=str, 
                        default=OUTPUT_CONFIG.get("logs_dir", "./logs"),
                        help="Directory to save log files")
    
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    # 设置日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.logs_dir, f"h2o_{args.language}_{args.dataset}_{timestamp}.log")
    global logger
    logger = setup_logging(log_file, level=getattr(logging, args.log_level))

    # 设置随机种子
    set_seed(args.random_seed)

    # 获取数据集配置
    try:
        dataset_config = DATASET_CONFIG[args.language][args.dataset]
        logger.info(f"使用数据集配置: {args.language}/{args.dataset}")
    except KeyError:
        logger.warning(f"未找到数据集配置: {args.language}/{args.dataset}，使用默认配置")
        # 创建一个基本的默认配置
        dataset_config = {
            "dataset_name": args.dataset,
            "split": "test",
            "max_samples": 100
        }

    # 创建模型配置
    model_config = {
        "model_name_or_path": args.model_name,
        "precision": args.precision,
        "device": args.device
    }

    # 解析KV缓存长度、批处理大小和H2O比率
    kv_cache_lengths = [int(x.strip()) for x in args.kv_cache_lengths.split(",")]
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    h2o_ratios = [float(x.strip()) for x in args.h2o_ratios.split(",")]

    logger.info(f"将测试以下配置:")
    logger.info(f"- 模型: {args.model_name} ({args.precision})")
    logger.info(f"- KV缓存长度: {kv_cache_lengths}")
    logger.info(f"- 批处理大小: {batch_sizes}")
    logger.info(f"- H2O比率: {h2o_ratios}")
    logger.info(f"- 每个配置重复: {args.num_repeats}次")

    # 所有实验结果
    all_results = []

    # 运行不同配置的H2O实验
    for kv_length in kv_cache_lengths:
        for batch_size in batch_sizes:
            for ratio in h2o_ratios:
                # 设置相同的heavy_ratio和recent_ratio
                heavy_ratio = ratio
                recent_ratio = ratio

                # 对每个配置重复实验多次
                for repeat in range(args.num_repeats):
                    logger.info(f"Running H2O experiment: KV Length={kv_length}, Batch Size={batch_size}, "
                                f"H2O Ratio={ratio}, Repeat={repeat + 1}/{args.num_repeats}")

                    # 运行H2O实验
                    metrics = run_h2o_experiment(
                        model_config,
                        args.dataset,
                        dataset_config,
                        kv_length,
                        batch_size,
                        args.max_new_tokens,
                        heavy_ratio,
                        recent_ratio,
                        args.output_dir,
                        repeat
                    )

                    all_results.append(metrics)

                    # 短暂暂停以确保资源释放
                    time.sleep(2)

    # 将所有结果保存为CSV
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(args.output_dir, f"h2o_results_{args.language}_{args.dataset}_{timestamp}.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info(f"All results saved to {csv_path}")

    # 打印汇总统计
    print("\nH2O Experiment Results Summary:")
    print(f"Total experiments: {len(all_results)}")

    # 只处理成功的实验
    success_results = results_df[results_df["success"] == True]

    if len(success_results) > 0:
        # 按配置分组的平均值
        summary = success_results.groupby(['kv_cache_length', 'batch_size', 'heavy_ratio']).agg({
            'ttft_ms': ['mean', 'std'],
            'tpot_ms': ['mean', 'std'],
            'peak_gpu_memory_mb': ['mean', 'std'],
            'throughput_tokens_per_sec': ['mean', 'std']
        })

        print("\nPerformance Summary by Configuration:")
        print(summary)

        # 将汇总结果保存为CSV
        summary_path = os.path.join(args.output_dir,
                                    f"h2o_summary_{args.language}_{args.dataset}_{timestamp}.csv")
        summary.to_csv(summary_path)
        logger.info(f"Summary saved to {summary_path}")
    else:
        logger.warning("No successful experiments to summarize")


if __name__ == "__main__":
    main()