# This is the new file content for baseline_main.py
# It is identical to the original main.py 

import sys
import os

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取项目目录的路径
project_dir = os.path.dirname(current_file_path)
# 获取项目根目录的路径
project_root_dir = os.path.dirname(project_dir)

# 如果项目根目录不在 sys.path 中，则添加它
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

"""
基线实验执行脚本 - 使用标准KV缓存机制
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
import config
MODEL_CONFIG = config.MODEL_CONFIG
EXPERIMENT_CONFIG = config.EXPERIMENT_CONFIG
DATASET_CONFIG = config.DATASET_CONFIG
OUTPUT_CONFIG = config.OUTPUT_CONFIG
MONITORING_CONFIG = config.MONITORING_CONFIG

# 导入模块 - 更新路径以匹配新的目录结构
from models.model_loader import (
    load_model_and_tokenizer,
    configure_model_for_kv_cache_length,
    prepare_model_for_baseline
)
from data.dataset_loader import load_dataset_split, prepare_samples_for_evaluation, prepare_batch
from metrics.metrics_collector import PerformanceMetricsCollector
from utils.monitoring_manager import MonitoringManager


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


def run_baseline_experiment(model_config, dataset_name, dataset_config,
                           kv_cache_length, batch_size, max_new_tokens,
                           output_dir, repeat_index=0):
    """
    运行单次基线实验

    Args:
        model_config: 模型配置
        dataset_name: 数据集名称
        dataset_config: 数据集配置
        kv_cache_length: KV缓存长度
        batch_size: 批处理大小
        max_new_tokens: 最大生成令牌数
        output_dir: 输出目录
        repeat_index: 重复实验的索引

    Returns:
        metrics: 性能指标
    """
    experiment_id = f"baseline_{dataset_name}_kv{kv_cache_length}_bs{batch_size}_rep{repeat_index}_{datetime.now().strftime('%H%M%S')}"
    logger.info(f"Starting baseline experiment: {experiment_id}")

    # 初始化性能指标收集器
    metrics_collector = PerformanceMetricsCollector(experiment_id)
    metrics_collector.record_config({
        "model_name": model_config["model_name_or_path"],
        "precision": model_config["precision"],
        "batch_size": batch_size,
        "kv_cache_length": kv_cache_length,
        "max_new_tokens": max_new_tokens,
        "use_baseline": True,
        "dataset": dataset_name
    })

    try:
        # 加载模型和分词器
        logger.info("Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer(model_config)

        # 配置模型的KV缓存长度
        model = configure_model_for_kv_cache_length(model, kv_cache_length)

        # 准备基线模型
        model = prepare_model_for_baseline(model)

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
            
            with open(os.path.join(outputs_dir, f"generated_texts_{experiment_id}.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "experiment_id": experiment_id,
                    "generated_texts": generated_texts,
                    "input_texts": [tokenizer.decode(input_ids, skip_special_tokens=True) for input_ids in batch["input_ids"]]
                }, f, indent=2, ensure_ascii=False)

        logger.info(f"Baseline experiment {experiment_id} completed successfully")
        return metrics

    except Exception as e:
        logger.error(f"Error during baseline experiment {experiment_id}: {e}")
        metrics_collector.record_error(str(e))
        return metrics_collector.compute_metrics()


def main():
    parser = argparse.ArgumentParser(description="Run Baseline KV Cache Experiments")
    parser.add_argument("--model_name", type=str, default=EXPERIMENT_CONFIG["model_name_or_path"], help="Name or path of the model to use.")
    parser.add_argument("--datasets", type=str, default=",".join(EXPERIMENT_CONFIG["datasets"]), help="Comma-separated list of datasets to use.")
    parser.add_argument("--kv_cache_lengths", type=str, default=",".join(map(str, EXPERIMENT_CONFIG["kv_cache_lengths"])), help="Comma-separated list of KV cache lengths.")
    parser.add_argument("--batch_sizes", type=str, default=",".join(map(str, EXPERIMENT_CONFIG["batch_sizes"])), help="Comma-separated list of batch sizes.")
    parser.add_argument("--max_new_tokens", type=int, default=EXPERIMENT_CONFIG["max_new_tokens"], help="Maximum number of new tokens to generate.")
    parser.add_argument("--repetitions", type=int, default=EXPERIMENT_CONFIG["repetitions"], help="Number of repetitions for each experiment configuration.")
    parser.add_argument("--output_dir", type=str, default=os.path.join(EXPERIMENT_CONFIG["output_base_dir"], "baseline_experiments"), help="Directory to save experiment results.")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--seed", type=int, default=EXPERIMENT_CONFIG.get("random_seed", 42), help="Random seed for reproducibility.")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置日志
    log_file_path = os.path.join(args.output_dir, "baseline_experiment_log.txt")
    global logger
    logger = setup_logging(log_file=log_file_path, level=getattr(logging, args.log_level.upper(), logging.INFO))

    logger.info(f"Starting baseline experiment suite")
    logger.info(f"Arguments: {args}")

    # 设置随机种子
    set_seed(args.seed)

    # 解析参数列表
    datasets_list = [d.strip() for d in args.datasets.split(',') if d.strip()]
    kv_lengths_list = [int(kv.strip()) for kv in args.kv_cache_lengths.split(',') if kv.strip()]
    batch_sizes_list = [int(bs.strip()) for bs in args.batch_sizes.split(',') if bs.strip()]

    all_results = []
    total_experiments = len(datasets_list) * len(kv_lengths_list) * len(batch_sizes_list) * args.repetitions
    logger.info(f"Total number of baseline experiment configurations to run: {total_experiments}")

    current_model_config = {
        "model_name_or_path": args.model_name,
        "precision": EXPERIMENT_CONFIG["precision"]
    }

    pbar = tqdm(total=total_experiments, desc="Running Baseline Experiments")

    for rep in range(args.repetitions):
        for dataset_name in datasets_list:
            dataset_config = DATASET_CONFIG.get("available_datasets", {}).get(dataset_name)
            if not dataset_config:
                logger.error(f"Dataset configuration for '{dataset_name}' not found. Skipping...")
                pbar.update(len(kv_lengths_list) * len(batch_sizes_list))
                continue
            
            for kv_len in kv_lengths_list:
                for bs in batch_sizes_list:
                    logger.info(f"Running baseline: Rep {rep+1}/{args.repetitions}, Dataset: {dataset_name}, KV_Len: {kv_len}, Batch: {bs}")
                    
                    experiment_metrics = run_baseline_experiment(
                        model_config=current_model_config,
                        dataset_name=dataset_name,
                        dataset_config=dataset_config,
                        kv_cache_length=kv_len,
                        batch_size=bs,
                        max_new_tokens=args.max_new_tokens,
                        output_dir=args.output_dir,
                        repeat_index=rep
                    )
                    all_results.append(experiment_metrics)
                    pbar.update(1)

    pbar.close()

    # 保存所有实验结果的汇总
    summary_file_path = os.path.join(args.output_dir, "all_baseline_experiments_summary.csv")
    if all_results and isinstance(all_results[0], dict):
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv(summary_file_path, index=False)
        logger.info(f"All baseline experiment summaries saved to {summary_file_path}")

    logger.info("Baseline experiment suite finished.")


if __name__ == "__main__":
    main() 