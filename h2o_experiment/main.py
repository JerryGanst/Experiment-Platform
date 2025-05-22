# print("DEBUG: h2o_experiment/main.py execution started") # REMOVE THIS LINE
import sys
import os

"""
主实验脚本 - 基线测试阶段
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
# from config import MODEL_CONFIG, EXPERIMENT_CONFIG, DATASET_CONFIG, OUTPUT_CONFIG, MONITORING_CONFIG # OLD WAY
import config as cfg # NEW WAY
MODEL_CONFIG = cfg.MODEL_CONFIG
EXPERIMENT_CONFIG = cfg.EXPERIMENT_CONFIG
DATASET_CONFIG = cfg.DATASET_CONFIG
OUTPUT_CONFIG = cfg.OUTPUT_CONFIG
MONITORING_CONFIG = cfg.MONITORING_CONFIG

from models.model_loader import load_model_and_tokenizer, configure_model_for_kv_cache_length, prepare_model_for_baseline
from data.dataset_loader import load_dataset_split, prepare_samples_for_evaluation, prepare_batch
from metrics.metrics_collector import PerformanceMetricsCollector
from utils.monitoring_manager import MonitoringManager

# 设置日志
def setup_logging(log_file=None, level=logging.INFO):
    """设置日志记录"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 创建日志目录
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    handlers = []
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    handlers.append(logging.StreamHandler())
    
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers
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
        "use_h2o": False,  # 基线不使用H2O优化
        "dataset": dataset_name
    })
    
    try:
        # 加载模型和分词器
        logger.info("Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer(model_config)
        
        # 配置模型的KV缓存长度
        model = configure_model_for_kv_cache_length(model, kv_cache_length)
        
        # 准备基线模型（不使用H2O优化）
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
        original_input_length = inputs["input_ids"].shape[1] # 获取原始输入长度
        logger.info(f"Original input length: {original_input_length}") # Log original length
        
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
            def __init__(self, collector, logger_instance, prompt_length):
                self.collector = collector
                self.logger = logger_instance
                self.prompt_length = prompt_length
                self.first_token_recorded = False
                self.last_seen_length = prompt_length # 初始化为 prompt 长度
                self.call_count = 0

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
                self.call_count += 1
                current_length = input_ids.shape[1]
                
                self.logger.debug(
                    f"TokenTimeLogitsProcessor call ({self.call_count}): "
                    f"Current length: {current_length}, Prompt length: {self.prompt_length}, "
                    f"Last seen length: {self.last_seen_length}, First token recorded: {self.first_token_recorded}"
                )

                if not self.first_token_recorded:
                    if current_length == self.prompt_length + 1:
                        self.collector.record_first_token()
                        self.first_token_recorded = True
                        self.last_seen_length = current_length
                        self.logger.info(
                            f"TokenTimeLogitsProcessor: Recorded first token at call {self.call_count}. "
                            f"New length: {current_length}"
                        )
                    elif current_length == self.prompt_length:
                         self.logger.debug(f"TokenTimeLogitsProcessor: Still at prompt length at call {self.call_count}.")
                    elif current_length > self.prompt_length + 1 :
                        self.logger.warning(
                            f"TokenTimeLogitsProcessor: Current length {current_length} skipped expected first token length {self.prompt_length + 1}. "
                            f"Recording first token now at call {self.call_count} as a fallback."
                        )
                        self.collector.record_first_token() # Fallback
                        self.first_token_recorded = True
                        self.last_seen_length = current_length
                elif self.first_token_recorded:
                    if current_length == self.last_seen_length + 1:
                        self.collector.record_token()
                        self.last_seen_length = current_length
                        self.logger.debug(
                            f"TokenTimeLogitsProcessor: Recorded subsequent token at call {self.call_count}. "
                            f"New length: {current_length}"
                        )
                    elif current_length > self.last_seen_length + 1:
                        num_new_tokens_skipped = current_length - (self.last_seen_length + 1)
                        self.logger.warning(
                            f"TokenTimeLogitsProcessor: Current length {current_length} jumped from last seen {self.last_seen_length}. "
                            f"Recording one token now, but {num_new_tokens_skipped} might have been generated in one step or missed."
                        )
                        self.collector.record_token() 
                        self.last_seen_length = current_length
                    elif current_length == self.last_seen_length:
                        self.logger.debug(f"TokenTimeLogitsProcessor: Length {current_length} unchanged from last seen at call {self.call_count}.")
                return scores

        # 创建 LogitsProcessor 实例
        token_time_processor = TokenTimeLogitsProcessor(metrics_collector, logger, original_input_length)
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
        if 'peak_gpu_memory_mb' in metrics and metrics['peak_gpu_memory_mb'] is not None and str(metrics['peak_gpu_memory_mb']).lower() != 'n/a':
            try:
                peak_memory_value = float(metrics['peak_gpu_memory_mb'])
                logger.info(f"Peak GPU Memory: {peak_memory_value:.2f}MB")
            except ValueError:
                logger.info(f"Peak GPU Memory: {metrics['peak_gpu_memory_mb']} (could not convert to float for formatting)")
        else:
            logger.info("Peak GPU Memory: N/A (CUDA not available, monitoring failed, or value was None/'N/A')")
        
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
    """主函数"""
    parser = argparse.ArgumentParser(description="Run Baseline Experiments")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_CONFIG.get("results_dir", "./results"),
                        help="Directory to save results")
    parser.add_argument("--dataset", type=str, default="mmlu",
                        help="Dataset to use for evaluation")
    parser.add_argument("--language", type=str, default="english",
                        choices=["english", "chinese"],
                        help="Language of the dataset")
    parser.add_argument("--kv_cache_lengths", type=str, default=None,
                        help="Comma-separated list of KV cache lengths to test (overrides config)")
    parser.add_argument("--batch_sizes", type=str, default=None,
                        help="Comma-separated list of batch sizes to test (overrides config)")
    parser.add_argument("--model_name", type=str, default=MODEL_CONFIG.get("model_name_or_path", "facebook/opt-125m"),
                        help="Name of the model to test (from config or Hugging Face)")
    parser.add_argument("--max_new_tokens", type=int, default=EXPERIMENT_CONFIG.get("max_new_tokens", 128),
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--num_repeats", type=int, default=EXPERIMENT_CONFIG.get("repeat_times", 1),
                        help="Number of times to repeat each experiment configuration")
    parser.add_argument("--log_level", type=str, default=MONITORING_CONFIG.get("logging_level", "INFO"),
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--random_seed", type=int, default=EXPERIMENT_CONFIG.get("random_seed", 42),
                        help="Random seed for reproducibility")

    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"baseline_{args.language}_{args.dataset}_{timestamp}.log"
    log_file = os.path.join(OUTPUT_CONFIG.get("logs_dir", "./logs"), log_file_name)
    global logger
    logger = setup_logging(log_file, level=getattr(logging, args.log_level.upper()))
    
    # 设置随机种子
    set_seed(args.random_seed)
    
    # 获取数据集配置
    dataset_config = DATASET_CONFIG[args.language][args.dataset]
    
    # 解析命令行参数覆盖配置
    kv_cache_lengths = EXPERIMENT_CONFIG.get("kv_cache_lengths", [512, 1024])
    if args.kv_cache_lengths:
        kv_cache_lengths = [int(x.strip()) for x in args.kv_cache_lengths.split(",")]
    
    batch_sizes = EXPERIMENT_CONFIG.get("batch_sizes", [1, 4])
    if args.batch_sizes:
        batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    
    # 所有实验结果
    all_results = []
    
    # 运行不同KV缓存长度和批处理大小的实验
    for kv_length in kv_cache_lengths:
        for batch_size in batch_sizes:
            # 对每个配置重复实验多次
            for repeat in range(args.num_repeats):
                logger.info(f"Running experiment: Model={args.model_name}, Dataset={args.dataset}, Lang={args.language}, KV Length={kv_length}, Batch Size={batch_size}, "
                           f"MaxTokens={args.max_new_tokens}, Repeat={repeat+1}/{args.num_repeats}")
                
                # MODIFICATION: Construct model_config for the current run
                current_run_model_config = MODEL_CONFIG.copy() 
                current_run_model_config["model_name_or_path"] = args.model_name
                # If precision or other model params need to be configurable via args, add them here.
                # For now, other params like 'precision' are taken from the global MODEL_CONFIG.

                # 运行基线实验
                metrics = run_baseline_experiment(
                    model_config=current_run_model_config, # MODIFIED to pass constructed config
                    dataset_name=args.dataset,
                    dataset_config=dataset_config,
                    kv_cache_length=kv_length,
                    batch_size=batch_size,
                    max_new_tokens=args.max_new_tokens, # MODIFIED to use args.max_new_tokens
                    output_dir=args.output_dir,
                    repeat_index=repeat
                )
                
                all_results.append(metrics)
                
                # 短暂暂停以确保资源释放
                time.sleep(2)  
    
    # 将所有结果保存为CSV
    results_df = pd.DataFrame(all_results)
    csv_file_name = f"baseline_results_model_{args.model_name.replace('/', '_')}_{args.language}_{args.dataset}_{timestamp}.csv"
    csv_path = os.path.join(args.output_dir, csv_file_name)
    results_df.to_csv(csv_path, index=False)
    logger.info(f"All results saved to {csv_path}")
    
    # 打印汇总统计
    print("\nExperiment Results Summary:")
    print(f"Total experiments: {len(all_results)}")
    
    # 只处理成功的实验
    success_results = results_df[results_df["success"] == True]
    
    if len(success_results) > 0:
        # 按KV缓存长度和批处理大小分组的平均值
        summary = success_results.groupby(['kv_cache_length', 'batch_size']).agg({
            'ttft_ms': ['mean', 'std'],
            'tpot_ms': ['mean', 'std'],
            'peak_gpu_memory_mb': ['mean', 'std'],
            'throughput_tokens_per_sec': ['mean', 'std']
        })
        
        print("\nPerformance Summary by Configuration:")
        print(summary)
        
        # 将汇总结果保存为CSV
        summary_file_name = f"baseline_summary_model_{args.model_name.replace('/', '_')}_{args.language}_{args.dataset}_{timestamp}.csv"
        summary_path = os.path.join(args.output_dir, summary_file_name)
        summary.to_csv(summary_path)
        logger.info(f"Summary saved to {summary_path}")
    else:
        logger.warning("No successful experiments to summarize")

if __name__ == "__main__":
    main() 