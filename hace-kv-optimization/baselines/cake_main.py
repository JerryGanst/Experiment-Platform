import sys
import os

# 获取当前文件 (e.g., .../h2o_hf/h2o_experiment/cake_main.py) 的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取 hace-kv-optimization 目录的路径
pkg_dir = os.path.dirname(current_file_path)
# 获取项目根目录
project_root_dir = os.path.dirname(pkg_dir)

# 如果项目根目录不在 sys.path 中，则添加它
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

"""
CAKE实验执行脚本 - 使用CAKE层级感知KV缓存优化
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
from hace_core import config  # 导入根目录的config
MODEL_CONFIG = config.MODEL_CONFIG
EXPERIMENT_CONFIG = config.EXPERIMENT_CONFIG
DATASET_CONFIG = config.DATASET_CONFIG
OUTPUT_CONFIG = config.OUTPUT_CONFIG
MONITORING_CONFIG = config.MONITORING_CONFIG
CAKE_MODEL_CONFIG = config.CAKE_MODEL_CONFIG # 新增CAKE模型配置

# 修改导入语句以匹配新的目录结构
from hace_core.models.model_loader import (
    load_model_and_tokenizer,
    configure_model_for_kv_cache_length,
    prepare_model_for_cake,  # 主要变化
    is_model_type_supported_by_cake # 主要变化
)
from hace_core.data.dataset_loader import load_dataset_split, prepare_samples_for_evaluation, prepare_batch
from hace_core.metrics.metrics_collector import PerformanceMetricsCollector
from hace_core.utils.monitoring_manager import MonitoringManager


# 设置日志 (与h2o_main.py相同)
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
    handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers
    )
    # 减少一些库的日志输出
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    return logging.getLogger(__name__)

logger = logging.getLogger(__name__) # 在定义setup_logging后获取logger

def set_seed(seed):
    """设置随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def run_cake_experiment(experiment_main_config, dataset_name, dataset_options,
                        kv_cache_length, batch_size, max_new_tokens,
                        allocation_strategy, cache_budget, 
                        output_dir, repeat_index=0, trial_params=None):
    """
    运行单次CAKE实验

    Args:
        experiment_main_config: 主实验配置 (EXPERIMENT_CONFIG)
        dataset_name: 数据集名称
        dataset_options: 数据集特定配置 (来自DATASET_CONFIG)
        kv_cache_length: KV缓存长度 (CAKE可能不直接使用此参数，但保留作比较或基础配置)
        batch_size: 批处理大小
        max_new_tokens: 最大生成令牌数
        allocation_strategy: CAKE的层分配策略
        cache_budget: CAKE的缓存预算比例
        output_dir: 输出目录
        repeat_index: 重复实验的索引
        trial_params: (可选) Optuna试验的参数

    Returns:
        metrics: 性能指标字典
    """
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_id = f"cake_{dataset_name}_kv{kv_cache_length}_bs{batch_size}_strat{allocation_strategy}_bud{cache_budget}_rep{repeat_index}_{run_timestamp}"
    if trial_params:
        experiment_id = f"cake_optuna_{dataset_name}_trial{trial_params['trial_number']}_{run_timestamp}"
        # 更新从trial_params中获取的参数
        allocation_strategy = trial_params.get('allocation_strategy', allocation_strategy)
        cache_budget = trial_params.get('cache_budget', cache_budget)
        # kv_cache_length = trial_params.get('kv_cache_length', kv_cache_length) # 如果也优化此参数

    logger.info(f"Starting CAKE experiment: {experiment_id}")

    # 初始化性能指标收集器
    metrics_collector = PerformanceMetricsCollector(experiment_id)
    metrics_collector.record_config({
        "model_name": experiment_main_config["model_name_or_path"],
        "precision": experiment_main_config["precision"],
        "batch_size": batch_size,
        "kv_cache_length": kv_cache_length, # 记录基础KV长度
        "max_new_tokens": max_new_tokens,
        "use_cake": True,
        "allocation_strategy": allocation_strategy,
        "cache_budget": cache_budget,
        "dataset": dataset_name,
        "repetition": repeat_index,
        "trial_params": trial_params if trial_params else {}
    })

    try:
        # 加载模型和分词器
        current_model_config = {
            "model_name_or_path": experiment_main_config.get("current_model_name") or experiment_main_config["model_name_or_path"],
            "precision": experiment_main_config["precision"]
        }
        logger.info("Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer(current_model_config)

        # 检查模型是否支持CAKE优化
        if not is_model_type_supported_by_cake(model, CAKE_MODEL_CONFIG):
            error_msg = f"模型 {current_model_config['model_name_or_path']} 类型不支持CAKE优化 (根据CAKE_MODEL_CONFIG)"
            logger.error(error_msg)
            metrics_collector.mark_failure(error_msg)
            return metrics_collector.finalize_metrics_on_error()

        # 配置模型的KV缓存长度 (CAKE可能内部管理，但外部配置可作为参考)
        # model = configure_model_for_kv_cache_length(model, kv_cache_length)
        # 对于CAKE，最大序列长度更重要，它会被CAKECacheManager内部使用
        # 确保模型配置 use_cache=True (CAKE转换器应处理，但再次确认)
        model.config.use_cache = True

        # 应用CAKE优化
        # cake_experiment_config 需要包含 "layer_allocation_strategies", "cache_budgets", "layer_analysis_configs"
        # 从主实验配置中提取CAKE部分
        cake_specific_exp_config = {
            "layer_allocation_strategies": [allocation_strategy], # 当前实验使用的策略
            "cache_budgets": [cache_budget], # 当前实验使用的预算
            "layer_analysis_configs": experiment_main_config.get("layer_analysis_configs", {})
        }
        model = prepare_model_for_cake(model, cake_specific_exp_config, CAKE_MODEL_CONFIG)

        # 加载数据集
        logger.info(f"Loading dataset {dataset_name}...")
        # dataset_options是从DATASET_CONFIG[dataset_name]获取的
        dataset = load_dataset_split(dataset_options, split="validation") # 使用validation分片，因为MMLU没有train分片

        # 准备评估样本
        # dataset_subset_size 应从 experiment_main_config 获取
        num_eval_samples = experiment_main_config.get("dataset_subset_size", {}).get(dataset_name)
        if num_eval_samples is None: # 如果未指定特定数量，则使用默认或少量
            num_eval_samples = min(100, len(dataset)) # 或其他合理默认值
        
        actual_num_samples_to_prepare = batch_size # 我们只需要一个batch进行性能测试
        if num_eval_samples < actual_num_samples_to_prepare:
            actual_num_samples_to_prepare = num_eval_samples
            logger.warning(f"请求的batch_size {batch_size} 大于可用/配置的样本数 {num_eval_samples} for {dataset_name}。使用 {actual_num_samples_to_prepare}。")
        
        if actual_num_samples_to_prepare == 0:
            error_msg = f"没有足够的样本为数据集 {dataset_name} 准备批处理 (需要 {batch_size}, 可用 {num_eval_samples})。"
            logger.error(error_msg)
            metrics_collector.record_error(error_msg)
            return metrics_collector.finalize_metrics_on_error()

        samples = prepare_samples_for_evaluation(
            dataset,
            dataset_name,
            num_samples=actual_num_samples_to_prepare, 
            random_seed=experiment_main_config.get("random_seed", 42)
        )

        # 准备批处理
        # 对于CAKE，最大序列长度由模型自身或其原始配置决定，CAKE在其预算内管理
        # max_length参数给prepare_batch应与模型能处理的最大长度（未被CAKE修改前）或测试用例的期望长度相关
        effective_max_length = model.config.max_position_embeddings 
        logger.info(f"Preparing batch with size {batch_size}, max_length {effective_max_length}...")
        
        batch = prepare_batch(
            samples,
            tokenizer,
            batch_size=actual_num_samples_to_prepare, # 使用实际准备的样本数作为batch_size
            max_length=effective_max_length
        )

        # 将批处理数据移至设备
        inputs = {
            "input_ids": batch["input_ids"].to(model.device),
            "attention_mask": batch["attention_mask"].to(model.device)
        }
        if "token_type_ids" in batch and batch["token_type_ids"] is not None:
             inputs["token_type_ids"] = batch["token_type_ids"].to(model.device)

        # 预热（可选）
        logger.info("Warming up CAKE model...")
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=min(5, max_new_tokens), # 短预热
                do_sample=False, # 预热不需要采样
                use_cache=True, # 确保预热时也使用缓存机制
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # 清理GPU缓存
        torch.cuda.empty_cache()
        time.sleep(0.5) # 短暂停止以确保监控器捕获基线

        # 初始化监控管理器并启动监控
        # monitoring_config 来自主config.py
        monitoring_manager = MonitoringManager(experiment_id=experiment_id, config=MONITORING_CONFIG)
        if experiment_main_config.get("enable_monitoring", True):
            monitoring_manager.start_monitoring()

        # 开始性能测量
        logger.info("Starting CAKE performance measurement...")
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

        token_time_processor = TokenTimeLogitsProcessor(metrics_collector)
        logits_processor_list = LogitsProcessorList([token_time_processor])
        
        # 生成配置，从DATASET_CONFIG获取，允许覆盖
        generate_kwargs = DATASET_CONFIG.get("generate_config", {}).copy()
        generate_kwargs.update({
            "max_new_tokens": max_new_tokens,
            "logits_processor": logits_processor_list,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True # 明确启用缓存
        })
        # 如果trial_params中包含生成参数，例如do_sample, temperature, top_p，则覆盖
        if trial_params:
            for gen_param in ["do_sample", "temperature", "top_p"]:
                if gen_param in trial_params:
                    generate_kwargs[gen_param] = trial_params[gen_param]

        # 生成文本
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **generate_kwargs
            )

        # 结束性能测量
        metrics_collector.end_generation()

        # 停止监控并收集指标
        if experiment_main_config.get("enable_monitoring", True):
            monitoring_manager.stop_monitoring()
            monitoring_metrics_data = monitoring_manager.get_metrics()
            if "gpu" in monitoring_metrics_data.get("metrics", {}):
                 metrics_collector.record_gpu_stats(monitoring_metrics_data["metrics"]["gpu"])
            
            # 保存监控数据 - 使用更短的路径避免Windows路径长度限制
            try:
                # 使用更短的目录名和文件名
                monitor_dir = os.path.join(output_dir, "monitor")
                os.makedirs(monitor_dir, exist_ok=True)
                monitoring_manager.save_metrics(
                    output_dir=monitor_dir,
                    filename="stats.json"
                )
            except Exception as monitor_error:
                logger.warning(f"无法保存监控数据: {monitor_error}")
                # 继续执行，不要因为监控数据保存失败而中断实验

        # 解码输出（可选，用于质量评估）
        # generated_texts = tokenizer.batch_decode(
        #     outputs[:, inputs["input_ids"].shape[1]:],
        #     skip_special_tokens=True
        # )
        # metrics_collector.record_generated_texts(generated_texts) # 如果需要保存

        # 计算和保存指标
        final_metrics = metrics_collector.compute_and_save_metrics(output_dir, filename_prefix="cake_metrics")
        logger.info(f"CAKE Experiment {experiment_id} completed. Metrics: {final_metrics}")
        return final_metrics

    except Exception as e:
        logger.error(f"Error during CAKE experiment {experiment_id}: {e}", exc_info=True)
        metrics_collector.record_error(str(e))
        return metrics_collector.finalize_metrics_on_error()
    finally:
        # 清理模型和GPU内存
        del model
        del tokenizer
        if 'inputs' in locals(): del inputs
        if 'outputs' in locals(): del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Cleaned up resources for experiment {experiment_id}")


def main():
    parser = argparse.ArgumentParser(description="Run CAKE KV Cache Experiments")
    parser.add_argument("--model_name", type=str, default=EXPERIMENT_CONFIG["model_name_or_path"], help="Name or path of the model to use.")
    parser.add_argument("--datasets", type=str, default=",".join(EXPERIMENT_CONFIG["datasets"]), help="Comma-separated list of datasets to use.")
    parser.add_argument("--kv_cache_lengths", type=str, default=",".join(map(str, EXPERIMENT_CONFIG["cake_kv_cache_lengths"]) if EXPERIMENT_CONFIG.get("cake_kv_cache_lengths") else ""), help="Comma-separated list of KV cache lengths for CAKE experiments (might be used as base for budget). Example: 1024,2048")
    parser.add_argument("--batch_sizes", type=str, default=",".join(map(str, EXPERIMENT_CONFIG["batch_sizes"])), help="Comma-separated list of batch sizes.")
    parser.add_argument("--max_new_tokens", type=int, default=EXPERIMENT_CONFIG["max_new_tokens"], help="Maximum number of new tokens to generate.")
    parser.add_argument("--allocation_strategies", type=str, default=",".join(EXPERIMENT_CONFIG["layer_allocation_strategies"]), help="Comma-separated list of CAKE allocation strategies (e.g., uniform,adaptive,attention_based).")
    parser.add_argument("--cache_budgets", type=str, default=",".join(map(str, EXPERIMENT_CONFIG["cache_budgets"])), help="Comma-separated list of CAKE cache budgets (e.g., 0.5,0.7,0.9).")
    parser.add_argument("--repetitions", type=int, default=EXPERIMENT_CONFIG["repetitions"], help="Number of repetitions for each experiment configuration.")
    parser.add_argument("--output_dir", type=str, default=os.path.join(EXPERIMENT_CONFIG["output_base_dir"], "cake_experiments"), help="Directory to save experiment results.")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--seed", type=int, default=config.EXPERIMENT_CONFIG.get("random_seed", 42), help="Random seed for reproducibility.")
    parser.add_argument("--run_name", type=str, default=f"cake_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}", help="A specific name for this run/sweep of experiments.")

    args = parser.parse_args()

    # 创建本次运行的总输出目录
    main_output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(main_output_dir, exist_ok=True)

    # 设置日志
    log_file_path = os.path.join(main_output_dir, "cake_experiment_log.txt")
    global logger # 更新全局logger实例
    logger = setup_logging(log_file=log_file_path, level=getattr(logging, args.log_level.upper(), logging.INFO))

    logger.info(f"Starting CAKE experiment suite with run name: {args.run_name}")
    logger.info(f"Arguments: {args}")
    logger.info(f"Global EXPERIMENT_CONFIG being used: {EXPERIMENT_CONFIG}")
    logger.info(f"Global CAKE_MODEL_CONFIG being used: {CAKE_MODEL_CONFIG}")

    # 设置随机种子
    set_seed(args.seed)

    # 解析参数列表
    datasets_list = [d.strip() for d in args.datasets.split(',') if d.strip()]
    kv_lengths_list = [int(kv.strip()) for kv in args.kv_cache_lengths.split(',') if kv.strip()] if args.kv_cache_lengths else EXPERIMENT_CONFIG.get("cake_kv_cache_lengths", [])
    batch_sizes_list = [int(bs.strip()) for bs in args.batch_sizes.split(',') if bs.strip()]
    strategies_list = [s.strip() for s in args.allocation_strategies.split(',') if s.strip()]
    budgets_list = [float(b.strip()) for b in args.cache_budgets.split(',') if b.strip()]

    all_results_summary = []
    total_experiments = len(datasets_list) * len(kv_lengths_list) * len(batch_sizes_list) * len(strategies_list) * len(budgets_list) * args.repetitions
    logger.info(f"Total number of CAKE experiment configurations to run: {total_experiments}")
    pbar = tqdm(total=total_experiments, desc="Running CAKE Experiments")

    current_model_config_for_run = {
        "model_name_or_path": args.model_name,
        "precision": EXPERIMENT_CONFIG["precision"] # 假设精度是全局的
    }

    # 创建一个临时的实验配置，包含命令行指定的模型名称
    temp_experiment_config = EXPERIMENT_CONFIG.copy()
    temp_experiment_config["current_model_name"] = args.model_name

    for rep in range(args.repetitions):
        for dataset_name in datasets_list:
            # 修复数据集配置获取逻辑，与baseline_main.py保持一致
            dataset_options = DATASET_CONFIG.get("available_datasets", {}).get(dataset_name)
            if not dataset_options:
                logger.error(f"Dataset configuration for '{dataset_name}' not found in DATASET_CONFIG. Skipping...")
                # 更新pbar跳过这个数据集的所有组合
                pbar.update(len(kv_lengths_list) * len(batch_sizes_list) * len(strategies_list) * len(budgets_list))
                continue
            
            for kv_len in kv_lengths_list:
                for bs in batch_sizes_list:
                    for strategy in strategies_list:
                        for budget in budgets_list:
                            logger.info(f"Running CAKE: Rep {rep+1}/{args.repetitions}, Dataset: {dataset_name}, KV_Len: {kv_len}, Batch: {bs}, Strategy: {strategy}, Budget: {budget}")
                            
                            # 为当前实验创建特定的输出子目录
                            exp_label = f"ds_{dataset_name}_kv{kv_len}_bs{bs}_strat{strategy}_bud{budget}_rep{rep}"
                            current_exp_output_dir = os.path.join(main_output_dir, exp_label)
                            os.makedirs(current_exp_output_dir, exist_ok=True)

                            experiment_metrics = run_cake_experiment(
                                experiment_main_config=temp_experiment_config, # 使用包含命令行模型名称的配置
                                dataset_name=dataset_name,
                                dataset_options=dataset_options,
                                kv_cache_length=kv_len,
                                batch_size=bs,
                                max_new_tokens=args.max_new_tokens,
                                allocation_strategy=strategy,
                                cache_budget=budget,
                                output_dir=current_exp_output_dir,
                                repeat_index=rep
                            )
                            all_results_summary.append(experiment_metrics)
                            pbar.update(1)
    pbar.close()

    # 保存所有实验结果的汇总
    summary_file_path = os.path.join(main_output_dir, "all_cake_experiments_summary.csv")
    if all_results_summary and isinstance(all_results_summary[0], dict):
        summary_df = pd.DataFrame(all_results_summary)
        summary_df.to_csv(summary_file_path, index=False)
        logger.info(f"All CAKE experiment summaries saved to {summary_file_path}")
    elif all_results_summary:
        logger.warning(f"Result summary items are not all dicts, cannot easily save to CSV. First item: {all_results_summary[0]}")
        # Fallback to JSON if not easily convertible to DataFrame
        summary_json_path = os.path.join(main_output_dir, "all_cake_experiments_summary.json")
        try:
            with open(summary_json_path, 'w') as f:
                json.dump(all_results_summary, f, indent=4)
            logger.info(f"All CAKE experiment summaries saved to {summary_json_path} as JSON.")
        except Exception as json_e:
            logger.error(f"Could not save summary as JSON: {json_e}")

    logger.info("CAKE experiment suite finished.")

if __name__ == "__main__":
    main()