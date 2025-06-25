import sys
import os

# 获取当前文件的绝对路径
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

# 导入基线评分工具
try:
    from eval_utils import (
        score_dataset,
        calculate_relative_score,
        aggregate_scores,
        format_score_report
    )
    BASELINE_SCORING_AVAILABLE = True
    print("[OK] CAKE基线评分工具加载成功")
except ImportError as e:
    print(f"[WARNING] CAKE基线评分工具加载失败: {e}")
    BASELINE_SCORING_AVAILABLE = False

# 修改导入语句以匹配新的目录结构
from hace_core.models.model_loader import (
    load_model_and_tokenizer,
    configure_model_for_kv_cache_length,
    prepare_model_for_cake,  # 主要变化
    is_model_type_supported_by_cake # 主要变化
)
from hace_core.data.dataset_loader import load_dataset_split, prepare_samples_for_evaluation, prepare_batch
from hace_core.utils.unified_monitor import UnifiedMonitor


def load_local_jsonl(dataset_name, data_dir=None):
    """
    从本地JSONL文件加载数据集
    
    Args:
        dataset_name: 数据集名称
        data_dir: 数据目录路径
        
    Returns:
        dataset: 数据列表
    """
    # 动态确定数据目录路径
    if data_dir is None:
        # 获取脚本所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 尝试多个可能的数据路径
        possible_data_dirs = [
            os.path.join(script_dir, "data"),  # 当前baselines/data/
            os.path.join(script_dir, "..", "..", "data"),  # 项目根目录的data/
            os.path.join(script_dir, "..", "data"),  # hace-kv-optimization/data/
        ]
        
        data_dir = None
        for possible_dir in possible_data_dirs:
            test_file = os.path.join(possible_dir, f"{dataset_name}.jsonl")
            if os.path.exists(test_file):
                data_dir = possible_dir
                logger.info(f"找到数据文件: {test_file}")
                break
        
        if data_dir is None:
            # 如果都找不到，使用默认路径并提供详细错误信息
            data_dir = possible_data_dirs[0]  # 使用第一个作为默认
            logger.warning(f"在以下路径中未找到 {dataset_name}.jsonl:")
            for path in possible_data_dirs:
                logger.warning(f"  - {os.path.join(path, f'{dataset_name}.jsonl')}")
    
    file_path = os.path.join(data_dir, f"{dataset_name}.jsonl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 本地文件不存在: {file_path}")

    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"跳过无效的JSON行: {line[:100]}... 错误: {e}")
    
    logger.info(f"✅ 从本地加载 {dataset_name}，共 {len(data)} 条样本")
    return data


def load_dataset_with_fallback(dataset_name, dataset_options, split="test"):
    """
    加载数据集，优先使用Hugging Face，失败时回退到本地JSONL文件
    
    Args:
        dataset_name: 数据集名称
        dataset_options: 数据集配置选项
        split: 数据分割名称
        
    Returns:
        dataset: 加载的数据集
    """
    # 首先尝试从本地JSONL文件加载（优先级更高，确保使用带答案的验证集）
    try:
        logger.info(f"尝试从本地JSONL文件加载数据集: {dataset_name}")
        local_data = load_local_jsonl(dataset_name)
        
        # 创建一个简单的数据集对象，模拟datasets库的格式
        class SimpleDataset:
            def __init__(self, data):
                self.data = data
                
            def __len__(self):
                return len(self.data)
                
            def __getitem__(self, idx):
                return self.data[idx]
                
            def __iter__(self):
                return iter(self.data)
        
        dataset = SimpleDataset(local_data)
        logger.info(f"✅ 成功从本地JSONL文件加载 {dataset_name} (来源: local)")
        return dataset
    except Exception as local_error:
        logger.warning(f"⚠️ 无法从本地加载 {dataset_name}: {local_error}")
        logger.info(f"回退到从Hugging Face加载数据集: {dataset_name}")
        try:
            # 回退到Hugging Face加载
            dataset = load_dataset_split(dataset_options, split=split)
            logger.info(f"✅ 成功从Hugging Face加载 {dataset_name} (来源: huggingface)")
            return dataset
        except Exception as hf_error:
            logger.error(f"❌ 无法从Hugging Face加载 {dataset_name}: {hf_error}")
            raise Exception(f"无法从任何来源加载数据集 {dataset_name}。本地错误: {local_error}. Hugging Face错误: {hf_error}")


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

    # 初始化统一监控器
    monitor = UnifiedMonitor(experiment_id=experiment_id)
    monitor.record_config({
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

    model = None  # 初始化model变量
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
            monitor.mark_failure(error_msg)
            return monitor.get_comprehensive_metrics()

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

        # 加载数据集，使用新的回退机制
        logger.info(f"Loading dataset {dataset_name}...")
        dataset = load_dataset_with_fallback(dataset_name, dataset_options, split="test")

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
            monitor.mark_failure(error_msg)
            return monitor.get_comprehensive_metrics()

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

        # 启动统一监控
        if experiment_main_config.get("enable_monitoring", True):
            monitor.start_monitoring()

        # 开始性能测量
        logger.info("Starting CAKE performance measurement...")
        monitor.start_generation()

        # 定义自定义 LogitsProcessor 来记录令牌生成时间
        class TokenTimeLogitsProcessor(LogitsProcessor):
            def __init__(self, monitor):
                self.monitor = monitor
                self.first_token_recorded = False

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
                if not self.first_token_recorded:
                    self.monitor.record_first_token()
                    self.first_token_recorded = True
                else:
                    self.monitor.record_token()
                return scores

        token_time_processor = TokenTimeLogitsProcessor(monitor)
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
        monitor.end_generation()

        # 停止监控并收集指标
        if experiment_main_config.get("enable_monitoring", True):
            monitor.stop_monitoring()

        # 计算和保存指标
        final_metrics = monitor.get_comprehensive_metrics()
        monitor.save_metrics(output_dir, filename=f"cake_metrics_{experiment_id}.json")
        logger.info(f"CAKE Experiment {experiment_id} completed. Metrics: {final_metrics}")
        return final_metrics

    except Exception as e:
        logger.error(f"Error during CAKE experiment {experiment_id}: {e}", exc_info=True)
        monitor.mark_failure(str(e))
        return monitor.get_comprehensive_metrics()
    finally:
        # 清理模型和GPU内存
        try:
            if 'model' in locals() and model is not None:
                del model
            if 'tokenizer' in locals():
                del tokenizer
            if 'inputs' in locals(): 
                del inputs
            if 'outputs' in locals(): 
                del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Cleaned up resources for experiment {experiment_id}")
        except Exception as cleanup_error:
            logger.warning(f"Error during cleanup: {cleanup_error}")


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
    parser.add_argument("--enable_scoring", action="store_true", help="Enable relative scoring against Full KV baseline")

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
            # 修复数据集配置获取逻辑
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

    # 处理相对评分（如果启用）
    if args.enable_scoring and BASELINE_SCORING_AVAILABLE:
        try:
            logger.info("开始处理CAKE相对评分...")
            
            # 收集所有实验的评分结果，计算相对于基线的分数
            relative_scores = []
            
            for result in all_results_summary:
                if isinstance(result, dict) and 'experiment_id' in result:
                    # 查找对应的评分文件
                    experiment_id = result['experiment_id']
                    
                    # 从实验ID中提取数据集名称
                    if 'cake_' in experiment_id:
                        parts = experiment_id.split('_')
                        if len(parts) > 1:
                            dataset_part = parts[1]  # cake_datasetname_...
                            
                            # 查找评分结果文件
                            for root, dirs, files in os.walk(main_output_dir):
                                for file in files:
                                    if file.startswith(f"evaluation_results_") and experiment_id in file:
                                        eval_file_path = os.path.join(root, file)
                                        try:
                                            with open(eval_file_path, 'r', encoding='utf-8') as f:
                                                eval_data = json.load(f)
                                                if eval_data.get("average_score") is not None:
                                                    score_result = calculate_relative_score(
                                                        dataset_name=dataset_part,
                                                        raw_score=eval_data["average_score"],
                                                        is_full_kv=False
                                                    )
                                                    # 添加策略信息
                                                    score_result["strategy"] = "CAKE"
                                                    score_result["experiment_id"] = experiment_id
                                                    relative_scores.append(score_result)
                                                    logger.info(f"CAKE相对分数: {dataset_part} = {score_result['relative_score']:.2f}/100")
                                        except Exception as e:
                                            logger.warning(f"处理CAKE评分文件时出错 {eval_file_path}: {e}")
            
            if relative_scores:
                # 生成CAKE相对评分报告
                aggregated = aggregate_scores(relative_scores)
                report = format_score_report(aggregated, "CAKE")
                
                # 保存CAKE评分报告
                cake_report_path = os.path.join(main_output_dir, "cake_relative_scoring_report.txt")
                with open(cake_report_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                
                logger.info(f"CAKE相对评分报告已保存到: {cake_report_path}")
                print(report)
                
                # 保存详细相对评分数据
                detailed_scores_path = os.path.join(main_output_dir, "cake_detailed_relative_scores.json")
                with open(detailed_scores_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "strategy": "CAKE",
                        "summary": aggregated,
                        "detailed_scores": relative_scores
                    }, f, indent=2, ensure_ascii=False)
                
                logger.info(f"CAKE详细相对评分数据已保存到: {detailed_scores_path}")
            else:
                logger.warning("未找到有效的CAKE评分结果，无法计算相对分数")
                
        except Exception as scoring_error:
            logger.error(f"处理CAKE相对评分时出错: {scoring_error}")
    
    elif args.enable_scoring and not BASELINE_SCORING_AVAILABLE:
        logger.warning("相对评分已启用，但基线评分工具不可用")
    
    logger.info("CAKE experiment suite finished.")

if __name__ == "__main__":
    main()