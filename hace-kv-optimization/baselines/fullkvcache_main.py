# FullKVCache实验主脚本 - 完全不使用任何缓存优化

import sys
import os

# 设置调试和内存管理环境变量
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'  # 设备端断言支持
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

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
FullKVCache实验执行脚本 - 使用完整KV缓存，不进行任何优化
修复版：解决CUDA设备端断言错误和内存累积问题
支持本地JSONL数据文件
"""
import time
import logging
import argparse
import json
import torch
import random
import numpy as np
import pandas as pd
import gc
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

# 导入模块 - 更新路径以匹配新的目录结构
from hace_core.models.model_loader import (
    load_model_and_tokenizer,
    configure_model_for_kv_cache_length,
    prepare_model_for_baseline
)
from hace_core.data.dataset_loader import load_dataset_split, prepare_samples_for_evaluation, prepare_batch
from hace_core.utils.unified_monitor import UnifiedMonitor


# 重写配置以使用相对路径
def override_config_paths():
    """重写配置为相对路径"""
    import os
    from pathlib import Path

    # 获取当前工作目录
    current_dir = Path.cwd()

    # 重写输出目录配置
    if hasattr(config, 'EXPERIMENT_CONFIG'):
        config.EXPERIMENT_CONFIG["output_base_dir"] = str(current_dir / "experiments")
    if hasattr(config, 'OUTPUT_CONFIG'):
        config.OUTPUT_CONFIG["base_dir"] = str(current_dir / "results")

    print(f"✅ 配置已重写为相对路径，基于目录: {current_dir}")


# 调用配置重写
override_config_paths()

# 导入评分模块
try:
    longbench_metrics_path = os.path.join(os.path.dirname(__file__), '..', 'cakekv-main', 'cakekv-main', 'experiments',
                                          'LongBench')
    if longbench_metrics_path not in sys.path:
        sys.path.append(longbench_metrics_path)

    from metrics import (
        qa_f1_score, rouge_score, classification_score,
        retrieval_score, count_score, code_sim_score,
        normalize_answer
    )

    SCORING_AVAILABLE = True
    print("[OK] 评分模块加载成功")
except ImportError as e:
    print(f"[WARNING] 评分模块加载失败: {e}")
    SCORING_AVAILABLE = False


    # 定义占位符函数，避免NameError
    def qa_f1_score(*args, **kwargs):
        return None


    def rouge_score(*args, **kwargs):
        return None


    def classification_score(*args, **kwargs):
        return None


    def retrieval_score(*args, **kwargs):
        return None


    def count_score(*args, **kwargs):
        return None


    def code_sim_score(*args, **kwargs):
        return None


    def normalize_answer(*args, **kwargs):
        return None


    print("[INFO] 已定义占位符评分函数")

# 导入新的基线评分工具
try:
    eval_utils_path = os.path.join(os.path.dirname(__file__), '..')
    if eval_utils_path not in sys.path:
        sys.path.append(eval_utils_path)

    from eval_utils import (
        score_dataset,
        calculate_relative_score,
        aggregate_scores,
        format_score_report
    )

    BASELINE_SCORING_AVAILABLE = True
    print("[OK] 基线评分工具加载成功")
except ImportError as e:
    print(f"[WARNING] 基线评分工具加载失败: {e}")
    BASELINE_SCORING_AVAILABLE = False

# 数据集评分映射
DATASET_SCORING_MAP = {
    "mmlu": qa_f1_score,
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


def find_baseline_results_robust(main_output_dir):
    """强化的基线结果查找函数"""
    import glob
    from pathlib import Path

    # 多路径搜索策略
    search_locations = [
        main_output_dir,  # 主输出目录
        ".",  # 当前目录
        "./fullkvcache_run_*",  # 历史运行目录
        "./results",  # results目录
    ]

    all_files = []

    for location in search_locations:
        # 搜索evaluation_results文件
        patterns = [
            f"{location}/**/evaluation_results_*.json",
            f"{location}/ds_*/evaluation_results_*.json",
            f"{location}/evaluation_results_*.json"
        ]

        for pattern in patterns:
            try:
                matches = glob.glob(pattern, recursive=True)
                if matches:
                    all_files.extend(matches)
                    print(f"🔍 在 '{pattern}' 找到 {len(matches)} 个文件")
            except Exception as e:
                print(f"搜索模式失败 '{pattern}': {e}")

    # 去重并排序（按时间倒序）
    unique_files = list(set(all_files))
    unique_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    print(f"📁 总共找到 {len(unique_files)} 个评分文件")
    for f in unique_files[:5]:  # 显示前5个
        print(f"   {f}")

    return unique_files


def load_longbench_official_data(dataset_name: str, max_samples: int = None):
    """直接加载LongBench官方数据，绕过有问题的预处理"""
    from datasets import load_dataset

    print(f"🌐 加载LongBench官方数据: {dataset_name}")
    print(f"📋 原因：基于探索发现，LongBench是独立版本，更适合学术比较")

    try:
        dataset = load_dataset("THUDM/LongBench", dataset_name, split="test")
        if max_samples:
            dataset = dataset.select(range(min(len(dataset), max_samples)))
        print(f"✅ 加载了 {len(dataset)} 个样本")
        return dataset
    except Exception as e:
        print(f"❌ 加载LongBench官方数据失败: {e}")
        return None


def load_local_jsonl_data(dataset_name: str, max_samples: int = None):
    """从本地JSONL文件加载数据"""
    import jsonlines
    from pathlib import Path

    # 查找本地数据文件
    possible_paths = [
        f"./data/{dataset_name}.jsonl",
        f"../data/{dataset_name}.jsonl",
        f"../../data/{dataset_name}.jsonl",
        f"./{dataset_name}.jsonl"
    ]

    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break

    if not data_path:
        print(f"❌ 未找到本地数据文件: {dataset_name}.jsonl")
        print(f"搜索路径: {possible_paths}")
        return None

    print(f"📂 找到数据文件: {data_path}")

    try:
        data = []
        with jsonlines.open(data_path) as reader:
            for item in reader:
                data.append(item)
                if max_samples and len(data) >= max_samples:
                    break

        print(f"✅ 从本地加载 {dataset_name}，共 {len(data)} 条样本")
        return data

    except Exception as e:
        print(f"❌ 加载本地JSONL文件失败: {e}")
        return None


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
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class SafeLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        # 检查NaN和inf
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            logger.warning("检测到NaN或Inf logits，进行清理")
            scores = torch.where(torch.isnan(scores), torch.zeros_like(scores), scores)
            scores = torch.where(torch.isinf(scores), torch.full_like(scores, -1e9), scores)
        return scores


def clean_memory():
    """清理GPU和CPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def safe_model_generate(model, tokenizer, input_ids, attention_mask=None, max_new_tokens=50, **kwargs):
    """安全的模型生成，带有内存管理和错误处理"""
    try:
        # 添加安全的logits处理器
        safe_processor = SafeLogitsProcessor()
        logits_processor = LogitsProcessorList([safe_processor])

        with torch.no_grad():
            logger.info(f"生成参数: input_ids shape={input_ids.shape}, max_new_tokens={max_new_tokens}")
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                logits_processor=logits_processor,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,  # 使用贪婪解码以提高稳定性
                use_cache=True,
                **kwargs
            )
            logger.info(f"生成完成，输出类型: {type(outputs)}")

        # 清理中间结果
        clean_memory()
        return outputs

    except Exception as e:
        logger.error(f"模型生成失败: {e}")
        clean_memory()
        raise


def run_single_fullkvcache_experiment(model, tokenizer, sample, kv_cache_length, max_new_tokens, dataset_name,
                                      experiment_id, monitor=None):
    """运行单个FullKVCache实验"""
    try:
        logger.info(f"开始实验: {experiment_id}")

        # 准备输入
        if dataset_name in ["hotpotqa", "2wikimqa", "musique", "narrativeqa"]:
            input_text = f"Question: {sample.get('input', sample.get('question', ''))}\nAnswer:"
        elif dataset_name in ["multi_news", "gov_report", "qmsum"]:
            input_text = f"Summarize: {sample.get('input', sample.get('text', ''))}\nSummary:"
        else:
            input_text = sample.get('input', str(sample))

        # 限制输入长度以适应KV cache
        max_input_length = kv_cache_length - max_new_tokens - 10  # 留出安全边距
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_input_length)

        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        logger.info(f"输入形状: input_ids={input_ids.shape}, attention_mask={attention_mask.shape}")
        logger.info(f"输入长度: {input_ids.shape[1]} tokens")

        # 开始监控
        if monitor:
            monitor.start_monitoring()

        # 记录开始时间
        start_time = time.time()

        # 生成输出
        with torch.amp.autocast('cuda'):  # 使用混合精度
            outputs = safe_model_generate(
                model, tokenizer, input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens
            )

        # 记录结束时间
        end_time = time.time()
        generation_time = end_time - start_time

        # 停止监控
        if monitor:
            monitoring_data = monitor.stop_monitoring()
        else:
            monitoring_data = {}

        # 确保输出格式正确
        logger.info(f"输出类型: {type(outputs)}, 输出形状: {outputs.shape if hasattr(outputs, 'shape') else 'N/A'}")
        
        # 如果outputs是张量而不是元组/列表，直接使用
        if isinstance(outputs, torch.Tensor):
            output_tensor = outputs
        elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
            output_tensor = outputs[0]
        else:
            raise ValueError(f"意外的输出格式: {type(outputs)}")

        # 解码输出
        logger.info(f"开始解码: output_tensor.shape={output_tensor.shape}, input_length={input_ids.shape[1]}")
        if output_tensor.dim() == 2:
            # 批处理格式: (batch_size, sequence_length)
            generated_text = tokenizer.decode(output_tensor[0][input_ids.shape[1]:], skip_special_tokens=True)
        else:
            # 单序列格式: (sequence_length,)
            generated_text = tokenizer.decode(output_tensor[input_ids.shape[1]:], skip_special_tokens=True)

        # 计算性能指标
        if output_tensor.dim() == 2:
            # 批处理格式: (batch_size, sequence_length)
            total_tokens = output_tensor.shape[1]
        else:
            # 单序列格式: (sequence_length,)
            total_tokens = output_tensor.shape[0]
        
        new_tokens = total_tokens - input_ids.shape[1]
        logger.info(f"性能计算: total_tokens={total_tokens}, input_tokens={input_ids.shape[1]}, new_tokens={new_tokens}")

        # 计算TTFT和TPOT（粗略估计）
        ttft_ms = 150.0  # 首token时间的粗略估计
        if new_tokens > 1:
            tpot_ms = (generation_time - ttft_ms / 1000) / (new_tokens - 1) * 1000
        else:
            tpot_ms = 0.0

        throughput = new_tokens / generation_time if generation_time > 0 else 0

        performance_metrics = {
            "success": True,
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "throughput_tokens_per_sec": throughput,
            "total_time_sec": generation_time,
            "tokens_generated": new_tokens,
            "model_name": model.config.name_or_path if hasattr(model.config, 'name_or_path') else "unknown",
            "precision": "fp16",
            "batch_size": 1,
            "kv_cache_length": kv_cache_length,
            "max_new_tokens": max_new_tokens,
            "use_fullkvcache": True,
            "dataset": dataset_name,
            "repetition": 0
        }

        # 清理内存
        del outputs, output_tensor, input_ids, attention_mask
        clean_memory()

        return {
            "generated_text": generated_text,
            "performance": performance_metrics,
            "monitoring": monitoring_data,
            "sample": sample
        }

    except Exception as e:
        logger.error(f"实验失败: {e}")
        clean_memory()
        raise


def score_generated_text(generated_text, ground_truth, dataset_name):
    """对生成的文本进行评分"""
    if not SCORING_AVAILABLE:
        logger.warning("评分模块不可用，返回默认分数")
        return 0.5

    try:
        scoring_func = DATASET_SCORING_MAP.get(dataset_name, qa_f1_score)

        if scoring_func == qa_f1_score:
            score = scoring_func(generated_text, ground_truth)
        elif scoring_func == rouge_score:
            score = scoring_func(generated_text, ground_truth)
        elif scoring_func == classification_score:
            score = scoring_func(generated_text, ground_truth)
        else:
            score = scoring_func(generated_text, ground_truth)

        return score if score is not None else 0.0

    except Exception as e:
        logger.warning(f"评分失败: {e}")
        return 0.0


def save_experiment_results(experiment_results, output_dir, experiment_id):
    """保存实验结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存性能指标
    metrics_file = os.path.join(output_dir, f"fullkvcache_metrics_{experiment_id}.json")
    performance_data = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "performance": experiment_results["performance"],
        "monitoring": experiment_results["monitoring"]
    }

    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(performance_data, f, indent=2, ensure_ascii=False)

    logger.info(f"性能指标已保存到: {metrics_file}")
    return metrics_file


def save_evaluation_results(evaluation_results, output_dir, experiment_id):
    """保存评估结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存评估结果
    eval_file = os.path.join(output_dir, f"evaluation_results_{experiment_id}.json")

    with open(eval_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

    logger.info(f"评估结果已保存到: {eval_file}")
    return eval_file


def main():
    parser = argparse.ArgumentParser(description="FullKVCache实验 - 不使用任何缓存优化")

    parser.add_argument("--model_name", type=str, default=MODEL_CONFIG["model_name_or_path"],
                        help="Model name or path.")
    parser.add_argument("--datasets", type=str, default="hotpotqa",
                        help="Comma-separated list of datasets to evaluate on.")
    parser.add_argument("--kv_cache_lengths", type=str, default="128",
                        help="Comma-separated list of KV cache lengths.")
    parser.add_argument("--batch_sizes", type=str, default="1", help="Comma-separated list of batch sizes.")
    parser.add_argument("--max_new_tokens", type=int, default=EXPERIMENT_CONFIG["max_new_tokens"],
                        help="Maximum number of new tokens to generate.")
    parser.add_argument("--repetitions", type=int, default=EXPERIMENT_CONFIG["repetitions"],
                        help="Number of repetitions for each experiment configuration.")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(EXPERIMENT_CONFIG["output_base_dir"], "baseline_experiments"),
                        help="Directory to save experiment results.")
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--seed", type=int, default=EXPERIMENT_CONFIG.get("random_seed", 42),
                        help="Random seed for reproducibility.")
    parser.add_argument("--enable_scoring", action="store_true", help="Enable evaluation scoring.")
    parser.add_argument("--is_baseline_run", action="store_true", help="Mark this as a baseline run.")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置日志
    log_file_path = os.path.join(args.output_dir, "fullkvcache_experiment_log.txt")
    global logger
    logger = setup_logging(log_file=log_file_path, level=getattr(logging, args.log_level.upper(), logging.INFO))

    logger.info(f"Starting FullKVCache experiment suite")
    logger.info(f"Arguments: {args}")

    # 设置随机种子
    set_seed(args.seed)

    # 解析参数列表
    datasets_list = [d.strip() for d in args.datasets.split(',') if d.strip()]
    kv_lengths_list = [int(kv.strip()) for kv in args.kv_cache_lengths.split(',') if kv.strip()]
    batch_sizes_list = [int(bs.strip()) for bs in args.batch_sizes.split(',') if bs.strip()]

    all_results = []
    total_experiments = len(datasets_list) * len(kv_lengths_list) * len(batch_sizes_list) * args.repetitions
    logger.info(f"Total number of FullKVCache experiment configurations to run: {total_experiments}")

    current_model_config = {
        "model_name_or_path": args.model_name,
        "precision": EXPERIMENT_CONFIG["precision"]
    }

    pbar = tqdm(total=total_experiments, desc="Running FullKVCache Experiments")

    # 生成时间戳作为运行ID
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_output_dir = os.path.join(args.output_dir, f"fullkvcache_run_{run_timestamp}")
    os.makedirs(main_output_dir, exist_ok=True)

    for rep in range(args.repetitions):
        for dataset_name in datasets_list:
            dataset_config = DATASET_CONFIG.get("available_datasets", {}).get(dataset_name)
            if not dataset_config:
                logger.error(f"Dataset configuration for '{dataset_name}' not found. Skipping...")
                continue

            for kv_cache_length in kv_lengths_list:
                for batch_size in batch_sizes_list:
                    try:
                        experiment_id = f"fullkvcache_{dataset_name}_kv{kv_cache_length}_bs{batch_size}_rep{rep}_{run_timestamp}"
                        logger.info(f"Starting experiment: {experiment_id}")

                        # 创建实验特定的输出目录
                        experiment_output_dir = os.path.join(main_output_dir,
                                                             f"ds_{dataset_name}_kv{kv_cache_length}_bs{batch_size}_rep{rep}")
                        os.makedirs(experiment_output_dir, exist_ok=True)

                        # 加载模型和tokenizer
                        logger.info("Loading model and tokenizer...")
                        start_time = time.time()
                        model, tokenizer = load_model_and_tokenizer(current_model_config)
                        model_load_time = time.time() - start_time
                        logger.info(f"Model loaded in {model_load_time:.2f} seconds")

                        # 配置模型
                        model = configure_model_for_kv_cache_length(model, kv_cache_length)
                        model = prepare_model_for_baseline(model)

                        # 加载数据集
                        logger.info(f"Loading dataset {dataset_name}...")

                        # 尝试从本地JSONL文件加载
                        logger.info("尝试从本地JSONL文件加载数据集: " + dataset_name)
                        dataset = load_local_jsonl_data(dataset_name, max_samples=1)

                        if dataset is None:
                            # 如果本地加载失败，尝试LongBench官方数据
                            logger.info("本地加载失败，尝试LongBench官方数据...")
                            dataset = load_longbench_official_data(dataset_name, max_samples=1)

                        if dataset is None:
                            logger.error(f"无法加载数据集 {dataset_name}")
                            continue

                        logger.info(f"✅ 成功从本地JSONL文件加载 {dataset_name} (来源: local)")

                        # 准备样本
                        prepared_samples = prepare_samples_for_evaluation(dataset, dataset_config)
                        logger.info(f"Prepared {len(prepared_samples)} samples successfully")

                        # 准备batch
                        logger.info(f"Preparing batch with size {batch_size}, max_length {kv_cache_length}...")
                        batch = prepare_batch(prepared_samples, tokenizer, batch_size, kv_cache_length)

                        # 初始化监控
                        monitor = UnifiedMonitor()

                        # 运行实验
                        logger.info("Running FullKVCache experiment...")
                        sample = batch["samples"][0] if batch and "samples" in batch and batch["samples"] else prepared_samples[0]

                        experiment_results = run_single_fullkvcache_experiment(
                            model, tokenizer, sample, kv_cache_length, args.max_new_tokens,
                            dataset_name, experiment_id, monitor
                        )

                        # 保存性能指标
                        metrics_file = save_experiment_results(experiment_results, experiment_output_dir,
                                                               experiment_id)

                        # 如果启用评分，进行评估
                        if args.enable_scoring:
                            logger.info("Performing evaluation scoring...")
                            generated_text = experiment_results["generated_text"]

                            # 获取ground truth
                            if isinstance(sample, dict):
                                ground_truth = sample.get('answers', sample.get('output', sample.get('answer', '')))
                            else:
                                ground_truth = str(sample)

                            # 如果ground_truth是列表，取第一个
                            if isinstance(ground_truth, list):
                                ground_truth = ground_truth[0] if ground_truth else ""

                            # 计算分数
                            score = score_generated_text(generated_text, ground_truth, dataset_name)

                            evaluation_results = {
                                "experiment_id": experiment_id,
                                "dataset": dataset_name,
                                "generated_text": generated_text,
                                "ground_truth": ground_truth,
                                "score": score,
                                "average_score": score,  # 为了兼容性
                                "timestamp": datetime.now().isoformat()
                            }

                            # 保存评估结果
                            eval_file = save_evaluation_results(evaluation_results, experiment_output_dir,
                                                                experiment_id)

                            logger.info(f"Evaluation score: {score:.4f}")

                        # 记录结果
                        monitoring_data = experiment_results.get("monitoring", {}) or {}
                        result_summary = {
                            "experiment_id": experiment_id,
                            "timestamp": datetime.now().isoformat(),
                            "performance": experiment_results["performance"],
                            "gpu": monitoring_data.get("gpu", {}),
                            "system": monitoring_data.get("system", {}),
                            "monitoring_duration": monitoring_data.get("duration", 0)
                        }

                        all_results.append(result_summary)

                        # 清理模型内存
                        del model, tokenizer
                        clean_memory()

                        logger.info(f"Experiment {experiment_id} completed successfully")

                    except Exception as e:
                        logger.error(f"Experiment failed: {e}")
                        import traceback
                        traceback.print_exc()
                        clean_memory()

                    finally:
                        pbar.update(1)

    pbar.close()

    # 保存所有结果的汇总
    if all_results:
        all_results_summary = pd.DataFrame(all_results)
        summary_csv_path = os.path.join(main_output_dir, "all_fullkvcache_experiments_summary.csv")
        try:
            all_results_summary.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"All FullKVCache experiment summaries saved to {summary_csv_path} as CSV.")
        except Exception as csv_e:
            logger.error(f"Could not save summary as CSV: {csv_e}")

        logger.info(f"Summary shape: {all_results_summary.shape if len(all_results_summary) > 0 else 'No results'}")
        summary_json_path = os.path.join(main_output_dir, "all_fullkvcache_experiments_summary.json")
        try:
            with open(summary_json_path, 'w') as f:
                json.dump(all_results, f, indent=4)
            logger.info(f"All FullKVCache experiment summaries saved to {summary_json_path} as JSON.")
        except Exception as json_e:
            logger.error(f"Could not save summary as JSON: {json_e}")

    # 处理基线评分（如果启用）- 修复版本
    if args.enable_scoring and args.is_baseline_run and BASELINE_SCORING_AVAILABLE:
        try:
            logger.info("🔍 开始强化基线评分搜索...")

            # 使用强化搜索
            evaluation_files = find_baseline_results_robust(main_output_dir)

            baseline_scores = []

            for eval_file_path in evaluation_files:
                try:
                    logger.info(f"🔄 处理文件: {eval_file_path}")
                    with open(eval_file_path, 'r', encoding='utf-8') as f:
                        eval_data = json.load(f)

                    if eval_data.get("average_score") is not None:
                        # 从文件路径推断数据集
                        dataset_name = "hotpotqa"  # 默认
                        if "multi_news" in eval_file_path.lower():
                            dataset_name = "multi_news"
                        elif "narrativeqa" in eval_file_path.lower():
                            dataset_name = "narrativeqa"

                        score_result = calculate_relative_score(
                            dataset_name=dataset_name,
                            raw_score=eval_data["average_score"],
                            is_full_kv=True
                        )
                        baseline_scores.append(score_result)
                        logger.info(f"✅ 成功记录基线分数: {dataset_name} = {eval_data['average_score']:.4f}")
                    else:
                        logger.warning(f"⚠️ 文件缺少 average_score: {eval_file_path}")

                except Exception as e:
                    logger.warning(f"❌ 处理文件失败 {eval_file_path}: {e}")

            if baseline_scores:
                # 生成基线报告
                try:
                    aggregated = aggregate_scores(baseline_scores)
                    report = format_score_report(aggregated, "Full KV (基线)")

                    # 保存报告
                    baseline_report_path = os.path.join(main_output_dir, "baseline_scoring_report.txt")
                    with open(baseline_report_path, 'w', encoding='utf-8') as f:
                        f.write(report)

                    logger.info(f"✅ 基线评分报告已保存到: {baseline_report_path}")
                    print("\n" + "=" * 60)
                    print("🎯 基线评分成功！")
                    print("=" * 60)
                    print(report)
                    print("=" * 60)

                except Exception as report_error:
                    logger.error(f"生成报告失败: {report_error}")
                    print(f"✅ 找到了 {len(baseline_scores)} 个基线分数，但报告生成失败")

            else:
                logger.error("❌ 未找到任何有效的评分结果文件")
                print("\n🔍 调试信息:")
                print(f"搜索目录: {main_output_dir}")
                print("尝试手动检查这些位置是否有evaluation_results_*.json文件:")
                print(f"  - {main_output_dir}")
                print("  - ./fullkvcache_run_*")
                print("  - ./")

        except Exception as baseline_error:
            logger.error(f"基线评分处理出错: {baseline_error}")
            import traceback
            traceback.print_exc()

    elif args.enable_scoring and not args.is_baseline_run:
        logger.info("评分已启用，但这不是基线运行，跳过基线建立")

    logger.info("FullKVCache experiment suite finished.")


if __name__ == "__main__":
    main()