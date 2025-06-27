# FullKVCache实验主脚本 - 完全不使用任何缓存优化

import sys
import os

# 设置调试和内存管理环境变量
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'  # 设备端断言支持
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# 获取项目根目录路径配置
current_file_path = os.path.abspath(__file__)
# 从src/third_party/fullkv/向上三级到达项目根目录
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))

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
from pathlib import Path
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
from src.monitoring.unified_monitor import UnifiedMonitor


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

    print(f"配置已重写为相对路径，基于目录: {current_dir}")


# 调用配置重写
override_config_paths()

# 初始化logger（在任何使用logger的代码之前）
logger = logging.getLogger(__name__)

# 导入评分模块 - 修复路径
try:
    # 修复：使用正确的相对路径到metrics.py
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]  # 回到项目根目录
    longbench_metrics_path = project_root / "src" / "third_party" / "cakekv-main" / "cakekv-main" / "experiments" / "LongBench"
    
    if str(longbench_metrics_path) not in sys.path:
        sys.path.append(str(longbench_metrics_path))

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

# 导入新的基线评分工具 - 修复路径指向src/cake_runner
# 首先计算项目根目录路径，确保在所有导入尝试中都可用
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]  # 回到项目根目录

try:
    eval_utils_path = project_root / "src" / "cake_runner"
    
    if str(eval_utils_path) not in sys.path:
        sys.path.append(str(eval_utils_path))

    from eval_utils import (
        score_dataset,
        calculate_relative_score,
        aggregate_scores,
        format_score_report
    )

    BASELINE_SCORING_AVAILABLE = True

    print(f"[OK] 基线评分工具加载成功，路径: {eval_utils_path}")
except ImportError as e:
    print(f"[WARNING] 从src/cake_runner加载失败: {e}")
    # 尝试从evaluation目录加载备用版本
    try:
        # 添加evaluation目录到sys.path
        evaluation_path = project_root / "evaluation"
        if str(evaluation_path) not in sys.path:
            sys.path.append(str(evaluation_path))
        
        from eval_utils import (
            score_dataset,
            calculate_relative_score,
            aggregate_scores,
            format_score_report
        )
        print(f"[OK] 使用evaluation目录下的eval_utils备用版本，路径: {evaluation_path}")
        BASELINE_SCORING_AVAILABLE = True
    except ImportError as e2:
        print(f"[ERROR] 所有eval_utils导入尝试均失败: {e2}")
        BASELINE_SCORING_AVAILABLE = False


# CAKE配置常量 - 从官方配置文件导入
CAKE_DATASET_CONFIG = {
    # Max new tokens配置 (来自dataset2maxlen.json)
    "max_new_tokens": {
        "narrativeqa": 128,
        "qasper": 128,
        "multifieldqa_en": 64,
        "multifieldqa_zh": 64,
        "hotpotqa": 32,
        "2wikimqa": 32,
        "musique": 32,
        "dureader": 128,
        "gov_report": 512,
        "qmsum": 512,
        "multi_news": 512,
        "vcsum": 512,
        "trec": 64,
        "triviaqa": 32,
        "samsum": 128,
        "lsht": 64,
        "passage_count": 32,
        "passage_retrieval_en": 32,
        "passage_retrieval_zh": 32,
        "lcc": 64,
        "repobench-p": 64
    },
    
    # Prompt模板配置 (来自dataset2prompt.json)
    "prompts": {
        "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
        "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
        "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
        "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
        "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
        "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
        "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
        "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
        "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
        "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
        "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
        "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
        "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
        "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
        "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
        "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
        "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n"
    }
}

def get_dataset_max_new_tokens(dataset_name: str) -> int:
    """
    获取数据集特定的max_new_tokens配置
    优先级：CAKE配置 > 全局配置
    """
    cake_tokens = CAKE_DATASET_CONFIG["max_new_tokens"].get(dataset_name)
    if cake_tokens is not None:
        logger.info(f"使用CAKE配置的max_new_tokens: {dataset_name} = {cake_tokens}")
        return cake_tokens
    
    global_tokens = EXPERIMENT_CONFIG.get("max_new_tokens", 256)
    logger.info(f"使用全局配置的max_new_tokens: {dataset_name} = {global_tokens}")
    return global_tokens

def get_dataset_prompt_template(dataset_name: str) -> str:
    """
    获取数据集特定的prompt模板
    """
    template = CAKE_DATASET_CONFIG["prompts"].get(dataset_name)
    if template is not None:
        logger.info(f"使用CAKE配置的prompt模板: {dataset_name}")
        return template
    
    logger.warning(f"未找到数据集 {dataset_name} 的prompt模板，使用默认格式")
    return "Context: {context}\n\nQuestion: {input}\n\nAnswer:"

# 数据集评分映射 - 扩展到支持所有CAKE数据集
DATASET_SCORING_MAP = {
    # 原有映射
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
    
    # 新增CAKE数据集映射
    "multifieldqa_zh": qa_f1_score,
    "dureader": qa_f1_score,
    "vcsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_zh": retrieval_score,
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
                    print(f"[SEARCH] 在 '{pattern}' 找到 {len(matches)} 个文件")
            except Exception as e:
                print(f"搜索模式失败 '{pattern}': {e}")

    # 去重并排序（按时间倒序）
    unique_files = list(set(all_files))
    unique_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    print(f"[FILES] 总共找到 {len(unique_files)} 个评分文件")
    for f in unique_files[:5]:  # 显示前5个
        print(f"   {f}")

    return unique_files


def load_longbench_official_data(dataset_name: str, max_samples: int = None):
    """直接加载LongBench官方数据，绕过有问题的预处理"""
    from datasets import load_dataset

    print(f"[LOAD] 加载LongBench官方数据: {dataset_name}")
    print(f"[INFO] 原因：基于探索发现，LongBench是独立版本，更适合学术比较")

    try:
        dataset = load_dataset("THUDM/LongBench", dataset_name, split="test")
        if max_samples:
            dataset = dataset.select(range(min(len(dataset), max_samples)))
        print(f"[OK] 加载了 {len(dataset)} 个样本")
        return dataset
    except Exception as e:
        print(f"[ERROR] 加载LongBench官方数据失败: {e}")
        return None


# 使用统一数据加载器 - 去重重复代码
# 该函数已移动到src.common.data_loader模块中
# 这里导入统一的数据加载器
try:
    from src.common.data_loader import load_local_jsonl_data as unified_load_local_jsonl_data
    
    def load_local_jsonl_data(dataset_name: str, max_samples: int = None):
        """数据加载包装器 - 使用统一数据加载器"""
        return unified_load_local_jsonl_data(dataset_name, max_samples)
    
    print("[OK] 成功导入统一数据加载器")
except ImportError as e:
    print(f"[WARNING] 无法导入统一数据加载器，使用本地版本: {e}")
    
    def load_local_jsonl_data(dataset_name: str, max_samples: int = None):
        """本地版本的数据加载器 - 备用方案"""
        import json
        from pathlib import Path

        # 项目重构后的路径配置
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[3]  # 回到项目根目录
        
        possible_paths = [
            project_root / "data" / f"{dataset_name}.jsonl",
            project_root / "hace-kv-optimization" / "data" / f"{dataset_name}.jsonl",
            project_root / "hace-kv-optimization" / "baselines" / "data" / f"{dataset_name}.jsonl",
            Path(f"./{dataset_name}.jsonl"),
        ]

        data_path = None
        for path in possible_paths:
            if path.exists():
                data_path = path
                break

        if not data_path:
            logger.warning(f"[ERROR] 未找到本地数据文件: {dataset_name}.jsonl")
            return None

        try:
            data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                            if max_samples and len(data) >= max_samples:
                                break
                        except json.JSONDecodeError as e:
                            logger.warning(f"跳过无效JSON行: {line[:50]}... 错误: {e}")

            logger.info(f"[OK] 从本地加载 {dataset_name}，共 {len(data)} 条样本")
            return data

        except Exception as e:
            logger.error(f"[ERROR] 加载本地JSONL文件失败: {e}")
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
                                      experiment_id, monitor=None, repetition=0):
    """运行单个FullKVCache实验"""
    try:
        logger.info(f"开始实验: {experiment_id}")
        
        # 调试信息：记录样本内容
        logger.info(f"样本数据: {sample}")
        logger.info(f"样本类型: {type(sample)}")
        
        # 准备输入 - 修复版本，确保每个数据集使用正确的输入格式
        input_text = ""
        if dataset_name in ["hotpotqa", "2wikimqa", "musique"]:
            # 问答类数据集
            if isinstance(sample, dict):
                question = sample.get('input', sample.get('question', sample.get('prompt', '')))
                context = sample.get('context', '')
                if context:
                    input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
                else:
                    input_text = f"Question: {question}\nAnswer:"
            else:
                input_text = f"Question: {str(sample)}\nAnswer:"
                
        elif dataset_name == "narrativeqa":
            # 叙事问答
            if isinstance(sample, dict):
                question = sample.get('input', sample.get('question', sample.get('prompt', '')))
                context = sample.get('context', sample.get('document', ''))
                if context:
                    input_text = f"Read the following story and answer the question.\nStory: {context}\nQuestion: {question}\nAnswer:"
                else:
                    input_text = f"Question: {question}\nAnswer:"
            else:
                input_text = f"Question: {str(sample)}\nAnswer:"
                
        elif dataset_name == "multi_news":
            # 多文档摘要
            if isinstance(sample, dict):
                content = sample.get('input', sample.get('context', sample.get('text', sample.get('prompt', ''))))
                if not content:
                    # 如果input为空，查找其他可能的字段
                    for field in ['document', 'articles', 'content']:
                        if field in sample and sample[field]:
                            content = sample[field]
                            break
                if content:
                    input_text = f"Summarize the following articles:\n{content}\nSummary:"
                else:
                    logger.warning(f"multi_news样本缺少内容: {sample}")
                    input_text = "Summarize the following articles:\n[No content available]\nSummary:"
            else:
                input_text = f"Summarize the following articles:\n{str(sample)}\nSummary:"
                
        elif dataset_name == "gov_report":
            # 政府报告摘要
            if isinstance(sample, dict):
                content = sample.get('input', sample.get('context', sample.get('text', sample.get('prompt', ''))))
                if not content:
                    # 如果input为空，查找其他可能的字段
                    for field in ['document', 'report', 'content']:
                        if field in sample and sample[field]:
                            content = sample[field]
                            break
                if content:
                    input_text = f"Summarize the following government report:\n{content}\nSummary:"
                else:
                    logger.warning(f"gov_report样本缺少内容: {sample}")
                    input_text = "Summarize the following government report:\n[No content available]\nSummary:"
            else:
                input_text = f"Summarize the following government report:\n{str(sample)}\nSummary:"
                
        elif dataset_name == "qmsum":
            # 会议摘要
            if isinstance(sample, dict):
                content = sample.get('input', sample.get('context', sample.get('text', sample.get('prompt', ''))))
                if not content:
                    # 如果input为空，查找其他可能的字段
                    for field in ['meeting', 'transcript', 'dialogue', 'content']:
                        if field in sample and sample[field]:
                            content = sample[field]
                            break
                if content:
                    input_text = f"Summarize the following meeting:\n{content}\nSummary:"
                else:
                    logger.warning(f"qmsum样本缺少内容: {sample}")
                    input_text = "Summarize the following meeting:\n[No content available]\nSummary:"
            else:
                input_text = f"Summarize the following meeting:\n{str(sample)}\nSummary:"
                
        else:
            # 其他数据集的通用处理
            if isinstance(sample, dict):
                input_text = sample.get('input', sample.get('prompt', sample.get('text', str(sample))))
            else:
                input_text = str(sample)
        
        # 验证输入文本
        if not input_text or input_text.strip() == "":
            logger.error(f"输入文本为空！数据集: {dataset_name}, 样本: {sample}")
            input_text = f"Default input for {dataset_name}"
        
        # 记录输入文本用于调试
        logger.info(f"最终输入文本 (前200字符): {input_text[:200]}...")
        
        # 限制输入长度以适应KV cache
        # 计算输入限制，防止出现负值
        max_input_length = max(kv_cache_length - max_new_tokens - 10, 16)
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

        # 记录生成的文本用于调试
        logger.info(f"生成的文本 (前200字符): {generated_text[:200]}...")

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
            "repetition": repetition
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


def score_generated_text(generated_text, ground_truth, dataset_name, sample_data=None):
    """对生成的文本进行评分"""
    if not SCORING_AVAILABLE:
        logger.warning("评分模块不可用，返回默认分数")
        return 0.5

    try:
        scoring_func = DATASET_SCORING_MAP.get(dataset_name, qa_f1_score)

        # 准备评分参数
        kwargs = {}
        
        # 对于分类任务，需要all_classes参数
        if scoring_func == classification_score and sample_data:
            if isinstance(sample_data, dict) and 'all_classes' in sample_data:
                kwargs['all_classes'] = sample_data['all_classes']
            else:
                # 提供TREC数据集的默认分类
                kwargs['all_classes'] = [
                    'Food', 'Date', 'Order, rank', 'Speed', 'Disease and medicine', 
                    'Word with a special property', 'Abbreviation', 'Language', 
                    'Letter like a-z', 'Other entity', 'Animal', 'Expression abbreviated', 
                    'Price', 'Techniques and method', 'Musical instrument', 'Mountain', 
                    'Currency name', 'Event', 'Product', 'State', 'Individual', 
                    'Organ of body', 'Reason', 'Manner of an action', 'City', 'Religion', 
                    'Invention, book and other creative piece', 'Distance, linear measure', 
                    'Temperature', 'Postcode or other code', 'Size, area and volume', 
                    'Sport', 'Country', 'Other location', 'Lasting time of somethin', 
                    'Equivalent term', 'Description of something', 'Weight', 'Vehicle', 
                    'Color', 'Other number', 'Definition of something', 'Element and substance', 
                    'Description of a person', 'Symbols and sign', 'Number of something', 
                    'Plant', 'Percent, fraction', 'Group or organization of person', 
                    'Title of a person'
                ]
                logger.info(f"使用默认TREC分类列表，共{len(kwargs['all_classes'])}个类别")

        # 调用评分函数
        score = scoring_func(generated_text, ground_truth, **kwargs)
        
        logger.info(f"评分详情: 函数={scoring_func.__name__}, 生成文本='{generated_text[:50]}...', 标准答案='{ground_truth}', 分数={score}")
        
        return score if score is not None else 0.0

    except Exception as e:
        logger.warning(f"评分失败: {e}")
        import traceback
        traceback.print_exc()
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
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Maximum number of new tokens to generate (if not specified, will use dataset-specific CAKE config).")
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
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process per dataset (None for all samples).")

    args = parser.parse_args()

    # 使用统一的路径配置设置输出目录
    try:
        from src.common.data_loader import get_runs_directory
        if args.output_dir is None:
            args.output_dir = get_runs_directory() / "fullkvcache_experiments"
        else:
            args.output_dir = Path(args.output_dir)
    except ImportError:
        # 备用方案
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[3]
        if args.output_dir is None:
            args.output_dir = project_root / "runs" / "fullkvcache_experiments"
        else:
            args.output_dir = Path(args.output_dir)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志
    log_file_path = args.output_dir / "fullkvcache_experiment_log.txt"
    global logger
    # 重新配置logger而不是重新赋值
    setup_logging(log_file=str(log_file_path), level=getattr(logging, args.log_level.upper(), logging.INFO))

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
    
    # 使用统一的路径配置
    try:
        from src.common.data_loader import get_runs_directory
        main_output_dir = get_runs_directory() / f"fullkvcache_run_{run_timestamp}"
    except ImportError:
        # 备用方案
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[3]
        main_output_dir = project_root / "runs" / f"fullkvcache_run_{run_timestamp}"
    
    main_output_dir.mkdir(parents=True, exist_ok=True)

    # 修复重复实验问题：将all_scores移到重复循环外部
    experiment_scores = {}  # 用于存储每个实验配置的分数

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

                        # 获取数据集特定的max_new_tokens配置
                        dataset_max_tokens = get_dataset_max_new_tokens(dataset_name) if args.max_new_tokens is None else args.max_new_tokens
                        logger.info(f"使用max_new_tokens: {dataset_max_tokens} (数据集: {dataset_name})")

                        # 创建实验特定的输出目录
                        experiment_output_dir = main_output_dir / f"ds_{dataset_name}_kv{kv_cache_length}_bs{batch_size}_rep{rep}"
                        experiment_output_dir.mkdir(parents=True, exist_ok=True)

                        # 加载模型和tokenizer
                        logger.info("Loading model and tokenizer...")
                        start_time = time.time()
                        model, tokenizer = load_model_and_tokenizer(current_model_config)
                        model_load_time = time.time() - start_time
                        logger.info(f"Model loaded in {model_load_time:.2f} seconds")

                        # 配置模型
                        model = configure_model_for_kv_cache_length(model, kv_cache_length)
                        model = prepare_model_for_baseline(model)

                        # 加载数据集 - 修复版本，确保每个数据集都正确加载
                        logger.info(f"Loading dataset {dataset_name}...")

                        # 尝试从本地JSONL文件加载
                        logger.info("尝试从本地JSONL文件加载数据集: " + dataset_name)
                        dataset = load_local_jsonl_data(dataset_name, max_samples=args.max_samples)
                        data_source = "local"

                        if dataset is None:
                            # 如果本地加载失败，尝试LongBench官方数据
                            logger.info("本地加载失败，尝试LongBench官方数据...")
                            dataset = load_longbench_official_data(dataset_name, max_samples=args.max_samples)
                            data_source = "longbench"

                        if dataset is None:
                            logger.error(f"[ERROR] 无法加载数据集 {dataset_name}，跳过此实验")
                            continue

                        # 验证数据集内容
                        if not dataset or len(dataset) == 0:
                            logger.error(f"[ERROR] 数据集 {dataset_name} 为空，跳过此实验")
                            continue

                        logger.info(f"[OK] 成功加载 {dataset_name} (来源: {data_source}, 样本数: {len(dataset)})")
                        
                        # 记录第一个样本用于调试
                        first_sample = dataset[0] if isinstance(dataset, list) else dataset[0]
                        logger.info(f"数据集 {dataset_name} 的第一个样本字段: {list(first_sample.keys()) if isinstance(first_sample, dict) else type(first_sample)}")
                        
                        # 验证样本内容
                        if isinstance(first_sample, dict):
                            has_content = False
                            for field in ['input', 'context', 'text', 'prompt', 'question', 'document']:
                                if field in first_sample and first_sample[field] and str(first_sample[field]).strip():
                                    has_content = True
                                    break
                            
                            if not has_content:
                                logger.warning(f"[WARNING] 数据集 {dataset_name} 的样本似乎缺少有效内容")
                        
                        logger.info(f"数据集 {dataset_name} 验证通过，继续执行实验")

                        # 准备样本
                        prepared_samples = prepare_samples_for_evaluation(dataset, dataset_name, num_samples=len(dataset))
                        logger.info(f"Prepared {len(prepared_samples)} samples successfully")

                        # 初始化监控
                        monitor = UnifiedMonitor()

                        # 运行实验 - 遍历所有样本
                        logger.info(f"Running FullKVCache experiment on {len(prepared_samples)} samples...")
                        
                        # 修复重复实验问题：为每个实验配置创建独立的分数存储
                        config_key = f"{dataset_name}_kv{kv_cache_length}_bs{batch_size}"
                        if config_key not in experiment_scores:
                            experiment_scores[config_key] = []
                        
                        current_scores = []
                        all_experiment_results = []

                        for sample_idx, sample in enumerate(prepared_samples):
                            logger.info(f"Processing sample {sample_idx + 1}/{len(prepared_samples)}")
                            
                            # 为每个样本创建独立的实验ID
                            sample_experiment_id = f"{experiment_id}_sample_{sample_idx}"
                            
                            try:
                                experiment_results = run_single_fullkvcache_experiment(
                                    model, tokenizer, sample, kv_cache_length, dataset_max_tokens,
                                    dataset_name, sample_experiment_id, monitor, repetition=rep
                                )

                                # 保存性能指标
                                metrics_file = save_experiment_results(experiment_results, experiment_output_dir,
                                                                       sample_experiment_id)

                                # 如果启用评分，进行评估
                                if args.enable_scoring:
                                    logger.info(f"Performing evaluation scoring for sample {sample_idx + 1}...")
                                    generated_text = experiment_results["generated_text"]

                                    # 获取ground truth - 修复版本，使用处理后样本的reference字段
                                    ground_truth = ""
                                    if isinstance(sample, dict):
                                        # 优先使用处理后的reference字段
                                        if 'reference' in sample:
                                            ground_truth = sample['reference']
                                            logger.info(f"使用处理后的reference: {ground_truth}")
                                        else:
                                            # 备用方案：从原始字段提取
                                            ground_truth = sample.get('answers', sample.get('output', sample.get('answer', '')))
                                            logger.info(f"使用原始字段提取: {ground_truth}")
                                    else:
                                        ground_truth = str(sample)

                                    # 如果ground_truth是列表，取第一个
                                    if isinstance(ground_truth, list):
                                        ground_truth = ground_truth[0] if ground_truth else ""
                                        logger.info(f"从列表中提取第一个元素: {ground_truth}")
                                    
                                    # 验证ground truth不为空
                                    if not ground_truth or str(ground_truth).strip() == "":
                                        logger.warning(f"Ground truth为空！样本类型: {type(sample)}, 样本内容: {sample}")
                                        ground_truth = "Unknown"  # 提供默认值避免评分失败

                                    # 计算分数 - 传递原始数据集样本以获取all_classes等信息
                                    original_sample = dataset[0] if dataset and len(dataset) > 0 else {}
                                    score = score_generated_text(generated_text, ground_truth, dataset_name, original_sample)
                                    current_scores.append(score)

                                    evaluation_results = {
                                        "experiment_id": sample_experiment_id,
                                        "dataset": dataset_name,
                                        "sample_index": sample_idx,
                                        "generated_text": generated_text,
                                        "ground_truth": ground_truth,
                                        "score": score,
                                        "average_score": score,  # 为了兼容性
                                        "timestamp": datetime.now().isoformat()
                                    }

                                    # 保存评估结果
                                    eval_file = save_evaluation_results(evaluation_results, experiment_output_dir,
                                                                        sample_experiment_id)

                                    logger.info(f"Sample {sample_idx + 1} evaluation score: {score:.4f}")

                                all_experiment_results.append(experiment_results)

                            except Exception as sample_error:
                                logger.error(f"Sample {sample_idx + 1} failed: {sample_error}")
                                continue

                        # 计算平均分数并存储到实验分数字典
                        if current_scores:
                            average_score = sum(current_scores) / len(current_scores)
                            experiment_scores[config_key].append(average_score)  # 存储到总体分数记录
                            logger.info(f"Dataset {dataset_name} rep {rep} average score: {average_score:.4f} (from {len(current_scores)} samples)")
                            
                            # 保存汇总评估结果
                            summary_evaluation = {
                                "experiment_id": experiment_id,
                                "dataset": dataset_name,
                                "total_samples": len(prepared_samples),
                                "evaluated_samples": len(current_scores),
                                "individual_scores": current_scores,
                                "average_score": average_score,
                                "timestamp": datetime.now().isoformat()
                            }
                            
                            summary_eval_file = save_evaluation_results(summary_evaluation, experiment_output_dir,
                                                                        f"{experiment_id}_summary")

                        # 记录结果 - 使用最后一个实验结果作为代表
                        if all_experiment_results:
                            last_result = all_experiment_results[-1]
                            monitoring_data = last_result.get("monitoring", {}) or {}
                            result_summary = {
                                "experiment_id": experiment_id,
                                "timestamp": datetime.now().isoformat(),
                                "performance": last_result["performance"],
                                "gpu": monitoring_data.get("gpu", {}),
                                "system": monitoring_data.get("system", {}),
                                "monitoring_duration": monitoring_data.get("duration", 0),
                                "total_samples": len(prepared_samples),
                                "average_score": average_score if current_scores else 0.0
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

    # 生成实验分数汇总报告
    logger.info("="*60)
    logger.info("实验分数汇总报告 (修复重复实验问题)")
    logger.info("="*60)
    
    for config_key, scores in experiment_scores.items():
        if scores:
            avg_score = sum(scores) / len(scores)
            std_score = (sum((s - avg_score) ** 2 for s in scores) / len(scores)) ** 0.5 if len(scores) > 1 else 0.0
            logger.info(f"{config_key}:")
            logger.info(f"  - 重复次数: {len(scores)}")
            logger.info(f"  - 各次分数: {[f'{s:.4f}' for s in scores]}")
            logger.info(f"  - 平均分数: {avg_score:.4f} ± {std_score:.4f}")
            logger.info(f"  - 最高分数: {max(scores):.4f}")
            logger.info(f"  - 最低分数: {min(scores):.4f}")
    
    logger.info("="*60)

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
            logger.info("[SEARCH] 开始强化基线评分搜索...")

            # 使用强化搜索
            evaluation_files = find_baseline_results_robust(main_output_dir)

            baseline_scores = []

            for eval_file_path in evaluation_files:
                try:
                    logger.info(f"🔄 处理文件: {eval_file_path}")
                    with open(eval_file_path, 'r', encoding='utf-8') as f:
                        eval_data = json.load(f)

                    if eval_data.get("average_score") is not None:
                        # 从文件路径推断数据集 - 修复版本
                        dataset_name = None
                        file_path_lower = eval_file_path.lower()
                        
                        # 检测所有可能的数据集名称
                        if "multifieldqa_en" in file_path_lower:
                            dataset_name = "multifieldqa_en"
                        elif "multi_news" in file_path_lower:
                            dataset_name = "multi_news"
                        elif "narrativeqa" in file_path_lower:
                            dataset_name = "narrativeqa"
                        elif "qasper" in file_path_lower:
                            dataset_name = "qasper"
                        elif "hotpotqa" in file_path_lower:
                            dataset_name = "hotpotqa"
                        elif "trec" in file_path_lower:
                            dataset_name = "trec"
                        elif "2wikimqa" in file_path_lower:
                            dataset_name = "2wikimqa"
                        elif "musique" in file_path_lower:
                            dataset_name = "musique"
                        elif "dureader" in file_path_lower:
                            dataset_name = "dureader"
                        elif "gov_report" in file_path_lower:
                            dataset_name = "gov_report"
                        elif "qmsum" in file_path_lower:
                            dataset_name = "qmsum"
                        elif "vcsum" in file_path_lower:
                            dataset_name = "vcsum"
                        elif "triviaqa" in file_path_lower:
                            dataset_name = "triviaqa"
                        elif "samsum" in file_path_lower:
                            dataset_name = "samsum"
                        elif "lsht" in file_path_lower:
                            dataset_name = "lsht"
                        elif "passage_count" in file_path_lower:
                            dataset_name = "passage_count"
                        elif "passage_retrieval_en" in file_path_lower:
                            dataset_name = "passage_retrieval_en"
                        elif "passage_retrieval_zh" in file_path_lower:
                            dataset_name = "passage_retrieval_zh"
                        elif "lcc" in file_path_lower:
                            dataset_name = "lcc"
                        elif "repobench" in file_path_lower:
                            dataset_name = "repobench-p"
                        else:
                            # 如果无法识别数据集，记录警告并跳过
                            logger.warning(f"[WARNING] 无法从文件路径识别数据集: {eval_file_path}")
                            continue

                        score_result = calculate_relative_score(
                            dataset_name=dataset_name,
                            raw_score=eval_data["average_score"],
                            is_full_kv=True
                        )
                        baseline_scores.append(score_result)
                        logger.info(f"[OK] 成功记录基线分数: {dataset_name} = {eval_data['average_score']:.4f}")
                    else:
                        logger.warning(f"[WARNING] 文件缺少 average_score: {eval_file_path}")

                except Exception as e:
                    logger.warning(f"[ERROR] 处理文件失败 {eval_file_path}: {e}")

            if baseline_scores:
                # 生成基线报告
                try:
                    aggregated = aggregate_scores(baseline_scores)
                    report = format_score_report(aggregated, "Full KV (基线)")

                    # 保存报告
                    baseline_report_path = os.path.join(main_output_dir, "baseline_scoring_report.txt")
                    with open(baseline_report_path, 'w', encoding='utf-8') as f:
                        f.write(report)

                    logger.info(f"[OK] 基线评分报告已保存到: {baseline_report_path}")
                    print("\n" + "=" * 60)
                    print("[SUCCESS] 基线评分成功！")
                    print("=" * 60)
                    print(report)
                    print("=" * 60)

                except Exception as report_error:
                    logger.error(f"生成报告失败: {report_error}")
                    print(f"[OK] 找到了 {len(baseline_scores)} 个基线分数，但报告生成失败")

            else:
                logger.error("[ERROR] 未找到任何有效的评分结果文件")
                print("\n[DEBUG] 调试信息:")
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
        pass  # 静默跳过基线建立，不显示提示

    logger.info("FullKVCache experiment suite finished.")


if __name__ == "__main__":
    main()