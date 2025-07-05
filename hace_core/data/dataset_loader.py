"""
数据集加载与预处理模块
"""
import logging
import random
from datasets import load_dataset

logger = logging.getLogger(__name__)

def load_dataset_split(dataset_config, split="validation", trust_remote_code=False):
    """
    加载指定的数据集
    
    Args:
        dataset_config: 数据集配置字典
        split: 要加载的数据分割，默认为validation
        trust_remote_code: 是否信任远程代码，对于某些数据集如hellaswag需要设置为True
        
    Returns:
        dataset: 加载的数据集
    """
    path = dataset_config["path"]
    subset = dataset_config["subset"]
    
    logger.info(f"Loading dataset: {path} (subset: {subset}) - split: {split}")
    
    try:
        if subset:
            # 修复HotpotQA加载问题：使用name参数而不是subset
            dataset = load_dataset(path, name=subset, split=split, trust_remote_code=True)
        else:
            dataset = load_dataset(path, split=split, trust_remote_code=trust_remote_code)
        logger.info(f"Dataset loaded successfully with {len(dataset)} samples")
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def prepare_samples_for_evaluation(dataset, dataset_info, num_samples=100, random_seed=42):
    """
    准备用于评估的样本 - 支持多种调用方式

    Args:
        dataset: 加载的数据集
        dataset_info: 可以是数据集名称(str)或数据集配置(dict)
        num_samples: 要准备的样本数量
        random_seed: 随机种子，用于可重复性

    Returns:
        samples: 处理后的样本列表，每个样本是一个字典，包含输入提示和参考答案
    """
    # -------- 兼容旧/新签名 -------------------------------------------------
    # 允许dataset_info既可以是名称字符串，也可以是配置字典。后续逻辑统一使用dataset_name
    if isinstance(dataset_info, str):
        dataset_name = dataset_info
    elif isinstance(dataset_info, dict):
        # 从配置字典中提取名称；优先级：显式字段 -> 回退到键名猜测
        dataset_name = dataset_info.get("name") or dataset_info.get("dataset_name")
        # 如果仍然为空，尝试根据全局配置反向查找
        if not dataset_name:
            try:
                from hace_core import config as _global_cfg
                for _name, _cfg in _global_cfg.DATASET_CONFIG.get("available_datasets", {}).items():
                    if _cfg == dataset_info:
                        dataset_name = _name
                        break
            except Exception:
                pass
        # 如果依旧未知，则设置占位符，后续仅用于日志
        dataset_name = dataset_name or "unknown_dataset"
    else:
        raise TypeError(f"dataset_info 应为 str 或 dict，实际收到: {type(dataset_info)}")

    # -----------------------------------------------------------------------
    random.seed(random_seed)
    logger.info(f"Preparing {num_samples} samples from {dataset_name}")
    
    # 处理不同的数据格式
    if hasattr(dataset, '__len__'):
        # 标准dataset对象
        dataset_items = dataset
    elif isinstance(dataset, list):
        # 直接的列表格式
        dataset_items = dataset
    else:
        logger.error(f"未知的数据集格式: {type(dataset)}")
        return []
    
    # 如果数据集样本数少于请求的样本数，使用所有样本
    num_available = len(dataset_items)
    if num_available < num_samples:
        num_samples = num_available
        logger.warning(f"Dataset contains only {num_samples} samples")
    
    # 随机选择样本索引
    indices = random.sample(range(num_available), num_samples)
    samples = []
    
    for idx in indices:
        item = dataset_items[idx]
        processed_sample = {}
        
        # 特殊数据集优先处理
        if dataset_name == "multi_news":
            # multi_news特殊处理 - input为空，使用context
            if "context" in item and "answers" in item:
                processed_sample["prompt"] = f"Summarize the following articles:\n{item['context']}\nSummary:"
                if isinstance(item["answers"], list):
                    processed_sample["reference"] = item["answers"][0] if item["answers"] else ""
                else:
                    processed_sample["reference"] = item["answers"]
            else:
                processed_sample["prompt"] = ""
                processed_sample["reference"] = ""
                
        elif dataset_name == "gov_report":
            # gov_report特殊处理 - input为空，使用context
            if "context" in item and "answers" in item:
                processed_sample["prompt"] = f"Summarize the following government report:\n{item['context']}\nSummary:"
                if isinstance(item["answers"], list):
                    processed_sample["reference"] = item["answers"][0] if item["answers"] else ""
                else:
                    processed_sample["reference"] = item["answers"]
            else:
                processed_sample["prompt"] = ""
                processed_sample["reference"] = ""
                
        elif dataset_name == "qmsum":
            # qmsum特殊处理 - 使用input作为问题，context作为会议内容
            if "input" in item and item["input"].strip() and "context" in item:
                query = item["input"].strip()
                processed_sample["prompt"] = f"Context: {item['context']}\nQuestion: {query}\nAnswer:"
            elif "context" in item:
                processed_sample["prompt"] = f"Summarize the following meeting:\n{item['context']}\nSummary:"
            else:
                processed_sample["prompt"] = ""
            
            if "answers" in item:
                if isinstance(item["answers"], list):
                    processed_sample["reference"] = item["answers"][0] if item["answers"] else ""
                else:
                    processed_sample["reference"] = item["answers"]
            else:
                processed_sample["reference"] = ""
        
        # LongBench标准格式优先处理
        elif "input" in item and "answers" in item and item["input"].strip():
            processed_sample["prompt"] = item["input"]
            
            # 处理答案 - LongBench的答案可能是列表
            if isinstance(item["answers"], list):
                processed_sample["reference"] = item["answers"][0] if item["answers"] else ""
            else:
                processed_sample["reference"] = item["answers"]
                
        elif "context" in item and "question" in item:
            # 问答格式
            processed_sample["prompt"] = f"Context: {item['context']}\nQuestion: {item['question']}\nAnswer:"
            if "answers" in item:
                answers = item["answers"]
                if isinstance(answers, list):
                    processed_sample["reference"] = answers[0] if answers else ""
                else:
                    processed_sample["reference"] = answers
            else:
                processed_sample["reference"] = item.get("answer", "")
                
        elif "dialogue" in item and "summary" in item:
            # 摘要格式（如samsum）
            processed_sample["prompt"] = f"Summarize the following dialogue:\n{item['dialogue']}\nSummary:"
            processed_sample["reference"] = item["summary"]
            
        elif "text" in item:
            # 通用文本格式
            processed_sample["prompt"] = item["text"]
            processed_sample["reference"] = item.get("summary", item.get("answer", ""))
            
        # 兼容旧格式处理
        elif dataset_name == "mmlu":
            # MMLU格式处理
            processed_sample["prompt"] = f"Question: {item['question']}\nChoices:\nA. {item['choices'][0]}\nB. {item['choices'][1]}\nC. {item['choices'][2]}\nD. {item['choices'][3]}\nAnswer:"
            processed_sample["reference"] = item["answer"]
        
        elif dataset_name == "hellaswag":
            # HellaSwag格式处理
            processed_sample["prompt"] = f"Context: {item['ctx']}\nComplete this with the most appropriate ending:"
            processed_sample["reference"] = item["endings"][item["label"]]
        
        elif dataset_name == "squad2":
            # SQuAD 2.0格式处理
            if "question" in item and "context" in item:
                processed_sample["prompt"] = f"Context: {item['context']}\nQuestion: {item['question']}\nAnswer:"
                processed_sample["reference"] = item.get("answers", {}).get("text", [""])[0] if "answers" in item else ""
            else:
                processed_sample["prompt"] = f"Question: {item.get('question', '')}\nContext: {item.get('context', '')}\nAnswer:"
                answers = item.get("answers", [])
                processed_sample["reference"] = answers[0]["text"] if answers else ""
        
        elif dataset_name == "ceval":
            # C-Eval格式处理
            processed_sample["prompt"] = f"题目: {item['question']}\n选项:\nA. {item['choices'][0]}\nB. {item['choices'][1]}\nC. {item['choices'][2]}\nD. {item['choices'][3]}\n答案:"
            processed_sample["reference"] = item["answer"]
        
        elif dataset_name == "race":
            # RACE格式处理
            if "article" in item and "question" in item and "options" in item:
                processed_sample["prompt"] = f"文章: {item['article']}\n问题: {item['question']}\n选项:\n"
                for i, option in enumerate(item['options']):
                    processed_sample["prompt"] += f"{chr(65+i)}. {option}\n"
                processed_sample["prompt"] += "答案:"
                
                answers = {"0": "A", "1": "B", "2": "C", "3": "D"}
                processed_sample["reference"] = answers.get(str(item.get("answer")), "")
            else:
                processed_sample["prompt"] = item.get("question", "")
                processed_sample["reference"] = ""
        
        else:
            # 未知格式 - 记录并使用备用方案
            logger.warning(f"未知的数据格式，数据集: {dataset_name}, 可用字段: {list(item.keys())}")
            # 使用第一个看起来像输入的字段
            for key in ["input", "question", "text", "context", "dialogue"]:
                if key in item:
                    processed_sample["prompt"] = str(item[key])
                    break
            else:
                processed_sample["prompt"] = str(item)
                
            # 尝试找到答案
            for key in ["answers", "answer", "summary", "output"]:
                if key in item:
                    ref = item[key]
                    if isinstance(ref, list):
                        processed_sample["reference"] = ref[0] if ref else ""
                    else:
                        processed_sample["reference"] = str(ref)
                    break
            else:
                processed_sample["reference"] = ""
        
        # 验证处理结果
        if not processed_sample.get("reference"):
            logger.warning(f"样本 {idx} 没有找到参考答案，数据集: {dataset_name}")
        
        # 确保都有内容
        if not processed_sample.get("prompt"):
            logger.warning(f"样本 {idx} 没有找到输入提示，数据集: {dataset_name}")
            processed_sample["prompt"] = ""
            
        samples.append(processed_sample)
    
    logger.info(f"Prepared {len(samples)} samples successfully")
    return samples

def prepare_batch(samples, tokenizer, batch_size, max_length=2048):
    """
    将样本处理成批次
    
    Args:
        samples: 样本列表
        tokenizer: 分词器
        batch_size: 批处理大小
        max_length: 最大序列长度
        
    Returns:
        batches: 包含输入ID和注意力掩码的字典
    """
    # 如果样本数小于批处理大小，复制样本以达到批处理大小
    if len(samples) < batch_size:
        samples_to_add = batch_size - len(samples)
        samples.extend(samples[:samples_to_add])
    
    # 选择批处理大小的样本
    batch_samples = samples[:batch_size]
    
    # 提取提示
    prompts = [sample["prompt"] for sample in batch_samples]
    
    # 对提示进行分词
    encodings = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "samples": batch_samples  # 保留原始样本，用于评估
    }

def get_dataset_info(dataset_name, language="english"):
    """
    获取数据集信息
    
    Args:
        dataset_name: 数据集名称
        language: 语言，'english'或'chinese'
        
    Returns:
        dict: 数据集信息
    """
    from .. import config
    DATASET_CONFIG = config.DATASET_CONFIG
    
    if language not in DATASET_CONFIG:
        raise ValueError(f"Unsupported language: {language}")
    
    if dataset_name not in DATASET_CONFIG[language]:
        raise ValueError(f"Unknown dataset: {dataset_name} for language {language}")
    
    dataset_info = DATASET_CONFIG[language][dataset_name].copy()
    dataset_info["name"] = dataset_name
    dataset_info["language"] = language
    
    return dataset_info

def get_available_datasets(language=None):
    """
    获取可用的数据集列表
    
    Args:
        language: 可选，指定语言
        
    Returns:
        dict: 按语言分组的可用数据集
    """
    from .. import config
    DATASET_CONFIG = config.DATASET_CONFIG
    
    if language:
        if language not in DATASET_CONFIG:
            raise ValueError(f"Unsupported language: {language}")
        return {language: list(DATASET_CONFIG[language].keys())}
    else:
        return {lang: list(datasets.keys()) for lang, datasets in DATASET_CONFIG.items()} 