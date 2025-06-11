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

def prepare_samples_for_evaluation(dataset, dataset_name, num_samples=100, random_seed=42):
    """
    准备用于评估的样本
    
    Args:
        dataset: 加载的数据集
        dataset_name: 数据集名称，用于确定处理方式
        num_samples: 要准备的样本数量
        random_seed: 随机种子，用于可重复性
        
    Returns:
        samples: 处理后的样本列表，每个样本是一个字典，包含输入提示和参考答案
    """
    random.seed(random_seed)
    logger.info(f"Preparing {num_samples} samples from {dataset_name}")
    
    # 如果数据集样本数少于请求的样本数，使用所有样本
    if len(dataset) < num_samples:
        num_samples = len(dataset)
        logger.warning(f"Dataset contains only {num_samples} samples")
    
    # 随机选择样本索引
    indices = random.sample(range(len(dataset)), num_samples)
    samples = []
    
    for idx in indices:
        sample = dataset[idx]
        processed_sample = {}
        
        # 根据不同数据集格式处理样本
        # 英文数据集
        if dataset_name == "mmlu":
            # MMLU格式处理
            processed_sample["prompt"] = f"Question: {sample['question']}\nChoices:\nA. {sample['choices'][0]}\nB. {sample['choices'][1]}\nC. {sample['choices'][2]}\nD. {sample['choices'][3]}\nAnswer:"
            processed_sample["reference"] = sample["answer"]
        
        elif dataset_name == "hellaswag":
            # HellaSwag格式处理
            processed_sample["prompt"] = f"Context: {sample['ctx']}\nComplete this with the most appropriate ending:"
            processed_sample["reference"] = sample["endings"][sample["label"]]
        
        elif dataset_name == "longbench":
            # LongBench格式处理
            if "input" in sample and "answer" in sample:
                processed_sample["prompt"] = sample["input"]
                processed_sample["reference"] = sample["answer"]
            elif "question" in sample and "answers" in sample:
                processed_sample["prompt"] = sample["question"]
                processed_sample["reference"] = sample["answers"][0] if isinstance(sample["answers"], list) else sample["answers"]
            else:
                # 处理其他格式的LongBench样本
                for key in sample:
                    if isinstance(sample[key], str) and len(sample[key]) > 20:
                        processed_sample["prompt"] = sample[key]
                        break
                processed_sample["reference"] = ""
        
        elif dataset_name == "squad2":
            # SQuAD 2.0格式处理
            if "question" in sample and "context" in sample:
                processed_sample["prompt"] = f"Context: {sample['context']}\nQuestion: {sample['question']}\nAnswer:"
                processed_sample["reference"] = sample.get("answers", {}).get("text", [""])[0] if "answers" in sample else ""
            else:
                # 兼容不同版本的SQuAD
                processed_sample["prompt"] = f"Question: {sample.get('question', '')}\nContext: {sample.get('context', '')}\nAnswer:"
                answers = sample.get("answers", [])
                processed_sample["reference"] = answers[0]["text"] if answers else ""
        
        # 中文数据集
        elif dataset_name == "cluewsc2020":
            # CLUEWSC2020格式处理
            if "text" in sample and "target" in sample:
                processed_sample["prompt"] = f"句子: {sample['text']}\n指代问题: {sample.get('target', {}).get('span1_text', '')}和{sample.get('target', {}).get('span2_text', '')}是否指代相同实体？"
                processed_sample["reference"] = "是" if sample.get("label") == 1 else "否"
            else:
                processed_sample["prompt"] = sample.get("text", "")
                processed_sample["reference"] = "是" if sample.get("label") == 1 else "否"
        
        elif dataset_name == "ceval":
            # C-Eval格式处理
            processed_sample["prompt"] = f"题目: {sample['question']}\n选项:\nA. {sample['choices'][0]}\nB. {sample['choices'][1]}\nC. {sample['choices'][2]}\nD. {sample['choices'][3]}\n答案:"
            processed_sample["reference"] = sample["answer"]
        
        elif dataset_name == "race":
            # RACE格式处理
            if "article" in sample and "question" in sample and "options" in sample:
                processed_sample["prompt"] = f"文章: {sample['article']}\n问题: {sample['question']}\n选项:\n"
                for i, option in enumerate(sample['options']):
                    processed_sample["prompt"] += f"{chr(65+i)}. {option}\n"
                processed_sample["prompt"] += "答案:"
                
                answers = {"0": "A", "1": "B", "2": "C", "3": "D"}
                processed_sample["reference"] = answers.get(str(sample.get("answer")), "")
            else:
                processed_sample["prompt"] = sample.get("question", "")
                processed_sample["reference"] = ""
        
        elif dataset_name == "openeval":
            # OpenEval格式处理
            if "question" in sample and "answer" in sample:
                processed_sample["prompt"] = sample["question"]
                processed_sample["reference"] = sample["answer"]
            else:
                processed_sample["prompt"] = next((sample[k] for k in sample if isinstance(sample[k], str) and len(sample[k]) > 20), "")
                processed_sample["reference"] = ""
        
        # 默认处理方式
        else:
            if "text" in sample:
                processed_sample["prompt"] = sample["text"]
            elif "question" in sample:
                processed_sample["prompt"] = sample["question"]
            else:
                # 使用第一个非特殊字段作为提示
                for key, value in sample.items():
                    if key != "label" and key != "idx" and isinstance(value, str):
                        processed_sample["prompt"] = value
                        break
            
            # 设置参考答案
            if "label" in sample:
                processed_sample["reference"] = str(sample["label"])
            elif "answer" in sample:
                processed_sample["reference"] = sample["answer"]
            else:
                processed_sample["reference"] = ""
        
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