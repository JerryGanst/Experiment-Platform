#!/usr/bin/env python3
"""
优化版FullKVCache实验主程序
按照用户建议修复的版本：
- 模型只加载一次，循环复用
- 修复HotpotQA数据集加载问题
- 优化GPU内存管理
"""

import os
import sys
import json
import argparse
import logging
import gc
from datetime import datetime
from pathlib import Path
import torch

# 获取项目根目录路径配置
current_file_path = os.path.abspath(__file__)
# 从src/cake_runner/向上两级到达项目根目录
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)
from hace_core.models.model_loader import load_model_and_tokenizer, prepare_model_for_baseline
from hace_core.data.dataset_loader import load_dataset_split, prepare_samples_for_evaluation
from hace_core import config

# 导入评分功能（如果可用）
try:
    # 添加当前目录到路径以导入eval_utils
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    from eval_utils import calculate_relative_score, aggregate_scores, format_score_report, score_dataset
    BASELINE_SCORING_AVAILABLE = True
    print("[OK] 基线评分工具加载成功")
except ImportError as e:
    print(f"[WARNING] 基线评分工具不可用: {e}")
    BASELINE_SCORING_AVAILABLE = False

# 全局配置
EXPERIMENT_CONFIG = config.EXPERIMENT_CONFIG
DATASET_CONFIG = config.DATASET_CONFIG

logger = logging.getLogger(__name__)

def main():
    """优化版主函数：模型只加载一次"""
    
    # 参数解析
    parser = argparse.ArgumentParser(description="优化版FullKVCache实验")
    parser.add_argument("--model_name", type=str, default=EXPERIMENT_CONFIG["model_name_or_path"])
    parser.add_argument("--datasets", type=str, default="hotpotqa,multi_news")
    parser.add_argument("--kv_cache_lengths", type=str, default="128,256,512,1024,2048")
    parser.add_argument("--enable_scoring", action="store_true")
    parser.add_argument("--is_baseline_run", action="store_true")
    
    args = parser.parse_args()
    
    # 解析参数
    datasets_list = [d.strip() for d in args.datasets.split(',') if d.strip()]
    kv_lengths_list = [int(kv.strip()) for kv in args.kv_cache_lengths.split(',') if kv.strip()]
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger.info("🚀 启动优化版FullKVCache实验")
    
    # ===== 关键优化1：只加载一次模型 =====
    logger.info("📚 加载模型（仅一次）...")
    model_config = {
        "model_name_or_path": args.model_name,
        "precision": EXPERIMENT_CONFIG["precision"]
    }
    
    try:
        model, tokenizer = load_model_and_tokenizer(model_config)
        model = prepare_model_for_baseline(model)
        logger.info(f"✅ 模型加载成功: {args.model_name}")
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        return
    
    all_results = []
    
    try:
        # ===== 核心循环：复用模型，只配置KV长度 =====
        for dataset_name in datasets_list:
            logger.info(f"📊 处理数据集: {dataset_name}")
            
            # 获取数据集配置
            dataset_options = DATASET_CONFIG.get("available_datasets", {}).get(dataset_name)
            if not dataset_options:
                logger.error(f"数据集配置未找到: {dataset_name}")
                continue
            
            for kv_len in kv_lengths_list:
                logger.info(f"⚙️ 配置KV长度: {kv_len}")
                
                # ===== 关键优化2：不重新加载模型，只配置KV长度 =====
                with torch.no_grad():
                    if hasattr(model.config, 'max_position_embeddings'):
                        model.config.max_position_embeddings = kv_len
                    model.config.use_cache = True
                
                try:
                    # 运行单次实验
                    result = run_single_experiment(
                        model, tokenizer, dataset_name, 
                        dataset_options, kv_len, args
                    )
                    all_results.append(result)
                    logger.info(f"✅ 实验完成: {dataset_name}_kv{kv_len}")
                    
                except Exception as e:
                    logger.error(f"❌ 实验失败: {dataset_name}_kv{kv_len}, 错误: {e}")
                    all_results.append({
                        "dataset": dataset_name,
                        "kv_length": kv_len,
                        "error": str(e),
                        "status": "failed"
                    })
    
    except KeyboardInterrupt:
        logger.info("⚠️ 实验被用户中断")
    
    finally:
        # ===== 关键优化3：最后才清理模型 =====
        logger.info("🧹 清理模型资源...")
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("✅ 资源清理完成")
    
    # 处理结果
    process_results(all_results, args)
    logger.info("🎉 优化版FullKVCache实验完成！")

def run_single_experiment(model, tokenizer, dataset_name, dataset_options, kv_length, args):
    """运行单次实验（模型已加载）"""
    
    experiment_id = f"optimized_{dataset_name}_kv{kv_length}_{datetime.now().strftime('%H%M%S')}"
    logger.info(f"开始实验: {experiment_id}")
    
    # 加载数据集
    try:
        dataset, dataset_source = load_dataset_with_fallback(dataset_name, dataset_options)
        logger.info(f"数据集加载成功: {dataset_name} (来源: {dataset_source})")
    except Exception as e:
        raise Exception(f"数据集加载失败: {e}")
    
    # 准备样本
    num_samples = min(10, len(dataset))  # 快速测试，只用10个样本
    samples = prepare_samples_for_evaluation(dataset, dataset_name, num_samples=num_samples)
    
    # 推理
    predictions = []
    references = []
    
    logger.info(f"开始推理，样本数: {len(samples)}")
    
    for i, sample in enumerate(samples):
        try:
            # 提取输入和参考答案
            prompt = sample.get("prompt", sample.get("input", ""))
            reference = extract_ground_truth(sample, dataset_source)
            
            # 推理
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=kv_length)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # 解码
            generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            predictions.append(generated_text.strip())
            references.append(reference.strip() if reference else "")
            
        except Exception as e:
            logger.warning(f"样本 {i} 推理失败: {e}")
            predictions.append("")
            references.append("")
    
    # 评分
    result = {
        "experiment_id": experiment_id,
        "dataset": dataset_name,
        "kv_length": kv_length,
        "num_samples": len(predictions)
    }
    
    if BASELINE_SCORING_AVAILABLE and predictions and references:
        try:
            raw_score = score_dataset(dataset_name, predictions, references)
            result["raw_score"] = raw_score
            logger.info(f"原始评分: {raw_score:.4f}")
            
            # 计算相对评分（如果是基线运行）
            if args.is_baseline_run:
                score_result = calculate_relative_score(dataset_name, raw_score, is_full_kv=True)
                result.update(score_result)
                logger.info(f"基线分数已记录: {dataset_name} = {raw_score:.4f}")
            
        except Exception as e:
            logger.warning(f"评分失败: {e}")
    
    return result

def load_dataset_with_fallback(dataset_name, dataset_options, split="test"):
    """加载数据集，优先HF，失败时本地JSONL"""
    try:
        # 尝试Hugging Face
        logger.info(f"尝试从Hugging Face加载: {dataset_name}")
        dataset = load_dataset_split(dataset_options, split=split)
        return dataset, "huggingface"
    except Exception as hf_error:
        logger.warning(f"HF加载失败: {hf_error}")
        try:
            # 回退到本地JSONL
            data = load_local_jsonl(dataset_name)
            return create_simple_dataset(data), "local"
        except Exception as local_error:
            raise Exception(f"所有数据源加载失败. HF: {hf_error}, Local: {local_error}")

def load_local_jsonl(dataset_name, data_dir="./data"):
    """从本地JSONL加载数据"""
    file_path = Path(data_dir) / f"{dataset_name}.jsonl"
    if not file_path.exists():
        raise FileNotFoundError(f"本地文件不存在: {file_path}")
    
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    logger.info(f"从本地加载 {dataset_name}: {len(data)} 条样本")
    return data

def create_simple_dataset(data):
    """创建简单数据集对象"""
    class SimpleDataset:
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]
        def __iter__(self):
            return iter(self.data)
    return SimpleDataset(data)

def extract_ground_truth(sample, dataset_source):
    """提取参考答案"""
    if "answers" in sample:
        answers = sample["answers"]
        if isinstance(answers, list) and answers:
            return answers[0]
        elif isinstance(answers, str):
            return answers
    elif "answer" in sample:
        return sample["answer"]
    return ""

def process_results(all_results, args):
    """处理实验结果"""
    if not all_results:
        logger.warning("没有实验结果")
        return
    
    # 保存结果
    output_file = f"optimized_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"结果已保存: {output_file}")
    
    # 生成基线报告（如果启用评分且是基线运行）
    if args.enable_scoring and args.is_baseline_run and BASELINE_SCORING_AVAILABLE:
        try:
            baseline_scores = [r for r in all_results if 'relative_score' in r]
            if baseline_scores:
                aggregated = aggregate_scores(baseline_scores)
                report = format_score_report(aggregated, "优化版Full KV基线")
                
                report_file = f"baseline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(report)
                
                logger.info(f"基线报告已保存: {report_file}")
                print(report)
        except Exception as e:
            logger.error(f"基线报告生成失败: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ 实验被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 实验出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 