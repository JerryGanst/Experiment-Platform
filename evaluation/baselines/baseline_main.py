# 基线实验主脚本
import sys
import os

# 获取项目根目录路径配置
current_file_path = os.path.abspath(__file__)
# 从src/third_party/fullkv/向上三级到达项目根目录
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))

# 如果项目根目录不在 sys.path 中，则添加它
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)


"""
基线实验执行脚本 - 使用标准KV缓存机制
"""
import copy
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
from pathlib import Path
from collections import defaultdict
from transformers import LogitsProcessor, LogitsProcessorList

# 导入项目模块
from src.hace import config
MODEL_CONFIG = config.MODEL_CONFIG
EXPERIMENT_CONFIG = config.EXPERIMENT_CONFIG
DATASET_CONFIG = config.DATASET_CONFIG
OUTPUT_CONFIG = config.OUTPUT_CONFIG
MONITORING_CONFIG = config.MONITORING_CONFIG

# 导入模块 - 更新路径以匹配新的目录结构
from src.hace.models.model_loader import (
    load_model_and_tokenizer,
    configure_model_for_kv_cache_length,
    prepare_model_for_baseline,
    load_inference_backend
)
from src.hace.data.dataset_loader import load_dataset_split, prepare_samples_for_evaluation, prepare_batch
from src.hace.utils.unified_monitor import UnifiedMonitor
from evaluation.eval_utils import score_dataset
from src.hace.models.inference_backend import GenerationConfig
import src.hace.config as hace_core_config


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


def normalize_backend_choice(backend: str) -> str:
    """归一化推理后端名称"""
    backend = (backend or "hf").strip().lower()
    if backend in {"hf", "huggingface", "transformers"}:
        return "hf"
    if backend == "vllm":
        return "vllm"
    raise ValueError(f"Unsupported backend: {backend} (expected: hf|vllm)")


def build_vllm_config_for_kv(
    base_vllm_config: dict,
    kv_cache_length: int,
    max_new_tokens: int,
    *,
    extra_len_buffer: int = 8,
) -> dict:
    """
    为单个 KV 长度实验构建 vLLM 配置。

    vLLM 的 max_model_len 是「prompt + completion」总长度上限，
    因此这里用 kv_cache_length + max_new_tokens (+ buffer) 来对齐实验含义。
    """
    vllm_cfg = copy.deepcopy(base_vllm_config or {})
    vllm_cfg["max_model_len"] = int(kv_cache_length) + int(max_new_tokens) + int(extra_len_buffer)
    return vllm_cfg


def truncate_prompts_by_tokens(tokenizer, prompts, max_length: int, mode: str = "middle"):
    """用 tokenizer 按 token 长度截断 prompt。mode: middle(默认)/head。"""
    if not prompts:
        return []

    if max_length <= 0:
        return list(prompts)

    if mode == "head":
        enc = tokenizer(
            prompts,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        input_ids = enc.get("input_ids", [])
        if input_ids and isinstance(input_ids[0], int):
            input_ids = [input_ids]
        return [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]

    truncated_prompts = []
    for prompt in prompts:
        if not prompt:
            truncated_prompts.append(prompt)
            continue

        enc = tokenizer(prompt, truncation=False, return_tensors=None)
        input_ids = enc.get("input_ids", [])
        if not input_ids:
            truncated_prompts.append(prompt)
            continue
        if isinstance(input_ids[0], list):
            input_ids = input_ids[0]

        if len(input_ids) <= max_length:
            truncated_prompts.append(prompt)
            continue

        half = max_length // 2
        truncated_ids = input_ids[:half] + input_ids[-half:]
        truncated_prompts.append(tokenizer.decode(truncated_ids, skip_special_tokens=True))

    return truncated_prompts


CAKE_NO_CHAT_DATASETS = {
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "lcc",
    "repobench-p",
}

_CAKE_DATASET_PROMPTS = None
_CAKE_DATASET_MAXLEN = None


def _get_logger():
    return globals().get("logger") or logging.getLogger(__name__)


def _load_cake_dataset_prompts():
    global _CAKE_DATASET_PROMPTS
    if _CAKE_DATASET_PROMPTS is not None:
        return _CAKE_DATASET_PROMPTS

    prompt_path = Path(project_root_dir) / "config" / "datasets" / "dataset2prompt.json"
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            _CAKE_DATASET_PROMPTS = json.load(f)
    except Exception as exc:
        _get_logger().warning(f"Failed to load CAKE LongBench prompts from {prompt_path}: {exc}")
        _CAKE_DATASET_PROMPTS = {}
    return _CAKE_DATASET_PROMPTS


def _load_cake_dataset_maxlen():
    global _CAKE_DATASET_MAXLEN
    if _CAKE_DATASET_MAXLEN is not None:
        return _CAKE_DATASET_MAXLEN

    maxlen_path = Path(project_root_dir) / "config" / "datasets" / "dataset2maxlen.json"
    try:
        with open(maxlen_path, "r", encoding="utf-8") as f:
            _CAKE_DATASET_MAXLEN = json.load(f)
    except Exception as exc:
        _get_logger().warning(f"Failed to load CAKE LongBench maxlen from {maxlen_path}: {exc}")
        _CAKE_DATASET_MAXLEN = {}
    return _CAKE_DATASET_MAXLEN


def _format_cake_prompt(prompt_format, raw_sample):
    data = defaultdict(str)
    if isinstance(raw_sample, dict):
        data.update(raw_sample)
    return prompt_format.format_map(data)


def _apply_cake_chat_template(tokenizer, prompt, dataset_name, model_name):
    if dataset_name in CAKE_NO_CHAT_DATASETS:
        return prompt

    model_id = (model_name or "").lower()
    if "llama3" in model_id:
        return (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    if "llama2" in model_id:
        return f"[INST] {prompt} [/INST]"
    if "mistral" in model_id:
        return f"<s>[INST] {prompt} [/INST]"
    if "qwen" in model_id:
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return prompt


def _resolve_chat_kind(model_name):
    model_id = (model_name or "").lower()
    if "llama3" in model_id:
        return "llama3"
    if "llama2" in model_id:
        return "llama2"
    if "mistral" in model_id:
        return "mistral"
    if "qwen" in model_id:
        return "qwen"
    return None


def _estimate_chat_overhead_tokens(tokenizer, chat_kind):
    if chat_kind is None:
        return 0

    if chat_kind == "qwen":
        if not hasattr(tokenizer, "apply_chat_template"):
            return 0
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": ""},
        ]
        chat_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    elif chat_kind == "llama3":
        chat_prompt = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    elif chat_kind == "llama2":
        chat_prompt = "[INST][/INST]"
    elif chat_kind == "mistral":
        chat_prompt = "<s>[INST]  [/INST]"
    else:
        return 0

    enc = tokenizer(chat_prompt, truncation=False, return_tensors=None)
    input_ids = enc.get("input_ids", [])
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return len(input_ids or [])


def apply_cake_prompting(samples, dataset_name, tokenizer, model_name, kv_cache_length):
    prompt_map = _load_cake_dataset_prompts()
    use_cake_prompt = dataset_name in prompt_map
    chat_kind = _resolve_chat_kind(model_name)
    use_chat = use_cake_prompt and dataset_name not in CAKE_NO_CHAT_DATASETS and chat_kind is not None
    chat_overhead = _estimate_chat_overhead_tokens(tokenizer, chat_kind) if use_chat else 0
    max_prompt_tokens = max(1, kv_cache_length - chat_overhead) if kv_cache_length else kv_cache_length

    for sample in samples:
        prompt = sample.get("prompt", "")
        raw_sample = sample.get("original_sample") if use_cake_prompt else None

        if use_cake_prompt and raw_sample:
            try:
                prompt = _format_cake_prompt(prompt_map[dataset_name], raw_sample)
            except Exception as exc:
                _get_logger().warning(f"CAKE prompt format failed for {dataset_name}: {exc}")

        prompt = truncate_prompts_by_tokens(tokenizer, [prompt], max_prompt_tokens)[0]
        if use_chat:
            prompt = _apply_cake_chat_template(tokenizer, prompt, dataset_name, model_name)

        sample["prompt"] = prompt


def apply_prompt_truncation(samples, tokenizer, kv_cache_length, mode="head"):
    if not kv_cache_length:
        return
    for sample in samples:
        prompt = sample.get("prompt", "")
        prompt = truncate_prompts_by_tokens(tokenizer, [prompt], kv_cache_length, mode=mode)[0]
        sample["prompt"] = prompt


def resolve_cake_max_new_tokens(dataset_name, fallback):
    maxlen_map = _load_cake_dataset_maxlen()
    if dataset_name in maxlen_map:
        try:
            return int(maxlen_map[dataset_name])
        except (TypeError, ValueError):
            pass
    return fallback


def run_baseline_experiment(model_config, dataset_name, dataset_config,
                           kv_cache_length, batch_size, max_new_tokens,
                           output_dir, repeat_index=0,
                           *,
                           backend=None,
                           tokenizer=None,
                           vllm_config=None,
                           cake_prompting=True,
                           cake_maxlen=True,
                           greedy_decode=True):
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
    backend_type = normalize_backend_choice(model_config.get("inference_backend") or "hf")

    # 初始化统一监控器
    monitor = UnifiedMonitor(experiment_id=experiment_id)
    monitor.record_config({
        "model_name": model_config["model_name_or_path"],
        "precision": model_config["precision"],
        "batch_size": batch_size,
        "kv_cache_length": kv_cache_length,
        "max_new_tokens": max_new_tokens,
        "use_baseline": True,
        "dataset": dataset_name
    })

    model = None  # 初始化model变量
    owns_backend = False
    local_backend = backend
    local_tokenizer = tokenizer
    try:
        # 加载模型/后端
        if backend_type == "vllm":
            if local_backend is None:
                logger.info("Loading VLLM backend...")
                local_backend = load_inference_backend(
                    model_config,
                    vllm_config or hace_core_config.VLLM_CONFIG
                )
                owns_backend = True
            if local_tokenizer is None:
                local_tokenizer = local_backend.get_tokenizer()
        else:
            logger.info("Loading model and tokenizer (HF)...")
            model, local_tokenizer = load_model_and_tokenizer(model_config)
            model = configure_model_for_kv_cache_length(model, kv_cache_length)
            model = prepare_model_for_baseline(model)

        # 加载数据集
        logger.info(f"Loading dataset {dataset_name}...")
        dataset = load_dataset_split(dataset_config)

        # 准备评估样本（全量评估）
        eval_sample_count = len(dataset)
        samples = prepare_samples_for_evaluation(
            dataset,
            dataset_name,
            num_samples=eval_sample_count,  # 使用全量样本进行评估
            random_seed=EXPERIMENT_CONFIG.get("random_seed", 42)
        )

        if not samples:
            logger.warning(f"Skipping experiment: no samples found for dataset {dataset_name}")
            return monitor.get_comprehensive_metrics()

        if cake_prompting:
            # 按 CAKE LongBench 规范构建 prompt（含中间截断与 chat 模板）
            apply_cake_prompting(
                samples,
                dataset_name,
                local_tokenizer,
                model_config.get("model_name_or_path", ""),
                kv_cache_length,
            )
        else:
            apply_prompt_truncation(samples, local_tokenizer, kv_cache_length, mode="head")

        if cake_maxlen:
            dataset_max_new_tokens = resolve_cake_max_new_tokens(dataset_name, max_new_tokens)
            if dataset_max_new_tokens != max_new_tokens:
                logger.info(
                    f"Using CAKE max_new_tokens={dataset_max_new_tokens} for dataset {dataset_name}"
                )
        else:
            dataset_max_new_tokens = max_new_tokens

        # 批次统计
        total_batches = (len(samples) + batch_size - 1) // batch_size
        logger.info(f"Preparing batches with size {batch_size}, total batches: {total_batches}")

        generated_texts = []
        references = []

        monitor.start_monitoring()

        if backend_type == "vllm":
            # VLLM：按文本分批生成（每次调用的 prompt 数量由 batch_size 控制）
            if not getattr(local_backend, "_baseline_warmed_up", False):
                warmup_cfg = GenerationConfig(
                    max_new_tokens=min(5, max_new_tokens),
                    temperature=0.0,
                    top_p=1.0,
                    top_k=-1,
                    do_sample=False,
                )
                try:
                    local_backend.generate("Hello", config=warmup_cfg)
                except Exception as e:
                    logger.warning(f"VLLM warmup failed (ignored): {e}")
                setattr(local_backend, "_baseline_warmed_up", True)

            gen_start = time.time()
            monitor.performance_metrics["start_time"] = gen_start
            monitor.performance_metrics["token_times"] = []
            monitor.performance_metrics["tokens_generated"] = 0
            monitor.performance_metrics["success"] = True

            total_tokens = 0
            token_times = []

            for start_idx in range(0, len(samples), batch_size):
                batch_samples = samples[start_idx:start_idx + batch_size]
                prompts = [s.get("prompt", "") for s in batch_samples]

                if greedy_decode:
                    gen_config = GenerationConfig(
                        max_new_tokens=dataset_max_new_tokens,
                        temperature=0.0,
                        top_p=1.0,
                        top_k=-1,
                        do_sample=False
                    )
                else:
                    gen_config = GenerationConfig(
                        max_new_tokens=dataset_max_new_tokens,
                        temperature=0.7,
                        top_p=1.0,
                        top_k=-1,
                        do_sample=True
                    )

                batch_start = time.time()
                outputs = local_backend.generate(prompts, config=gen_config)
                batch_end = time.time()

                if not isinstance(outputs, list):
                    outputs = [outputs]
                if len(outputs) != len(batch_samples):
                    logger.warning(
                        f"VLLM output count mismatch: got {len(outputs)} expected {len(batch_samples)}"
                    )

                for out, sample in zip(outputs, batch_samples):
                    generated_texts.append(out.text)
                    references.append(sample.get("reference", ""))

                    out_tokens = int(getattr(out, "completion_tokens", 0) or 0)
                    if out_tokens <= 0 and getattr(out, "token_ids", None) is not None:
                        out_tokens = len(out.token_ids or [])
                    total_tokens += out_tokens

                    if out_tokens > 0:
                        meta = getattr(out, "metadata", None) or {}
                        first_t = meta.get("first_token_time") or meta.get("first_token_ts")
                        finish_t = meta.get("finished_time") or meta.get("finish_time") or meta.get("finished_ts")
                        first_t = float(first_t) if first_t is not None else batch_start
                        finish_t = float(finish_t) if finish_t is not None else batch_end
                        if finish_t < first_t:
                            first_t, finish_t = batch_start, batch_end

                        if out_tokens == 1:
                            token_times.append(first_t)
                        else:
                            interval = (finish_t - first_t) / max(1, out_tokens - 1)
                            token_times.extend(first_t + i * interval for i in range(out_tokens))

            gen_end = time.time()
            total_time = gen_end - gen_start
            monitor.performance_metrics["total_time"] = total_time

            if token_times:
                token_times.sort()
                monitor.performance_metrics["token_times"] = token_times
                monitor.performance_metrics["first_token_time"] = token_times[0]
            else:
                monitor.performance_metrics["token_times"] = [gen_end]
                monitor.performance_metrics["first_token_time"] = gen_end

            monitor.performance_metrics["tokens_generated"] = total_tokens

        else:
            # HF 路径：预热 + LogitsProcessor 记录 token 时间
            warmup_batch = prepare_batch(
                samples[:batch_size],
                local_tokenizer,
                batch_size,
                max_length=kv_cache_length,
                drop_last=False
            )
            if warmup_batch is None:
                logger.warning(f"Skipping experiment: insufficient samples ({len(samples)}) for batch size {batch_size}")
                return monitor.get_comprehensive_metrics()
            warmup_inputs = {
                "input_ids": warmup_batch["input_ids"].to(model.device),
                "attention_mask": warmup_batch["attention_mask"].to(model.device)
            }

            logger.info("Warming up model...")
            with torch.no_grad():
                model.generate(
                    **warmup_inputs,
                    max_new_tokens=5,
                    do_sample=False
                )

            torch.cuda.empty_cache()

            monitor.start_generation()

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

            for start_idx in range(0, len(samples), batch_size):
                batch_samples = samples[start_idx:start_idx + batch_size]
                batch = prepare_batch(
                    batch_samples,
                    local_tokenizer,
                    batch_size,
                    max_length=kv_cache_length,
                    drop_last=False
                )
                if batch is None:
                    continue

                inputs = {
                    "input_ids": batch["input_ids"].to(model.device),
                    "attention_mask": batch["attention_mask"].to(model.device)
                }

                with torch.no_grad():
                    if greedy_decode:
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=dataset_max_new_tokens,
                            do_sample=False,
                            temperature=0.0,
                            logits_processor=logits_processor_list,
                            pad_token_id=local_tokenizer.pad_token_id,
                            eos_token_id=local_tokenizer.eos_token_id
                        )
                    else:
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=dataset_max_new_tokens,
                            do_sample=True,
                            temperature=0.7,
                            logits_processor=logits_processor_list,
                            pad_token_id=local_tokenizer.pad_token_id,
                            eos_token_id=local_tokenizer.eos_token_id
                        )

                batch_generated = local_tokenizer.batch_decode(
                    outputs[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )
                generated_texts.extend(batch_generated)
                references.extend([s.get("reference", "") for s in batch["samples"]])

            monitor.end_generation()

        # 停止监控并收集指标
        monitor.stop_monitoring()

        # 计算任务分数（粗略评估）
        try:
            raw_score = score_dataset(dataset_name, generated_texts, references)
            logger.info(f"Task score ({dataset_name}): {raw_score:.4f}")
        except Exception as e:
            logger.warning(f"评分失败，跳过得分记录: {e}")
            raw_score = None

        # 计算和保存指标
        metrics = monitor.get_comprehensive_metrics()
        metrics_filepath = monitor.save_metrics(output_dir)
        if raw_score is not None:
            # 将任务分数写回保存的指标文件
            try:
                with open(metrics_filepath, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                saved["task_score"] = raw_score
                saved["dataset"] = dataset_name
                with open(metrics_filepath, "w", encoding="utf-8") as f:
                    json.dump(saved, f, indent=2, ensure_ascii=False)
                metrics = saved
            except Exception as e:
                logger.warning(f"写入任务分数失败，保持原始指标文件: {e}")

        # 保存生成的文本（可选）
        if OUTPUT_CONFIG.get("save_model_outputs", False):
            outputs_dir = os.path.join(output_dir, "model_outputs")
            os.makedirs(outputs_dir, exist_ok=True)
            
            with open(os.path.join(outputs_dir, f"generated_texts_{experiment_id}.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "experiment_id": experiment_id,
                    "dataset": dataset_name,
                    "kv_cache_length": kv_cache_length,
                    "generated_texts": generated_texts,
                    "input_texts": [s.get("prompt", "") for s in samples],
                    "references": references,
                    "task_score": raw_score,
                }, f, indent=2, ensure_ascii=False)

        logger.info(f"Baseline experiment {experiment_id} completed successfully")
        return metrics

    except Exception as e:
        logger.error(f"Error during baseline experiment {experiment_id}: {e}")
        monitor.mark_failure(str(e))
        return monitor.get_comprehensive_metrics()
    finally:
        # 资源清理：避免多轮实验时 GPU 显存累积
        if backend_type == "vllm":
            if owns_backend and local_backend is not None:
                try:
                    local_backend.cleanup()
                except Exception as e:
                    logger.warning(f"Cleanup VLLM backend failed (ignored): {e}")
        else:
            try:
                if model is not None:
                    del model
                if local_tokenizer is not None:
                    del local_tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


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
    parser.add_argument("--backend", type=str, default=None, help="Inference backend: hf or vllm")
    parser.add_argument("--cake_prompting", dest="cake_prompting", action="store_true", help="Enable CAKE prompt alignment (default: on)")
    parser.add_argument("--no_cake_prompting", dest="cake_prompting", action="store_false", help="Disable CAKE prompt alignment")
    parser.add_argument("--cake_maxlen", dest="cake_maxlen", action="store_true", help="Enable CAKE per-dataset max_new_tokens (default: on)")
    parser.add_argument("--no_cake_maxlen", dest="cake_maxlen", action="store_false", help="Disable CAKE per-dataset max_new_tokens")
    parser.add_argument("--greedy_decode", dest="greedy_decode", action="store_true", help="Use greedy decoding (default: on)")
    parser.add_argument("--no_greedy_decode", dest="greedy_decode", action="store_false", help="Use sampling decoding")
    parser.set_defaults(cake_prompting=True, cake_maxlen=True, greedy_decode=True)

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

    backend_choice = normalize_backend_choice(args.backend or EXPERIMENT_CONFIG.get("inference_backend", "hf"))

    current_model_config = {
        "model_name_or_path": args.model_name,
        "precision": EXPERIMENT_CONFIG["precision"],
        "inference_backend": backend_choice
    }

    vllm_max_new_tokens = args.max_new_tokens
    if backend_choice == "vllm" and args.cake_maxlen:
        dataset_max_new_tokens = [
            resolve_cake_max_new_tokens(dataset_name, args.max_new_tokens)
            for dataset_name in datasets_list
        ]
        if dataset_max_new_tokens:
            vllm_max_new_tokens = max(dataset_max_new_tokens)
        if vllm_max_new_tokens != args.max_new_tokens:
            logger.info(
                "Using vLLM max_new_tokens=%s to cover CAKE per-dataset overrides",
                vllm_max_new_tokens,
            )

    pbar = tqdm(total=total_experiments, desc="Running Baseline Experiments")

    if backend_choice == "vllm":
        # vLLM 的 max_model_len 需要覆盖「prompt + completion」总长度，
        # 且无法在同一引擎实例上动态变更；因此按 kv_len 分组初始化引擎以避免重复加载模型。
        for kv_len in kv_lengths_list:
            vllm_cfg = build_vllm_config_for_kv(
                hace_core_config.VLLM_CONFIG,
                kv_cache_length=kv_len,
                max_new_tokens=vllm_max_new_tokens,
            )
            logger.info(
                f"Initializing VLLM backend for kv_len={kv_len} (max_model_len={vllm_cfg.get('max_model_len')})"
            )

            backend = load_inference_backend(current_model_config, vllm_cfg)
            tokenizer = backend.get_tokenizer()
            try:
                for rep in range(args.repetitions):
                    for dataset_name in datasets_list:
                        dataset_config = DATASET_CONFIG.get("available_datasets", {}).get(dataset_name)
                        if not dataset_config:
                            logger.error(f"Dataset configuration for '{dataset_name}' not found. Skipping...")
                            pbar.update(len(batch_sizes_list))
                            continue

                        for bs in batch_sizes_list:
                            logger.info(
                                f"Running baseline(vllm): Rep {rep+1}/{args.repetitions}, Dataset: {dataset_name}, "
                                f"KV_Len: {kv_len}, Batch: {bs}"
                            )
                            experiment_metrics = run_baseline_experiment(
                                model_config=current_model_config,
                                dataset_name=dataset_name,
                                dataset_config=dataset_config,
                                kv_cache_length=kv_len,
                                batch_size=bs,
                                max_new_tokens=args.max_new_tokens,
                                output_dir=args.output_dir,
                                repeat_index=rep,
                                backend=backend,
                                tokenizer=tokenizer,
                                vllm_config=vllm_cfg,
                                cake_prompting=args.cake_prompting,
                                cake_maxlen=args.cake_maxlen,
                                greedy_decode=args.greedy_decode,
                            )
                            all_results.append(experiment_metrics)
                            pbar.update(1)
            finally:
                try:
                    backend.cleanup()
                except Exception as e:
                    logger.warning(f"Cleanup VLLM backend failed (ignored): {e}")
    else:
        for rep in range(args.repetitions):
            for dataset_name in datasets_list:
                dataset_config = DATASET_CONFIG.get("available_datasets", {}).get(dataset_name)
                if not dataset_config:
                    logger.error(f"Dataset configuration for '{dataset_name}' not found. Skipping...")
                    pbar.update(len(kv_lengths_list) * len(batch_sizes_list))
                    continue

                for kv_len in kv_lengths_list:
                    for bs in batch_sizes_list:
                        logger.info(
                            f"Running baseline(hf): Rep {rep+1}/{args.repetitions}, Dataset: {dataset_name}, "
                            f"KV_Len: {kv_len}, Batch: {bs}"
                        )
                        experiment_metrics = run_baseline_experiment(
                            model_config=current_model_config,
                            dataset_name=dataset_name,
                            dataset_config=dataset_config,
                            kv_cache_length=kv_len,
                            batch_size=bs,
                            max_new_tokens=args.max_new_tokens,
                            output_dir=args.output_dir,
                            repeat_index=rep,
                            cake_prompting=args.cake_prompting,
                            cake_maxlen=args.cake_maxlen,
                            greedy_decode=args.greedy_decode,
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
