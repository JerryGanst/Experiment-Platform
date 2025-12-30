#!/usr/bin/env python3
"""
HACE 头级差异化实验 - VLLM 版本

使用 VLLM 高性能推理引擎运行 HACE 实验
支持层级和头级的 KV Cache 预算分配策略
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "hace_core"))

def parse_args():
    parser = argparse.ArgumentParser(description="HACE Experiments with VLLM")

    # 模型配置
    parser.add_argument("--model", type=str, default="qwen2.5-7b-instruct",
                        help="Model name or path")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Explicit model path (overrides model name)")

    # 数据集配置
    parser.add_argument("--dataset", type=str, default="qmsum",
                        choices=["qmsum", "narrativeqa", "qasper", "multifieldqa_en",
                                 "hotpotqa", "2wikimqa", "musique", "gov_report",
                                 "multi_news", "trec", "triviaqa", "samsum"],
                        help="Dataset to evaluate")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Custom data path")
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Number of samples to evaluate")

    # HACE 策略配置
    parser.add_argument("--pref_mode", type=str, default="normal",
                        choices=["normal", "reverse_disp", "uniform"],
                        help="Layer-level preference mode")
    parser.add_argument("--head_mode", type=str, default="",
                        choices=["", "high_entropy", "low_entropy", "adakv"],
                        help="Head-level allocation mode")

    # Cache 配置
    parser.add_argument("--cache_size", type=int, default=128,
                        help="Total cache size per layer")
    parser.add_argument("--window_size", type=int, default=32,
                        help="Sliding window size")

    # VLLM 配置
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90,
                        help="VLLM GPU memory utilization")
    parser.add_argument("--max_model_len", type=int, default=32768,
                        help="Maximum model context length")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Tensor parallel size")

    # 生成配置
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature")

    # 输出配置
    parser.add_argument("--output_dir", type=str, default="results_clean",
                        help="Output directory")
    parser.add_argument("--pred_name", type=str, default=None,
                        help="Prediction output name")

    return parser.parse_args()


def get_model_path(model_name: str, explicit_path: str = None) -> str:
    """获取模型路径"""
    if explicit_path:
        return explicit_path

    # 模型路径候选
    path_candidates = [
        # GCP 路径
        f"/mnt/disks/experiment/models/{model_name}",
        f"/mnt/disks/experiment/models/Qwen2.5-7B-Instruct",
        # 旧服务器路径
        f"/cloud/cloud-ssd1/models/{model_name}",
        "/cloud/cloud-ssd1/models/Qwen2.5-7B-Instruct",
        # 本地路径
        f"~/models/{model_name}",
        # HuggingFace Hub
        f"Qwen/Qwen2.5-7B-Instruct",
    ]

    for path in path_candidates:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            return expanded_path

    # 回退到 HuggingFace Hub
    return "Qwen/Qwen2.5-7B-Instruct"


def get_data_path(dataset: str, custom_path: str = None) -> str:
    """获取数据路径"""
    if custom_path:
        return custom_path

    candidates = [
        PROJECT_ROOT / "data" / f"{dataset}.jsonl",
        PROJECT_ROOT / "src/third_party/cakekv-main/cakekv-main/data" / f"{dataset}.jsonl",
        Path(f"/mnt/disks/experiment/Experiment-Platform/data/{dataset}.jsonl"),
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    raise FileNotFoundError(f"Dataset {dataset} not found in any known locations")


def load_dataset(data_path: str, num_samples: int):
    """加载数据集"""
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
                if len(data) >= num_samples:
                    break
    return data


def build_prompt(sample: dict, dataset: str) -> str:
    """构建 prompt"""
    # LongBench 格式的 prompt 模板
    prompt_templates = {
        "qmsum": "You are given a meeting transcript and a query. Answer the query based on the transcript.\n\nTranscript:\n{context}\n\nQuery: {input}\n\nAnswer:",
        "narrativeqa": "You are given a story and a question. Answer the question based on the story.\n\nStory:\n{context}\n\nQuestion: {input}\n\nAnswer:",
        "hotpotqa": "Answer the following question based on the provided context.\n\nContext:\n{context}\n\nQuestion: {input}\n\nAnswer:",
        "multi_news": "Write a summary of the following news articles.\n\nArticles:\n{context}\n\nSummary:",
        "gov_report": "Write a summary of the following government report.\n\nReport:\n{context}\n\nSummary:",
    }

    template = prompt_templates.get(dataset, "{context}\n\nQuestion: {input}\n\nAnswer:")

    context = sample.get("context", "")
    input_text = sample.get("input", "")

    return template.format(context=context, input=input_text)


def main():
    args = parse_args()

    # 设置环境变量传递 HACE 配置
    os.environ["HACE_PREF_MODE"] = args.pref_mode
    os.environ["HACE_HEAD_MODE"] = args.head_mode or ""
    os.environ["HACE_CACHE_SIZE"] = str(args.cache_size)
    os.environ["HACE_WINDOW_SIZE"] = str(args.window_size)

    print("=" * 60)
    print("HACE Experiment with VLLM")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Pref Mode (Layer): {args.pref_mode}")
    print(f"Head Mode: {args.head_mode or 'uniform'}")
    print(f"Cache Size: {args.cache_size}")
    print(f"Window Size: {args.window_size}")
    print(f"Samples: {args.num_samples}")
    print("=" * 60)

    # 获取路径
    model_path = get_model_path(args.model, args.model_path)
    data_path = get_data_path(args.dataset, args.data_path)
    print(f"Model Path: {model_path}")
    print(f"Data Path: {data_path}")

    # 导入 VLLM
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("Error: VLLM not installed. Run: pip install vllm")
        sys.exit(1)

    # 初始化 VLLM
    print("\nInitializing VLLM...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enforce_eager=True,  # 便于调试
        enable_prefix_caching=False,  # HACE 需要禁用 prefix caching
    )

    tokenizer = llm.get_tokenizer()

    # 采样参数
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=1.0,
        stop=["<|endoftext|>", "<|im_end|>"],
    )

    # 加载数据
    print(f"\nLoading dataset: {args.dataset}")
    data = load_dataset(data_path, args.num_samples)
    print(f"Loaded {len(data)} samples")

    # 准备输出目录
    if args.pred_name:
        output_name = args.pred_name
    else:
        head_suffix = f"_{args.head_mode}" if args.head_mode else "_uniform"
        output_name = f"hace_{args.pref_mode}{head_suffix}"

    output_dir = Path(args.output_dir) / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.dataset}.jsonl"

    print(f"Output: {output_file}")

    # 运行推理
    print("\nRunning inference...")
    results = []

    for sample in tqdm(data, desc="Processing"):
        prompt = build_prompt(sample, args.dataset)

        # 应用 chat 模板
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 截断过长的输入
        input_ids = tokenizer.encode(formatted_prompt)
        max_input_len = args.max_model_len - args.max_new_tokens - 100
        if len(input_ids) > max_input_len:
            # 截断中间部分，保留开头和结尾
            half = max_input_len // 2
            input_ids = input_ids[:half] + input_ids[-half:]
            formatted_prompt = tokenizer.decode(input_ids)

        # 生成
        outputs = llm.generate([formatted_prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text

        # 保存结果
        result = {
            "pred": generated_text,
            "answers": sample.get("answers", []),
            "all_classes": sample.get("all_classes", None),
            "length": sample.get("length", len(prompt)),
        }
        results.append(result)

        # 实时写入
        with open(output_file, "a", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")

    print(f"\nCompleted! Results saved to: {output_file}")
    print(f"Total samples: {len(results)}")

    # 运行评估
    print("\nRunning evaluation...")
    eval_script = PROJECT_ROOT / "src/third_party/cakekv-main/cakekv-main/experiments/LongBench/eval.py"
    if eval_script.exists():
        import subprocess
        result = subprocess.run([
            sys.executable, str(eval_script),
            "--dir_path", str(output_dir)
        ], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

    # 显示结果
    result_json = output_dir / "result.json"
    if result_json.exists():
        with open(result_json, "r") as f:
            scores = json.load(f)
        print("\n" + "=" * 40)
        print("SCORES:")
        for k, v in scores.items():
            print(f"  {k}: {v:.2f}")
        print("=" * 40)


if __name__ == "__main__":
    main()
