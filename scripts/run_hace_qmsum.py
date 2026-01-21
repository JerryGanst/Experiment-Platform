#!/usr/bin/env python3
"""Run HACE (Glue) variant on QMSum using CAKE LongBench pipeline."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
CAKE_ROOT = ROOT / "vendor" / "cake"
LONG_BENCH_DIR = CAKE_ROOT / "longbench"
CONFIG_DIR = ROOT / "config"

if str(ROOT / "vendor") not in sys.path:
    sys.path.insert(0, str(ROOT / "vendor"))

from cake.utils import CompressConfig  # noqa: E402
from cake.longbench.pred_cake import build_chat, load_model_and_tokenizer  # noqa: E402


NO_CHAT_DATASETS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HACE on QMSum (LongBench format)")
    parser.add_argument("--model", default="qwen2.5-7b-instruct", help="Model key used in CAKE config")
    parser.add_argument("--model_path", default=None, help="Override model path (optional)")
    parser.add_argument("--dataset", default="qmsum", help="Dataset name (default: qmsum)")
    parser.add_argument("--cache_size", type=int, default=128, help="Total cache size (Btotal)")
    parser.add_argument("--window_size", type=int, default=32, help="Window size")
    parser.add_argument("--pred_name", default="qwen2.5_hace_128", help="Prediction run name")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--max_length", type=int, default=None, help="Override max prompt length")
    parser.add_argument("--max_gen", type=int, default=None, help="Override max generation length")
    parser.add_argument("--pref_mode", default="reverse_disp", help="HACE pref mode: normal | reverse | reverse_disp")
    parser.add_argument("--head_mode", default="", help="HACE head mode: '' (disabled) | high_entropy (分数加权) | low_entropy (分数加权) | budget_realloc (真正的预算重分配) | adakv (Ada-KV counting)")
    parser.add_argument("--output_dir", default=None, help="Output dir")
    parser.add_argument("--use_cascading", action="store_true", help="Enable CAKE cascading")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (enables reproducible randomness)")
    return parser.parse_args()


def build_prompt(tokenizer, prompt_format: str, sample: dict, dataset: str, max_length: int, model_name: str) -> str:
    prompt = prompt_format.format(**sample)
    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(tokenized_prompt) > max_length:
        half = int(max_length / 2)
        prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + \
                 tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
    if dataset not in NO_CHAT_DATASETS:
        prompt = build_chat(tokenizer, prompt, model_name)
    return prompt


def main() -> None:
    args = parse_args()

    # Set random seed for reproducible sampling
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.output_dir is None:
        args.output_dir = str(ROOT / "results_clean" / args.pred_name)

    os.environ["HACE_PREF_MODE"] = args.pref_mode
    os.environ["HACE_HEAD_MODE"] = args.head_mode

    model2path = json.load(open(LONG_BENCH_DIR / "config" / "model2path.json", "r"))
    model2maxlen = json.load(open(LONG_BENCH_DIR / "config" / "model2maxlen.json", "r"))
    dataset2prompt = json.load(open(LONG_BENCH_DIR / "config" / "dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open(LONG_BENCH_DIR / "config" / "dataset2maxlen.json", "r"))
    model2tau = json.load(open(LONG_BENCH_DIR / "config" / "model2tau.json", "r"))

    model_name = args.model
    model_path = args.model_path or model2path[model_name]

    if model_path and Path(model_path).is_absolute() and not Path(model_path).exists():
        print(f"WARNING: model path not found: {model_path}")
        candidates = []
        if model_name == "qwen2.5-7b-instruct":
            candidates.extend([
                "/cloud/cloud-ssd1/Experiment-Platform/model/ModelScope/Qwen/Qwen2.5-7B-Instruct",
                "/model/ModelScope/Qwen/Qwen2.5-7B-Instruct",
                "/model/Qwen2.5-7B-Instruct",
                str(ROOT / "model" / "Qwen2.5-7B-Instruct"),
            ])
        base = Path(model_path).name
        candidates.extend([
            f"/model/{base}",
            str(ROOT / "model" / base),
        ])
        for cand in candidates:
            if Path(cand).exists():
                model_path = cand
                print(f"Using fallback model path: {model_path}")
                break

    max_length = args.max_length or model2maxlen[model_name]
    max_gen = args.max_gen or dataset2maxlen[args.dataset]

    compress_config = CompressConfig(True, args.use_cascading)
    compress_config.cache_size = args.cache_size
    compress_config.window_size = args.window_size

    tau1 = model2tau.get(model_name, {}).get(str(args.cache_size), {}).get("tau1", 1.0)
    tau2 = model2tau.get(model_name, {}).get(str(args.cache_size), {}).get("tau2", 1.0)
    gamma = 200.0
    compress_config.hyper = [tau1, tau2, gamma]

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(model_path, model_name, device, compress_config)

    data_files = {"train": str(ROOT / "data" / f"{args.dataset}.jsonl")}
    data = load_dataset("json", data_files=data_files, split="train")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{args.dataset}.jsonl"

    # Clear old results to prevent pollution
    if out_path.exists():
        print(f"WARNING: {out_path} already exists, removing...")
        out_path.unlink()

    prompt_format = dataset2prompt[args.dataset]

    print(f"\n{'='*60}")
    print(f"HACE Experiment Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Pref Mode (Layer): {args.pref_mode}")
    print(f"  Head Mode: {args.head_mode if args.head_mode else 'disabled (uniform)'}")
    print(f"  Cache Size: {args.cache_size}")
    print(f"  Window Size: {args.window_size}")
    print(f"  Seed: {args.seed}")
    print(f"  Sampling: do_sample=True, temperature=0.7, top_p=0.95")
    print(f"  Output: {out_path}")
    print(f"{'='*60}\n")

    # Support sample limit for quick testing
    max_samples = int(os.environ.get("HACE_MAX_SAMPLES", "0"))
    data_list = list(data)
    if max_samples > 0:
        data_list = data_list[:max_samples]
        print(f"[HACE] Limited to {max_samples} samples for testing")

    print(f"DEBUG: About to start loop with {len(data_list)} samples"); import sys; sys.stdout.flush()
    for sample in data_list:
        prompt = build_prompt(tokenizer, prompt_format, sample, args.dataset, max_length, model_name)
        inputs = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = inputs.input_ids.shape[-1]

        output = model.generate(
            **inputs,
            max_new_tokens=max_gen,
            num_beams=1,
            do_sample=True,       # Enable sampling for randomness
            temperature=0.7,      # Moderate randomness (industry default)
            top_p=0.95,           # Nucleus sampling
        )[0]

        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump(
                {
                    "pred": pred,
                    "answers": sample["answers"],
                    "all_classes": sample.get("all_classes"),
                    "length": sample.get("length"),
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")

        # Reset prefill state for next sample (CRITICAL for budget_realloc to work!)
        layers = len(model.model.layers)
        for i in range(layers):
            model.model.layers[i].self_attn.config.prefill = [True]*layers
            model.model.layers[i].self_attn.config.decoding_evict = [None]*layers

        # Clear GPU cache and trigger garbage collection after each sample
        del output, inputs
        torch.cuda.empty_cache()
        gc.collect()

    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
