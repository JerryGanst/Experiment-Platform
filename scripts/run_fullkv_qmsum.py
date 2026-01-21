#!/usr/bin/env python3
"""Run Full KV Cache baseline on QMSum using same data as HACE."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
CAKE_ROOT = ROOT / "vendor" / "cake"
LONG_BENCH_DIR = CAKE_ROOT / "longbench"
CONFIG_DIR = ROOT / "config"

if str(ROOT / "vendor") not in sys.path:
    sys.path.insert(0, str(ROOT / "vendor"))

from cake.longbench.pred_cake import build_chat  # noqa: E402

NO_CHAT_DATASETS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Full KV Cache baseline on QMSum")
    parser.add_argument("--model", default="qwen2.5-7b-instruct", help="Model key")
    parser.add_argument("--model_path", default=None, help="Override model path")
    parser.add_argument("--dataset", default="qmsum", help="Dataset name")
    parser.add_argument("--pred_name", default="fullkv_baseline", help="Prediction run name")
    parser.add_argument("--device", type=int, default=0, help="CUDA device")
    parser.add_argument("--max_length", type=int, default=None, help="Max prompt length")
    parser.add_argument("--max_gen", type=int, default=None, help="Max generation length")
    parser.add_argument("--output_dir", default=None, help="Output dir")
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

    if args.output_dir is None:
        args.output_dir = str(ROOT / "results_clean" / args.pred_name)

    model2path = json.load(open(CONFIG_DIR / "models" / "model2path.json", "r"))
    model2maxlen = json.load(open(CONFIG_DIR / "models" / "model2maxlen.json", "r"))
    dataset2prompt = json.load(open(CONFIG_DIR / "datasets" / "dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open(CONFIG_DIR / "datasets" / "dataset2maxlen.json", "r"))

    model_name = args.model
    model_path = args.model_path or model2path[model_name]

    # Model path fallback
    if model_path and Path(model_path).is_absolute() and not Path(model_path).exists():
        print(f"WARNING: model path not found: {model_path}")
        candidates = [
            "/cloud/cloud-ssd1/Experiment-Platform/model/ModelScope/Qwen/Qwen2.5-7B-Instruct",
            "/model/ModelScope/Qwen/Qwen2.5-7B-Instruct",
            str(ROOT / "model" / "ModelScope" / "Qwen" / "Qwen2.5-7B-Instruct"),
        ]
        for cand in candidates:
            if Path(cand).exists():
                model_path = cand
                print(f"Using fallback model path: {model_path}")
                break

    max_length = args.max_length or model2maxlen[model_name]
    max_gen = args.max_gen or dataset2maxlen[args.dataset]

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # Load model WITHOUT compression (Full KV Cache)
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    ).to(device)

    # Load data - same as HACE
    data_files = {"train": str(ROOT / "data" / f"{args.dataset}.jsonl")}
    data = load_dataset("json", data_files=data_files, split="train")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{args.dataset}.jsonl"

    if out_path.exists():
        print(f"WARNING: {out_path} exists, removing...")
        out_path.unlink()

    prompt_format = dataset2prompt[args.dataset]

    print(f"\n{'='*60}")
    print(f"Full KV Cache Baseline Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Max Length: {max_length}")
    print(f"  Max Gen: {max_gen}")
    print(f"  Output: {out_path}")
    print(f"{'='*60}\n")

    for i, sample in enumerate(data):
        prompt = build_prompt(tokenizer, prompt_format, sample, args.dataset, max_length, model_name)
        inputs = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = inputs.input_ids.shape[-1]

        output = model.generate(
            **inputs,
            max_new_tokens=max_gen,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
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

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(data)} samples")

        del output, inputs
        torch.cuda.empty_cache()
        gc.collect()

    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
