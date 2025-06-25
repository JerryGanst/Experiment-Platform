import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoConfig, Gemma2ForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse

import torch.distributed as dist
import torch.multiprocessing as mp

import sys
from pathlib import Path

from cake.cake_cache import CakeprefillKVCache 
from cake.utils import CompressConfig 

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="llama3.1-8b-128k", 
                        choices=["llama2-7b-chat-4k", 
                                 "llama2-13b-chat-4k", 
                                 "llama3.1-8b-128k", 
                                 "mistral-0.3-7b-32k",
                                 "qwen2.5-7b-instruct"])
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--compress', action='store_true', help="Comrpess kv cache with CAKE")
    parser.add_argument('--cascading', action='store_true', help="Using cascading cache mangement")
    parser.add_argument('--pred_name', type=str, default="pred", help="Pred Output Name")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--cache_size', type=int, default=1024)
    parser.add_argument('--window_size', type=int, default=32)
    parser.add_argument('--tau1', type=float, default=1.0)
    parser.add_argument('--tau2', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=200.0)
    return parser.parse_args(args)

    
# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "llama3" in model_name:
        print("======== llama3 build chat ========")
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "mistral" in model_name:
        print("======== mistral build chat ========")
        prompt = f'<s>[INST] {prompt} [/INST]'
    elif "qwen" in model_name:
        print("======== qwen build chat ========")
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    return prompt



@torch.inference_mode()
def get_pred(model, tokenizer, compress, data, max_length, max_gen, prompt_format, dataset, model_name, model2path, out_path):

    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]

        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)

        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]

        if dataset == "samsum":
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]
  

        if compress:
            layers = len(model.model.layers)
            for i in range(layers):
                model.model.layers[i].self_attn.config.prefill = [True]*layers
                model.model.layers[i].self_attn.config.decoding_evict = [None]*layers

        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device, compress_config):

    if compress_config.compress:
        if "llama" in model_name:
            from cake.monkeypatch import replace_flashllama_attn_with_cakeattn
            replace_flashllama_attn_with_cakeattn()
        elif "mistral" in model_name:
            from cake.monkeypatch import replace_flashmistral_attn_with_cakeattn
            replace_flashmistral_attn_with_cakeattn()
        elif "qwen2" in model_name:
            from cake.monkeypatch import replace_flashqwen2_attn_with_cakeattn
            replace_flashqwen2_attn_with_cakeattn()

    if "qwen2" in model_name:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
   
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype,
        attn_implementation="flash_attention_2"
    ).to(device)
    config = AutoConfig.from_pretrained(path)
    if hasattr(config, 'num_hidden_layers'):
        layers = config.num_hidden_layers

    if compress_config.compress:
        for i in range(layers):
            model.model.layers[i].self_attn.config.key_size = [compress_config.cache_size - compress_config.window_size]*layers
            model.model.layers[i].self_attn.config.window_size = [compress_config.window_size]*layers
            model.model.layers[i].self_attn.config.prefill = [True]*layers
            model.model.layers[i].self_attn.config.decoding_evict = [None]*layers
            model.model.layers[i].self_attn.config.tau1 = compress_config.hyper[0]
            model.model.layers[i].self_attn.config.tau2 = compress_config.hyper[1] 
            model.model.layers[i].self_attn.config.gamma = compress_config.hyper[2] 
            model.model.layers[i].self_attn.config.prefill_cake_evict = [CakeprefillKVCache(
                cache_size=compress_config.cache_size,
                window_size=compress_config.window_size,
                k_seq_dim=2,
                v_seq_dim=2,
                num_heads=model.model.layers[i].self_attn.num_heads,
                num_layers=layers,
                use_cascading=compress_config.cascading
            )]*layers

    model = model.eval()

    
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    pred_name = args.pred_name
    model_name = args.model
    compress = args.compress
    cascading = args.cascading
    compress_config = CompressConfig(compress, cascading)
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    # define your model
    max_length = model2maxlen[model_name]
    if compress:
        compress_config.cache_size = args.cache_size
        compress_config.window_size = args.window_size
        cache_name = f"cache{args.cache_size}"
        model2tau = json.load(open("config/model2tau.json", "r"))
        try:
            tau1 = model2tau[model_name][f"{args.cache_size}"]["tau1"]
            tau2 = model2tau[model_name][f"{args.cache_size}"]["tau2"]
            print(tau1,tau2)
        except Exception as e:
            print(f"Error loading tau values: {e}")
            tau1, tau2 = 1.0, 1.0 
        gamma = args.gamma
        hyper = [tau1,tau2,gamma]
        compress_config.hyper = hyper
    else:
        cache_name ="cachefull"

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device, compress_config)

    datasets = ["narrativeqa", "qasper", "multifieldqa_en",  "hotpotqa", "2wikimqa", "musique", \
                    "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
                "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
                

    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists(f"./pred_result/{cache_name}/{pred_name}"):
        os.makedirs(f"./pred_result/{cache_name}/{pred_name}")
    
    for dataset in datasets:
        #load offline 
        data_files = {"test": f"{dataset}.jsonl"}
        data = load_dataset("json", data_dir='./datasets/longbench/data', split='test', data_files=data_files)

        if not os.path.exists(f"./pred_result/{cache_name}/{pred_name}/{model_name}"):
            os.makedirs(f"./pred_result/{cache_name}/{pred_name}/{model_name}")
        out_path = f"./pred_result/{cache_name}/{pred_name}/{model_name}/{dataset}.jsonl"
    
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]

        if os.path.exists(out_path):
            with open(out_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            if len(data_all) == len(lines):
                continue
            else:
                data_all=data_all[len(lines):]

        get_pred(model, tokenizer, compress, data_all, max_length, \
                                    max_gen, prompt_format, dataset, model_name, model2path, out_path)
