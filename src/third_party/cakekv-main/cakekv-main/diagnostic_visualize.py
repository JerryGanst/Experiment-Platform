#!/usr/bin/env python3
"""
诊断可视化：对比 Baseline vs Ada 的 token 保留差异

Usage:
    python diagnostic_visualize.py --baseline baseline.json --ada ada.json --sample 0 --layer 0
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_diagnostic(path: str) -> List[Dict]:
    """加载诊断文件"""
    with open(path, 'r') as f:
        return json.load(f)


def compare_two_runs(baseline_path: str, ada_path: str, 
                     sample_id: int = 0, layer_idx: int = 0) -> Tuple[Optional[List[Dict]], int]:
    """
    对比两次运行的 token 保留差异
    
    Returns:
        Tuple of (head_diffs, seq_len) or (None, 0) if no data
    """
    baseline = load_diagnostic(baseline_path)
    ada = load_diagnostic(ada_path)
    
    # 过滤指定 sample 和 layer
    baseline_layer = [r for r in baseline 
                      if r['sample_id'] == sample_id and r['layer_idx'] == layer_idx]
    ada_layer = [r for r in ada 
                 if r['sample_id'] == sample_id and r['layer_idx'] == layer_idx]
    
    if not baseline_layer or not ada_layer:
        print(f"No data for sample_id={sample_id}, layer_idx={layer_idx}")
        return None, 0
    
    # Sort by head_idx to ensure alignment
    baseline_layer = sorted(baseline_layer, key=lambda x: x['head_idx'])
    ada_layer = sorted(ada_layer, key=lambda x: x['head_idx'])
    
    num_heads = len(baseline_layer)
    seq_len = baseline_layer[0]['total_seq_len']
    
    print(f"\n{'='*60}")
    print(f"Sample {sample_id}, Layer {layer_idx}, SeqLen {seq_len}")
    print(f"{'='*60}")
    
    # 统计差异
    total_only_baseline = 0
    total_only_ada = 0
    total_common = 0
    
    head_diffs = []
    
    for head_idx in range(num_heads):
        b_record = baseline_layer[head_idx]
        a_record = ada_layer[head_idx]
        
        b_set = set(b_record['kept_indices'])
        a_set = set(a_record['kept_indices'])
        
        only_baseline = b_set - a_set  # Baseline 保留但 Ada 丢弃
        only_ada = a_set - b_set       # Ada 新增保留
        common = b_set & a_set         # 两者都保留
        
        total_only_baseline += len(only_baseline)
        total_only_ada += len(only_ada)
        total_common += len(common)
        
        head_diffs.append({
            'head_idx': head_idx,
            'baseline_budget': b_record['head_budget'],
            'ada_budget': a_record['head_budget'],
            'baseline_entropy': b_record.get('head_entropy', 0),
            'ada_entropy': a_record.get('head_entropy', 0),
            'only_baseline': sorted(only_baseline),
            'only_ada': sorted(only_ada),
            'common_count': len(common),
            'jaccard': len(common) / len(b_set | a_set) if (b_set | a_set) else 1.0
        })
    
    # 打印总体统计
    print(f"\n📊 Overall Statistics:")
    print(f"   Common tokens:        {total_common}")
    print(f"   Only in Baseline:     {total_only_baseline} (Ada dropped these)")
    print(f"   Only in Ada:          {total_only_ada} (Ada added these)")
    
    # 打印每个 head 的差异
    print(f"\n📋 Per-Head Differences:")
    print(f"{'Head':>4} | {'B_Bud':>5} | {'A_Bud':>5} | {'Entropy':>7} | {'Jaccard':>7} | {'B_Only':>6} | {'A_Only':>6}")
    print("-" * 60)
    
    for d in head_diffs:
        print(f"{d['head_idx']:>4} | {d['baseline_budget']:>5} | {d['ada_budget']:>5} | "
              f"{d['ada_entropy']:>7.3f} | {d['jaccard']:>7.3f} | "
              f"{len(d['only_baseline']):>6} | {len(d['only_ada']):>6}")
    
    # Print summary
    avg_jaccard = np.mean([d['jaccard'] for d in head_diffs])
    print(f"\n   Average Jaccard similarity: {avg_jaccard:.3f}")
    
    return head_diffs, seq_len


def visualize_token_retention(head_diffs: List[Dict], seq_len: int, 
                               output_path: str = None):
    """
    可视化 token 保留热力图
    """
    num_heads = len(head_diffs)
    
    # 创建三个矩阵: common, only_baseline, only_ada
    baseline_matrix = np.zeros((num_heads, seq_len))
    ada_matrix = np.zeros((num_heads, seq_len))
    
    for d in head_diffs:
        head_idx = d['head_idx']
        for idx in d['only_baseline']:
            if idx < seq_len:
                baseline_matrix[head_idx, idx] = 1
        for idx in d['only_ada']:
            if idx < seq_len:
                ada_matrix[head_idx, idx] = 1
    
    # 创建合并视图: -1=baseline_only, 0=neither, 1=ada_only
    combined = np.zeros((num_heads, seq_len))
    combined[baseline_matrix == 1] = -1  # 红色：被 Ada 丢弃
    combined[ada_matrix == 1] = 1        # 绿色：被 Ada 新增
    
    # 绘图
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    
    # 图1: 差异热力图
    ax1 = axes[0]
    im1 = ax1.imshow(combined, aspect='auto', cmap='RdYlGn', 
                     vmin=-1, vmax=1, interpolation='nearest')
    ax1.set_xlabel('Token Position')
    ax1.set_ylabel('Head Index')
    ax1.set_title('Token Retention Difference: Red=Dropped by Ada, Green=Added by Ada')
    plt.colorbar(im1, ax=ax1, label='Difference')
    
    # 图2: Budget 和 Entropy 对比
    ax2 = axes[1]
    x = range(num_heads)
    baseline_budgets = [d['baseline_budget'] for d in head_diffs]
    ada_budgets = [d['ada_budget'] for d in head_diffs]
    entropies = [d['ada_entropy'] for d in head_diffs]
    
    ax2_twin = ax2.twinx()
    
    width = 0.35
    ax2.bar([i - width/2 for i in x], baseline_budgets, width, 
            label='Baseline Budget', alpha=0.7, color='blue')
    ax2.bar([i + width/2 for i in x], ada_budgets, width, 
            label='Ada Budget', alpha=0.7, color='orange')
    ax2_twin.plot(x, entropies, 'g.-', label='Head Entropy', linewidth=2)
    
    ax2.set_xlabel('Head Index')
    ax2.set_ylabel('Budget (tokens)')
    ax2_twin.set_ylabel('Entropy', color='green')
    ax2.set_title('Budget Allocation Comparison')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n📈 Visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_answer_overlap(head_diffs: List[Dict], 
                           answer_positions: List[int],
                           context: str = None):
    """
    分析被丢弃/新增的 token 与答案位置的关系
    
    answer_positions: 答案在序列中的 token 位置
    """
    print(f"\n🎯 Answer Position Analysis:")
    print(f"   Answer positions: {answer_positions}")
    
    answer_set = set(answer_positions)
    
    dropped_answer_tokens = 0
    added_answer_tokens = 0
    
    dropped_heads = []
    added_heads = []
    
    for d in head_diffs:
        dropped_in_answer = set(d['only_baseline']) & answer_set
        added_in_answer = set(d['only_ada']) & answer_set
        
        if dropped_in_answer:
            print(f"   ⚠️  Head {d['head_idx']}: DROPPED answer tokens at {sorted(dropped_in_answer)}")
            dropped_answer_tokens += len(dropped_in_answer)
            dropped_heads.append(d['head_idx'])
        
        if added_in_answer:
            print(f"   ✅ Head {d['head_idx']}: ADDED answer tokens at {sorted(added_in_answer)}")
            added_answer_tokens += len(added_in_answer)
            added_heads.append(d['head_idx'])
    
    print(f"\n   Summary:")
    print(f"   - Answer tokens dropped by Ada: {dropped_answer_tokens} (in heads: {dropped_heads})")
    print(f"   - Answer tokens added by Ada: {added_answer_tokens} (in heads: {added_heads})")
    
    if dropped_answer_tokens > added_answer_tokens:
        print(f"   ❌ Ada is HURTING answer retention!")
    elif added_answer_tokens > dropped_answer_tokens:
        print(f"   ✅ Ada is HELPING answer retention!")
    else:
        print(f"   ➖ Ada has neutral effect on answer retention")
    
    return {
        'dropped': dropped_answer_tokens,
        'added': added_answer_tokens,
        'dropped_heads': dropped_heads,
        'added_heads': added_heads
    }


def compare_across_layers(baseline_path: str, ada_path: str, 
                          sample_id: int = 0,
                          output_path: str = None):
    """
    对比所有层的差异，生成层级汇总
    """
    baseline = load_diagnostic(baseline_path)
    ada = load_diagnostic(ada_path)
    
    # Find all layers for this sample
    baseline_layers = set(r['layer_idx'] for r in baseline if r['sample_id'] == sample_id)
    ada_layers = set(r['layer_idx'] for r in ada if r['sample_id'] == sample_id)
    common_layers = sorted(baseline_layers & ada_layers)
    
    if not common_layers:
        print(f"No common layers found for sample {sample_id}")
        return
    
    print(f"\n{'='*70}")
    print(f"Cross-Layer Analysis for Sample {sample_id}")
    print(f"{'='*70}")
    
    layer_stats = []
    
    for layer_idx in common_layers:
        head_diffs, seq_len = compare_two_runs(baseline_path, ada_path, sample_id, layer_idx)
        if head_diffs:
            avg_jaccard = np.mean([d['jaccard'] for d in head_diffs])
            total_dropped = sum(len(d['only_baseline']) for d in head_diffs)
            total_added = sum(len(d['only_ada']) for d in head_diffs)
            avg_entropy = np.mean([d['ada_entropy'] for d in head_diffs])
            
            layer_stats.append({
                'layer_idx': layer_idx,
                'avg_jaccard': avg_jaccard,
                'total_dropped': total_dropped,
                'total_added': total_added,
                'avg_entropy': avg_entropy
            })
    
    # Print summary table
    print(f"\n📊 Layer-wise Summary:")
    print(f"{'Layer':>5} | {'Jaccard':>7} | {'Dropped':>8} | {'Added':>8} | {'Entropy':>8}")
    print("-" * 50)
    for s in layer_stats:
        print(f"{s['layer_idx']:>5} | {s['avg_jaccard']:>7.3f} | {s['total_dropped']:>8} | {s['total_added']:>8} | {s['avg_entropy']:>8.3f}")
    
    # Plot layer-wise summary
    if output_path and layer_stats:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        layers = [s['layer_idx'] for s in layer_stats]
        
        # Jaccard similarity across layers
        ax1 = axes[0, 0]
        ax1.plot(layers, [s['avg_jaccard'] for s in layer_stats], 'b.-', linewidth=2)
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Average Jaccard Similarity')
        ax1.set_title('Token Overlap: Baseline vs Ada')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Dropped vs Added tokens
        ax2 = axes[0, 1]
        ax2.bar([l - 0.2 for l in layers], [s['total_dropped'] for s in layer_stats], 
                width=0.4, label='Dropped by Ada', color='red', alpha=0.7)
        ax2.bar([l + 0.2 for l in layers], [s['total_added'] for s in layer_stats], 
                width=0.4, label='Added by Ada', color='green', alpha=0.7)
        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Token Count')
        ax2.set_title('Token Changes per Layer')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Average entropy per layer
        ax3 = axes[1, 0]
        ax3.plot(layers, [s['avg_entropy'] for s in layer_stats], 'g.-', linewidth=2)
        ax3.set_xlabel('Layer Index')
        ax3.set_ylabel('Average Head Entropy')
        ax3.set_title('Head Entropy Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Net change (added - dropped)
        ax4 = axes[1, 1]
        net_change = [s['total_added'] - s['total_dropped'] for s in layer_stats]
        colors = ['green' if n >= 0 else 'red' for n in net_change]
        ax4.bar(layers, net_change, color=colors, alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.set_xlabel('Layer Index')
        ax4.set_ylabel('Net Token Change (Added - Dropped)')
        ax4.set_title('Net Token Change per Layer')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n📈 Cross-layer visualization saved to: {output_path}")
        plt.close()
    
    return layer_stats


def visualize_single_sample_detail(baseline_path: str, ada_path: str,
                                   sample_id: int, 
                                   input_text: str = None,
                                   answer_span: tuple = None,
                                   tokenizer_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    """
    详细可视化单个样本
    
    input_text: 原始输入文本
    answer_span: (start_char, end_char) 答案在文本中的位置
    """
    print(f"\n📝 Sample {sample_id} Detail Analysis")
    
    answer_positions = None
    
    if input_text and answer_span:
        try:
            from transformers import AutoTokenizer
            
            # 加载 tokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            
            # Tokenize
            tokens = tokenizer.encode(input_text)
            token_strs = [tokenizer.decode([t]) for t in tokens]
            
            print(f"   Total tokens: {len(tokens)}")
            
            # 如果有答案位置，找到对应的 token 位置
            # 简化处理：遍历找到答案
            char_to_token = []
            current_char = 0
            for i, t_str in enumerate(token_strs):
                for _ in t_str:
                    char_to_token.append(i)
                    current_char += 1
            
            answer_start_token = char_to_token[min(answer_span[0], len(char_to_token)-1)]
            answer_end_token = char_to_token[min(answer_span[1], len(char_to_token)-1)]
            answer_positions = list(range(answer_start_token, answer_end_token + 1))
            
            print(f"   Answer token positions: {answer_positions}")
            print(f"   Answer tokens: {token_strs[answer_start_token:answer_end_token+1]}")
        except Exception as e:
            print(f"   Warning: Could not tokenize: {e}")
    
    # 加载并对比
    head_diffs, seq_len = compare_two_runs(baseline_path, ada_path, 
                                           sample_id=sample_id, layer_idx=0)
    
    if head_diffs and answer_positions:
        analyze_answer_overlap(head_diffs, answer_positions)
    
    if head_diffs:
        visualize_token_retention(head_diffs, seq_len, 
                                  f"diagnostic_sample{sample_id}.png")


# === 主入口 ===
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HACE Diagnostic Visualization")
    parser.add_argument("--baseline", type=str, required=True,
                        help="Path to baseline diagnostic JSON")
    parser.add_argument("--ada", type=str, required=True,
                        help="Path to Ada diagnostic JSON")
    parser.add_argument("--sample", type=int, default=0,
                        help="Sample ID to analyze")
    parser.add_argument("--layer", type=int, default=0,
                        help="Layer index to analyze")
    parser.add_argument("--output", type=str, default="diagnostic.png",
                        help="Output image path")
    parser.add_argument("--all-layers", action="store_true",
                        help="Generate cross-layer analysis")
    parser.add_argument("--answer-positions", type=str, default=None,
                        help="Comma-separated answer token positions (e.g., '100,101,102')")
    
    args = parser.parse_args()
    
    if args.all_layers:
        # Cross-layer analysis
        compare_across_layers(
            args.baseline, args.ada,
            sample_id=args.sample,
            output_path=args.output.replace('.png', '_layers.png')
        )
    else:
        # Single layer analysis
        head_diffs, seq_len = compare_two_runs(
            args.baseline, args.ada,
            sample_id=args.sample,
            layer_idx=args.layer
        )
        
        if head_diffs:
            visualize_token_retention(head_diffs, seq_len, args.output)
            
            # Answer position analysis if provided
            if args.answer_positions:
                positions = [int(p.strip()) for p in args.answer_positions.split(',')]
                analyze_answer_overlap(head_diffs, positions)
