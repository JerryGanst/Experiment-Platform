'''
H2O实验结果报告生成脚本

生成包含表格、图表和分析文本的完整实验报告
'''
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import jinja2

# 导入表格生成模块
from generate_tables import (
    load_experiment_results,
    create_performance_summary_table,
    create_language_comparison_table,
    create_best_config_table
)

def create_memory_comparison_plots(baseline_df, h2o_df, output_dir):
    """
    创建内存使用对比图
    
    Args:
        baseline_df: 基线数据框
        h2o_df: H2O数据框
        output_dir: 输出目录
    
    Returns:
        plot_paths: 图表文件路径列表
    """
    plot_paths = []
    
    # 创建KV缓存长度对内存使用的影响图
    plt.figure(figsize=(12, 8))
    
    # 合并数据以便于绘图
    if h2o_df is not None:
        baseline_df = baseline_df.copy()
        baseline_df['配置'] = '基线'
        
        h2o_df = h2o_df.copy()
        h2o_df['配置'] = h2o_df['heavy_ratio'].apply(lambda x: f'H2O-{int(x*100)}%')
        
        combined_df = pd.concat([baseline_df, h2o_df])
    else:
        baseline_df = baseline_df.copy()
        baseline_df['配置'] = '基线'
        combined_df = baseline_df
    
    # 按KV缓存长度分组计算平均内存使用
    grouped = combined_df.groupby(['配置', 'kv_cache_length']).agg({
        'peak_gpu_memory_mb': 'mean'
    }).reset_index()
    
    # 为每个配置绘制一条线
    for config in sorted(grouped['配置'].unique()):
        config_data = grouped[grouped['配置'] == config]
        plt.plot(
            config_data['kv_cache_length'],
            config_data['peak_gpu_memory_mb'],
            marker='o',
            label=config
        )
    
    plt.title('KV缓存长度对内存使用的影响')
    plt.xlabel('KV缓存长度')
    plt.ylabel('峰值GPU内存 (MB)')
    plt.grid(True)
    plt.legend()
    
    # 保存图表
    memory_kv_path = os.path.join(output_dir, 'memory_vs_kv_length.png')
    plt.savefig(memory_kv_path, dpi=300)
    plt.close()
    plot_paths.append(memory_kv_path)
    
    # 创建批处理大小对内存使用的影响图
    plt.figure(figsize=(12, 8))
    
    # 按批处理大小分组计算平均内存使用
    grouped = combined_df.groupby(['配置', 'batch_size']).agg({
        'peak_gpu_memory_mb': 'mean'
    }).reset_index()
    
    # 为每个配置绘制一条线
    for config in sorted(grouped['配置'].unique()):
        config_data = grouped[grouped['配置'] == config]
        plt.plot(
            config_data['batch_size'],
            config_data['peak_gpu_memory_mb'],
            marker='o',
            label=config
        )
    
    plt.title('批处理大小对内存使用的影响')
    plt.xlabel('批处理大小')
    plt.ylabel('峰值GPU内存 (MB)')
    plt.grid(True)
    plt.legend()
    
    # 保存图表
    memory_bs_path = os.path.join(output_dir, 'memory_vs_batch_size.png')
    plt.savefig(memory_bs_path, dpi=300)
    plt.close()
    plot_paths.append(memory_bs_path)
    
    # 创建内存节省热力图
    if h2o_df is not None:
        # 在不同KV缓存长度和批处理大小下的内存节省热力图
        plt.figure(figsize=(14, 10))
        
        # 计算各配置下的内存节省
        memory_savings = []
        
        for kv_length in sorted(combined_df['kv_cache_length'].unique()):
            for bs in sorted(combined_df['batch_size'].unique()):
                # 获取基线内存
                baseline_mem = baseline_df[
                    (baseline_df['kv_cache_length'] == kv_length) &
                    (baseline_df['batch_size'] == bs)
                ]['peak_gpu_memory_mb'].mean()
                
                # 获取不同heavy_ratio下的H2O内存
                for ratio in sorted(h2o_df['heavy_ratio'].unique()):
                    h2o_mem = h2o_df[
                        (h2o_df['kv_cache_length'] == kv_length) &
                        (h2o_df['batch_size'] == bs) &
                        (h2o_df['heavy_ratio'] == ratio)
                    ]['peak_gpu_memory_mb'].mean()
                    
                    if not np.isnan(baseline_mem) and not np.isnan(h2o_mem):
                        saving = (baseline_mem - h2o_mem) / baseline_mem * 100
                        memory_savings.append({
                            'KV缓存长度': kv_length,
                            '批处理大小': bs,
                            'Heavy Ratio': f'{int(ratio*100)}%',
                            '内存节省(%)': saving
                        })
        
        if memory_savings:
            savings_df = pd.DataFrame(memory_savings)
            
            # 创建透视表
            pivot = pd.pivot_table(
                savings_df,
                values='内存节省(%)',
                index='批处理大小',
                columns=['KV缓存长度', 'Heavy Ratio'],
                aggfunc='mean'
            )
            
            sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu")
            plt.title('内存节省热力图 (%)，按KV缓存长度、批处理大小和Heavy Ratio')
            
            # 保存图表
            memory_heatmap_path = os.path.join(output_dir, 'memory_saving_heatmap.png')
            plt.savefig(memory_heatmap_path, dpi=300)
            plt.close()
            plot_paths.append(memory_heatmap_path)
    
    return plot_paths

def create_latency_plots(baseline_df, h2o_df, output_dir):
    """
    创建延迟对比图
    
    Args:
        baseline_df: 基线数据框
        h2o_df: H2O数据框
        output_dir: 输出目录
    
    Returns:
        plot_paths: 图表文件路径列表
    """
    plot_paths = []
    
    # 合并数据以便于绘图
    if h2o_df is not None:
        baseline_df = baseline_df.copy()
        baseline_df['配置'] = '基线'
        
        h2o_df = h2o_df.copy()
        h2o_df['配置'] = h2o_df['heavy_ratio'].apply(lambda x: f'H2O-{int(x*100)}%')
        
        combined_df = pd.concat([baseline_df, h2o_df])
    else:
        baseline_df = baseline_df.copy()
        baseline_df['配置'] = '基线'
        combined_df = baseline_df
    
    # 创建TTFT图表
    plt.figure(figsize=(12, 8))
    
    # 按批处理大小分组计算平均TTFT
    grouped = combined_df.groupby(['配置', 'batch_size']).agg({
        'ttft_ms': 'mean'
    }).reset_index()
    
    # 为每个配置绘制一条线
    for config in sorted(grouped['配置'].unique()):
        config_data = grouped[grouped['配置'] == config]
        plt.plot(
            config_data['batch_size'],
            config_data['ttft_ms'],
            marker='o',
            label=config
        )
    
    plt.title('批处理大小对首个令牌生成时间的影响')
    plt.xlabel('批处理大小')
    plt.ylabel('TTFT (ms)')
    plt.grid(True)
    plt.legend()
    
    # 保存图表
    ttft_path = os.path.join(output_dir, 'ttft_vs_batch_size.png')
    plt.savefig(ttft_path, dpi=300)
    plt.close()
    plot_paths.append(ttft_path)
    
    # 创建TPOT图表
    plt.figure(figsize=(12, 8))
    
    # 按批处理大小分组计算平均TPOT
    grouped = combined_df.groupby(['配置', 'batch_size']).agg({
        'tpot_ms': 'mean'
    }).reset_index()
    
    # 为每个配置绘制一条线
    for config in sorted(grouped['配置'].unique()):
        config_data = grouped[grouped['配置'] == config]
        plt.plot(
            config_data['batch_size'],
            config_data['tpot_ms'],
            marker='o',
            label=config
        )
    
    plt.title('批处理大小对每令牌生成时间的影响')
    plt.xlabel('批处理大小')
    plt.ylabel('TPOT (ms)')
    plt.grid(True)
    plt.legend()
    
    # 保存图表
    tpot_path = os.path.join(output_dir, 'tpot_vs_batch_size.png')
    plt.savefig(tpot_path, dpi=300)
    plt.close()
    plot_paths.append(tpot_path)
    
    # 创建吞吐量图表
    plt.figure(figsize=(12, 8))
    
    # 按批处理大小分组计算平均吞吐量
    grouped = combined_df.groupby(['配置', 'batch_size']).agg({
        'throughput_tokens_per_sec': 'mean'
    }).reset_index()
    
    # 为每个配置绘制一条线
    for config in sorted(grouped['配置'].unique()):
        config_data = grouped[grouped['配置'] == config]
        plt.plot(
            config_data['batch_size'],
            config_data['throughput_tokens_per_sec'],
            marker='o',
            label=config
        )
    
    plt.title('批处理大小对吞吐量的影响')
    plt.xlabel('批处理大小')
    plt.ylabel('吞吐量 (tokens/sec)')
    plt.grid(True)
    plt.legend()
    
    # 保存图表
    throughput_path = os.path.join(output_dir, 'throughput_vs_batch_size.png')
    plt.savefig(throughput_path, dpi=300)
    plt.close()
    plot_paths.append(throughput_path)
    
    return plot_paths

def create_tradeoff_plot(baseline_df, h2o_df, output_dir):
    """
    创建内存-质量权衡散点图
    
    Args:
        baseline_df: 基线数据框
        h2o_df: H2O数据框
        output_dir: 输出目录
    
    Returns:
        plot_path: 图表文件路径
    """
    # 如果没有H2O数据，则无法创建权衡图
    if h2o_df is None:
        return None
    
    plt.figure(figsize=(12, 8))
    
    # 计算各配置下的内存节省和吞吐量变化
    tradeoff_data = []
    
    for kv_length in sorted(baseline_df['kv_cache_length'].unique()):
        for bs in sorted(baseline_df['batch_size'].unique()):
            # 获取基线性能
            baseline_metrics = baseline_df[
                (baseline_df['kv_cache_length'] == kv_length) &
                (baseline_df['batch_size'] == bs)
            ].agg({
                'peak_gpu_memory_mb': 'mean',
                'throughput_tokens_per_sec': 'mean'
            })
            
            # 获取不同heavy_ratio下的H2O性能
            for ratio in sorted(h2o_df['heavy_ratio'].unique()):
                h2o_metrics = h2o_df[
                    (h2o_df['kv_cache_length'] == kv_length) &
                    (h2o_df['batch_size'] == bs) &
                    (h2o_df['heavy_ratio'] == ratio)
                ].agg({
                    'peak_gpu_memory_mb': 'mean',
                    'throughput_tokens_per_sec': 'mean'
                })
                
                if not baseline_metrics.empty and not h2o_metrics.empty:
                    memory_saving = ((baseline_metrics['peak_gpu_memory_mb'] - h2o_metrics['peak_gpu_memory_mb']) / 
                                   baseline_metrics['peak_gpu_memory_mb'] * 100)
                    
                    throughput_change = ((h2o_metrics['throughput_tokens_per_sec'] - baseline_metrics['throughput_tokens_per_sec']) / 
                                      baseline_metrics['throughput_tokens_per_sec'] * 100)
                    
                    tradeoff_data.append({
                        'KV缓存长度': kv_length,
                        '批处理大小': bs,
                        'Heavy Ratio': ratio,
                        '内存节省(%)': memory_saving,
                        '吞吐量变化(%)': throughput_change
                    })
    
    if not tradeoff_data:
        return None
    
    # 创建数据框
    tradeoff_df = pd.DataFrame(tradeoff_data)
    
    # 绘制散点图
    sns.scatterplot(
        data=tradeoff_df,
        x='内存节省(%)',
        y='吞吐量变化(%)',
        hue='Heavy Ratio',
        size='批处理大小',
        sizes=(50, 200),
        alpha=0.7
    )
    
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
    
    plt.title('内存节省与吞吐量变化权衡图')
    plt.grid(True, alpha=0.3)
    
    # 添加注释，标记最佳点
    best_memory = tradeoff_df.loc[tradeoff_df['内存节省(%)'].idxmax()]
    best_throughput = tradeoff_df.loc[tradeoff_df['吞吐量变化(%)'].idxmax()]
    best_balanced = tradeoff_df.loc[(tradeoff_df['内存节省(%)'] + tradeoff_df['吞吐量变化(%)']).idxmax()]
    
    plt.annotate(
        f"最大内存节省\nKV={best_memory['KV缓存长度']}, BS={best_memory['批处理大小']}, HR={best_memory['Heavy Ratio']:.1f}",
        xy=(best_memory['内存节省(%)'], best_memory['吞吐量变化(%)']),
        xytext=(10, 10),
        textcoords='offset points',
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2')
    )
    
    plt.annotate(
        f"最大吞吐量提升\nKV={best_throughput['KV缓存长度']}, BS={best_throughput['批处理大小']}, HR={best_throughput['Heavy Ratio']:.1f}",
        xy=(best_throughput['内存节省(%)'], best_throughput['吞吐量变化(%)']),
        xytext=(10, -10),
        textcoords='offset points',
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2')
    )
    
    plt.annotate(
        f"最佳平衡点\nKV={best_balanced['KV缓存长度']}, BS={best_balanced['批处理大小']}, HR={best_balanced['Heavy Ratio']:.1f}",
        xy=(best_balanced['内存节省(%)'], best_balanced['吞吐量变化(%)']),
        xytext=(-10, 10),
        textcoords='offset points',
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2')
    )
    
    # 保存图表
    tradeoff_path = os.path.join(output_dir, 'memory_throughput_tradeoff.png')
    plt.savefig(tradeoff_path, dpi=300)
    plt.close()
    
    return tradeoff_path

def generate_html_report(tables, plots, output_dir):
    """
    生成HTML报告
    
    Args:
        tables: 表格字典，键为表格名称，值为数据框
        plots: 图表文件路径列表
        output_dir: 输出目录
    """
    # HTML模板
    template_str = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>H2O缓存优化技术实验报告</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            h1, h2, h3 {
                color: #2c3e50;
            }
            h1 {
                text-align: center;
                border-bottom: 2px solid #eee;
                padding-bottom: 10px;
            }
            .section {
                margin-bottom: 30px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            .plot {
                margin: 20px 0;
                text-align: center;
            }
            .plot img {
                max-width: 100%;
                height: auto;
            }
            .footer {
                text-align: center;
                margin-top: 30px;
                padding-top: 10px;
                border-top: 1px solid #eee;
                font-size: 0.9em;
                color: #777;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>H2O缓存优化技术实验报告</h1>
            
            <div class="section">
                <h2>实验概述</h2>
                <p>本实验评估了Heavy-Hitter Oracle (H2O) 缓存优化技术对大语言模型推理性能的影响。H2O作为一种KV缓存驱逐策略，通过动态保留重要命中(Heavy Hitter)令牌和最近使用的令牌，以减少内存占用并维持模型性能。</p>
                <p>实验使用了OPT系列模型，在不同KV缓存长度和批处理大小下，比较了标准KV缓存和H2O优化KV缓存的性能差异。</p>
            </div>
            
            <div class="section">
                <h2>性能指标汇总</h2>
                {{ main_table_html }}
                
                <div class="plot">
                    <h3>内存使用对比</h3>
                    {% if 'memory_vs_kv_length.png' in plots %}
                    <img src="{{ plots['memory_vs_kv_length.png'] }}" alt="KV缓存长度对内存使用的影响">
                    {% endif %}
                    
                    {% if 'memory_vs_batch_size.png' in plots %}
                    <img src="{{ plots['memory_vs_batch_size.png'] }}" alt="批处理大小对内存使用的影响">
                    {% endif %}
                    
                    {% if 'memory_saving_heatmap.png' in plots %}
                    <img src="{{ plots['memory_saving_heatmap.png'] }}" alt="内存节省热力图">
                    {% endif %}
                </div>
                
                <div class="plot">
                    <h3>延迟和吞吐量对比</h3>
                    {% if 'ttft_vs_batch_size.png' in plots %}
                    <img src="{{ plots['ttft_vs_batch_size.png'] }}" alt="批处理大小对TTFT的影响">
                    {% endif %}
                    
                    {% if 'tpot_vs_batch_size.png' in plots %}
                    <img src="{{ plots['tpot_vs_batch_size.png'] }}" alt="批处理大小对TPOT的影响">
                    {% endif %}
                    
                    {% if 'throughput_vs_batch_size.png' in plots %}
                    <img src="{{ plots['throughput_vs_batch_size.png'] }}" alt="批处理大小对吞吐量的影响">
                    {% endif %}
                </div>
                
                {% if 'memory_throughput_tradeoff.png' in plots %}
                <div class="plot">
                    <h3>内存-吞吐量权衡分析</h3>
                    <img src="{{ plots['memory_throughput_tradeoff.png'] }}" alt="内存节省与吞吐量变化权衡图">
                </div>
                {% endif %}
            </div>
            
            <div class="section">
                <h2>不同语言性能对比</h2>
                {{ language_table_html }}
            </div>
            
            <div class="section">
                <h2>最佳配置推荐</h2>
                {{ config_table_html }}
                <p>以上配置推荐基于实验结果，具体应用时应考虑实际场景需求和硬件限制。</p>
            </div>
            
            <div class="section">
                <h2>结论与建议</h2>
                <ul>
                    <li>H2O缓存优化在保持模型性能的同时，显著减少了内存使用，在长序列场景中效果尤为明显。</li>
                    <li>不同的Heavy Hitter比例设置适合不同的应用场景，实时应用应选择较高的比例，批处理场景可选择较低的比例。</li>
                    <li>KV缓存长度和批处理大小对H2O优化效果有显著影响，应根据具体需求调整这些参数。</li>
                    <li>H2O优化在降低内存使用的同时，在某些配置下还能提高吞吐量，实现双重优化效果。</li>
                </ul>
            </div>
            
            <div class="footer">
                <p>报告生成时间: {{ generation_time }}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # 创建Jinja2模板
    template = jinja2.Template(template_str)
    
    # 准备表格HTML
    tables_html = {}
    for name, df in tables.items():
        tables_html[name] = df.to_html(index=False)
    
    # 准备图表路径字典
    plots_dict = {}
    for plot_path in plots:
        if plot_path:
            base_name = os.path.basename(plot_path)
            plots_dict[base_name] = plot_path
    
    # 渲染HTML
    html = template.render(
        main_table_html=tables_html['主要性能指标汇总表'],
        language_table_html=tables_html['不同语言性能对比表'],
        config_table_html=tables_html['最佳配置推荐表'],
        plots=plots_dict,
        generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    # 保存HTML报告
    output_path = os.path.join(output_dir, "h2o_experiment_report.html")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"HTML report saved to {output_path}")
    return output_path

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Generate H2O Experiment Report")
    parser.add_argument("--baseline_dir", type=str, required=True,
                       help="Directory containing baseline experiment results")
    parser.add_argument("--h2o_dir", type=str, default=None,
                       help="Directory containing H2O experiment results (optional)")
    parser.add_argument("--output_dir", type=str, default="./report",
                       help="Directory to save report")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 加载实验结果
    baseline_df, h2o_df = load_experiment_results(args.baseline_dir, args.h2o_dir)
    
    # 创建表格
    tables = {
        "主要性能指标汇总表": create_performance_summary_table(baseline_df, h2o_df),
        "不同语言性能对比表": create_language_comparison_table(baseline_df, h2o_df),
        "最佳配置推荐表": create_best_config_table(baseline_df, h2o_df)
    }
    
    # 创建图表
    plots = []
    plots.extend(create_memory_comparison_plots(baseline_df, h2o_df, plots_dir))
    plots.extend(create_latency_plots(baseline_df, h2o_df, plots_dir))
    
    # 创建权衡图（仅当有H2O数据时）
    if h2o_df is not None:
        tradeoff_plot = create_tradeoff_plot(baseline_df, h2o_df, plots_dir)
        if tradeoff_plot:
            plots.append(tradeoff_plot)
    
    # 生成HTML报告
    report_path = generate_html_report(tables, plots, args.output_dir)
    
    # 保存Excel版表格
    excel_path = os.path.join(args.output_dir, "h2o_experiment_tables.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        for name, df in tables.items():
            df.to_excel(writer, sheet_name=name.replace("表", ""), index=False)
    
    print(f"Report generation completed! Excel tables saved to {excel_path}")

if __name__ == "__main__":
    main() 