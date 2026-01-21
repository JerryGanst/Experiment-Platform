"""
批量实验调度器 - 核心调度逻辑
"""
import os
import sys
import json
import logging
import datetime
import argparse
from pathlib import Path
import pandas as pd
import yaml

from .monitor_wrapper import UnifiedMonitor

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

class BatchRunner:
    """批量实验调度器"""
    
    def __init__(self, configs_dir: str = "configs", runs_dir: str = "runs"):
        self.configs_dir = Path(configs_dir)
        self.runs_dir = Path(runs_dir)
        
        # 确保目录存在
        self.runs_dir.mkdir(exist_ok=True)
        
        # 加载配置
        self.datasets = self._load_datasets()
        self.kv_lengths = self._load_kv_lengths()
        self.cake_grid = self._load_cake_grid()
        
        # 模型相关
        self.model = None
        self.tokenizer = None
        
    def _load_datasets(self) -> pd.DataFrame:
        """加载数据集配置"""
        datasets_file = self.configs_dir / "datasets.csv"
        if not datasets_file.exists():
            raise FileNotFoundError(f"数据集配置文件不存在: {datasets_file}")
        return pd.read_csv(datasets_file)
    
    def _load_kv_lengths(self) -> list:
        """加载KV长度配置"""
        kv_file = self.configs_dir / "kv_lengths.yaml"
        if not kv_file.exists():
            raise FileNotFoundError(f"KV长度配置文件不存在: {kv_file}")
        
        with open(kv_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config['kv_lengths']
    
    def _load_cake_grid(self) -> pd.DataFrame:
        """加载CAKE策略配置"""
        grid_file = self.configs_dir / "cake_grid.csv"
        if not grid_file.exists():
            raise FileNotFoundError(f"CAKE策略配置文件不存在: {grid_file}")
        return pd.read_csv(grid_file)
    
    def load_model(self, model_path: str):
        """加载模型（只加载一次）"""
        logger.info(f"正在加载模型: {model_path}")
        
        try:
            # 这里需要根据实际的模型加载逻辑进行调整
            # 导入现有的模型加载代码
            from hacer_core.models import load_model_and_tokenizer
            self.model, self.tokenizer = load_model_and_tokenizer(model_path)
            logger.info("模型加载成功")
        except ImportError:
            logger.warning("无法导入模型加载函数，使用占位符")
            self.model = "placeholder_model"
            self.tokenizer = "placeholder_tokenizer"
    
    def apply_strategy(self, strategy: str, budget: float):
        """应用缓存策略"""
        if strategy == "fullkv":
            self._reset_fullkv()
        else:
            self._apply_cake(strategy, budget)
    
    def _reset_fullkv(self):
        """重置为Full KV缓存"""
        logger.info("切换到Full KV策略")
        # 这里需要实现实际的Full KV重置逻辑
        pass
    
    def _apply_cake(self, strategy: str, budget: float):
        """应用CAKE策略"""
        logger.info(f"应用CAKE策略: {strategy}, budget={budget}")
        # 这里需要实现实际的CAKE应用逻辑
        pass
    
    def run_single_dataset(self, dataset: str, kv_length: int, output_path: str):
        """运行单个数据集的实验"""
        logger.info(f"运行数据集: {dataset}, KV长度: {kv_length}")
        
        # 这里需要实现实际的数据集运行逻辑
        # 暂时创建一个占位符输出
        output_data = {
            "dataset": dataset,
            "kv_length": kv_length,
            "results": "placeholder_results",
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    def run_batch_experiments(self, tag: str = "experiment", 
                            model_path: str = "checkpoints/mistral-7b-instruct-v0.3",
                            selected_datasets: list = None,
                            selected_strategies: list = None):
        """运行批量实验"""
        
        # 创建运行目录
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{timestamp}_{tag}"
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(exist_ok=True)
        
        logger.info(f"开始批量实验: {run_id}")
        logger.info(f"输出目录: {run_dir}")
        
        # 加载模型（只加载一次）
        self.load_model(model_path)
        
        # 过滤数据集和策略
        datasets_to_run = self.datasets
        if selected_datasets:
            datasets_to_run = datasets_to_run[datasets_to_run['dataset'].isin(selected_datasets)]
        
        strategies_to_run = self.cake_grid
        if selected_strategies:
            strategies_to_run = strategies_to_run[strategies_to_run['strategy'].isin(selected_strategies)]
        
        # 执行实验网格
        total_experiments = len(datasets_to_run) * len(self.kv_lengths) * len(strategies_to_run)
        logger.info(f"总实验数量: {total_experiments}")
        
        experiment_count = 0
        
        for _, dataset_row in datasets_to_run.iterrows():
            dataset = dataset_row['dataset']
            
            for kv_length in self.kv_lengths:
                for _, strategy_row in strategies_to_run.iterrows():
                    strategy = strategy_row['strategy']
                    budget = strategy_row['budget']
                    
                    experiment_count += 1
                    
                    # 创建实验子目录
                    exp_name = f"kv{kv_length}_{strategy}_{int(budget*100):03d}"
                    exp_dir = run_dir / exp_name
                    exp_dir.mkdir(exist_ok=True)
                    
                    logger.info(f"[{experiment_count}/{total_experiments}] 运行实验: {exp_name}")
                    
                    # 应用策略
                    self.apply_strategy(strategy, budget)
                    
                    # 监控和运行
                    with UnifiedMonitor(exp_name) as monitor:
                        self.run_single_dataset(
                            dataset=dataset,
                            kv_length=kv_length,
                            output_path=str(exp_dir / "outputs.jsonl")
                        )
                        
                        # 添加实验参数到监控
                        monitor.add_metric("dataset", dataset)
                        monitor.add_metric("kv_length", kv_length)
                        monitor.add_metric("strategy", strategy)
                        monitor.add_metric("budget", budget)
                    
                    # 保存监控数据
                    monitor.save_metrics(str(exp_dir / "metrics.json"))
                    
                    # 保存日志
                    with open(exp_dir / "logs.txt", 'w', encoding='utf-8') as f:
                        f.write(monitor.summary())
        
        logger.info(f"批量实验完成: {run_id}")
        return run_dir

def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="批量实验调度器")
    parser.add_argument("--tag", default="experiment", help="实验标签")
    parser.add_argument("--model", default="checkpoints/mistral-7b-instruct-v0.3", help="模型路径")
    parser.add_argument("--datasets", nargs="+", help="选择的数据集")
    parser.add_argument("--strategies", nargs="+", help="选择的策略")
    parser.add_argument("--configs-dir", default="configs", help="配置目录")
    parser.add_argument("--runs-dir", default="runs", help="运行输出目录")
    parser.add_argument("--log-level", default="INFO", help="日志级别")
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建并运行批处理器
    runner = BatchRunner(configs_dir=args.configs_dir, runs_dir=args.runs_dir)
    
    try:
        run_dir = runner.run_batch_experiments(
            tag=args.tag,
            model_path=args.model,
            selected_datasets=args.datasets,
            selected_strategies=args.strategies
        )
        print(f"实验完成，结果保存在: {run_dir}")
    except Exception as e:
        logger.error(f"实验执行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 