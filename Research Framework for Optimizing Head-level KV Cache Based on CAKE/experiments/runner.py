import os
import sys
from pathlib import Path
import yaml
import pandas as pd

# Add project root to sys.path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import config
from baseline_main import run_baseline_experiment
from cake_main import run_cake_experiment

class BaseMethod:
    """Common interface for experiment methods."""

    def __init__(self, model_config, batch_size, max_new_tokens, output_dir):
        self.model_config = model_config
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.output_dir = output_dir

    def run(self, dataset, kv_length, budget=None):
        raise NotImplementedError

class BaselineMethod(BaseMethod):
    """Standard KV cache baseline."""

    def run(self, dataset, kv_length, budget=None):
        dataset_cfg = config.DATASET_CONFIG["available_datasets"].get(dataset)
        if not dataset_cfg:
            raise ValueError(f"Unknown dataset: {dataset}")
        return run_baseline_experiment(
            model_config=self.model_config,
            dataset_name=dataset,
            dataset_config=dataset_cfg,
            kv_cache_length=kv_length,
            batch_size=self.batch_size,
            max_new_tokens=self.max_new_tokens,
            output_dir=self.output_dir,
            repeat_index=0,
        )

class HACEMethod(BaseMethod):
    """HACE method using CAKE as underlying implementation."""

    def __init__(self, model_config, batch_size, max_new_tokens, output_dir, allocation_strategy):
        super().__init__(model_config, batch_size, max_new_tokens, output_dir)
        self.allocation_strategy = allocation_strategy

    def run(self, dataset, kv_length, budget):
        dataset_cfg = config.DATASET_CONFIG["available_datasets"].get(dataset)
        if not dataset_cfg:
            raise ValueError(f"Unknown dataset: {dataset}")
        return run_cake_experiment(
            experiment_main_config=self.model_config,
            dataset_name=dataset,
            dataset_options=dataset_cfg,
            kv_cache_length=kv_length,
            batch_size=self.batch_size,
            max_new_tokens=self.max_new_tokens,
            allocation_strategy=self.allocation_strategy,
            cache_budget=budget,
            output_dir=self.output_dir,
            repeat_index=0,
        )

def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    cfg_path = Path(project_root) / "experiments" / "config.yaml"
    cfg = load_config(cfg_path)

    output_dir = Path(cfg.get("output_dir", "results/raw"))
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = {
        "model_name_or_path": cfg.get("model_name_or_path", config.MODEL_CONFIG["model_name_or_path"]),
        "precision": cfg.get("precision", config.MODEL_CONFIG.get("precision", "fp16")),
    }
    batch_size = cfg.get("batch_size", 1)
    max_new_tokens = cfg.get("max_new_tokens", config.EXPERIMENT_CONFIG.get("max_new_tokens", 128))

    methods = []
    methods_cfg = cfg.get("methods", {})
    if methods_cfg.get("baseline"):
        methods.append(("baseline", BaselineMethod(model_config, batch_size, max_new_tokens, output_dir)))
    hace_cfg = methods_cfg.get("hace", {})
    if hace_cfg.get("enabled"):
        strategy = hace_cfg.get("allocation_strategy", "adaptive")
        methods.append(("hace", HACEMethod(model_config, batch_size, max_new_tokens, output_dir, strategy)))

    datasets = cfg.get("datasets", [])
    kv_lengths = cfg.get("kv_cache_lengths", [])
    budgets = cfg.get("cache_budgets", [1.0])

    results = []
    for dataset in datasets:
        for kv_len in kv_lengths:
            for name, method in methods:
                if name == "baseline":
                    metrics = method.run(dataset, kv_len)
                    metrics["method"] = name
                    metrics["cache_budget"] = 1.0
                    results.append(metrics)
                else:
                    for bud in budgets:
                        metrics = method.run(dataset, kv_len, bud)
                        metrics["method"] = name
                        metrics["cache_budget"] = bud
                        results.append(metrics)

    df = pd.DataFrame(results)
    csv_path = output_dir / "all_experiments.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    main()
