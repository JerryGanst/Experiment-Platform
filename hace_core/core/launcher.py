#!/usr/bin/env python3
"""
Cake-AdaKV 集成框架 – 启动器

该脚本提供一个简洁的命令行接口 (CLI) ，用于快速调用
`CakeAdaKVIntegration` 来完成缓存分配优化、性能自动调优等功能。

基本用法
--------
1. 使用合成数据运行优化：
   $ python src/core-code/launcher.py --synthetic

2. 使用现成的注意力权重 (NumPy .npy, 或包含多个 numpy.ndarray 的 *.npz)：
   $ python src/core-code/launcher.py --input /path/attn_weights.npy --output result.json

3. 开启监控与自动调优：
   $ python src/core-code/launcher.py --input weights.npz --monitor --auto-tune --detailed

参数说明
--------
--input/-i        :  指向 .npy 或 .npz 文件，若未指定将自动生成合成数据
--output/-o       :  输出 JSON 文件路径，默认打印到标准输出
--cache-size/-c   :  KV 缓存总大小 (默认 4096)
--monitor         :  启用性能监控
--auto-tune       :  启用自动调优
--detailed        :  返回详细结果(包含性能监控、分配详情)
--synthetic       :  使用合成数据(可搭配 --synthetic-* 参数)
--synthetic-layers:  合成数据层数 (默认 12)
--synthetic-heads :  合成数据头数 (默认 32)
--synthetic-seq   :  合成数据序列长度 (默认 512)
--synthetic-batch :  合成数据 batch 大小 (默认 1)

依赖
----
- numpy
- 本仓库 src/core-code 下的所有依赖模块

"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from typing import Any, List, Dict


# 尝试导入 numpy，如未安装则给出友好提示
try:
    import numpy as np  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    print("❌ 检测到未安装依赖: numpy\n   请先执行 `pip install -r requirements.txt` 安装所需依赖后再运行。")
    raise exc


# 尝试导入 yaml（可选依赖）
try:
    import yaml  # type: ignore
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# 动态修改 sys.path, 保证直接从仓库根目录执行时可以导入
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent  # /workspace/src
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 导入核心框架
try:
    # 直接顶级导入（与 example_usage.py 风格保持一致）
    from integration_framework import CakeAdaKVIntegration, IntegrationConfig  # type: ignore
except ImportError:  # pragma: no cover
    # 如果用户通过 "python -m core-code" 形式运行，则允许相对导入
    from .integration_framework import CakeAdaKVIntegration, IntegrationConfig  # type: ignore


def generate_synthetic_attention_weights(
    num_layers: int = 12,
    num_heads: int = 32,
    seq_len: int = 512,
    batch_size: int = 1,
) -> List[np.ndarray]:
    """生成合成注意力权重 (与 example_usage.py 中实现基本一致)。"""
    attention_weights_list: List[np.ndarray] = []
    for layer_idx in range(num_layers):
        if layer_idx < num_layers // 3:
            attn = np.random.exponential(0.1, (batch_size, num_heads, seq_len, seq_len))
        elif layer_idx < 2 * num_layers // 3:
            attn = np.random.gamma(2, 0.5, (batch_size, num_heads, seq_len, seq_len))
        else:
            attn = np.random.uniform(0, 1, (batch_size, num_heads, seq_len, seq_len))
        attn = attn / np.sum(attn, axis=-1, keepdims=True)
        attn = np.clip(attn, 1e-8, 1.0)
        attention_weights_list.append(attn)
    return attention_weights_list


def load_config_file(config_path: str) -> Dict[str, Any]:
    """从YAML文件加载配置"""
    if not YAML_AVAILABLE:
        print("⚠️  未安装PyYAML，无法加载配置文件。请运行: pip install pyyaml")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ 已加载配置文件: {config_path}")
        return config or {}
    except Exception as e:
        print(f"❌ 加载配置文件失败: {e}")
        return {}


def load_attention_weights(path: Path) -> List[Any]:
    """从 .npy 或 .npz 文件加载注意力权重数据。"""
    if not path.exists():
        raise FileNotFoundError(f"未找到输入文件: {path}")

    if path.suffix == ".npy":
        data = np.load(path, allow_pickle=True)
        # 若保存的是 ndarray list, data 可能本身就是对象数组
        if isinstance(data, np.ndarray) and data.dtype == object:
            return list(data)
        # 假设形状为 (L, B, H, S, S) ，拆分为 list
        if data.ndim == 5:
            return [data[i] for i in range(data.shape[0])]
        raise ValueError("无法解析 .npy 文件格式")

    elif path.suffix == ".npz":
        # 加载多个数组，按键排序后合并
        npz_data = np.load(path, allow_pickle=True)
        data_list = [npz_data[key] for key in sorted(npz_data.files)]
        return data_list
    else:
        raise ValueError("仅支持 .npy 或 .npz 文件")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cake-AdaKV Integration Launcher")


    # 配置文件选项（优先级最高）
    parser.add_argument("--config", type=str, help="YAML配置文件路径")

    io_group = parser.add_argument_group("I/O")
    io_group.add_argument("-i", "--input", type=str, help="注意力权重文件 (.npy/.npz)")
    io_group.add_argument("-o", "--output", type=str, help="输出 JSON 文件路径")

    cfg_group = parser.add_argument_group("Integration Config")
    cfg_group.add_argument("-c", "--cache-size", type=int, default=4096, help="总缓存大小")
    cfg_group.add_argument("--monitor", action="store_true", help="启用性能监控")
    cfg_group.add_argument("--auto-tune", action="store_true", help="启用自动调优")
    cfg_group.add_argument("--detailed", action="store_true", help="返回详细结果")

    synth_group = parser.add_argument_group("Synthetic Data")
    synth_group.add_argument("--synthetic", action="store_true", help="使用合成数据 (若未指定 --input，将自动启用)")
    synth_group.add_argument("--synthetic-layers", type=int, default=12, help="合成数据层数")
    synth_group.add_argument("--synthetic-heads", type=int, default=32, help="合成数据头数")

    # 为了判断用户是否显式传入 --synthetic-seq，我们将其默认值设为 None，
    # 后续手动处理默认 512 的逻辑。
    synth_group.add_argument("--synthetic-seq", type=int, default=None, help="合成数据序列长度 (默认 512)")

    # BL 参数（KV缓存长度），与论文/脚本中使用的一致
    parser.add_argument("--bl", type=int, dest="bl", default=None,
                        help="KV缓存长度 B_L (例如 128 或 1024)。若同时启用 --synthetic 且未显式指定 --synthetic-seq，将覆盖合成数据序列长度。")

    synth_group.add_argument("--synthetic-batch", type=int, default=1, help="合成数据 batch 大小")


    # 新增：性能配置组
    perf_group = parser.add_argument_group("Performance Config")
    perf_group.add_argument("--warmup-samples", type=int, default=5, help="预热样本数")
    perf_group.add_argument("--performance-tracking", action="store_true", help="启用性能跟踪")
    perf_group.add_argument("--enable-fallback", action="store_true", default=True, help="启用回退机制")
    perf_group.add_argument("--memory-efficient", action="store_true", help="启用内存高效模式")

    # 新增：实验配置组
    exp_group = parser.add_argument_group("Experiment Config")
    exp_group.add_argument("--experiment-mode", action="store_true", help="启用实验模式")
    exp_group.add_argument("--detailed-logging", action="store_true", help="启用详细日志")
    exp_group.add_argument("--v-metric", type=str, default="var", 
                          choices=["var", "scaled_var", "std", "entropy"],
                          help="V指标计算方式")

    # 新增：高级配置组
    adv_group = parser.add_argument_group("Advanced Config")
    adv_group.add_argument("--high-dispersion-threshold", type=float, default=0.7,
                          help="H指标高分散阈值")
    adv_group.add_argument("--high-dynamics-threshold", type=float, default=0.5,
                          help="V指标高动态阈值")
    adv_group.add_argument("--key-head-ratio", type=float, default=0.2,
                          help="关键头比例")
    
    # 新增：环境预设
    parser.add_argument("--env", type=str, choices=["dev", "prod", "custom"], default="custom",
                       help="环境预设：dev(开发), prod(生产), custom(自定义)")


    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 加载配置文件（如果提供）
    config_data = {}
    if args.config:
        config_data = load_config_file(args.config)
    
    # 合并配置：命令行参数优先级高于配置文件
    # 处理缓存大小
    effective_cache_size = args.cache_size
    if args.bl is not None:
        effective_cache_size = args.bl
    elif config_data.get('cache', {}).get('bl'):
        effective_cache_size = config_data['cache']['bl']
    elif config_data.get('cache', {}).get('total_size'):
        effective_cache_size = config_data['cache']['total_size']

    # 环境预设配置
    if args.env == "dev":
        # 开发环境：启用所有监控和日志
        env_overrides = {
            "enable_monitoring": True,
            "enable_fallback": True,
            "detailed_logging": True,
            "performance_tracking": True,
            "experiment_mode": True,
        }
    elif args.env == "prod":
        # 生产环境：关闭调试功能，优化性能
        env_overrides = {
            "enable_monitoring": False,
            "enable_fallback": True,
            "detailed_logging": False,
            "performance_tracking": False,
            "experiment_mode": False,
        }
    else:  # custom
        env_overrides = {}

    # 构建自定义阈值
    custom_thresholds = {
        "high_dispersion_threshold": args.high_dispersion_threshold,
        "high_dynamics_threshold": args.high_dynamics_threshold,
        "key_head_ratio": args.key_head_ratio,
    }


    # 构建 IntegrationConfig
    integration_cfg = IntegrationConfig(
        total_cache_size=effective_cache_size,

        enable_monitoring=env_overrides.get("enable_monitoring", args.monitor),
        enable_auto_tuning=args.auto_tune,
        warmup_samples=args.warmup_samples,
        performance_tracking=env_overrides.get("performance_tracking", args.performance_tracking),
        enable_fallback=env_overrides.get("enable_fallback", args.enable_fallback),
        experiment_mode=env_overrides.get("experiment_mode", args.experiment_mode),
        detailed_logging=env_overrides.get("detailed_logging", args.detailed_logging),
        custom_thresholds=custom_thresholds,

    )

    integration = CakeAdaKVIntegration(integration_cfg)

    # 准备注意力权重
    attention_weights_list: List[Any]
    if args.input:
        attention_weights_list = load_attention_weights(Path(args.input))
    else:

        # 根据以下优先级决定合成序列长度 (seq_len):
        # 1. 用户通过 --synthetic-seq 显式指定 (最高优先级)
        # 2. 用户通过 --bl 指定 (若未显式指定 --synthetic-seq)
        # 3. 默认值 512
        if args.synthetic_seq is not None:
            seq_len = args.synthetic_seq
        elif args.bl is not None:
            seq_len = args.bl
        else:
            seq_len = 512

        attention_weights_list = generate_synthetic_attention_weights(
            num_layers=args.synthetic_layers,
            num_heads=args.synthetic_heads,
            seq_len=seq_len,
            batch_size=args.synthetic_batch,
        )

    detailed = bool(args.detailed)

    # 优化缓存
    result = integration.optimize_cache(
        attention_weights_list, return_detailed_info=detailed
    )

    # 处理输出
    if detailed:
        output_obj = result  # type: ignore[assignment]
    else:
        layer_budgets, head_budgets_list = result  # type: ignore[misc]
        output_obj = {
            "layer_budgets": layer_budgets,
            "head_budgets_list": head_budgets_list,
        }

    json_str = json.dumps(output_obj, ensure_ascii=False, indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_str, encoding="utf-8")
        print(f"✅ 结果已保存: {output_path} (size: {output_path.stat().st_size} bytes)")
    else:
        # 打印到 stdout
        print(json_str)


if __name__ == "__main__":
    main()