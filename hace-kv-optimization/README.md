# HACE KV 缓存优化实验

此目录包含 HACE 及其基线方法的实现代码，提供统一的运行入口和配置示例。

## 目录结构

```text
├── main.py               # 入口脚本
├── config.yaml           # 默认实验配置
├── run_experiments.sh    # 批量执行脚本
├── hace_core/            # HACE 算法实现
├── baselines/            # 各种基线方法
├── experiments/          # 实验运行框架
├── analysis/             # 结果分析脚本
└── results/              # 输出结果目录
```

## 运行方式

在仓库根目录执行：

```bash
bash setup_env.sh            # 安装依赖
./run_experiments.sh         # 运行实验
```

实验完成后，可在 `results/` 查看生成的数据，并使用 `analysis/` 里的脚本进行可视化与统计测试。

