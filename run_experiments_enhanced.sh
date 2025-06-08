#!/bin/bash

# 增强的实验运行脚本
# 基于现有的 hace-kv-optimization 基础设施
# 支持容错继续运行，避免单个实验失败影响整体进度

# 移除 set -e，改为手动错误处理以支持容错运行

echo "=========================================="
echo "  HACE vs Baselines 实验平台"
echo "  基于现有基础设施的增强版本"
echo "=========================================="

# 检查 hace-kv-optimization 目录是否存在
if [ ! -d "hace-kv-optimization" ]; then
    echo "❌ 错误: 未找到 hace-kv-optimization 目录"
    echo "请确保在正确的项目根目录下运行此脚本"
    exit 1
fi

# 默认参数
EXPERIMENT_TYPE="compare_all"
DATASETS="mmlu"
RUN_NAME="experiment_$(date +%m%d_%H%M)"
QUICK_TEST=false
INTERACTIVE=false

# 实验状态跟踪
declare -a EXPERIMENT_LOG=()
declare -a SUCCESS_EXPERIMENTS=()
declare -a FAILED_EXPERIMENTS=()
TOTAL_EXPERIMENTS=0

# 设置信号处理，确保中断时也能看到报告
cleanup_on_interrupt() {
    echo ""
    echo "⚠️  收到中断信号，正在生成当前进度报告..."
    generate_final_report
    echo ""
    echo "💡 实验被中断，但已保存的结果仍然有效"
    exit 130
}

trap cleanup_on_interrupt SIGINT SIGTERM

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -t, --type TYPE       实验类型 (baseline, cake, compare_all)"
    echo "  -d, --datasets LIST   数据集列表，逗号分隔 (默认: mmlu)"
    echo "  -n, --name NAME       运行名称 (默认: experiment_MMDD_HHMM)"
    echo "  -q, --quick          快速测试模式"
    echo "  -i, --interactive    交互模式"
    echo "  -h, --help           显示此帮助"
    echo ""
    echo "示例:"
    echo "  $0                                    # 运行默认对比实验"
    echo "  $0 -t cake -d mmlu,gsm8k            # 运行CAKE实验"
    echo "  $0 -t compare_all -n my_exp         # 运行完整对比"
    echo "  $0 -q                                # 快速测试"
    echo "  $0 -i                                # 交互模式"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            EXPERIMENT_TYPE="$2"
            shift 2
            ;;
        -d|--datasets)
            DATASETS="$2"
            shift 2
            ;;
        -n|--name)
            RUN_NAME="$2"
            shift 2
            ;;
        -q|--quick)
            QUICK_TEST=true
            shift
            ;;
        -i|--interactive)
            INTERACTIVE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 交互模式
if [ "$INTERACTIVE" = true ]; then
    echo ""
    echo "=== 交互模式 ==="
    echo "1. 完整对比实验 (推荐)"
    echo "2. 单独运行 CAKE"
    echo "3. 单独运行 Baseline"
    echo "4. 快速测试"
    echo ""
    read -p "请选择 (1-4): " choice
    
    case $choice in
        1)
            EXPERIMENT_TYPE="compare_all"
            ;;
        2)
            EXPERIMENT_TYPE="cake"
            ;;
        3)
            EXPERIMENT_TYPE="baseline"
            ;;
        4)
            QUICK_TEST=true
            EXPERIMENT_TYPE="baseline"
            ;;
        *)
            echo "无效选择"
            exit 1
            ;;
    esac
    
    echo ""
    read -p "数据集 (默认: mmlu): " input_datasets
    if [ -n "$input_datasets" ]; then
        DATASETS="$input_datasets"
    fi
    
    echo ""
    read -p "运行名称 (默认: $RUN_NAME): " input_name
    if [ -n "$input_name" ]; then
        RUN_NAME="$input_name"
    fi
fi

echo ""
echo "实验配置:"
echo "  类型: $EXPERIMENT_TYPE"
echo "  数据集: $DATASETS"
echo "  运行名称: $RUN_NAME"
echo "  快速测试: $QUICK_TEST"
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 python3"
    exit 1
fi

# 添加实验日志函数
log_experiment() {
    local status="$1"
    local experiment="$2"
    local message="$3"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    
    EXPERIMENT_LOG+=("[$timestamp] $status: $experiment - $message")
    
    if [ "$status" = "✅ SUCCESS" ]; then
        SUCCESS_EXPERIMENTS+=("$experiment")
        echo "✅ $experiment 成功完成"
    elif [ "$status" = "❌ FAILED" ]; then
        FAILED_EXPERIMENTS+=("$experiment")
        echo "❌ $experiment 失败: $message"
    elif [ "$status" = "🚀 START" ]; then
        echo "🚀 开始运行: $experiment"
        TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + 1))
    fi
}

# 生成最终报告
generate_final_report() {
    echo ""
    echo "=========================================="
    echo "  实验批处理完成 - 最终报告"
    echo "=========================================="
    echo "总实验数: $TOTAL_EXPERIMENTS"
    echo "成功: ${#SUCCESS_EXPERIMENTS[@]}"
    echo "失败: ${#FAILED_EXPERIMENTS[@]}"
    echo ""
    
    if [ ${#SUCCESS_EXPERIMENTS[@]} -gt 0 ]; then
        echo "✅ 成功的实验:"
        for exp in "${SUCCESS_EXPERIMENTS[@]}"; do
            echo "  - $exp"
        done
        echo ""
    fi
    
    if [ ${#FAILED_EXPERIMENTS[@]} -gt 0 ]; then
        echo "❌ 失败的实验:"
        for exp in "${FAILED_EXPERIMENTS[@]}"; do
            echo "  - $exp"
        done
        echo ""
        echo "🔍 建议检查以下日志文件获取详细错误信息:"
        cd hace-kv-optimization 2>/dev/null || true
        find "results/cli_runs/$RUN_NAME" -name "*.txt" -type f 2>/dev/null | head -5 | while read logfile; do
            echo "  📄 $logfile"
        done
        cd .. 2>/dev/null || true
        echo ""
        echo "🔧 重新运行失败实验的建议:"
        for exp in "${FAILED_EXPERIMENTS[@]}"; do
            case $exp in
                "baseline")
                    echo "  ./run_experiments_enhanced.sh -t baseline -n retry_baseline_$(date +%H%M)"
                    ;;
                "cake")
                    echo "  ./run_experiments_enhanced.sh -t cake -n retry_cake_$(date +%H%M)"
                    ;;
                "compare_all")
                    echo "  ./run_experiments_enhanced.sh -t compare_all -n retry_compare_$(date +%H%M)"
                    ;;
            esac
        done
    else
        echo "🎉 所有实验都成功完成！"
    fi
    
    echo ""
    echo "详细运行日志:"
    for log_entry in "${EXPERIMENT_LOG[@]}"; do
        echo "  $log_entry"
    done
}

# 构建命令
cd hace-kv-optimization

CMD="python3 experiments/run_experiment.py $EXPERIMENT_TYPE"
CMD="$CMD --datasets $DATASETS"
CMD="$CMD --run_name $RUN_NAME"

# 快速测试模式参数
if [ "$QUICK_TEST" = true ]; then
    echo "启用快速测试模式..."
    CMD="$CMD --repetitions 1"
    CMD="$CMD --max_new_tokens 10"
    if [ "$EXPERIMENT_TYPE" = "cake" ] || [ "$EXPERIMENT_TYPE" = "compare_all" ]; then
        CMD="$CMD --cache_budgets 0.7"
        CMD="$CMD --allocation_strategies adaptive"
    fi
fi

echo "执行命令: $CMD"
echo ""

# 记录实验开始
log_experiment "🚀 START" "$EXPERIMENT_TYPE" "开始运行实验"

# 运行实验（容错模式）
echo "⏳ 正在运行实验，请耐心等待..."
echo "💡 提示: 即使某个子实验失败，脚本也会继续运行后续实验"
if eval "$CMD" 2>&1; then
    log_experiment "✅ SUCCESS" "$EXPERIMENT_TYPE" "实验成功完成"
    
    # 查找结果目录
    RESULTS_BASE="results/cli_runs/$RUN_NAME"
    if [ -d "$RESULTS_BASE" ]; then
        echo ""
        echo "📁 结果保存在: $RESULTS_BASE"
        
        # 列出生成的文件
        echo ""
        echo "📄 生成的文件:"
        find "$RESULTS_BASE" -name "*.csv" -o -name "*.png" -o -name "*.json" 2>/dev/null | head -10 | while read file; do
            echo "  📄 $file"
        done
        
        # 检查是否有对比报告
        if [ -d "$RESULTS_BASE/comparison_report" ]; then
            echo ""
            echo "📊 对比报告已生成: $RESULTS_BASE/comparison_report/"
        fi
        
        echo ""
        echo "💡 下一步操作建议:"
        echo "  1. 查看结果: ls -la $RESULTS_BASE"
        echo "  2. 运行分析: python3 analysis/generate_report.py"
        echo "  3. 查看图表: 打开 .png 文件"
    fi
else
    # 实验失败，但不退出
    EXPERIMENT_EXIT_CODE=$?
    log_experiment "❌ FAILED" "$EXPERIMENT_TYPE" "退出代码: $EXPERIMENT_EXIT_CODE"
    echo ""
    echo "⚠️  警告: 当前实验失败，但继续执行后续步骤..."
    echo "详细错误信息请查看日志文件"
fi

# 生成最终报告
generate_final_report

# 根据失败数量决定退出代码
if [ ${#FAILED_EXPERIMENTS[@]} -gt 0 ]; then
    echo ""
    echo "⚠️  注意: 有 ${#FAILED_EXPERIMENTS[@]} 个实验失败，但已完成所有可能的实验"
    echo "建议检查失败的实验日志文件并重新运行"
    exit 2  # 部分失败
else
    echo ""
    echo "🎊 所有实验都成功完成！实验平台运行良好。"
    exit 0  # 全部成功
fi

cd .. 