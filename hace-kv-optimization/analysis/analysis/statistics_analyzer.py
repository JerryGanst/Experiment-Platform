"""
统计分析模块

用于分析实验结果的统计显著性和相关性
"""
import os
import logging
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from hace_core import config

STATISTICS_CONFIG = config.STATISTICS_CONFIG
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class StatisticsAnalyzer:
    """统计分析器类"""
    
    def __init__(self, config=None):
        """
        初始化统计分析器
        
        Args:
            config: 统计分析配置，默认使用STATISTICS_CONFIG
        """
        self.config = config or STATISTICS_CONFIG
        self.significance_level = self.config.get("significance_level", 0.05)
        self.outlier_config = self.config.get("outlier_detection", {})
        logger.info(f"统计分析器初始化完成，显著性水平: {self.significance_level}")
    
    def perform_paired_ttest(self, baseline_data, h2o_data, metric_name, alternative="two-sided"):
        """
        执行配对t检验
        
        Args:
            baseline_data: 基线数据数组
            h2o_data: H2O数据数组
            metric_name: 指标名称
            alternative: 备选假设，可选"two-sided", "less", "greater"
            
        Returns:
            dict: 包含t统计量、p值和假设是否成立的结果
        """
        if len(baseline_data) != len(h2o_data):
            logger.warning(f"配对t检验要求样本长度相同，当前长度: baseline={len(baseline_data)}, h2o={len(h2o_data)}")
            # 截取相同长度
            min_len = min(len(baseline_data), len(h2o_data))
            baseline_data = baseline_data[:min_len]
            h2o_data = h2o_data[:min_len]
        
        # 执行配对t检验
        t_stat, p_value = stats.ttest_rel(baseline_data, h2o_data, alternative=alternative)
        
        # 确定假设是否成立
        hypothesis_holds = p_value < self.significance_level
        
        result = {
            "metric": metric_name,
            "test": "paired_ttest",
            "t_statistic": t_stat,
            "p_value": p_value,
            "alternative": alternative,
            "significant": hypothesis_holds,
            "confidence_level": 1 - self.significance_level
        }
        
        logger.info(f"配对t检验结果 (指标: {metric_name}): t={t_stat:.4f}, p={p_value:.4f}, 显著: {hypothesis_holds}")
        return result
    
    def perform_anova(self, data_groups, metric_name):
        """
        执行单因素方差分析(ANOVA)
        
        Args:
            data_groups: 不同组的数据列表
            metric_name: 指标名称
            
        Returns:
            dict: 包含F统计量、p值和假设是否成立的结果
        """
        # 执行单因素ANOVA
        f_stat, p_value = stats.f_oneway(*data_groups)
        
        # 确定假设是否成立
        hypothesis_holds = p_value < self.significance_level
        
        result = {
            "metric": metric_name,
            "test": "anova",
            "f_statistic": f_stat,
            "p_value": p_value,
            "significant": hypothesis_holds,
            "confidence_level": 1 - self.significance_level,
            "group_means": [np.mean(group) for group in data_groups],
            "group_std": [np.std(group) for group in data_groups]
        }
        
        logger.info(f"方差分析结果 (指标: {metric_name}): F={f_stat:.4f}, p={p_value:.4f}, 显著: {hypothesis_holds}")
        return result
    
    def regression_analysis(self, df, x_vars, y_var):
        """
        执行多元回归分析
        
        Args:
            df: 包含数据的DataFrame
            x_vars: 自变量列名列表
            y_var: 因变量列名
            
        Returns:
            dict: 包含回归结果的字典
        """
        # 准备数据
        X = df[x_vars].values
        y = df[y_var].values
        
        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 执行回归分析
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # 计算决定系数(R²)
        r_squared = model.score(X_scaled, y)
        
        # 计算调整后的R²
        n = X.shape[0]  # 样本数量
        p = X.shape[1]  # 特征数量
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
        
        # 预测值
        y_pred = model.predict(X_scaled)
        
        # 计算残差
        residuals = y - y_pred
        
        # 计算标准误差和t值
        mse = np.mean(residuals ** 2)
        se = np.sqrt(np.sum(residuals ** 2) / (n - p - 1))
        
        # 构建结果
        result = {
            "dependent_var": y_var,
            "independent_vars": x_vars,
            "coefficients": dict(zip(x_vars, model.coef_)),
            "intercept": model.intercept_,
            "r_squared": r_squared,
            "adj_r_squared": adj_r_squared,
            "standard_error": se,
            "sample_size": n
        }
        
        # 添加统计显著性信息
        for i, var in enumerate(x_vars):
            # 计算标准误差
            x_i = X_scaled[:, i]
            se_i = se / np.sqrt(np.sum((x_i - np.mean(x_i)) ** 2))
            
            # 计算t值
            t_value = model.coef_[i] / se_i
            
            # 计算p值
            p_value = 2 * (1 - stats.t.cdf(abs(t_value), n - p - 1))
            
            # 记录显著性信息
            result[f"{var}_t_value"] = t_value
            result[f"{var}_p_value"] = p_value
            result[f"{var}_significant"] = p_value < self.significance_level
        
        logger.info(f"回归分析结果 (因变量: {y_var}): R²={r_squared:.4f}, 调整R²={adj_r_squared:.4f}")
        return result
    
    def correlation_analysis(self, df, variables=None, method='pearson'):
        """
        执行相关性分析
        
        Args:
            df: 包含数据的DataFrame
            variables: 要分析的变量列表，默认使用所有数值列
            method: 相关系数计算方法，'pearson', 'spearman' 或 'kendall'
            
        Returns:
            dict: 包含相关系数矩阵和显著性信息的结果
        """
        # 如果未指定变量，使用所有数值列
        if variables is None:
            variables = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 计算相关系数矩阵
        corr_matrix = df[variables].corr(method=method)
        
        # 计算p值矩阵
        p_matrix = pd.DataFrame(np.zeros((len(variables), len(variables))), 
                               index=variables, columns=variables)
        
        # 填充p值矩阵
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i == j:
                    p_matrix.loc[var1, var2] = 0.0
                else:
                    if method == 'pearson':
                        corr, p_value = stats.pearsonr(df[var1], df[var2])
                    elif method == 'spearman':
                        corr, p_value = stats.spearmanr(df[var1], df[var2])
                    else:  # kendall
                        corr, p_value = stats.kendalltau(df[var1], df[var2])
                    
                    p_matrix.loc[var1, var2] = p_value
        
        # 构建显著性矩阵
        significance_matrix = p_matrix < self.significance_level
        
        result = {
            "method": method,
            "correlation_matrix": corr_matrix,
            "p_value_matrix": p_matrix,
            "significance_matrix": significance_matrix
        }
        
        # 添加强相关对
        # 提取强相关系数（绝对值大于0.7）且统计显著的变量对
        strong_correlations = []
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables[i+1:], i+1):
                corr_value = corr_matrix.loc[var1, var2]
                p_value = p_matrix.loc[var1, var2]
                
                if abs(corr_value) > 0.7 and p_value < self.significance_level:
                    strong_correlations.append({
                        "var1": var1,
                        "var2": var2,
                        "correlation": corr_value,
                        "p_value": p_value
                    })
        
        result["strong_correlations"] = strong_correlations
        
        logger.info(f"相关性分析完成，使用方法: {method}，发现{len(strong_correlations)}对强相关变量")
        return result
    
    def detect_outliers(self, data, method=None, threshold=None):
        """
        检测异常值
        
        Args:
            data: 数据数组或Series
            method: 检测方法，'zscore'或'iqr'，默认使用配置
            threshold: 异常阈值，默认使用配置
            
        Returns:
            tuple: (异常值索引, 异常值)
        """
        # 使用配置值作为默认值
        if method is None:
            method = self.outlier_config.get("method", "zscore")
        
        if threshold is None:
            threshold = self.outlier_config.get("threshold", 3.0)
        
        # 确保数据是一维数组
        if isinstance(data, pd.DataFrame):
            raise ValueError("detect_outliers需要一维数组或Series，而不是DataFrame")
        
        if isinstance(data, pd.Series):
            values = data.values
            indices = data.index
        else:
            values = np.array(data)
            indices = np.arange(len(values))
        
        outlier_indices = []
        outlier_values = []
        
        # 使用Z分数方法
        if method == 'zscore':
            z_scores = np.abs((values - np.mean(values)) / np.std(values))
            for i, z in enumerate(z_scores):
                if z > threshold:
                    outlier_indices.append(indices[i])
                    outlier_values.append(values[i])
        
        # 使用IQR方法
        elif method == 'iqr':
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            for i, value in enumerate(values):
                if value < lower_bound or value > upper_bound:
                    outlier_indices.append(indices[i])
                    outlier_values.append(value)
        
        else:
            raise ValueError(f"不支持的异常值检测方法: {method}")
        
        logger.info(f"异常值检测完成 (方法: {method}, 阈值: {threshold})，发现{len(outlier_indices)}个异常值")
        return outlier_indices, outlier_values
    
    def analyze_method_performance(self, baseline_df: pd.DataFrame, 
                                 method_df: pd.DataFrame, 
                                 method_name: str, 
                                 method_params_cols: list[str], 
                                 metrics_to_compare: list[str] = None,
                                 output_dir: str = None):
        """
        通用的方法性能分析（支持H2O、CAKE等）。
        对比特定方法与基线的性能，并分析方法内部参数的影响。

        Args:
            baseline_df: 基线实验结果的DataFrame。
            method_df: 特定方法实验结果的DataFrame。
            method_name: 方法名称 (例如 "H2O", "CAKE", "CustomMethod")。
            method_params_cols: 方法特定的参数列名列表 (例如 ["heavy_ratio", "recent_ratio"] for H2O, 
                                or ["allocation_strategy", "cache_budget"] for CAKE)。
            metrics_to_compare: 需要进行对比和分析的指标列名列表。
                                默认使用 STATISTICS_CONFIG["comparison_metrics"]。
            output_dir: 保存分析结果（如图表、报告）的目录。如果为None，则不保存。

        Returns:
            dict: 包含各项分析结果的字典。
        """
        logger.info(f"开始分析方法 '{method_name}' 的性能，对比基线并分析参数影响。")
        analysis_results = {}

        if metrics_to_compare is None:
            metrics_to_compare = self.config.get("comparison_metrics", 
                                               ["TTFT_ms", "TPOT_ms", "Throughput_tok_sec", "GPU_Memory_Peak_MB"])

        if output_dir:
            method_analysis_dir = os.path.join(output_dir, method_name.lower() + "_analysis")
            os.makedirs(method_analysis_dir, exist_ok=True)
            logger.info(f"分析结果将保存到: {method_analysis_dir}")
        else:
            method_analysis_dir = None # 明确设置为None，以便后续检查

        # 1. 基线 vs 方法对比 (总体)
        logger.info(f"对比 {method_name} (总体) vs 基线...")
        overall_comparison_results = {}
        for metric in metrics_to_compare:
            if metric not in baseline_df.columns or metric not in method_df.columns:
                logger.warning(f"指标 '{metric}' 在基线或方法DataFrame中不存在，跳过对比。")
                continue
            
            baseline_metric_data = baseline_df[metric].dropna()
            method_metric_data = method_df[metric].dropna()

            if len(baseline_metric_data) < 2 or len(method_metric_data) < 2: # t-test至少需要两个样本
                logger.warning(f"指标 '{metric}' 的基线或方法数据样本不足 (<2)，无法进行t检验。 Baseline: {len(baseline_metric_data)}, Method: {len(method_metric_data)}")
                overall_comparison_results[metric] = {
                    "test": "independent_ttest",
                    "method_mean": np.mean(method_metric_data) if len(method_metric_data) > 0 else None,
                    "baseline_mean": np.mean(baseline_metric_data) if len(baseline_metric_data) > 0 else None,
                    "error": "Insufficient data for t-test"
                }
                continue

            alternative = "less" if "TTFT" in metric or "TPOT" in metric or "Memory" in metric or "Latency" in metric else "greater"
            
            t_stat, p_value = stats.ttest_ind(method_metric_data, baseline_metric_data, alternative=alternative, equal_var=False) 
            significant = p_value < self.significance_level
            
            overall_comparison_results[metric] = {
                "test": "independent_ttest (Welch's)",
                "method_mean": np.mean(method_metric_data),
                "method_std": np.std(method_metric_data),
                "baseline_mean": np.mean(baseline_metric_data),
                "baseline_std": np.std(baseline_metric_data),
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant_improvement": significant,
                "alternative": alternative,
                "method_N": len(method_metric_data),
                "baseline_N": len(baseline_metric_data)
            }
            logger.info(f"  {method_name} vs Baseline - {metric}: Method Mean={np.mean(method_metric_data):.2f} (N={len(method_metric_data)}), Baseline Mean={np.mean(baseline_metric_data):.2f} (N={len(baseline_metric_data)}), p-value={p_value:.4f}, Significant Imprv.={significant}")
        analysis_results[f"overall_{method_name.lower()}_vs_baseline"] = overall_comparison_results

        # 2. 方法内部参数影响分析
        logger.info(f"分析 {method_name} 内部参数 ({', '.join(method_params_cols)}) 的影响...")
        parameter_impact_results = {}
        for param_col in method_params_cols:
            if param_col not in method_df.columns:
                logger.warning(f"参数列 '{param_col}' 在 {method_name} DataFrame中不存在，跳过分析。")
                continue
            
            parameter_impact_results[param_col] = {}
            unique_param_values = method_df[param_col].unique()
            try:
                # 尝试对唯一参数值进行排序，以便图表更美观
                # 如果参数是混合类型（例如数字和字符串'None'），排序可能会失败
                sorted_unique_param_values = sorted(unique_param_values, key=lambda x: (isinstance(x, str), x))
            except TypeError:
                logger.warning(f"参数 '{param_col}' 的值包含不可比较的混合类型，将使用原始顺序。")
                sorted_unique_param_values = unique_param_values

            logger.info(f"分析参数 '{param_col}' 的影响，唯一值 (排序后): {sorted_unique_param_values}")

            for metric in metrics_to_compare:
                if metric not in method_df.columns:
                    logger.warning(f"指标 '{metric}' 在 {method_name} DataFrame中不存在，跳过参数影响分析。")
                    continue
                
                metric_param_analysis = {"means": {}, "stds": {}, "counts":{}, "anova_results": None}
                data_groups_for_anova = []
                group_labels_for_plot = []

                for val in sorted_unique_param_values:
                    # 处理 val 可能为 np.nan 或 None 的情况，确保键是字符串
                    val_key = str(val) if not pd.isna(val) else "NaN/None"
                    group_data = method_df[method_df[param_col].fillna("NaN/None") == val_key][metric].dropna()
                    
                    metric_param_analysis["means"][val_key] = np.mean(group_data) if len(group_data) > 0 else None
                    metric_param_analysis["stds"][val_key] = np.std(group_data) if len(group_data) > 0 else None
                    metric_param_analysis["counts"][val_key] = len(group_data)
                    if len(group_data) > 1: # ANOVA和箱线图至少需要一个组有多个数据点
                        data_groups_for_anova.append(group_data)
                        group_labels_for_plot.append(val_key)

                if len(data_groups_for_anova) > 1:
                    min_group_size = min(len(g) for g in data_groups_for_anova)
                    if min_group_size > 0: 
                        anova_res = self.perform_anova(data_groups_for_anova, metric_name=f"{metric}_by_{param_col}")
                        metric_param_analysis["anova_results"] = anova_res
                        logger.info(f"  ANOVA for {metric} by {param_col}: F={anova_res['f_statistic']:.2f}, p={anova_res['p_value']:.4f}, Significant Diff.={anova_res['significant']}")
                    else:
                        logger.warning(f"  无法为 {metric} by {param_col} 执行ANOVA，因为部分组数据不足 (min_group_size=0)。")    
                elif len(data_groups_for_anova) <=1:
                    logger.info(f"  对于参数 '{param_col}' 和指标 '{metric}'，不足两个有效数据组 (有>1个样本的组数量: {len(data_groups_for_anova)}) 进行ANOVA分析。已记录均值和标准差。")
                
                parameter_impact_results[param_col][metric] = metric_param_analysis

                if method_analysis_dir and len(data_groups_for_anova) > 1 and len(group_labels_for_plot) > 1:
                    try:
                        plt.figure(figsize=self.config.get("plot_figsize", (max(10, len(group_labels_for_plot) * 1.2), 6)))
                        sns.boxplot(data=data_groups_for_anova, notch=False) # notch=True 可能因数据导致错误
                        plt.xticks(ticks=np.arange(len(group_labels_for_plot)), labels=group_labels_for_plot, rotation=45, ha="right")
                        plt.title(f"{method_name}: Impact of {param_col} on {metric}")
                        plt.xlabel(str(param_col))
                        plt.ylabel(metric)
                        plt.tight_layout()
                        plot_filename = os.path.join(method_analysis_dir, f"boxplot_{method_name.lower()}_{param_col}_vs_{metric}.png")
                        plt.savefig(plot_filename)
                        plt.close()
                        logger.info(f"    参数影响箱线图已保存到: {plot_filename}")
                    except Exception as e:
                        logger.error(f"    生成参数影响图表失败 for {param_col} on {metric}: {e}", exc_info=True)
        
        analysis_results[f"{method_name.lower()}_parameter_impact"] = parameter_impact_results

        logger.info(f"方法 '{method_name}' 的性能分析完成。")
        if method_analysis_dir:
            summary_report_path = os.path.join(method_analysis_dir, f"analysis_summary_{method_name.lower()}.json")
            try:
                # 自定义JSON序列化函数处理Numpy类型、datetime等
                def json_serializable(obj):
                    if isinstance(obj, (np.integer, np.bool_)):
                        return int(obj) if isinstance(obj, np.integer) else bool(obj)
                    elif isinstance(obj, np.floating):
                        # 处理 np.nan, np.inf, -np.inf
                        if np.isnan(obj):
                            return None # or "NaN" as string
                        if np.isinf(obj):
                            return "Infinity" if obj > 0 else "-Infinity"
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif pd.isna(obj):
                        return None 
                    elif isinstance(obj, (datetime, pd.Timestamp)):
                        return obj.isoformat()
                    elif isinstance(obj, (pd.Series, pd.DataFrame)):
                        return obj.to_dict() # 基本序列化
                    raise TypeError(f"Type {type(obj)} with value '{obj}' not serializable for JSON output")

                with open(summary_report_path, 'w', encoding='utf-8') as f:
                    json.dump(analysis_results, f, indent=4, default=json_serializable, ensure_ascii=False)
                logger.info(f"完整的分析报告已保存到: {summary_report_path}")
            except Exception as e:
                logger.error(f"保存分析报告失败: {e}", exc_info=True)

        return analysis_results

    def analyze_h2o_performance(self, baseline_df: pd.DataFrame, 
                                h2o_df: pd.DataFrame, 
                                metrics: list[str] = None, 
                                output_dir: str = None):
        """
        分析H2O实验的性能，对比基线并评估H2O参数（如heavy_ratio, recent_ratio）的影响。
        此方法现在是 analyze_method_performance 的一个特定包装器。

        Args:
            baseline_df: 基线实验结果的DataFrame。
            h2o_df: H2O实验结果的DataFrame。
            metrics: 需要进行对比和分析的指标列名列表。
                     默认使用 STATISTICS_CONFIG["comparison_metrics"]。
            output_dir: 保存分析结果（如图表、报告）的目录。

        Returns:
            dict: 包含各项分析结果的字典。
        """
        logger.info("调用 analyze_method_performance 进行H2O性能分析...")
        # H2O特定的参数列，从config中获取，如果未定义则使用默认值
        h2o_group_name = self.config.get("h2o_group_name", "H2O")
        default_params_config = self.config.get("default_method_params", {})
        h2o_specific_params = default_params_config.get(h2o_group_name.lower(), 
                                                        default_params_config.get("h2o", 
                                सरदार["heavy_ratio", "recent_ratio", "eviction_strategy"])) # 支持小写或大写H2O键
        
        return self.analyze_method_performance(
            baseline_df=baseline_df,
            method_df=h2o_df,
            method_name=h2o_group_name,
            method_params_cols=h2o_specific_params,
            metrics_to_compare=metrics,
            output_dir=output_dir
        )

    # 新增: 为CAKE实验调用通用分析方法的包装器
    def analyze_cake_performance(self, baseline_df: pd.DataFrame, 
                                 cake_df: pd.DataFrame, 
                                 metrics: list[str] = None, 
                                 output_dir: str = None):
        """
        分析CAKE实验的性能，对比基线并评估CAKE参数（如allocation_strategy, cache_budget）的影响。
        此方法是 analyze_method_performance 的一个特定包装器。

        Args:
            baseline_df: 基线实验结果的DataFrame。
            cake_df: CAKE实验结果的DataFrame。
            metrics: 需要进行对比和分析的指标列名列表。
                     默认使用 STATISTICS_CONFIG["comparison_metrics"]。
            output_dir: 保存分析结果（如图表、报告）的目录。

        Returns:
            dict: 包含各项分析结果的字典。
        """
        logger.info("调用 analyze_method_performance 进行CAKE性能分析...")
        # CAKE特定的参数列，从config中获取
        cake_group_name = self.config.get("cake_group_name", "CAKE")
        default_params_config = self.config.get("default_method_params", {})
        cake_specific_params = default_params_config.get(cake_group_name.lower(), 
                                                       default_params_config.get("cake", 
                               ["allocation_strategy", "cache_budget"])) # 支持小写或大写CAKE键
        
        return self.analyze_method_performance(
            baseline_df=baseline_df,
            method_df=cake_df,
            method_name=cake_group_name,
            method_params_cols=cake_specific_params,
            metrics_to_compare=metrics,
            output_dir=output_dir
        )

    def generate_report(self, analysis_results, output_dir, report_name="statistics_report"):
        # 实现生成报告的逻辑
        pass 