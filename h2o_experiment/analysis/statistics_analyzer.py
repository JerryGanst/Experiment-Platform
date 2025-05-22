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
from config import STATISTICS_CONFIG

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
    
    def analyze_h2o_performance(self, baseline_df, h2o_df, metrics=None, output_dir=None):
        """
        分析H2O性能改进的统计显著性
        
        Args:
            baseline_df: 基线实验结果DataFrame
            h2o_df: H2O实验结果DataFrame
            metrics: 要分析的指标列表，默认为主要性能指标
            output_dir: 输出目录，用于保存分析结果
            
        Returns:
            dict: 包含分析结果的字典
        """
        # 过滤成功的实验
        baseline_df = baseline_df[baseline_df["success"] == True].copy()
        h2o_df = h2o_df[h2o_df["success"] == True].copy()
        
        if baseline_df.empty or h2o_df.empty:
            logger.warning("无法进行性能分析：基线或H2O结果为空")
            return None
        
        # 默认分析的指标
        if metrics is None:
            metrics = [
                "peak_gpu_memory_mb", 
                "ttft_ms", 
                "tpot_ms", 
                "throughput_tokens_per_sec"
            ]
        
        # 创建结果目录
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 准备结果容器
        results = {
            "t_tests": {},
            "regression_analysis": {},
            "correlation_analysis": None,
            "improvements": {}
        }
        
        # 计算平均改进率
        for metric in metrics:
            # 在某些指标中，较小的值表示更好的性能（如内存使用和延迟）
            lower_better = metric in ["peak_gpu_memory_mb", "ttft_ms", "tpot_ms"]
            
            baseline_mean = baseline_df[metric].mean()
            h2o_mean = h2o_df[metric].mean()
            
            if lower_better:
                improvement = (baseline_mean - h2o_mean) / baseline_mean * 100
            else:
                improvement = (h2o_mean - baseline_mean) / baseline_mean * 100
            
            results["improvements"][metric] = {
                "baseline_mean": baseline_mean,
                "h2o_mean": h2o_mean,
                "improvement_percent": improvement,
                "lower_better": lower_better
            }
        
        # 执行配对t检验
        for metric in metrics:
            # 在某些指标中，较小的值表示更好的性能，需要调整备选假设
            if metric in ["peak_gpu_memory_mb", "ttft_ms", "tpot_ms"]:
                # H2O应该小于基线
                alternative = "less"
            else:
                # H2O应该大于基线
                alternative = "greater"
            
            # 为了公平比较，我们需要对相同的配置进行配对
            merged_data = pd.merge(
                baseline_df[["kv_cache_length", "batch_size", metric]],
                h2o_df[["kv_cache_length", "batch_size", metric]],
                on=["kv_cache_length", "batch_size"],
                suffixes=('_baseline', '_h2o')
            )
            
            if not merged_data.empty:
                baseline_values = merged_data[f"{metric}_baseline"].values
                h2o_values = merged_data[f"{metric}_h2o"].values
                
                # 执行配对t检验
                test_result = self.perform_paired_ttest(
                    baseline_values, 
                    h2o_values, 
                    metric, 
                    alternative=alternative
                )
                
                results["t_tests"][metric] = test_result
        
        # 分析影响H2O性能提升的因素
        if not h2o_df.empty and "heavy_ratio" in h2o_df.columns:
            for metric in metrics:
                x_vars = ["kv_cache_length", "batch_size", "heavy_ratio"]
                if "recent_ratio" in h2o_df.columns:
                    x_vars.append("recent_ratio")
                
                # 执行回归分析
                regression_result = self.regression_analysis(h2o_df, x_vars, metric)
                results["regression_analysis"][metric] = regression_result
        
        # 相关性分析
        if not h2o_df.empty:
            corr_vars = metrics.copy()
            # 添加实验参数
            for param in ["kv_cache_length", "batch_size"]:
                if param in h2o_df.columns:
                    corr_vars.append(param)
            
            # H2O特定参数
            for h2o_param in ["heavy_ratio", "recent_ratio"]:
                if h2o_param in h2o_df.columns:
                    corr_vars.append(h2o_param)
            
            # 执行相关性分析
            corr_result = self.correlation_analysis(h2o_df, corr_vars)
            results["correlation_analysis"] = corr_result
            
            # 生成相关性热图
            if output_dir:
                plt.figure(figsize=(10, 8))
                mask = np.zeros_like(corr_result["correlation_matrix"], dtype=bool)
                mask[np.triu_indices_from(mask)] = True
                
                sns.heatmap(
                    corr_result["correlation_matrix"], 
                    annot=True, 
                    fmt=".2f", 
                    cmap="coolwarm", 
                    mask=mask,
                    vmin=-1, 
                    vmax=1
                )
                plt.title("Correlation Matrix of Performance Metrics")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
                plt.close()
        
        # 保存分析结果
        if output_dir:
            with open(os.path.join(output_dir, "statistical_analysis.json"), "w") as f:
                import json
                
                # 将不可序列化的对象转换为字符串
                def json_serializable(obj):
                    if isinstance(obj, (pd.DataFrame, pd.Series)):
                        return obj.to_dict()
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.bool_):
                        return bool(obj)
                    else:
                        return str(obj)
                
                # 递归转换字典
                def convert_dict(d):
                    result = {}
                    for k, v in d.items():
                        if isinstance(v, dict):
                            result[k] = convert_dict(v)
                        elif isinstance(v, list):
                            result[k] = [convert_dict(item) if isinstance(item, dict) else json_serializable(item) for item in v]
                        else:
                            result[k] = json_serializable(v)
                    return result
                
                # 转换结果并保存
                serializable_results = convert_dict(results)
                json.dump(serializable_results, f, indent=2)
        
        return results 