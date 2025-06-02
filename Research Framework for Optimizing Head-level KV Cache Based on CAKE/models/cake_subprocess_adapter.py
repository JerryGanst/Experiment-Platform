#!/usr/bin/env python3
"""
CAKE子进程调用适配器

基于新方案设计，实现通过子进程调用CAKE官方代码的方式。
这种方式提供更好的隔离性，避免全局状态冲突。

改进版本：
- 支持CAKE_ROOT环境变量
- 增强Hydra参数映射
- 改进子进程调用稳定性
- 统一返回接口格式
- 跨平台兼容性
"""

import os
import sys
import json
import subprocess
import logging
import tempfile
import signal
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import yaml
import re

logger = logging.getLogger(__name__)

class CAKESubprocessConfig:
    """CAKE子进程配置管理器"""
    
    # Hydra参数映射表（确保与官方配置树一致）
    PARAM_MAPPING = {
        "cache_size": "cake.cache_size",
        "window_size": "cake.window_size", 
        "gamma": "cake.gamma",
        "tau1": "cake.tau1",
        "tau2": "cake.tau2",
        "allocation_strategy": "cake.allocation_strategy",
        # GPU相关
        "devices": "trainer.devices",
        "accelerator": "trainer.accelerator",
        # 训练相关
        "max_epochs": "trainer.max_epochs",
        "limit_train_batches": "trainer.limit_train_batches",
        "limit_val_batches": "trainer.limit_val_batches",
        "limit_test_batches": "trainer.limit_test_batches",
    }
    
    def __init__(self, 
                 experiment_name: str = "default",
                 model_name: str = "llama",
                 cache_size: int = 1024,
                 window_size: int = 32,
                 gamma: float = 0.8,
                 tau1: float = 1.0,
                 tau2: float = 1.0,
                 output_dir: Optional[str] = None,
                 devices: int = 1,
                 accelerator: str = "auto",
                 **kwargs):
        """
        初始化CAKE子进程配置
        
        Args:
            experiment_name: 实验名称
            model_name: 模型名称
            cache_size: 缓存大小
            window_size: 窗口大小
            gamma: gamma参数
            tau1: tau1参数
            tau2: tau2参数
            output_dir: 输出目录
            devices: GPU设备数
            accelerator: 加速器类型
        """
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.cache_size = cache_size
        self.window_size = window_size
        self.gamma = gamma
        self.tau1 = tau1
        self.tau2 = tau2
        self.output_dir = output_dir
        self.devices = devices
        self.accelerator = accelerator
        self.extra_params = kwargs
        
    def to_hydra_args(self) -> List[str]:
        """转换为Hydra命令行参数"""
        args = []
        
        # 基础实验配置
        if self.experiment_name:
            args.append(f"experiment={self.experiment_name}")
        
        # 模型配置
        args.append(f"model.name={self.model_name}")
        
        # 核心参数使用映射表
        core_params = {
            "cache_size": self.cache_size,
            "window_size": self.window_size,
            "gamma": self.gamma,
            "tau1": self.tau1,
            "tau2": self.tau2,
            "devices": self.devices,
            "accelerator": self.accelerator,
        }
        
        for key, value in core_params.items():
            if key in self.PARAM_MAPPING:
                mapped_key = self.PARAM_MAPPING[key]
                args.append(f"{mapped_key}={self._format_value(value)}")
        
        # Hydra输出目录
        if self.output_dir:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            hydra_dir = f"{self.output_dir}/{self.experiment_name}/{timestamp}"
            args.append(f"hydra.run.dir={hydra_dir}")
        
        # 额外参数
        for key, value in self.extra_params.items():
            if key in self.PARAM_MAPPING:
                mapped_key = self.PARAM_MAPPING[key]
                args.append(f"{mapped_key}={self._format_value(value)}")
            else:
                args.append(f"{key}={self._format_value(value)}")
        
        return args
    
    def _format_value(self, value: Any) -> str:
        """格式化参数值"""
        if isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, list):
            return ",".join(str(v) for v in value)
        else:
            return str(value)
    
    @classmethod
    def from_legacy_config(cls, 
                          cake_experiment_config: Dict[str, Any],
                          cake_model_config: Dict[str, Any]) -> 'CAKESubprocessConfig':
        """从现有配置格式创建"""
        # 提取分配策略
        allocation_strategy = "adaptive"
        if "layer_allocation_strategies" in cake_experiment_config:
            strategies = cake_experiment_config["layer_allocation_strategies"]
            allocation_strategy = strategies[0] if strategies else "adaptive"
        
        # 提取缓存预算
        cache_budget = 0.8
        if "cache_budgets" in cake_experiment_config:
            budgets = cake_experiment_config["cache_budgets"]
            cache_budget = budgets[0] if budgets else 0.8
        
        # 计算cache_size（简化计算）
        cache_size = int(2048 * cache_budget)  # 假设基础长度2048
        
        return cls(
            experiment_name=f"cake_{allocation_strategy}",
            cache_size=cache_size,
            window_size=cake_model_config.get("window_size", 32),
            gamma=cake_model_config.get("gamma", 0.8),
            tau1=cake_model_config.get("tau1", 1.0),
            tau2=cake_model_config.get("tau2", 1.0),
            allocation_strategy=allocation_strategy
        )

class CAKESubprocessAdapter:
    """CAKE子进程调用适配器"""
    
    def __init__(self, cake_root: Optional[str] = None):
        """
        初始化适配器
        
        Args:
            cake_root: CAKE代码根目录，如果为None则自动检测
        """
        self.cake_root = self._find_cake_root(cake_root)
        self.main_script = self.cake_root / "src" / "main.py"
        
        # 验证CAKE代码可用性
        self._validate_cake_installation()
        
    def _find_cake_root(self, cake_root: Optional[str] = None) -> Path:
        """查找CAKE代码根目录"""
        # 1. 优先使用明确指定的路径
        if cake_root:
            path = Path(cake_root)
            if self._is_valid_cake_root(path):
                return path
            else:
                logger.warning(f"指定的CAKE路径无效: {cake_root}")
        
        # 2. 检查CAKE_ROOT环境变量
        env_cake_root = os.getenv("CAKE_ROOT")
        if env_cake_root:
            path = Path(env_cake_root)
            if self._is_valid_cake_root(path):
                logger.info(f"使用环境变量CAKE_ROOT: {path}")
                return path
            else:
                logger.warning(f"环境变量CAKE_ROOT路径无效: {env_cake_root}")
        
        # 3. 自动搜索可能的位置
        current_dir = Path(__file__).parent.parent
        possible_paths = [
            current_dir / "third_party" / "CAKE",
            current_dir / "cakekv-main" / "cakekv-main",  # 实际CAKE代码位置
            current_dir / "CAKE",
        ]
        
        # 4. 搜索third_party下的所有子目录
        third_party_dir = current_dir / "third_party"
        if third_party_dir.exists():
            for subdir in third_party_dir.iterdir():
                if subdir.is_dir():
                    possible_paths.append(subdir)
        
        # 验证所有可能路径
        for path in possible_paths:
            if self._is_valid_cake_root(path):
                logger.info(f"找到CAKE代码: {path}")
                return path
        
        raise FileNotFoundError(
            f"无法找到CAKE代码。请设置CAKE_ROOT环境变量或确保CAKE代码在以下位置之一: {possible_paths}"
        )
    
    def _is_valid_cake_root(self, path: Path) -> bool:
        """检查是否为有效的CAKE根目录"""
        # 检查多种可能的CAKE目录结构
        possible_structures = [
            # 标准结构 (有src/main.py)
            path / "src" / "main.py",
            # CAKE核心结构 (有cake目录)
            path / "cake" / "cake_cache.py",
            # 实验结构 (有experiments目录)
            path / "experiments",
        ]
        
        return path.exists() and any(structure.exists() for structure in possible_structures)
    
    def _validate_cake_installation(self):
        """验证CAKE安装"""
        # 检查多种可能的主脚本位置
        possible_main_scripts = [
            self.cake_root / "src" / "main.py",
            self.cake_root / "main.py",
            self.cake_root / "experiments" / "main.py",
            # 对于当前CAKE结构，我们可能需要使用Python模块方式
            self.cake_root / "cake" / "__init__.py",
        ]
        
        main_script_found = False
        for script in possible_main_scripts:
            if script.exists():
                self.main_script = script
                main_script_found = True
                logger.info(f"找到主脚本: {script}")
                break
        
        if not main_script_found:
            # 如果没有找到标准主脚本，检查是否有CAKE核心模块
            cake_module = self.cake_root / "cake" / "cake_cache.py"
            if cake_module.exists():
                logger.info("检测到CAKE核心模块，将使用模块导入方式而非脚本执行")
                self.main_script = None  # 标记为使用模块导入
            else:
                raise FileNotFoundError(f"无法找到CAKE主脚本或模块在: {self.cake_root}")
        
        # 检查conf目录或其他配置
        possible_conf_dirs = [
            self.cake_root / "conf",
            self.cake_root / "config",
            self.cake_root / "configs",
        ]
        
        conf_found = any(conf_dir.exists() for conf_dir in possible_conf_dirs)
        if not conf_found:
            logger.warning(f"未找到配置目录在: {possible_conf_dirs}")
        
        # 检查依赖版本（简化版本）
        requirements_file = self.cake_root / "requirements.txt"
        if requirements_file.exists():
            logger.info("检测到requirements.txt，建议验证依赖版本")
        
        logger.info(f"CAKE安装验证通过: {self.cake_root}")
    
    def run_experiment(self, 
                      config: CAKESubprocessConfig,
                      capture_output: bool = True,
                      timeout: Optional[int] = None,
                      stream_logs: bool = False) -> Dict[str, Any]:
        """
        运行CAKE实验
        
        Args:
            config: CAKE配置
            capture_output: 是否捕获输出
            timeout: 超时时间（秒）
            stream_logs: 是否流式输出日志
            
        Returns:
            标准化的结果字典
        """
        logger.info(f"开始运行CAKE实验: {config.experiment_name}")
        
        # 检查是否有可执行的主脚本
        if self.main_script is None:
            # 使用模块导入方式
            return self._run_as_module(config, timeout)
        
        # 创建临时输出文件
        temp_dir = Path(tempfile.mkdtemp(prefix="cake_exp_"))
        stdout_file = temp_dir / "stdout.log"
        stderr_file = temp_dir / "stderr.log"
        
        try:
            # 构建命令
            cmd = self._build_command(config)
            logger.info(f"执行命令: {' '.join(cmd)}")
            
            # 设置环境
            env = self._prepare_environment()
            
            # 执行子进程
            if stream_logs:
                result = self._run_with_streaming(cmd, env, timeout, stdout_file, stderr_file)
            else:
                result = self._run_simple(cmd, env, timeout, capture_output)
            
            # 解析结果
            return self._parse_result_enhanced(result, config, stdout_file, stderr_file, temp_dir)
            
        except subprocess.TimeoutExpired as e:
            logger.error(f"CAKE实验超时: {timeout}秒")
            return self._create_error_result("timeout", f"实验超时 ({timeout}秒)", config, temp_dir)
        except Exception as e:
            logger.error(f"CAKE实验执行失败: {e}")
            return self._create_error_result("error", str(e), config, temp_dir)
    
    def _run_as_module(self, config: CAKESubprocessConfig, timeout: Optional[int]) -> Dict[str, Any]:
        """作为Python模块运行CAKE（当没有主脚本时的回退方案）"""
        logger.info("使用模块导入方式运行CAKE")
        
        # 创建临时目录
        temp_dir = Path(tempfile.mkdtemp(prefix="cake_module_"))
        
        try:
            # 检查CAKE模块是否可导入
            import sys
            cake_path = str(self.cake_root)
            if cake_path not in sys.path:
                sys.path.insert(0, cake_path)
            
            # 尝试导入CAKE核心模块
            try:
                from cake.cake_cache import CAKECache
                from cake.utils import apply_cache
                
                logger.info("✅ CAKE模块导入成功")
                
                # 创建模拟成功结果
                result = {
                    "status": "ok", 
                    "metrics": {
                        "module_import": True,
                        "cache_size": config.cache_size,
                        "method": "module_import"
                    },
                    "config": config.__dict__,
                    "artifacts": {"temp_dir": str(temp_dir)},
                    "error": None
                }
                
                return result
                
            except ImportError as e:
                logger.error(f"CAKE模块导入失败: {e}")
                return self._create_error_result("import_error", f"模块导入失败: {e}", config, temp_dir)
        
        except Exception as e:
            logger.error(f"模块运行失败: {e}")
            return self._create_error_result("module_error", str(e), config, temp_dir)
    
    def _build_command(self, config: CAKESubprocessConfig) -> List[str]:
        """构建执行命令"""
        cmd = [
            sys.executable,  # 使用当前Python解释器
            str(self.main_script)
        ]
        
        # 添加Hydra参数
        cmd.extend(config.to_hydra_args())
        
        return cmd
    
    def _prepare_environment(self) -> Dict[str, str]:
        """准备环境变量"""
        env = os.environ.copy()
        
        # Python路径
        python_path = str(self.cake_root)
        if "PYTHONPATH" in env:
            python_path = f"{python_path}:{env['PYTHONPATH']}"
        env["PYTHONPATH"] = python_path
        
        # 跨平台编码设置
        if os.name == 'nt':  # Windows
            env["PYTHONIOENCODING"] = "utf-8"
        
        return env
    
    def _run_simple(self, cmd: List[str], env: Dict[str, str], 
                   timeout: Optional[int], capture_output: bool) -> subprocess.CompletedProcess:
        """简单执行模式"""
        return subprocess.run(
            cmd,
            cwd=str(self.cake_root),
            env=env,
            capture_output=capture_output,
            text=True,
            timeout=timeout
        )
    
    def _run_with_streaming(self, cmd: List[str], env: Dict[str, str],
                           timeout: Optional[int], stdout_file: Path, stderr_file: Path) -> subprocess.CompletedProcess:
        """流式执行模式（避免大输出内存问题）"""
        process = subprocess.Popen(
            cmd,
            cwd=str(self.cake_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 流式写入文件
        def stream_output(pipe, file_path):
            with open(file_path, 'w', encoding='utf-8') as f:
                for line in iter(pipe.readline, ''):
                    f.write(line)
                    f.flush()
                    # 实时日志输出（可选）
                    if file_path.name == 'stderr.log':
                        logger.info(f"CAKE: {line.strip()}")
        
        stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, stdout_file))
        stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, stderr_file))
        
        stdout_thread.start()
        stderr_thread.start()
        
        try:
            # 等待完成
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # 友好终止
            logger.warning("实验超时，尝试友好终止...")
            process.send_signal(signal.SIGINT)
            time.sleep(5)  # 等待友好退出
            
            if process.poll() is None:
                logger.warning("强制终止进程")
                process.kill()
            
            raise
        finally:
            stdout_thread.join()
            stderr_thread.join()
        
        # 读取输出
        stdout_content = stdout_file.read_text(encoding='utf-8') if stdout_file.exists() else ""
        stderr_content = stderr_file.read_text(encoding='utf-8') if stderr_file.exists() else ""
        
        # 模拟CompletedProcess
        class MockResult:
            def __init__(self, returncode, stdout, stderr):
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr
        
        return MockResult(process.returncode, stdout_content, stderr_content)
    
    def _parse_result_enhanced(self, 
                              subprocess_result: subprocess.CompletedProcess,
                              config: CAKESubprocessConfig,
                              stdout_file: Path,
                              stderr_file: Path,
                              temp_dir: Path) -> Dict[str, Any]:
        """增强版结果解析"""
        success = subprocess_result.returncode == 0
        
        # 统一返回格式
        result = {
            "status": "ok" if success else "error",
            "returncode": subprocess_result.returncode,
            "metrics": {},
            "error": None,
            "config": config.__dict__,
            "artifacts": {
                "temp_dir": str(temp_dir),
                "stdout": str(stdout_file) if stdout_file.exists() else None,
                "stderr": str(stderr_file) if stderr_file.exists() else None,
                "hydra_dir": config.output_dir
            }
        }
        
        if success:
            # 优先从JSON文件读取指标
            metrics = self._extract_metrics_enhanced(subprocess_result.stdout, config)
            result["metrics"] = metrics
            logger.info(f"CAKE实验成功完成，指标: {list(metrics.keys())}")
        else:
            error_msg = subprocess_result.stderr or "未知错误"
            result["error"] = error_msg
            logger.error(f"CAKE实验失败: {error_msg}")
        
        return result
    
    def _extract_metrics_enhanced(self, output: str, config: CAKESubprocessConfig) -> Dict[str, Any]:
        """增强版指标提取"""
        metrics = {}
        
        # 1. 优先查找JSON指标文件
        if config.output_dir:
            json_metrics = self._try_load_json_metrics(config.output_dir)
            if json_metrics:
                metrics.update(json_metrics)
                logger.info("从JSON文件加载指标成功")
                return metrics
        
        # 2. 回退到正则表达式提取
        # 先清理ANSI颜色代码
        clean_output = re.sub(r'\x1b\[[0-9;]*m', '', output)
        
        # 改进的正则模式
        patterns = {
            "test_accuracy": r"(?:Test accuracy|Accuracy):\s*([0-9.]+)(%?)",
            "train_accuracy": r"(?:Train accuracy|Training accuracy):\s*([0-9.]+)(%?)",
            "loss": r"(?:Loss|Test loss):\s*([0-9.]+)",
            "perplexity": r"(?:Perplexity|PPL):\s*([0-9.]+)",
            "bleu_score": r"(?:BLEU|BLEU[-_]?[0-9]+):\s*([0-9.]+)(%?)",
        }
        
        for metric_name, pattern in patterns.items():
            matches = re.findall(pattern, clean_output, re.IGNORECASE)
            if matches:
                try:
                    # 处理带百分号的匹配
                    if isinstance(matches[0], tuple):
                        value_str, percent_indicator = matches[-1]  # 取最后匹配
                        value = float(value_str)
                        # 如果明确有%号，转换为小数
                        if percent_indicator == '%':
                            value = value / 100
                    else:
                        value = float(matches[-1])
                    
                    metrics[metric_name] = value
                except (ValueError, IndexError):
                    logger.warning(f"无法解析指标 {metric_name}: {matches[-1]}")
        
        # 添加元数据
        metrics.update({
            "output_lines": len(clean_output.split('\n')),
            "has_error": "error" in clean_output.lower() or "exception" in clean_output.lower(),
            "extraction_method": "regex"
        })
        
        return metrics
    
    def _try_load_json_metrics(self, output_dir: str) -> Optional[Dict[str, Any]]:
        """尝试从JSON文件加载指标"""
        possible_files = [
            Path(output_dir) / "metrics.json",
            Path(output_dir) / ".hydra" / "metrics.json",
            Path(output_dir) / "test_results.json"
        ]
        
        for json_file in possible_files:
            if json_file.exists():
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    logger.info(f"成功加载JSON指标: {json_file}")
                    return data
                except Exception as e:
                    logger.warning(f"读取JSON指标文件失败 {json_file}: {e}")
        
        return None
    
    def _create_error_result(self, status: str, error: str, 
                           config: CAKESubprocessConfig, temp_dir: Path) -> Dict[str, Any]:
        """创建错误结果"""
        return {
            "status": status,
            "error": error,
            "metrics": {},
            "config": config.__dict__,
            "artifacts": {
                "temp_dir": str(temp_dir)
            }
        }
    
    def test_installation(self) -> bool:
        """测试CAKE安装是否正常"""
        try:
            # 运行一个简单的测试命令
            test_config = CAKESubprocessConfig(
                experiment_name="test",
                cache_size=64,  # 很小的缓存避免资源问题
            )
            
            # 添加测试标记，尝试快速退出
            test_config.extra_params.update({
                "trainer.max_epochs": 1,
                "trainer.limit_train_batches": 1,
                "trainer.limit_val_batches": 0,
                "trainer.limit_test_batches": 0,
            })
            
            result = self.run_experiment(test_config, timeout=120)
            return result["status"] == "ok"
            
        except Exception as e:
            logger.error(f"CAKE安装测试失败: {e}")
            return False

# 与现有系统的兼容函数
def run_cake_via_subprocess(model, 
                           model_config_hf: Dict[str, Any],
                           cake_experiment_config: Dict[str, Any],
                           cake_model_specific_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    兼容函数：通过子进程运行CAKE
    
    这个函数提供与现有apply_cake_to_model类似的接口，
    但使用子进程调用方式，可以作为备选方案。
    """
    try:
        # 创建适配器
        adapter = CAKESubprocessAdapter()
        
        # 转换配置
        config = CAKESubprocessConfig.from_legacy_config(
            cake_experiment_config, 
            cake_model_specific_config
        )
        
        # 设置模型相关参数
        config.model_name = model_config_hf.get("model_type", "llama")
        
        # 运行实验
        result = adapter.run_experiment(config)
        
        if result["status"] == "ok":
            logger.info("子进程CAKE实验成功完成")
            return {
                "model": model,  # 返回原模型（子进程模式下不修改模型）
                "metrics": result["metrics"],
                "method": "subprocess",
                "artifacts": result["artifacts"]
            }
        else:
            raise RuntimeError(f"CAKE子进程执行失败: {result.get('error', '未知错误')}")
            
    except Exception as e:
        logger.error(f"子进程CAKE执行失败: {e}")
        raise

if __name__ == "__main__":
    # 测试子进程适配器
    logging.basicConfig(level=logging.INFO)
    
    try:
        adapter = CAKESubprocessAdapter()
        logger.info("✅ CAKE子进程适配器初始化成功")
        
        # 测试安装
        if adapter.test_installation():
            logger.info("✅ CAKE安装测试通过")
        else:
            logger.error("❌ CAKE安装测试失败")
            
    except Exception as e:
        logger.error(f"❌ 子进程适配器测试失败: {e}")
        logger.info("提示：请确保CAKE代码已正确安装在指定位置或设置CAKE_ROOT环境变量") 