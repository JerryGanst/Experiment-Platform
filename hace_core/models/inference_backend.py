"""
推理后端抽象层

提供统一的推理接口，支持 HuggingFace transformers 和 VLLM 两种后端。
通过抽象层解耦上层代码与具体推理实现，便于切换后端和扩展新引擎。

核心组件：
- BaseInferenceBackend: 推理后端抽象基类
- HuggingFaceBackend: HuggingFace transformers 后端实现
- VLLMBackend: VLLM 后端实现（进程内模式和Server模式）
- create_inference_backend: 后端工厂函数
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GenerationOutput:
    """生成输出的统一数据结构"""
    text: str  # 生成的文本
    token_ids: Optional[List[int]] = None  # token IDs
    attention_weights: Optional[np.ndarray] = None  # 注意力权重 (可选)
    finish_reason: str = "unknown"  # 完成原因: "stop", "length", "error"
    prompt_tokens: int = 0  # prompt token数
    completion_tokens: int = 0  # 生成 token数
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据


@dataclass
class GenerationConfig:
    """生成配置的统一数据结构"""
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    do_sample: bool = False
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None
    return_attention: bool = False  # 是否返回注意力权重


class BaseInferenceBackend(ABC):
    """
    推理后端抽象基类

    定义统一的推理接口，所有后端实现必须继承此类。
    """

    def __init__(self, model_config: Dict[str, Any], backend_config: Optional[Dict[str, Any]] = None):
        """
        初始化推理后端

        Args:
            model_config: 模型配置字典（包含 model_name_or_path, precision 等）
            backend_config: 后端特定配置（如 VLLM_CONFIG）
        """
        self.model_config = model_config
        self.backend_config = backend_config or {}
        self.is_initialized = False
        self._tokenizer = None

    @property
    def backend_name(self) -> str:
        """返回后端名称"""
        return "base"

    @abstractmethod
    def initialize(self) -> None:
        """初始化后端（加载模型等）"""
        pass

    @abstractmethod
    def generate(
        self,
        prompts: Union[str, List[str]],
        config: Optional[GenerationConfig] = None
    ) -> Union[GenerationOutput, List[GenerationOutput]]:
        """
        执行文本生成

        Args:
            prompts: 单个prompt或prompt列表
            config: 生成配置

        Returns:
            单个或多个 GenerationOutput
        """
        pass

    @abstractmethod
    def get_tokenizer(self):
        """获取tokenizer实例"""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """清理资源"""
        pass

    def encode(self, text: str) -> List[int]:
        """编码文本为token IDs"""
        tokenizer = self.get_tokenizer()
        return tokenizer.encode(text, add_special_tokens=True)

    def decode(self, token_ids: List[int]) -> str:
        """解码token IDs为文本"""
        tokenizer = self.get_tokenizer()
        return tokenizer.decode(token_ids, skip_special_tokens=True)

    def supports_attention_output(self) -> bool:
        """是否支持输出注意力权重"""
        return False


class HuggingFaceBackend(BaseInferenceBackend):
    """
    HuggingFace transformers 后端实现

    封装现有的 model_loader.py 逻辑，提供标准推理接口。
    支持 output_attentions=True 获取注意力权重。
    """

    def __init__(self, model_config: Dict[str, Any], backend_config: Optional[Dict[str, Any]] = None):
        super().__init__(model_config, backend_config)
        self._model = None
        self._tokenizer = None

    @property
    def backend_name(self) -> str:
        return "huggingface"

    def initialize(self) -> None:
        """初始化 HuggingFace 模型和 tokenizer"""
        if self.is_initialized:
            return

        logger.info(f"Initializing HuggingFace backend for model: {self.model_config['model_name_or_path']}")

        # 使用现有的 model_loader 逻辑
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = self.model_config["model_name_or_path"]
        precision = self.model_config.get("precision", "fp16")

        # 确定数据类型
        torch_dtype = torch.float16
        if precision == "bf16" and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        elif precision == "fp32":
            torch_dtype = torch.float32

        # 显存管理
        memory_mgmt = self.model_config.get("memory_management", {})
        auto_max_memory = memory_mgmt.get("auto_max_memory", True)
        buffer_ratio = memory_mgmt.get("memory_buffer_ratio", 0.05)

        max_memory_config = None
        if torch.cuda.is_available() and auto_max_memory:
            max_memory_config = {}
            for i in range(torch.cuda.device_count()):
                total_memory = torch.cuda.get_device_properties(i).total_memory
                total_memory_mb = int(total_memory / (1024 * 1024))
                available_ratio = 1.0 - buffer_ratio
                max_memory_mb = int(total_memory_mb * available_ratio)

                # 根据GPU大小设置上限
                if total_memory_mb >= 24000:
                    max_memory_mb = min(23000, max_memory_mb)
                elif total_memory_mb >= 16000:
                    max_memory_mb = min(15000, max_memory_mb)
                elif total_memory_mb >= 12000:
                    max_memory_mb = min(11000, max_memory_mb)
                elif total_memory_mb >= 8000:
                    max_memory_mb = min(7000, max_memory_mb)

                max_memory_config[i] = f"{max_memory_mb}MB"
                logger.info(f"GPU {i}: 设置显存上限 {max_memory_mb}MB")

        # 加载模型
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": "auto",
            "low_cpu_mem_usage": True
        }
        if max_memory_config:
            model_kwargs["max_memory"] = max_memory_config

        self._model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

        # 加载tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self.is_initialized = True
        logger.info("HuggingFace backend initialized successfully")

    def generate(
        self,
        prompts: Union[str, List[str]],
        config: Optional[GenerationConfig] = None
    ) -> Union[GenerationOutput, List[GenerationOutput]]:
        """执行文本生成"""
        if not self.is_initialized:
            self.initialize()

        config = config or GenerationConfig()
        single_input = isinstance(prompts, str)
        if single_input:
            prompts = [prompts]

        results = []
        for prompt in prompts:
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            prompt_length = inputs["input_ids"].shape[1]

            # 构建生成参数
            gen_kwargs = {
                "max_new_tokens": config.max_new_tokens,
                "do_sample": config.do_sample or config.temperature > 0,
                "pad_token_id": self._tokenizer.pad_token_id,
            }

            if config.temperature > 0:
                gen_kwargs["temperature"] = config.temperature
            if config.top_p < 1.0:
                gen_kwargs["top_p"] = config.top_p
            if config.top_k > 0:
                gen_kwargs["top_k"] = config.top_k
            if config.return_attention:
                gen_kwargs["output_attentions"] = True
                gen_kwargs["return_dict_in_generate"] = True

            # 执行生成
            with torch.no_grad():
                outputs = self._model.generate(inputs["input_ids"], **gen_kwargs)

            # 处理输出
            if config.return_attention and hasattr(outputs, "attentions"):
                generated_ids = outputs.sequences[0]
                attention_weights = outputs.attentions
                # 将注意力权重转换为numpy
                attn_np = [
                    [layer_attn[0].cpu().numpy() for layer_attn in step_attn]
                    for step_attn in attention_weights
                ] if attention_weights else None
            else:
                generated_ids = outputs[0] if isinstance(outputs, tuple) else outputs[0]
                attn_np = None

            # 解码生成的文本
            generated_text = self._tokenizer.decode(
                generated_ids[prompt_length:],
                skip_special_tokens=True
            )

            completion_tokens = len(generated_ids) - prompt_length

            results.append(GenerationOutput(
                text=generated_text,
                token_ids=generated_ids.tolist(),
                attention_weights=attn_np,
                finish_reason="stop" if completion_tokens < config.max_new_tokens else "length",
                prompt_tokens=prompt_length,
                completion_tokens=completion_tokens,
            ))

        return results[0] if single_input else results

    def get_tokenizer(self):
        """获取tokenizer"""
        if not self.is_initialized:
            self.initialize()
        return self._tokenizer

    def get_model(self):
        """获取原始HuggingFace模型（用于CAKE/H2O等优化）"""
        if not self.is_initialized:
            self.initialize()
        return self._model

    def cleanup(self) -> None:
        """清理资源"""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.is_initialized = False
        logger.info("HuggingFace backend cleaned up")

    def supports_attention_output(self) -> bool:
        """HuggingFace后端支持注意力输出"""
        return True


class VLLMBackend(BaseInferenceBackend):
    """
    VLLM 后端实现

    支持两种模式：
    1. 进程内模式 (inprocess): 直接在Python进程内加载VLLM引擎
    2. Server模式 (server): 通过HTTP请求调用VLLM API服务器
    """

    def __init__(self, model_config: Dict[str, Any], backend_config: Optional[Dict[str, Any]] = None):
        super().__init__(model_config, backend_config)
        self._llm = None  # VLLM LLM实例（进程内模式）
        self._client = None  # HTTP客户端（Server模式）
        self._tokenizer = None
        self._mode = backend_config.get("mode", "inprocess") if backend_config else "inprocess"

    @property
    def backend_name(self) -> str:
        return f"vllm-{self._mode}"

    def initialize(self) -> None:
        """初始化VLLM后端"""
        if self.is_initialized:
            return

        if self._mode == "inprocess":
            self._initialize_inprocess()
        elif self._mode == "server":
            self._initialize_server()
        else:
            raise ValueError(f"Unknown VLLM mode: {self._mode}")

        self.is_initialized = True

    def _initialize_inprocess(self) -> None:
        """初始化进程内VLLM引擎"""
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "VLLM is not installed. Please install it with: pip install vllm\n"
                f"Original error: {e}"
            )

        model_path = self.model_config["model_name_or_path"]
        vllm_config = self.backend_config or {}

        logger.info(f"Initializing VLLM inprocess engine for model: {model_path}")

        # 构建VLLM参数
        llm_kwargs = {
            "model": model_path,
            "tensor_parallel_size": vllm_config.get("tensor_parallel_size", 1),
            "gpu_memory_utilization": vllm_config.get("gpu_memory_utilization", 0.90),
            "max_model_len": vllm_config.get("max_model_len", 4096),
            "enforce_eager": vllm_config.get("enforce_eager", False),
            "trust_remote_code": vllm_config.get("trust_remote_code", True),
            "seed": vllm_config.get("seed", 42),
            "swap_space": vllm_config.get("swap_space", 4),  # CPU交换空间大小(GB)
        }

        # 数据类型处理
        dtype = vllm_config.get("dtype", "auto")
        if dtype != "auto":
            llm_kwargs["dtype"] = dtype

        # 量化配置
        quantization = vllm_config.get("quantization")
        if quantization:
            llm_kwargs["quantization"] = quantization

        # 流水线并行
        pp_size = vllm_config.get("pipeline_parallel_size", 1)
        if pp_size > 1:
            llm_kwargs["pipeline_parallel_size"] = pp_size

        self._llm = LLM(**llm_kwargs)

        # 获取tokenizer
        self._tokenizer = self._llm.get_tokenizer()

        logger.info("VLLM inprocess engine initialized successfully")

    def _initialize_server(self) -> None:
        """初始化VLLM Server客户端"""
        try:
            import httpx
        except ImportError as e:
            raise ImportError(
                "httpx is not installed. Please install it with: pip install httpx\n"
                f"Original error: {e}"
            )

        vllm_config = self.backend_config or {}
        server_url = vllm_config.get("server_url", "http://localhost:8000")
        timeout = vllm_config.get("timeout", 300)

        logger.info(f"Initializing VLLM server client for: {server_url}")

        # 创建HTTP客户端
        self._client = httpx.Client(
            base_url=server_url,
            timeout=timeout,
        )

        # 尝试连接验证
        try:
            response = self._client.get("/health")
            if response.status_code != 200:
                logger.warning(f"VLLM server health check returned status: {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not verify VLLM server connection: {e}")

        # Server模式下，tokenizer需要单独加载
        from transformers import AutoTokenizer
        model_path = self.model_config["model_name_or_path"]
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        logger.info("VLLM server client initialized successfully")

    def generate(
        self,
        prompts: Union[str, List[str]],
        config: Optional[GenerationConfig] = None
    ) -> Union[GenerationOutput, List[GenerationOutput]]:
        """执行文本生成"""
        if not self.is_initialized:
            self.initialize()

        config = config or GenerationConfig()
        single_input = isinstance(prompts, str)
        if single_input:
            prompts = [prompts]

        if self._mode == "inprocess":
            results = self._generate_inprocess(prompts, config)
        else:
            results = self._generate_server(prompts, config)

        return results[0] if single_input else results

    def _generate_inprocess(
        self,
        prompts: List[str],
        config: GenerationConfig
    ) -> List[GenerationOutput]:
        """进程内模式生成"""
        from vllm import SamplingParams

        # 构建采样参数
        sampling_params = SamplingParams(
            max_tokens=config.max_new_tokens,
            temperature=config.temperature if config.temperature > 0 else 0.0,
            top_p=config.top_p,
            top_k=config.top_k if config.top_k > 0 else -1,
            presence_penalty=config.presence_penalty,
            frequency_penalty=config.frequency_penalty,
            stop=config.stop_sequences,
        )

        # 执行生成
        outputs = self._llm.generate(prompts, sampling_params)

        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            token_ids = output.outputs[0].token_ids
            finish_reason = output.outputs[0].finish_reason

            results.append(GenerationOutput(
                text=generated_text,
                token_ids=list(token_ids) if token_ids else None,
                attention_weights=None,  # VLLM默认不返回注意力
                finish_reason=finish_reason or "unknown",
                prompt_tokens=len(output.prompt_token_ids),
                completion_tokens=len(token_ids) if token_ids else 0,
            ))

        return results

    def _generate_server(
        self,
        prompts: List[str],
        config: GenerationConfig
    ) -> List[GenerationOutput]:
        """Server模式生成，支持VLLM原生API和OpenAI兼容API"""
        vllm_config = self.backend_config or {}
        max_retries = vllm_config.get("max_retries", 3)
        # 检测API类型：优先使用OpenAI兼容API
        api_type = vllm_config.get("api_type", "openai")  # "openai" 或 "native"

        results = []
        for prompt in prompts:
            # 根据API类型构建请求
            if api_type == "openai":
                # OpenAI兼容API格式
                request_body = {
                    "model": self.model_config.get("model_name_or_path", "default"),
                    "prompt": prompt,
                    "max_tokens": config.max_new_tokens,
                    "temperature": config.temperature if config.temperature > 0 else 0.0,
                    "top_p": config.top_p,
                    "presence_penalty": config.presence_penalty,
                    "frequency_penalty": config.frequency_penalty,
                }
                endpoint = "/v1/completions"
            else:
                # VLLM原生API格式
                request_body = {
                    "prompt": prompt,
                    "max_tokens": config.max_new_tokens,
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "presence_penalty": config.presence_penalty,
                    "frequency_penalty": config.frequency_penalty,
                }
                endpoint = "/generate"

            if config.top_k > 0:
                request_body["top_k"] = config.top_k
            if config.stop_sequences:
                request_body["stop"] = config.stop_sequences

            # 发送请求（带重试）
            last_error = None
            for attempt in range(max_retries):
                try:
                    response = self._client.post(endpoint, json=request_body)
                    response.raise_for_status()
                    data = response.json()

                    # 根据API类型解析响应
                    if api_type == "openai":
                        # OpenAI兼容格式: {"choices": [{"text": "...", "finish_reason": "..."}], "usage": {...}}
                        choices = data.get("choices", [{}])
                        if choices:
                            choice = choices[0]
                            generated_text = choice.get("text", "")
                            finish_reason = choice.get("finish_reason", "unknown")
                        else:
                            generated_text = ""
                            finish_reason = "error"
                        usage = data.get("usage", {})
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                    else:
                        # VLLM原生格式: {"text": [...]} 或 {"text": "..."}
                        text_data = data.get("text", "")
                        if isinstance(text_data, list):
                            generated_text = text_data[0] if text_data else ""
                        else:
                            generated_text = text_data
                        finish_reason = data.get("finish_reason", "unknown")
                        prompt_tokens = data.get("prompt_tokens", 0)
                        completion_tokens = data.get("completion_tokens", 0)

                    results.append(GenerationOutput(
                        text=generated_text,
                        token_ids=None,
                        attention_weights=None,
                        finish_reason=finish_reason,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    ))
                    break

                except Exception as e:
                    last_error = e
                    logger.warning(f"VLLM server request failed (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(2 ** attempt)  # 指数退避

            else:
                # 所有重试都失败
                logger.error(f"VLLM server request failed after {max_retries} attempts")
                results.append(GenerationOutput(
                    text="",
                    finish_reason="error",
                    metadata={"error": str(last_error)},
                ))

        return results

    def get_tokenizer(self):
        """获取tokenizer"""
        if not self.is_initialized:
            self.initialize()
        return self._tokenizer

    def cleanup(self) -> None:
        """清理资源"""
        if self._llm is not None:
            del self._llm
            self._llm = None
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # 确保所有CUDA操作完成
            torch.cuda.empty_cache()
        self.is_initialized = False
        logger.info("VLLM backend cleaned up")

    def supports_attention_output(self) -> bool:
        """VLLM后端默认不支持注意力输出"""
        return False

    def get_llm(self):
        """获取VLLM LLM实例（仅进程内模式）"""
        if self._mode != "inprocess":
            raise RuntimeError("get_llm() is only available in inprocess mode")
        if not self.is_initialized:
            self.initialize()
        return self._llm


def create_inference_backend(
    model_config: Dict[str, Any],
    vllm_config: Optional[Dict[str, Any]] = None
) -> BaseInferenceBackend:
    """
    推理后端工厂函数

    根据配置创建对应的推理后端实例。

    Args:
        model_config: 模型配置（必须包含 inference_backend 字段）
        vllm_config: VLLM特定配置（当使用VLLM后端时）

    Returns:
        BaseInferenceBackend 实例

    Example:
        >>> from hace_core.config import MODEL_CONFIG, VLLM_CONFIG
        >>> backend = create_inference_backend(MODEL_CONFIG, VLLM_CONFIG)
        >>> output = backend.generate("Hello, world!")
        >>> print(output.text)
    """
    backend_type = model_config.get("inference_backend", "hf").lower()

    if backend_type in ("hf", "huggingface", "transformers"):
        logger.info("Creating HuggingFace backend")
        return HuggingFaceBackend(model_config, vllm_config)

    elif backend_type == "vllm":
        if vllm_config is None:
            # 使用默认VLLM配置
            from hace_core.config import VLLM_CONFIG
            vllm_config = VLLM_CONFIG
        logger.info(f"Creating VLLM backend (mode: {vllm_config.get('mode', 'inprocess')})")
        return VLLMBackend(model_config, vllm_config)

    else:
        raise ValueError(
            f"Unknown inference backend: {backend_type}. "
            "Supported backends: 'hf' (HuggingFace), 'vllm' (VLLM)"
        )


def check_vllm_available() -> Tuple[bool, str]:
    """
    检查VLLM是否可用

    Returns:
        (is_available, message): 是否可用及说明信息
    """
    try:
        import vllm
        version = getattr(vllm, "__version__", "unknown")
        return True, f"VLLM {version} is available"
    except ImportError:
        return False, "VLLM is not installed. Install with: pip install vllm"
    except Exception as e:
        return False, f"VLLM import failed: {e}"
