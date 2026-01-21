#!/usr/bin/env python3
"""
推理后端集成测试

测试 HuggingFace 和 VLLM 两种推理后端的功能。
支持自动检测后端可用性并跳过不可用的测试。

使用方法：
    # 运行所有可用后端测试
    pytest tests/integration/test_inference_backend.py -v

    # 只运行 HuggingFace 后端测试
    pytest tests/integration/test_inference_backend.py -v -k "hf"

    # 只运行 VLLM 后端测试（需要安装VLLM）
    pytest tests/integration/test_inference_backend.py -v -k "vllm"
"""

import os
import sys
import pytest
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import Mock, patch, MagicMock

# 添加项目路径
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# ============== 后端可用性检测 ==============

def is_vllm_available() -> bool:
    """检查 VLLM 是否可用"""
    try:
        import vllm
        return True
    except ImportError:
        return False


def is_torch_cuda_available() -> bool:
    """检查 CUDA 是否可用"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# 测试标记
requires_cuda = pytest.mark.skipif(
    not is_torch_cuda_available(),
    reason="CUDA not available"
)

requires_vllm = pytest.mark.skipif(
    not is_vllm_available(),
    reason="VLLM not installed"
)


# ============== 测试配置 ==============

@pytest.fixture
def model_config() -> Dict[str, Any]:
    """基础模型配置"""
    return {
        "model_name_or_path": os.environ.get(
            "TEST_MODEL_PATH",
            "mistralai/Mistral-7B-Instruct-v0.3"
        ),
        "precision": "fp16",
        "inference_backend": "hf",
    }


@pytest.fixture
def vllm_config() -> Dict[str, Any]:
    """VLLM 配置"""
    return {
        "mode": "inprocess",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.90,
        "max_model_len": 2048,
        "enforce_eager": True,  # 测试时使用eager模式
        "trust_remote_code": True,
        "seed": 42,
    }


# ============== 基础功能测试 ==============

class TestInferenceBackendImport:
    """测试模块导入"""

    def test_import_inference_backend(self):
        """测试 inference_backend 模块导入"""
        from src.hace.models.inference_backend import (
            BaseInferenceBackend,
            HuggingFaceBackend,
            VLLMBackend,
            create_inference_backend,
            GenerationConfig,
            GenerationOutput,
        )

        assert BaseInferenceBackend is not None
        assert HuggingFaceBackend is not None
        assert VLLMBackend is not None
        assert create_inference_backend is not None

    def test_import_attention_collector(self):
        """测试 attention_collector 模块导入"""
        from src.hace.models.attention_collector import (
            AttentionCollector,
            AttentionCollectionConfig,
            AttentionData,
            VLLMAttentionAdapter,
        )

        assert AttentionCollector is not None
        assert AttentionCollectionConfig is not None
        assert AttentionData is not None
        assert VLLMAttentionAdapter is not None

    def test_import_model_loader_new_functions(self):
        """测试 model_loader 新增函数导入"""
        from src.hace.models.model_loader import (
            load_inference_backend,
            get_backend_for_optimization,
            detect_available_backends,
            get_recommended_backend,
            create_hybrid_inference_pipeline,
        )

        assert load_inference_backend is not None
        assert get_backend_for_optimization is not None
        assert detect_available_backends is not None


class TestBackendDetection:
    """测试后端检测功能"""

    def test_detect_available_backends(self):
        """测试后端可用性检测"""
        from src.hace.models.model_loader import detect_available_backends

        backends = detect_available_backends()

        assert isinstance(backends, dict)
        assert "huggingface" in backends
        assert backends["huggingface"] is True  # HF应该始终可用

    def test_get_recommended_backend_default(self, model_config):
        """测试默认后端推荐"""
        from src.hace.models.model_loader import get_recommended_backend

        model_config["inference_backend"] = "hf"
        recommended = get_recommended_backend(model_config)

        assert recommended == "hf"

    @requires_vllm
    def test_get_recommended_backend_vllm(self, model_config):
        """测试 VLLM 后端推荐"""
        from src.hace.models.model_loader import get_recommended_backend

        model_config["inference_backend"] = "vllm"
        recommended = get_recommended_backend(model_config)

        assert recommended == "vllm"


class TestGenerationConfig:
    """测试生成配置"""

    def test_generation_config_defaults(self):
        """测试默认生成配置"""
        from src.hace.models.inference_backend import GenerationConfig

        config = GenerationConfig()

        assert config.max_new_tokens == 256
        assert config.temperature == 0.0
        assert config.top_p == 1.0
        assert config.do_sample is False

    def test_generation_config_custom(self):
        """测试自定义生成配置"""
        from src.hace.models.inference_backend import GenerationConfig

        config = GenerationConfig(
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

        assert config.max_new_tokens == 128
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.do_sample is True


class TestGenerationOutput:
    """测试生成输出"""

    def test_generation_output_basic(self):
        """测试基本生成输出"""
        from src.hace.models.inference_backend import GenerationOutput

        output = GenerationOutput(
            text="Hello, world!",
            finish_reason="stop",
            prompt_tokens=5,
            completion_tokens=3,
        )

        assert output.text == "Hello, world!"
        assert output.finish_reason == "stop"
        assert output.prompt_tokens == 5
        assert output.completion_tokens == 3


# ============== HuggingFace 后端测试 ==============

class TestHuggingFaceBackendMock:
    """HuggingFace 后端模拟测试（不加载实际模型）"""

    def test_hf_backend_creation(self, model_config):
        """测试 HF 后端创建"""
        from src.hace.models.inference_backend import HuggingFaceBackend

        backend = HuggingFaceBackend(model_config)

        assert backend.backend_name == "huggingface"
        assert backend.is_initialized is False

    def test_hf_backend_supports_attention(self, model_config):
        """测试 HF 后端支持注意力输出"""
        from src.hace.models.inference_backend import HuggingFaceBackend

        backend = HuggingFaceBackend(model_config)

        assert backend.supports_attention_output() is True

    @patch('src.hace.models.inference_backend.AutoModelForCausalLM')
    @patch('src.hace.models.inference_backend.AutoTokenizer')
    def test_hf_backend_initialize_mock(self, mock_tokenizer, mock_model, model_config):
        """测试 HF 后端初始化（模拟）"""
        from src.hace.models.inference_backend import HuggingFaceBackend

        # 设置模拟
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()

        backend = HuggingFaceBackend(model_config)
        backend.initialize()

        assert backend.is_initialized is True
        mock_model.from_pretrained.assert_called_once()
        mock_tokenizer.from_pretrained.assert_called_once()


# ============== VLLM 后端测试 ==============

class TestVLLMBackendMock:
    """VLLM 后端模拟测试"""

    def test_vllm_backend_creation(self, model_config, vllm_config):
        """测试 VLLM 后端创建"""
        from src.hace.models.inference_backend import VLLMBackend

        model_config["inference_backend"] = "vllm"
        backend = VLLMBackend(model_config, vllm_config)

        assert "vllm" in backend.backend_name
        assert backend.is_initialized is False

    def test_vllm_backend_modes(self, model_config, vllm_config):
        """测试 VLLM 后端模式"""
        from src.hace.models.inference_backend import VLLMBackend

        # 进程内模式
        vllm_config["mode"] = "inprocess"
        backend = VLLMBackend(model_config, vllm_config)
        assert backend.backend_name == "vllm-inprocess"

        # Server 模式
        vllm_config["mode"] = "server"
        backend = VLLMBackend(model_config, vllm_config)
        assert backend.backend_name == "vllm-server"

    def test_vllm_backend_no_attention_support(self, model_config, vllm_config):
        """测试 VLLM 后端默认不支持注意力输出"""
        from src.hace.models.inference_backend import VLLMBackend

        backend = VLLMBackend(model_config, vllm_config)

        assert backend.supports_attention_output() is False


# ============== 工厂函数测试 ==============

class TestBackendFactory:
    """测试后端工厂函数"""

    def test_create_hf_backend(self, model_config):
        """测试创建 HF 后端"""
        from src.hace.models.inference_backend import create_inference_backend

        model_config["inference_backend"] = "hf"
        backend = create_inference_backend(model_config)

        assert backend.backend_name == "huggingface"

    def test_create_vllm_backend(self, model_config, vllm_config):
        """测试创建 VLLM 后端"""
        from src.hace.models.inference_backend import create_inference_backend

        model_config["inference_backend"] = "vllm"
        backend = create_inference_backend(model_config, vllm_config)

        assert "vllm" in backend.backend_name

    def test_create_backend_invalid_type(self, model_config):
        """测试无效后端类型"""
        from src.hace.models.inference_backend import create_inference_backend

        model_config["inference_backend"] = "invalid_backend"

        with pytest.raises(ValueError, match="Unknown inference backend"):
            create_inference_backend(model_config)


# ============== 注意力收集器测试 ==============

class TestAttentionCollector:
    """测试注意力收集器"""

    def test_attention_config_defaults(self):
        """测试注意力收集配置默认值"""
        from src.hace.models.attention_collector import AttentionCollectionConfig

        config = AttentionCollectionConfig()

        assert config.mode == "external_warmup"
        assert config.warmup_samples == 10
        assert config.max_seq_length == 2048

    def test_attention_data_structure(self):
        """测试注意力数据结构"""
        import numpy as np
        from src.hace.models.attention_collector import AttentionData

        # 创建模拟注意力数据
        attention_weights = [
            np.random.rand(1, 32, 64, 64).astype(np.float32)
            for _ in range(4)  # 4层
        ]

        data = AttentionData(
            attention_weights_list=attention_weights,
            num_layers=4,
            num_heads=32,
            seq_length=64,
            model_name="test_model",
            collection_mode="external_warmup",
        )

        assert data.num_layers == 4
        assert data.num_heads == 32
        assert data.seq_length == 64
        assert len(data.attention_weights_list) == 4

    def test_vllm_attention_adapter(self):
        """测试 VLLM 注意力适配器"""
        import numpy as np
        from src.hace.models.attention_collector import (
            AttentionData,
            VLLMAttentionAdapter,
        )

        # 创建模拟注意力数据
        attention_weights = [
            np.random.rand(1, 32, 64, 64).astype(np.float32)
            for _ in range(4)
        ]

        data = AttentionData(
            attention_weights_list=attention_weights,
            num_layers=4,
            num_heads=32,
            seq_length=64,
            model_name="test_model",
            collection_mode="external_warmup",
        )

        adapter = VLLMAttentionAdapter(data)
        weights = adapter.get_attention_weights_for_allocator()

        assert len(weights) == 4


# ============== 配置系统测试 ==============

class TestVLLMConfig:
    """测试 VLLM 配置系统"""

    def test_vllm_config_exists(self):
        """测试 VLLM_CONFIG 存在"""
        from src.hace.config import VLLM_CONFIG

        assert VLLM_CONFIG is not None
        assert isinstance(VLLM_CONFIG, dict)

    def test_vllm_config_structure(self):
        """测试 VLLM_CONFIG 结构"""
        from src.hace.config import VLLM_CONFIG

        # 必要字段
        assert "mode" in VLLM_CONFIG
        assert "tensor_parallel_size" in VLLM_CONFIG
        assert "gpu_memory_utilization" in VLLM_CONFIG
        assert "server_url" in VLLM_CONFIG
        assert "sampling_params" in VLLM_CONFIG
        assert "attention_collection" in VLLM_CONFIG

    def test_model_config_backend_field(self):
        """测试 MODEL_CONFIG 中的 inference_backend 字段"""
        from src.hace.config import MODEL_CONFIG

        assert "inference_backend" in MODEL_CONFIG
        assert MODEL_CONFIG["inference_backend"] in ("hf", "vllm")

    def test_experiment_config_backend_field(self):
        """测试 EXPERIMENT_CONFIG 中的 inference_backend 字段"""
        from src.hace.config import EXPERIMENT_CONFIG

        assert "inference_backend" in EXPERIMENT_CONFIG


# ============== 实际加载测试（需要 GPU）==============

@requires_cuda
class TestHuggingFaceBackendReal:
    """HuggingFace 后端实际测试（需要 GPU）"""

    @pytest.mark.slow
    def test_hf_backend_load_small_model(self):
        """测试加载小型模型"""
        from src.hace.models.inference_backend import HuggingFaceBackend

        # 使用小型模型进行测试
        config = {
            "model_name_or_path": "gpt2",  # 小型模型
            "precision": "fp32",
            "inference_backend": "hf",
        }

        backend = HuggingFaceBackend(config)
        backend.initialize()

        assert backend.is_initialized is True
        assert backend.get_tokenizer() is not None

        backend.cleanup()
        assert backend.is_initialized is False


@requires_cuda
@requires_vllm
class TestVLLMBackendReal:
    """VLLM 后端实际测试（需要 GPU 和 VLLM）"""

    @pytest.mark.slow
    def test_vllm_check_available(self):
        """测试 VLLM 可用性检查"""
        from src.hace.models.inference_backend import check_vllm_available

        is_available, message = check_vllm_available()

        assert is_available is True
        assert "VLLM" in message


# ============== 混合管道测试 ==============

class TestHybridPipeline:
    """测试混合推理管道"""

    def test_create_pipeline_hf(self, model_config):
        """测试创建 HF 管道"""
        from src.hace.models.model_loader import create_hybrid_inference_pipeline

        model_config["inference_backend"] = "hf"
        pipeline = create_hybrid_inference_pipeline(model_config)

        assert pipeline["inference_backend"] is not None
        assert pipeline["optimization_method"] == "none"

    def test_create_pipeline_vllm_with_optimization(self, model_config, vllm_config):
        """测试创建带优化的 VLLM 管道"""
        from src.hace.models.model_loader import create_hybrid_inference_pipeline

        model_config["inference_backend"] = "vllm"
        pipeline = create_hybrid_inference_pipeline(
            model_config,
            vllm_config,
            optimization_method="cake"
        )

        assert pipeline["inference_backend"] is not None
        assert pipeline["attention_collector"] is not None
        assert pipeline["optimization_method"] == "cake"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
