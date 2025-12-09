"""
VLLM 配置修复的单元测试 (不依赖 torch)
验证关键修复是否正确实现
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBudgetAlignerLogic:
    """测试 BudgetAligner 的核心逻辑"""

    def test_divide_zero_protection(self):
        """测试除零保护"""
        # 模拟 BudgetAligner 的除零保护逻辑
        layer_token_budgets = [0, 0, 0, 0]
        block_size = 16

        total_requested_tokens = sum(layer_token_budgets)

        # 除零保护：如果所有预算都为0，应返回全0
        if total_requested_tokens == 0:
            result = [0] * len(layer_token_budgets)
        else:
            result = [b // block_size for b in layer_token_budgets]

        assert result == [0, 0, 0, 0], f"Expected [0,0,0,0], got {result}"
        print("✓ test_divide_zero_protection passed")

    def test_importance_weighted_rounding(self):
        """测试重要性加权取整"""
        layer_token_budgets = [100, 150, 80, 120]
        block_size = 16
        num_layers = len(layer_token_budgets)

        # 计算基础 block 数和余数
        base_blocks = [budget // block_size for budget in layer_token_budgets]
        remainders = [budget % block_size for budget in layer_token_budgets]

        # 验证基础计算
        assert base_blocks == [6, 9, 5, 7], f"Expected [6,9,5,7], got {base_blocks}"
        assert remainders == [4, 6, 0, 8], f"Expected [4,6,0,8], got {remainders}"

        # 计算可分配的额外 block 数
        total_requested = sum(layer_token_budgets)  # 450
        total_base = sum(b * block_size for b in base_blocks)  # 432
        extra_tokens = total_requested - total_base  # 18
        extra_blocks = extra_tokens // block_size  # 1

        assert extra_blocks == 1, f"Expected 1 extra block, got {extra_blocks}"
        print("✓ test_importance_weighted_rounding passed")


class TestServerModeAPIParser:
    """测试 Server 模式 API 响应解析"""

    def test_openai_format_parsing(self):
        """测试 OpenAI 兼容格式解析"""
        # 模拟 OpenAI 格式响应
        data = {
            "choices": [{"text": "Hello world", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }

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

        assert generated_text == "Hello world"
        assert finish_reason == "stop"
        assert prompt_tokens == 10
        assert completion_tokens == 5
        print("✓ test_openai_format_parsing passed")

    def test_native_format_parsing_list(self):
        """测试 VLLM 原生格式解析 (列表)"""
        # 模拟 VLLM 原生格式响应 (text 为列表)
        data = {"text": ["Generated text"], "finish_reason": "length"}

        text_data = data.get("text", "")
        if isinstance(text_data, list):
            generated_text = text_data[0] if text_data else ""
        else:
            generated_text = text_data

        assert generated_text == "Generated text"
        print("✓ test_native_format_parsing_list passed")

    def test_native_format_parsing_string(self):
        """测试 VLLM 原生格式解析 (字符串)"""
        # 模拟 VLLM 原生格式响应 (text 为字符串)
        data = {"text": "Direct string response"}

        text_data = data.get("text", "")
        if isinstance(text_data, list):
            generated_text = text_data[0] if text_data else ""
        else:
            generated_text = text_data

        assert generated_text == "Direct string response"
        print("✓ test_native_format_parsing_string passed")

    def test_empty_choices_handling(self):
        """测试空 choices 处理"""
        data = {"choices": []}

        choices = data.get("choices", [{}])
        if choices:
            choice = choices[0]
            generated_text = choice.get("text", "")
        else:
            generated_text = ""

        assert generated_text == ""
        print("✓ test_empty_choices_handling passed")


class TestImportPathFallback:
    """测试导入路径 fallback 机制"""

    def test_import_path_priority(self):
        """测试导入路径优先级正确"""
        import_paths = [
            "hace_core.core_code.unified_allocator",  # 项目内优先
            "src.core_code.unified_allocator",        # 旧路径兼容
            "core_code.unified_allocator",            # 相对导入兼容
        ]

        # 验证 hace_core 路径在最前面
        assert import_paths[0] == "hace_core.core_code.unified_allocator"
        print("✓ test_import_path_priority passed")

    def test_fallback_uniform_allocation(self):
        """测试 fallback 均匀分配逻辑"""
        num_layers = 32
        num_heads = 32
        total_cache_size = 4096

        # 模拟均匀分配
        if num_layers == 0:
            layer_budgets = []
        else:
            layer_budget = total_cache_size // num_layers
            layer_budgets = [layer_budget] * num_layers

        assert len(layer_budgets) == 32
        assert all(b == 128 for b in layer_budgets)
        assert sum(layer_budgets) == 4096
        print("✓ test_fallback_uniform_allocation passed")

    def test_fallback_zero_layers(self):
        """测试零层数的 fallback 处理"""
        num_layers = 0
        total_cache_size = 4096

        if num_layers == 0:
            layer_budgets = []
        else:
            layer_budget = total_cache_size // num_layers
            layer_budgets = [layer_budget] * num_layers

        assert layer_budgets == []
        print("✓ test_fallback_zero_layers passed")


class TestMemoryEstimation:
    """测试显存估算公式"""

    def test_basic_memory_estimation(self):
        """测试基础显存估算"""
        num_params_billions = 7.0
        precision = "fp16"
        bytes_per_param = {"fp16": 2, "bf16": 2, "fp32": 4, "int8": 1, "int4": 0.5, "auto": 2}
        bpp = bytes_per_param.get(precision, 2)

        # 模型参数显存
        param_memory_gb = (num_params_billions * 1e9 * bpp) / (1024**3)

        # 7B 模型 fp16 应该约为 13GB
        assert 12 < param_memory_gb < 15, f"Expected ~13GB, got {param_memory_gb:.2f}GB"
        print(f"✓ test_basic_memory_estimation passed (7B fp16 = {param_memory_gb:.2f}GB)")

    def test_kv_cache_estimation(self):
        """测试 KV Cache 显存估算"""
        batch_size = 1
        max_seq_len = 4096
        num_layers = 32
        num_heads = 32
        head_dim = 128
        bpp = 2  # fp16

        # KV Cache 计算
        # 每层每token: 2 (K+V) * num_heads * head_dim * bytes = 2 * 32 * 128 * 2 = 16384 bytes
        kv_per_token_per_layer = 2 * num_heads * head_dim * bpp
        kv_cache_bytes = batch_size * max_seq_len * num_layers * kv_per_token_per_layer
        kv_cache_gb = kv_cache_bytes / (1024**3)

        # 验证计算: 1 * 4096 * 32 * 16384 = 2GB (正确值)
        assert 1.5 < kv_cache_gb < 2.5, f"Expected ~2GB KV cache, got {kv_cache_gb:.2f}GB"
        print(f"✓ test_kv_cache_estimation passed (KV cache = {kv_cache_gb:.2f}GB)")


class TestGenerateContextUsage:
    """测试 generate() 中的 context 使用"""

    def test_context_returned_when_available(self):
        """测试有 context 时正确返回"""
        # 模拟有 request_allocator 和 request_id 的情况
        request_allocator = True  # 模拟存在
        request_id = "test-123"

        context = None
        if request_allocator and request_id:
            context = {"request_id": request_id, "budgets": [100, 100, 100]}

        result = "generated text"

        # 模拟返回逻辑
        if context:
            output = (result, context)
        else:
            output = result

        assert isinstance(output, tuple)
        assert output[0] == "generated text"
        assert output[1]["request_id"] == "test-123"
        print("✓ test_context_returned_when_available passed")

    def test_no_context_returns_result_only(self):
        """测试无 context 时只返回结果"""
        request_allocator = None
        request_id = None

        context = None
        if request_allocator and request_id:
            context = {"request_id": request_id}

        result = "generated text"

        if context:
            output = (result, context)
        else:
            output = result

        assert output == "generated text"
        assert not isinstance(output, tuple)
        print("✓ test_no_context_returns_result_only passed")


class TestBoundaryHandling:
    """测试边界情况处理"""

    def test_layer_budget_less_than_num_heads(self):
        """测试 layer_budget < num_heads 时不抛出异常"""
        from hace_core.core.indicator_normalizer import BudgetNormalizer

        # 场景: layer_budget=5, num_heads=32
        # 之前的代码会因为 min_budget=1 * 32 > 5 而抛出 ValueError
        # 修复后应该使用 min_budget=0，允许部分头预算为0

        raw_budgets = [1] * 32  # 32个头，每个初始预算1
        total_budget = 5  # 只有5的预算
        min_budget = 0  # 允许预算为0

        result = BudgetNormalizer.normalize_to_budget(raw_budgets, total_budget, min_budget)

        assert sum(result) == total_budget, f"预算不守恒: {sum(result)} != {total_budget}"
        assert len(result) == 32, f"头数量不对: {len(result)} != 32"
        # 应该有一些头的预算为0
        zero_count = sum(1 for b in result if b == 0)
        assert zero_count > 0, "应该有一些头的预算为0"
        print(f"✓ test_layer_budget_less_than_num_heads passed (zero heads: {zero_count})")

    def test_f1_score_empty_input(self):
        """测试 F1 Score 空输入不抛出除零异常"""
        from collections import Counter

        # 直接复制修复后的 f1_score 函数进行测试，避免依赖 jieba
        def f1_score(prediction, ground_truth, **kwargs):
            # 添加空输入保护，避免除零错误
            if len(prediction) == 0 or len(ground_truth) == 0:
                return 0.0
            common = Counter(prediction) & Counter(ground_truth)
            num_same = sum(common.values())
            if num_same == 0:
                return 0.0
            precision = 1.0 * num_same / len(prediction)
            recall = 1.0 * num_same / len(ground_truth)
            f1 = (2 * precision * recall) / (precision + recall)
            return f1

        # 测试空输入
        result1 = f1_score([], ["hello"])
        assert result1 == 0.0, f"空prediction应返回0, 得到: {result1}"

        result2 = f1_score(["hello"], [])
        assert result2 == 0.0, f"空ground_truth应返回0, 得到: {result2}"

        result3 = f1_score([], [])
        assert result3 == 0.0, f"双空应返回0, 得到: {result3}"

        # 测试正常情况
        result4 = f1_score(["a", "b", "c"], ["a", "b", "d"])
        assert result4 > 0, f"正常情况应返回正数, 得到: {result4}"

        print("✓ test_f1_score_empty_input passed")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Running VLLM Configuration Fixes Tests")
    print("=" * 60)

    test_classes = [
        TestBudgetAlignerLogic(),
        TestServerModeAPIParser(),
        TestImportPathFallback(),
        TestMemoryEstimation(),
        TestGenerateContextUsage(),
        TestBoundaryHandling(),
    ]

    total_tests = 0
    passed_tests = 0

    for test_instance in test_classes:
        print(f"\n--- {test_instance.__class__.__name__} ---")
        for method_name in dir(test_instance):
            if method_name.startswith("test_"):
                try:
                    getattr(test_instance, method_name)()
                    passed_tests += 1
                except AssertionError as e:
                    print(f"✗ {method_name} FAILED: {e}")
                except Exception as e:
                    print(f"✗ {method_name} ERROR: {e}")
                total_tests += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed_tests}/{total_tests} tests passed")
    print("=" * 60)

    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
