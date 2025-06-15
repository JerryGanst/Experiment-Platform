
# 将此函数替换到 hace-kv-optimization/baselines/fullkvcache_main.py 中

def extract_ground_truth_from_sample(sample, dataset_source):
    """
    修复版：专门处理LongBench官方数据
    处理所有可能的答案字段格式
    """

    # 调试信息（可以在生产环境中移除）
    # logger.debug(f"提取答案 - 样本键: {list(sample.keys())}")

    # 1. 优先处理标准LongBench格式
    if 'answers' in sample and sample['answers']:
        answers = sample['answers']
        if isinstance(answers, list) and len(answers) > 0:
            # 过滤空答案
            valid_answers = [str(ans).strip() for ans in answers if ans and str(ans).strip()]
            if valid_answers:
                return valid_answers
        elif isinstance(answers, str) and answers.strip():
            return [answers.strip()]

    # 2. 处理processed sample的reference字段
    if 'reference' in sample and sample['reference']:
        reference = sample['reference']
        if isinstance(reference, list) and len(reference) > 0:
            valid_refs = [str(ref).strip() for ref in reference if ref and str(ref).strip()]
            if valid_refs:
                return valid_refs
        elif isinstance(reference, str) and reference.strip():
            return [reference.strip()]

    # 3. 处理嵌套的原始样本
    if 'original_sample' in sample:
        original = sample['original_sample']
        if isinstance(original, dict):
            for field in ['answers', 'answer', 'output', 'target']:
                if field in original and original[field]:
                    value = original[field]
                    if isinstance(value, list) and len(value) > 0:
                        valid_vals = [str(v).strip() for v in value if v and str(v).strip()]
                        if valid_vals:
                            return valid_vals
                    elif isinstance(value, str) and value.strip():
                        return [value.strip()]

    # 4. 其他常见答案字段
    answer_fields = ['answer', 'output', 'gold', 'target', 'label', 'ground_truth']
    for field in answer_fields:
        if field in sample and sample[field] is not None:
            value = sample[field]
            if isinstance(value, list) and len(value) > 0:
                valid_vals = [str(v).strip() for v in value if v and str(v).strip()]
                if valid_vals:
                    return valid_vals
            elif isinstance(value, str) and value.strip():
                return [value.strip()]
            elif isinstance(value, (int, float)):
                return [str(value)]

    # 5. 如果都没找到，记录警告但不返回["Unknown"]
    logger.warning(f"未找到有效答案字段，样本键: {list(sample.keys())}")

    # 返回空列表而不是["Unknown"]，让上层处理
    return []
