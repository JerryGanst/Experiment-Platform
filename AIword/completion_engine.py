"""
垂直化提示词补全引擎
实现智能提示词补全功能，支持前缀触发和领域术语补全
"""

import yaml
import re
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CompletionSuggestion:
    """补全建议数据结构"""
    text: str                # 显示文本
    template: str           # 补全模板
    category: str           # 分类
    confidence: float       # 置信度
    trigger_type: str       # 触发类型（prefix/domain）
    description: str = ""   # 选项描述


class CompletionEngine:
    """垂直化提示词补全引擎"""
    
    def __init__(self, config_path: str = "completion_config.yaml"):
        """初始化补全引擎"""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.last_input_time = 0
        self.input_history = []
        
        # 构建领域关键词映射表，用于智能识别
        self._build_domain_keywords()
        
    def _load_config(self) -> Dict:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件未找到: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件格式错误: {e}")
    
    def _build_domain_keywords(self):
        """构建领域关键词映射表，用于智能识别"""
        self.domain_keywords = {}
        
        # 从domain_triggers中提取关键词
        for domain in self.config.get('domain_triggers', []):
            term = domain['term'].lower()
            category = domain['category']
            
            # 主要关键词
            self.domain_keywords[term] = category
            
            # 添加相关关键词
            keywords = []
            if 'AI' in term or '人工智能' in term:
                keywords.extend(['ai', 'artificial', 'intelligence', '智能', '算法'])
            elif '机器学习' in term or 'machine' in term:
                keywords.extend(['ml', 'machine', 'learning', '训练', '模型'])
            elif '数据' in term or 'data' in term:
                keywords.extend(['data', 'analytics', '分析', '统计', '可视化'])
            elif '区块链' in term or 'blockchain' in term:
                keywords.extend(['blockchain', 'crypto', '比特币', '以太坊', '智能合约'])
            elif '量子' in term or 'quantum' in term:
                keywords.extend(['quantum', '量子', 'qubit', '纠缠'])
            elif '微服务' in term or 'microservice' in term:
                keywords.extend(['microservice', 'docker', 'k8s', 'kubernetes', '容器'])
            elif '容器' in term or 'container' in term:
                keywords.extend(['docker', 'container', 'k8s', 'kubernetes', 'pod'])
            
            for keyword in keywords:
                self.domain_keywords[keyword] = category
    
    def detect_completion(self, text: str, pause_time: float = 0) -> List[CompletionSuggestion]:
        """
        检测并生成补全建议
        
        Args:
            text: 用户输入文本
            pause_time: 停顿时间（毫秒）
            
        Returns:
            补全建议列表
        """
        suggestions = []
        
        # 检查最小输入长度
        if len(text.strip()) < 2:
            return suggestions
        
        # 检查停顿时间
        trigger_delay = 300  # 使用固定值300ms
        if pause_time < trigger_delay:
            return suggestions
        
        # 规则1：检测前缀关键词补全
        prefix_suggestions = self._get_prefix_suggestions(text)
        suggestions.extend(prefix_suggestions)
        
        # 规则2：智能检测领域术语补全
        domain_suggestions = self._get_smart_domain_suggestions(text)
        suggestions.extend(domain_suggestions)
        
        # 限制建议数量
        max_suggestions = 6
        suggestions = suggestions[:max_suggestions]
        
        return suggestions
    
    def _get_prefix_suggestions(self, text: str) -> List[CompletionSuggestion]:
        """获取前缀触发的补全建议"""
        suggestions = []
        text_lower = text.lower().strip()
        
        for trigger in self.config.get('prefix_triggers', []):
            prefix = trigger['prefix']
            
            # 检查是否以前缀结尾
            if text_lower.endswith(prefix.lower()):
                for option in trigger['options']:
                    if isinstance(option, dict):
                        option_text = option['text']
                        option_desc = option.get('description', '')
                    else:
                        option_text = str(option)
                        option_desc = ''
                    
                    suggestion = CompletionSuggestion(
                        text=f"🔹 {option_text}",
                        template=trigger.get('template', f"{prefix}{option_text}"),
                        category=trigger['category'],
                        confidence=0.9,
                        trigger_type="prefix",
                        description=option_desc
                    )
                    suggestions.append(suggestion)
        
        return suggestions
    
    def _get_smart_domain_suggestions(self, text: str) -> List[CompletionSuggestion]:
        """智能获取领域术语触发的补全建议"""
        suggestions = []
        text_lower = text.lower().strip()
        
        # 智能识别领域
        detected_domain = self._detect_domain(text_lower)
        
        for domain_term in self.config.get('domain_triggers', []):
            term = domain_term['term'].lower()
            
            # 检查条件：
            # 1. 直接包含术语
            # 2. 检测到相关领域
            # 3. 术语是输入的最后一个词
            last_word = self._extract_last_word(text_lower)
            
            should_trigger = (
                term in text_lower or 
                last_word == term or
                (detected_domain and detected_domain == domain_term['category'])
            )
            
            if should_trigger:
                for option in domain_term['options']:
                    if isinstance(option, dict):
                        option_text = option['text']
                        option_desc = option.get('description', '')
                    else:
                        option_text = str(option)
                        option_desc = ''
                    
                    # 根据匹配类型设置置信度
                    confidence = 0.85
                    if term in text_lower:
                        confidence = 0.9  # 直接匹配更高置信度
                    elif detected_domain:
                        confidence = 0.8  # 领域推测稍低
                    
                    suggestion = CompletionSuggestion(
                        text=f"📊 {option_text}",
                        template=domain_term.get('template', f"{domain_term['term']}{option_text}"),
                        category=domain_term['category'],
                        confidence=confidence,
                        trigger_type="domain",
                        description=option_desc
                    )
                    suggestions.append(suggestion)
        
        return suggestions
    
    def _detect_domain(self, text: str) -> Optional[str]:
        """智能检测输入文本的领域分类"""
        text_words = re.findall(r'\b\w+\b', text.lower())
        
        # 统计各领域关键词出现次数
        domain_scores = {}
        
        for word in text_words:
            if word in self.domain_keywords:
                domain = self.domain_keywords[word]
                domain_scores[domain] = domain_scores.get(domain, 0) + 1
        
        # 返回得分最高的领域
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    def _extract_last_word(self, text: str) -> str:
        """提取文本中的最后一个词"""
        words = re.findall(r'\b\w+\b', text)
        return words[-1] if words else ""
    
    def generate_complete_question(self, text: str, selected_option: str, 
                                 template: str, trigger_type: str) -> str:
        """
        生成完整的问句
        
        Args:
            text: 原始输入文本
            selected_option: 选择的选项
            template: 模板字符串
            trigger_type: 触发类型
            
        Returns:
            完整的问句
        """
        # 清理选项文本（移除图标）
        clean_option = re.sub(r'[🔹📊]', '', selected_option).strip()
        
        if trigger_type == "prefix":
            # 前缀补全：如何 -> 如何学习机器学习？
            topic = ""
            words = text.split()
            if len(words) > 1:
                topic = " ".join(words[1:])  # 除了前缀的其他词
            return template.format(option=clean_option, topic=topic)
        
        elif trigger_type == "domain":
            # 领域补全：机器学习 -> 机器学习算法对比有哪些？
            return template.format(option=clean_option)
        
        return text + " " + clean_option
    
    def update_user_history(self, text: str, selected_option: str = None):
        """更新用户输入历史，用于优化建议"""
        self.input_history.append({
            'text': text,
            'selected_option': selected_option,
            'timestamp': time.time()
        })
        
        # 保持历史记录在合理范围内
        if len(self.input_history) > 100:
            self.input_history = self.input_history[-50:]
    
    def get_popular_completions(self, category: str = None) -> List[str]:
        """获取热门补全选项"""
        popular = []
        
        # 基于历史使用频率分析（简化版）
        if not self.input_history:
            return self._get_default_popular(category)
        
        # 统计选择频率
        option_counts = {}
        for item in self.input_history:
            if item.get('selected_option'):
                option = item['selected_option']
                option_counts[option] = option_counts.get(option, 0) + 1
        
        # 返回最热门的选项
        sorted_options = sorted(option_counts.items(), key=lambda x: x[1], reverse=True)
        popular = [option for option, count in sorted_options[:6]]
        
        return popular
    
    def _get_default_popular(self, category: str = None) -> List[str]:
        """获取默认热门选项"""
        defaults = [
            "最佳实践", "入门路径", "常见问题", 
            "应用场景", "核心原理", "实战案例"
        ]
        return defaults
    
    def get_categories(self) -> Dict:
        """获取分类信息"""
        categories = {}
        
        # 从domain_triggers中提取分类
        for domain in self.config.get('domain_triggers', []):
            category = domain['category']
            if category not in categories:
                categories[category] = {
                    'name': category,
                    'common_patterns': []
                }
            categories[category]['common_patterns'].append(domain['term'])
        
        return categories
    
    def add_domain_term(self, term: str, options: List[str], 
                       template: str, category: str):
        """动态添加领域术语"""
        new_term = {
            'term': term,
            'category': category,
            'options': [{'text': opt, 'description': ''} for opt in options],
            'template': template
        }
        
        if 'domain_triggers' not in self.config:
            self.config['domain_triggers'] = []
        
        self.config['domain_triggers'].append(new_term)
        
        # 重新构建关键词映射
        self._build_domain_keywords()
        
        # 保存到配置文件
        self._save_config()
    
    def _save_config(self):
        """保存配置到文件"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, 
                     allow_unicode=True, sort_keys=False)


# 使用示例函数
def example_usage():
    """使用示例"""
    engine = CompletionEngine()
    
    # 示例1：前缀补全
    print("=== 前缀补全示例 ===")
    suggestions = engine.detect_completion("如何", pause_time=600)
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion.text} ({suggestion.category})")
        if suggestion.description:
            print(f"   描述: {suggestion.description}")
    
    # 示例2：领域术语补全
    print("\n=== 领域术语补全示例 ===")
    suggestions = engine.detect_completion("机器学习", pause_time=600)
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion.text} ({suggestion.category})")
        if suggestion.description:
            print(f"   描述: {suggestion.description}")
    
    # 示例3：智能领域识别
    print("\n=== 智能领域识别示例 ===")
    test_cases = ["AI模型", "数据分析", "docker部署", "区块链应用"]
    for case in test_cases:
        suggestions = engine.detect_completion(case, pause_time=600)
        print(f"输入'{case}' -> 识别到{len(suggestions)}个建议")
        for suggestion in suggestions[:2]:  # 只显示前2个
            print(f"  {suggestion.text} ({suggestion.category})")


if __name__ == "__main__":
    example_usage() 