#!/usr/bin/env python3
"""
AIword 垂直化提示词补全系统 - 命令行演示
展示核心补全功能和效果
"""

import time
from completion_engine import CompletionEngine
from colorama import Colorama, Fore, Style, init

# 初始化颜色输出
init(autoreset=True)

class AIWordDemo:
    def __init__(self):
        self.engine = CompletionEngine()
        self.demo_cases = [
            {
                'input': '如何',
                'description': '前缀触发补全示例',
                'type': 'prefix'
            },
            {
                'input': '什么是',
                'description': '概念查询补全示例',
                'type': 'prefix'
            },
            {
                'input': '机器学习',
                'description': '领域术语补全示例',
                'type': 'domain'
            },
            {
                'input': 'Python',
                'description': '编程语言补全示例',
                'type': 'domain'
            },
            {
                'input': '数据分析',
                'description': '数据科学补全示例',
                'type': 'domain'
            }
        ]
    
    def print_header(self):
        """打印演示标题"""
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}    🤖 AIword 垂直化提示词补全系统演示")
        print(f"{Fore.CYAN}{'='*60}")
        print(f"{Fore.GREEN}✨ 智能补全 | 🎯 精准提问 | 🚀 高效交互")
        print(f"{Fore.CYAN}{'='*60}\n")
    
    def simulate_typing(self, text, delay=0.1):
        """模拟打字效果"""
        for char in text:
            print(char, end='', flush=True)
            time.sleep(delay)
        print()
    
    def demo_completion(self, test_case):
        """演示单个补全案例"""
        input_text = test_case['input']
        description = test_case['description']
        
        print(f"\n{Fore.YELLOW}📝 {description}")
        print(f"{Fore.BLUE}输入文本: {Style.BRIGHT}{input_text}")
        
        # 模拟用户输入过程
        print(f"{Fore.WHITE}正在输入", end="")
        for _ in range(3):
            print(".", end="", flush=True)
            time.sleep(0.5)
        print()
        
        # 获取补全建议
        suggestions = self.engine.detect_completion(input_text, pause_time=600)
        
        if suggestions:
            print(f"{Fore.GREEN}🔍 智能补全建议:")
            for i, suggestion in enumerate(suggestions[:4], 1):
                category_color = Fore.MAGENTA if suggestion.trigger_type == 'domain' else Fore.CYAN
                print(f"  {i}. {suggestion.text} {category_color}({suggestion.category})")
            
            # 模拟选择第一个建议
            if suggestions:
                selected = suggestions[0]
                print(f"\n{Fore.GREEN}👆 选择建议: {selected.text}")
                
                # 生成完整问句
                complete_question = self.engine.generate_complete_question(
                    input_text,
                    selected.text.replace('🔹', '').replace('📊', '').strip(),
                    selected.template,
                    selected.trigger_type
                )
                
                print(f"{Fore.YELLOW}📋 生成完整问句:")
                print(f"{Fore.WHITE}{Style.BRIGHT}   → {complete_question}")
                
                # 更新使用历史
                self.engine.update_user_history(input_text, selected.text)
        else:
            print(f"{Fore.RED}❌ 未找到匹配的补全建议")
        
        print(f"{Fore.BLUE}{'-'*50}")
    
    def demo_interactive_mode(self):
        """交互式演示模式"""
        print(f"\n{Fore.CYAN}🎮 进入交互式演示模式")
        print(f"{Fore.WHITE}输入问题片段，体验智能补全（输入 'quit' 退出）:")
        
        while True:
            try:
                user_input = input(f"\n{Fore.GREEN}请输入: {Style.BRIGHT}")
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print(f"{Fore.YELLOW}👋 感谢体验AIword系统！")
                    break
                
                if not user_input.strip():
                    continue
                
                # 检测补全
                suggestions = self.engine.detect_completion(user_input, pause_time=600)
                
                if suggestions:
                    print(f"{Fore.GREEN}💡 智能建议:")
                    for i, suggestion in enumerate(suggestions, 1):
                        print(f"  {i}. {suggestion.text} ({suggestion.category})")
                    
                    # 让用户选择
                    try:
                        choice = input(f"{Fore.CYAN}选择编号 (1-{len(suggestions)}) 或按Enter跳过: ")
                        if choice.isdigit() and 1 <= int(choice) <= len(suggestions):
                            selected = suggestions[int(choice) - 1]
                            complete_question = self.engine.generate_complete_question(
                                user_input,
                                selected.text.replace('🔹', '').replace('📊', '').strip(),
                                selected.template,
                                selected.trigger_type
                            )
                            print(f"{Fore.YELLOW}✨ 完整问句: {Style.BRIGHT}{complete_question}")
                    except (ValueError, IndexError):
                        continue
                else:
                    print(f"{Fore.RED}😅 暂无匹配的补全建议，试试其他关键词")
                    
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}👋 感谢体验AIword系统！")
                break
    
    def show_statistics(self):
        """显示系统统计信息"""
        print(f"\n{Fore.CYAN}📊 系统统计信息:")
        
        config = self.engine.config
        
        # 配置统计
        trigger_count = len(config.get('triggers', []))
        domain_count = len(config.get('domain_terms', []))
        category_count = len(config.get('categories', {}))
        
        print(f"  🔧 配置的触发器数量: {Fore.WHITE}{trigger_count}")
        print(f"  🏷️  支持的领域术语: {Fore.WHITE}{domain_count}")
        print(f"  📂 专业分类数量: {Fore.WHITE}{category_count}")
        
        # 使用统计
        history_count = len(self.engine.input_history)
        print(f"  📈 演示使用次数: {Fore.WHITE}{history_count}")
        
        # 热门补全
        popular = self.engine.get_popular_completions()
        if popular:
            print(f"  🔥 热门补全选项: {Fore.WHITE}{', '.join(popular[:3])}")
    
    def run_demo(self):
        """运行完整演示"""
        self.print_header()
        
        # 1. 自动演示预设案例
        print(f"{Fore.MAGENTA}🎬 自动演示模式\n")
        
        for i, case in enumerate(self.demo_cases, 1):
            print(f"{Fore.CYAN}演示 {i}/{len(self.demo_cases)}")
            self.demo_completion(case)
            time.sleep(1)
        
        # 2. 显示统计信息
        self.show_statistics()
        
        # 3. 交互式演示
        print(f"\n{Fore.MAGENTA}🎯 想要亲自体验吗？")
        choice = input(f"{Fore.GREEN}按Enter开始交互式演示，或输入'skip'跳过: ")
        
        if choice.lower() != 'skip':
            self.demo_interactive_mode()
        
        # 4. 结束语
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.GREEN}🎉 演示完成！AIword让提问更智能！")
        print(f"{Fore.WHITE}💡 启动Web版本: python web_interface.py")
        print(f"{Fore.WHITE}🌐 访问地址: http://localhost:5000")
        print(f"{Fore.CYAN}{'='*60}\n")


def main():
    """主函数"""
    try:
        demo = AIWordDemo()
        demo.run_demo()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}演示被中断")
    except Exception as e:
        print(f"\n{Fore.RED}演示出错: {e}")


if __name__ == "__main__":
    main() 