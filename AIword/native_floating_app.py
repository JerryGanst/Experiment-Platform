#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIword 原生悬浮窗桌面应用
基于tkinter，无需浏览器，真正的桌面原生应用
支持：
- 原生GUI悬浮窗
- 窗口拖拽移动
- 始终置顶
- 透明背景
- 键盘快捷键
- 智能补全
"""

import tkinter as tk
from tkinter import ttk, messagebox, Menu
import threading
import json
import time
import sys
import os
from completion_engine import CompletionEngine

class NativeFloatingAIWord:
    def __init__(self):
        self.root = None
        self.completion_engine = CompletionEngine()
        self.current_suggestions = []
        self.selected_index = -1
        self.last_input_time = 0
        self.completion_window = None
        self.is_dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        
        # 设置
        self.settings = {
            'trigger_delay': 300,
            'max_suggestions': 6,
            'always_on_top': True,
            'window_opacity': 0.95,
            'auto_complete': True
        }
        
        # 统计
        self.stats = {
            'total_questions': 0,
            'completion_used': 0,
            'total_response_time': 0,
            'request_count': 0
        }
        
        self.load_settings()
        self.create_window()
        
    def create_window(self):
        """创建主悬浮窗"""
        self.root = tk.Tk()
        self.root.title("AIword - 智能提示词补全")
        
        # 窗口设置
        self.root.geometry("480x650+100+100")
        self.root.minsize(380, 500)
        self.root.maxsize(600, 800)
        
        # 悬浮窗属性
        self.root.attributes('-topmost', self.settings['always_on_top'])
        self.root.attributes('-alpha', self.settings['window_opacity'])
        
        # 去除标题栏装饰（可选）
        # self.root.overrideredirect(True)
        
        # 设置图标和样式
        self.setup_styles()
        self.create_widgets()
        self.bind_events()
        
        # 加载数据
        self.load_categories()
        self.update_stats_display()
        
    def setup_styles(self):
        """设置样式主题"""
        style = ttk.Style()
        
        # 配置主题
        try:
            style.theme_use('clam')  # 使用现代主题
        except:
            pass
            
        # 自定义样式
        style.configure('Title.TLabel', 
                       font=('Segoe UI', 16, 'bold'),
                       foreground='#667eea')
        style.configure('Subtitle.TLabel',
                       font=('Segoe UI', 10),
                       foreground='#7f8c8d')
        style.configure('Header.TFrame',
                       background='#667eea')
        
    def create_widgets(self):
        """创建界面组件"""
        # 主容器
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题区域（可拖拽）
        self.create_header(main_frame)
        
        # 输入区域
        self.create_input_section(main_frame)
        
        # 热门提示区域
        self.create_popular_section(main_frame)
        
        # 分类区域
        self.create_categories_section(main_frame)
        
        # 统计区域
        self.create_stats_section(main_frame)
        
        # 底部区域
        self.create_footer(main_frame)
        
        # 创建右键菜单
        self.create_context_menu()
        
    def create_header(self, parent):
        """创建标题区域"""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 标题
        title_label = ttk.Label(header_frame, text="🧠 AIword", style='Title.TLabel')
        title_label.pack()
        
        subtitle_label = ttk.Label(header_frame, text="智能提示词补全", style='Subtitle.TLabel')
        subtitle_label.pack()
        
        # 绑定拖拽事件
        for widget in [header_frame, title_label, subtitle_label]:
            widget.bind('<Button-1>', self.start_drag)
            widget.bind('<B1-Motion>', self.on_drag)
            widget.bind('<ButtonRelease-1>', self.stop_drag)
        
    def create_input_section(self, parent):
        """创建输入区域"""
        input_frame = ttk.LabelFrame(parent, text="问题输入", padding="10")
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 输入框
        self.text_widget = tk.Text(input_frame, 
                                  height=4, 
                                  font=('Segoe UI', 11),
                                  wrap=tk.WORD,
                                  relief='solid',
                                  borderwidth=1,
                                  takefocus=True)  # 确保能获得焦点
        self.text_widget.pack(fill=tk.X, pady=(0, 5))
        
        # 禁用Text widget的默认Tab行为
        self.text_widget.bind('<Tab>', self.on_tab_key)
        
        # 提示信息
        hint_frame = ttk.Frame(input_frame)
        hint_frame.pack(fill=tk.X)
        
        self.char_count_label = ttk.Label(hint_frame, text="字符: 0")
        self.char_count_label.pack(side=tk.LEFT)
        
        hint_label = ttk.Label(hint_frame, text="Tab触发补全 | ↑↓选择 | Enter确认")
        hint_label.pack(side=tk.RIGHT)
        
        # 提交按钮
        self.submit_btn = ttk.Button(input_frame, 
                                   text="🚀 提交问题",
                                   command=self.submit_question)
        self.submit_btn.pack(pady=(5, 0))
        
    def create_popular_section(self, parent):
        """创建热门提示区域"""
        popular_frame = ttk.LabelFrame(parent, text="🔥 热门提示", padding="5")
        popular_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 热门标签容器
        self.popular_container = ttk.Frame(popular_frame)
        self.popular_container.pack(fill=tk.X)
        
        # 加载热门提示
        self.load_popular_completions()
        
    def create_categories_section(self, parent):
        """创建分类区域"""
        cat_frame = ttk.LabelFrame(parent, text="📚 专业领域", padding="5")
        cat_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 分类按钮容器
        self.categories_container = ttk.Frame(cat_frame)
        self.categories_container.pack(fill=tk.X)
        
    def create_stats_section(self, parent):
        """创建统计区域"""
        stats_frame = ttk.LabelFrame(parent, text="📊 使用统计", padding="5")
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 统计网格
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill=tk.X)
        
        # 统计项
        self.total_questions_label = ttk.Label(stats_grid, text="总问题: 0", font=('Segoe UI', 9))
        self.total_questions_label.grid(row=0, column=0, padx=5, sticky=tk.W)
        
        self.completion_rate_label = ttk.Label(stats_grid, text="成功率: 0%", font=('Segoe UI', 9))
        self.completion_rate_label.grid(row=0, column=1, padx=5, sticky=tk.W)
        
        self.response_time_label = ttk.Label(stats_grid, text="响应: 0ms", font=('Segoe UI', 9))
        self.response_time_label.grid(row=0, column=2, padx=5, sticky=tk.W)
        
    def create_footer(self, parent):
        """创建底部区域"""
        footer_frame = ttk.Frame(parent)
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # 功能按钮
        btn_frame = ttk.Frame(footer_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="⚙️", width=3, command=self.show_settings).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="📌", width=3, command=self.toggle_topmost).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="➖", width=3, command=self.minimize_window).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="❌", width=3, command=self.close_app).pack(side=tk.RIGHT, padx=2)
        
        # 版权信息
        copyright_label = ttk.Label(footer_frame, text="© 2024 AIword | 智能补全助手", font=('Segoe UI', 8))
        copyright_label.pack(pady=2)
        
    def create_context_menu(self):
        """创建右键菜单"""
        self.context_menu = Menu(self.root, tearoff=0)
        self.context_menu.add_command(label="设置", command=self.show_settings)
        self.context_menu.add_command(label="切换置顶", command=self.toggle_topmost)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="重新加载", command=self.reload_data)
        self.context_menu.add_command(label="关于", command=self.show_about)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="退出", command=self.close_app)
        
    def bind_events(self):
        """绑定事件"""
        # 输入框事件
        self.text_widget.bind('<KeyRelease>', self.on_text_change)
        self.text_widget.bind('<KeyPress>', self.on_key_press)
        self.text_widget.bind('<Button-3>', self.show_context_menu)
        
        # 全局快捷键
        self.root.bind('<Control-comma>', lambda e: self.show_settings())
        self.root.bind('<Control-slash>', lambda e: self.show_help())
        self.root.bind('<Escape>', lambda e: self.hide_completion())
        
        # 右键菜单
        self.root.bind('<Button-3>', self.show_context_menu)
        
        # 窗口事件
        self.root.protocol("WM_DELETE_WINDOW", self.close_app)
        
        # 确保输入框获得焦点
        self.text_widget.focus_set()
        
    def start_drag(self, event):
        """开始拖拽"""
        self.is_dragging = True
        self.drag_start_x = event.x_root - self.root.winfo_x()
        self.drag_start_y = event.y_root - self.root.winfo_y()
        
    def on_drag(self, event):
        """拖拽过程"""
        if self.is_dragging:
            x = event.x_root - self.drag_start_x
            y = event.y_root - self.drag_start_y
            self.root.geometry(f"+{x}+{y}")
            
    def stop_drag(self, event):
        """停止拖拽"""
        self.is_dragging = False
        # 拖拽结束后重新设置焦点
        self.text_widget.focus_set()
        
    def on_text_change(self, event):
        """文本变化事件"""
        text = self.text_widget.get("1.0", tk.END).strip()
        
        # 更新字符计数
        self.char_count_label.config(text=f"字符: {len(text)}")
        
        # 记录输入时间
        self.last_input_time = time.time() * 1000
        
        # 重置选择
        self.selected_index = -1
        
        # 延迟触发补全
        if len(text) >= 2:
            self.root.after(self.settings['trigger_delay'], 
                          lambda: self.detect_completion(text))
        else:
            self.hide_completion()
            
    def on_key_press(self, event):
        """按键事件"""
        print(f"键盘事件: {event.keysym}")  # 调试输出
        
        # Tab键触发补全
        if event.keysym == 'Tab':
            print("触发Tab补全")  # 调试输出
            self.trigger_manual_completion()
            return 'break'
            
        # 处理补全窗口的键盘导航
        if self.completion_window and self.completion_window.winfo_exists():
            if event.keysym == 'Up':
                print("向上导航")  # 调试输出
                self.navigate_completion(-1)
                return 'break'
            elif event.keysym == 'Down':
                print("向下导航")  # 调试输出
                self.navigate_completion(1)
                return 'break'
            elif event.keysym == 'Return':
                print("确认选择")  # 调试输出
                self.select_current_completion()
                return 'break'
            elif event.keysym == 'Escape':
                print("关闭补全")  # 调试输出
                self.hide_completion()
                return 'break'
        
        # 如果是普通文本输入，允许正常处理
        return None
            
    def detect_completion(self, text):
        """检测补全"""
        current_time = time.time() * 1000
        
        # 检查是否是最新的输入
        if current_time - self.last_input_time < self.settings['trigger_delay']:
            return
            
        try:
            start_time = time.time() * 1000
            
            # 获取补全建议
            suggestions = self.completion_engine.detect_completion(
                text, int(current_time - self.last_input_time)
            )
            
            response_time = time.time() * 1000 - start_time
            self.update_response_time(response_time)
            
            if suggestions:
                self.current_suggestions = suggestions
                self.show_completion(suggestions)
            else:
                self.hide_completion()
                
        except Exception as e:
            print(f"补全检测失败: {e}")
            self.hide_completion()
            
    def trigger_manual_completion(self):
        """手动触发补全"""
        text = self.text_widget.get("1.0", tk.END).strip()
        if len(text) >= 1:
            self.detect_completion(text)
            
    def show_completion(self, suggestions):
        """显示补全窗口"""
        self.hide_completion()  # 先隐藏已存在的
        
        if not suggestions:
            return
            
        # 创建补全窗口
        self.completion_window = tk.Toplevel(self.root)
        self.completion_window.title("智能补全")
        self.completion_window.geometry("400x200")
        self.completion_window.transient(self.root)
        self.completion_window.attributes('-topmost', True)
        
        # 定位到输入框下方
        input_x = self.root.winfo_x()
        input_y = self.root.winfo_y() + 150
        self.completion_window.geometry(f"+{input_x}+{input_y}")
        
        # 补全选项
        frame = ttk.Frame(self.completion_window, padding="5")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="💡 智能建议", font=('Segoe UI', 10, 'bold')).pack(anchor=tk.W)
        
        self.completion_listbox = tk.Listbox(frame, 
                                           font=('Segoe UI', 10),
                                           height=min(6, len(suggestions)))
        self.completion_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 填充选项
        for suggestion in suggestions:
            display_text = f"{suggestion.text} ({suggestion.category})"
            self.completion_listbox.insert(tk.END, display_text)
            
        # 绑定事件 - 同时绑定在listbox和窗口上
        self.completion_listbox.bind('<Double-Button-1>', self.on_completion_select)
        self.completion_listbox.bind('<Return>', self.on_completion_select)
        self.completion_listbox.bind('<KeyPress>', self.on_completion_key_press)
        
        # 补全窗口的键盘事件
        self.completion_window.bind('<KeyPress>', self.on_completion_key_press)
        self.completion_window.bind('<Up>', lambda e: self.navigate_completion(-1))
        self.completion_window.bind('<Down>', lambda e: self.navigate_completion(1))
        self.completion_window.bind('<Return>', lambda e: self.select_current_completion())
        self.completion_window.bind('<Escape>', lambda e: self.hide_completion())
        
        # 选择第一项
        if suggestions:
            self.completion_listbox.selection_set(0)
            self.selected_index = 0
            
        # 键盘提示
        hint_label = ttk.Label(frame, text="↑↓选择 | Enter确认 | Esc关闭", font=('Segoe UI', 8))
        hint_label.pack(pady=2)
        
        # 确保补全窗口获得焦点，但保持主窗口的输入框焦点
        self.completion_window.focus_set()
        
    def on_completion_key_press(self, event):
        """补全窗口的键盘事件处理"""
        print(f"补全窗口键盘事件: {event.keysym}")  # 调试输出
        
        if event.keysym == 'Up':
            self.navigate_completion(-1)
            return 'break'
        elif event.keysym == 'Down':
            self.navigate_completion(1)
            return 'break'
        elif event.keysym == 'Return':
            self.select_current_completion()
            return 'break'
        elif event.keysym == 'Escape':
            self.hide_completion()
            return 'break'
        elif event.keysym == 'Tab':
            self.select_current_completion()
            return 'break'
            
        return None
            
    def hide_completion(self):
        """隐藏补全窗口"""
        if self.completion_window:
            self.completion_window.destroy()
            self.completion_window = None
        self.current_suggestions = []
        self.selected_index = -1
        
        # 重新设置焦点到输入框
        self.text_widget.focus_set()
        
    def navigate_completion(self, direction):
        """导航补全选项"""
        if not self.completion_window or not self.current_suggestions:
            return
            
        max_index = len(self.current_suggestions) - 1
        
        if direction > 0:  # 向下
            self.selected_index = min(max_index, self.selected_index + 1)
        else:  # 向上
            self.selected_index = max(0, self.selected_index - 1)
            
        self.completion_listbox.selection_clear(0, tk.END)
        self.completion_listbox.selection_set(self.selected_index)
        self.completion_listbox.see(self.selected_index)
        
        print(f"导航到选项 {self.selected_index}: {self.current_suggestions[self.selected_index].text}")  # 调试输出
        
    def select_current_completion(self):
        """选择当前补全项"""
        if self.selected_index >= 0 and self.selected_index < len(self.current_suggestions):
            print(f"选择补全项: {self.current_suggestions[self.selected_index].text}")  # 调试输出
            self.apply_completion(self.current_suggestions[self.selected_index])
            
    def on_completion_select(self, event):
        """补全选项选择事件"""
        selection = self.completion_listbox.curselection()
        if selection:
            index = selection[0]
            if index < len(self.current_suggestions):
                self.selected_index = index  # 更新选择索引
                self.apply_completion(self.current_suggestions[index])
                
    def apply_completion(self, suggestion):
        """应用补全"""
        try:
            current_text = self.text_widget.get("1.0", tk.END).strip()
            
            print(f"应用补全: {suggestion.text}")  # 调试输出
            
            # 生成完整问句
            complete_text = self.completion_engine.generate_complete_question(
                current_text,
                suggestion.text.replace('🔹', '').replace('📊', '').strip(),
                suggestion.template,
                suggestion.trigger_type
            )
            
            # 更新输入框
            self.text_widget.delete("1.0", tk.END)
            self.text_widget.insert("1.0", complete_text)
            
            # 更新统计
            self.stats['completion_used'] += 1
            self.update_stats_display()
            
            # 隐藏补全窗口
            self.hide_completion()
            
            # 将焦点重新设置到输入框
            self.text_widget.focus_set()
            
            # 将光标移到文本末尾
            self.text_widget.mark_set(tk.INSERT, tk.END)
            
            print("补全应用成功")  # 调试输出
            
        except Exception as e:
            print(f"应用补全失败: {e}")
            self.hide_completion()
            self.text_widget.focus_set()
            
    def submit_question(self):
        """提交问题"""
        text = self.text_widget.get("1.0", tk.END).strip()
        
        if not text:
            self.show_notification("请输入问题内容", "warning")
            return
            
        # 更新统计
        self.stats['total_questions'] += 1
        self.update_stats_display()
        self.save_stats()
        
        self.show_notification("问题已提交处理", "success")
        
    def load_popular_completions(self):
        """加载热门补全"""
        try:
            popular = self.completion_engine.get_popular_completions()
            
            # 清空现有按钮
            for widget in self.popular_container.winfo_children():
                widget.destroy()
                
            # 创建标签按钮
            for i, completion in enumerate(popular[:8]):  # 最多8个
                btn = ttk.Button(self.popular_container,
                               text=completion,
                               command=lambda c=completion: self.insert_popular(c))
                btn.grid(row=i//4, column=i%4, padx=2, pady=2, sticky=tk.W)
                
        except Exception as e:
            print(f"加载热门补全失败: {e}")
            
    def insert_popular(self, text):
        """插入热门提示"""
        current_text = self.text_widget.get("1.0", tk.END).strip()
        if current_text:
            new_text = f"{current_text} {text}"
        else:
            new_text = text
            
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.insert("1.0", new_text)
        self.text_widget.focus_set()
        
    def load_categories(self):
        """加载分类"""
        try:
            categories = self.completion_engine.get_categories()
            
            # 清空现有按钮
            for widget in self.categories_container.winfo_children():
                widget.destroy()
                
            icons = {
                'AI/ML': '🤖',
                'Data Science': '📊', 
                'Development': '💻',
                'Emerging Tech': '🚀'
            }
            
            for i, (key, category) in enumerate(categories.items()):
                icon = icons.get(key, '📁')
                btn_text = f"{icon} {category['name']}"
                
                btn = ttk.Button(self.categories_container,
                               text=btn_text,
                               command=lambda k=key: self.filter_by_category(k))
                btn.grid(row=i//2, column=i%2, padx=2, pady=2, sticky=tk.EW)
                
            # 配置列权重
            self.categories_container.columnconfigure(0, weight=1)
            self.categories_container.columnconfigure(1, weight=1)
            
        except Exception as e:
            print(f"加载分类失败: {e}")
            
    def filter_by_category(self, category):
        """按分类筛选"""
        self.show_notification(f"已选择 {category} 分类", "info")
        
    def update_stats_display(self):
        """更新统计显示"""
        self.total_questions_label.config(text=f"总问题: {self.stats['total_questions']}")
        
        completion_rate = 0
        if self.stats['total_questions'] > 0:
            completion_rate = round((self.stats['completion_used'] / self.stats['total_questions']) * 100)
        self.completion_rate_label.config(text=f"成功率: {completion_rate}%")
        
        avg_response_time = 0
        if self.stats['request_count'] > 0:
            avg_response_time = round(self.stats['total_response_time'] / self.stats['request_count'])
        self.response_time_label.config(text=f"响应: {avg_response_time}ms")
        
    def update_response_time(self, time_ms):
        """更新响应时间"""
        self.stats['total_response_time'] += time_ms
        self.stats['request_count'] += 1
        self.update_stats_display()
        
    def show_settings(self):
        """显示设置对话框"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("设置")
        settings_window.geometry("350x300")
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        frame = ttk.Frame(settings_window, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # 触发延迟
        ttk.Label(frame, text="触发延迟（毫秒）:").pack(anchor=tk.W)
        delay_var = tk.IntVar(value=self.settings['trigger_delay'])
        delay_scale = ttk.Scale(frame, from_=100, to=2000, variable=delay_var, orient=tk.HORIZONTAL)
        delay_scale.pack(fill=tk.X, pady=(0, 10))
        
        # 最大建议数
        ttk.Label(frame, text="最大建议数:").pack(anchor=tk.W)
        max_var = tk.IntVar(value=self.settings['max_suggestions'])
        max_scale = ttk.Scale(frame, from_=3, to=10, variable=max_var, orient=tk.HORIZONTAL)
        max_scale.pack(fill=tk.X, pady=(0, 10))
        
        # 透明度
        ttk.Label(frame, text="窗口透明度:").pack(anchor=tk.W)
        opacity_var = tk.DoubleVar(value=self.settings['window_opacity'])
        opacity_scale = ttk.Scale(frame, from_=0.3, to=1.0, variable=opacity_var, orient=tk.HORIZONTAL)
        opacity_scale.pack(fill=tk.X, pady=(0, 10))
        
        # 复选框选项
        topmost_var = tk.BooleanVar(value=self.settings['always_on_top'])
        ttk.Checkbutton(frame, text="始终置顶", variable=topmost_var).pack(anchor=tk.W, pady=5)
        
        autocomplete_var = tk.BooleanVar(value=self.settings['auto_complete'])
        ttk.Checkbutton(frame, text="自动补全", variable=autocomplete_var).pack(anchor=tk.W, pady=5)
        
        # 按钮
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        def save_settings():
            self.settings['trigger_delay'] = int(delay_var.get())
            self.settings['max_suggestions'] = int(max_var.get())
            self.settings['window_opacity'] = opacity_var.get()
            self.settings['always_on_top'] = topmost_var.get()
            self.settings['auto_complete'] = autocomplete_var.get()
            
            # 应用设置
            self.root.attributes('-alpha', self.settings['window_opacity'])
            self.root.attributes('-topmost', self.settings['always_on_top'])
            
            self.save_settings()
            self.show_notification("设置已保存", "success")
            settings_window.destroy()
            
        ttk.Button(btn_frame, text="保存", command=save_settings).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="取消", command=settings_window.destroy).pack(side=tk.RIGHT)
        
    def toggle_topmost(self):
        """切换置顶状态"""
        current = self.root.attributes('-topmost')
        self.root.attributes('-topmost', not current)
        self.settings['always_on_top'] = not current
        status = "开启" if not current else "关闭"
        self.show_notification(f"置顶状态: {status}", "info")
        
    def minimize_window(self):
        """最小化窗口"""
        self.root.iconify()
        
    def show_context_menu(self, event):
        """显示右键菜单"""
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()
            
    def show_notification(self, message, type_="info"):
        """显示通知"""
        # 简单的messagebox通知，后续可以改为自定义通知
        if type_ == "error":
            messagebox.showerror("错误", message)
        elif type_ == "warning":
            messagebox.showwarning("警告", message)
        elif type_ == "success":
            messagebox.showinfo("成功", message)
        else:
            messagebox.showinfo("提示", message)
            
    def show_help(self):
        """显示帮助"""
        help_text = """
AIword 智能提示词补全 - 快捷键帮助

补全操作：
• Tab - 触发智能补全
• ↑↓ - 选择补全选项
• Enter - 确认选择
• Esc - 关闭补全层

窗口操作：
• Ctrl+, - 打开设置
• Ctrl+/ - 显示帮助
• 右键 - 显示菜单

使用方法：
1. 在输入框中输入问题
2. 系统会自动或手动（Tab）触发补全
3. 使用方向键选择建议选项
4. 按Enter确认补全
        """
        messagebox.showinfo("帮助", help_text)
        
    def show_about(self):
        """显示关于信息"""
        about_text = """
AIword v1.3.0 - 原生悬浮窗版本

垂直化提示词补全系统
基于tkinter原生GUI，无需浏览器

功能特色：
• 🎯 智能前缀触发
• 🧠 专业领域识别  
• ⌨️ 键盘导航体验
• 🎈 原生悬浮窗
• 📊 使用统计

© 2024 AIword Team
        """
        messagebox.showinfo("关于", about_text)
        
    def reload_data(self):
        """重新加载数据"""
        try:
            self.completion_engine = CompletionEngine()  # 重新初始化引擎
            self.load_categories()
            self.load_popular_completions()
            self.show_notification("数据已重新加载", "success")
        except Exception as e:
            self.show_notification(f"重新加载失败: {e}", "error")
            
    def load_settings(self):
        """加载设置"""
        try:
            with open('settings.json', 'r', encoding='utf-8') as f:
                saved_settings = json.load(f)
                self.settings.update(saved_settings)
        except:
            pass  # 使用默认设置
            
    def save_settings(self):
        """保存设置"""
        try:
            with open('settings.json', 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存设置失败: {e}")
            
    def save_stats(self):
        """保存统计"""
        try:
            with open('stats.json', 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存统计失败: {e}")
            
    def close_app(self):
        """关闭应用"""
        self.save_settings()
        self.save_stats()
        self.root.quit()
        self.root.destroy()
        
    def run(self):
        """运行应用"""
        print("🚀 启动 AIword 原生悬浮窗...")
        print("💡 功能说明：")
        print("   🖱️  拖拽标题移动窗口")
        print("   📌 点击📌按钮切换置顶")
        print("   ⌨️  Tab键触发智能补全")
        print("   ↕️  ↑↓键选择补全选项") 
        print("   ✅ Enter确认补全")
        print("   🖱️  右键显示菜单")
        print("   ⚙️  Ctrl+, 打开设置")
        print("   ❓ Ctrl+/ 显示帮助")
        print("")
        print("🛑 关闭窗口退出应用")
        
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\n👋 感谢使用 AIword！")
        except Exception as e:
            print(f"❌ 应用运行失败: {e}")

    def on_tab_key(self, event):
        """专门处理Tab键"""
        print("Tab键被按下 - 直接触发补全")  # 调试输出
        self.trigger_manual_completion()
        return 'break'  # 阻止默认Tab行为

def main():
    """主函数"""
    try:
        app = NativeFloatingAIWord()
        app.run()
    except Exception as e:
        print(f"❌ 应用启动失败: {e}")
        input("按回车键退出...")

if __name__ == '__main__':
    main() 