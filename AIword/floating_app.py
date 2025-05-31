#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIword 悬浮窗桌面应用
支持：
- 悬浮窗显示
- 窗口拖拽
- 始终置顶
- 透明背景
- 快捷键激活
"""

import webview
import threading
import time
import sys
import os
from web_interface import app as flask_app

class FloatingAIWord:
    def __init__(self):
        self.window = None
        self.is_minimized = False
        self.opacity = 0.95
        self.always_on_top = True
        
    def create_window(self):
        """创建悬浮窗"""
        self.window = webview.create_window(
            title='AIword - 智能提示词补全',
            url='http://localhost:5000/floating',  # 使用专门的悬浮窗页面
            width=480,
            height=650,
            min_size=(380, 500),
            max_size=(600, 800),
            resizable=True,
            shadow=True,
            on_top=self.always_on_top,
            text_select=False,
        )
        
        # 绑定事件
        self.bind_events()
        
        return self.window
    
    def bind_events(self):
        """绑定窗口事件"""
        pass
    
    def toggle_window(self):
        """切换窗口显示/隐藏"""
        if self.window:
            if self.is_minimized:
                self.window.restore()
                self.is_minimized = False
            else:
                self.window.minimize()
                self.is_minimized = True
    
    def set_opacity(self, opacity):
        """设置窗口透明度"""
        self.opacity = max(0.3, min(1.0, opacity))
        # 注意：webview的透明度控制有限，主要通过CSS实现
        
    def toggle_always_on_top(self):
        """切换是否始终置顶"""
        self.always_on_top = not self.always_on_top
        # 注意：动态切换置顶可能需要重启窗口
        print(f"窗口置顶: {'开启' if self.always_on_top else '关闭'}")
    
    def minimize_window(self):
        """最小化窗口"""
        if self.window:
            self.window.minimize()
            self.is_minimized = True

# 创建API类用于JavaScript调用
class WindowAPI:
    def __init__(self, floating_app):
        self.floating_app = floating_app
    
    def toggle_window(self):
        """JavaScript调用：切换窗口"""
        self.floating_app.toggle_window()
        return "窗口状态已切换"
    
    def toggle_always_on_top(self):
        """JavaScript调用：切换置顶"""
        self.floating_app.toggle_always_on_top()
        return f"置顶状态: {'开启' if self.floating_app.always_on_top else '关闭'}"
    
    def minimize_window(self):
        """JavaScript调用：最小化"""
        self.floating_app.minimize_window()
        return "窗口已最小化"
    
    def set_opacity(self, opacity):
        """JavaScript调用：设置透明度"""
        self.floating_app.set_opacity(opacity / 100.0)
        return f"透明度设置为: {opacity}%"

def start_flask_server():
    """启动Flask服务器"""
    print("正在启动 AIword 后端服务...")
    try:
        flask_app.run(
            host='127.0.0.1',
            port=5000,
            debug=False,
            use_reloader=False,
            threaded=True
        )
    except Exception as e:
        print(f"Flask服务器启动失败: {e}")

def check_flask_server():
    """检查Flask服务器是否启动"""
    import requests
    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            response = requests.get('http://localhost:5000/floating', timeout=2)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(0.5)
        print(f"等待后端服务启动... ({attempt + 1}/{max_attempts})")
    return False

def main():
    """主函数"""
    print("🚀 启动 AIword 悬浮窗应用...")
    
    # 在单独线程中启动Flask服务器
    flask_thread = threading.Thread(target=start_flask_server, daemon=True)
    flask_thread.start()
    
    # 检查Flask服务器是否启动成功
    print("⏳ 等待后端服务启动...")
    if not check_flask_server():
        print("❌ 后端服务启动失败，请检查端口5000是否被占用")
        sys.exit(1)
    
    # 创建悬浮窗应用
    floating_app = FloatingAIWord()
    window = floating_app.create_window()
    
    # 创建API实例
    api = WindowAPI(floating_app)
    
    print("✨ AIword 悬浮窗已启动")
    print("💡 功能说明：")
    print("   🖱️  拖拽头部移动窗口")
    print("   📌 窗口始终置顶显示")
    print("   ⌨️  Tab键触发智能补全")
    print("   ↕️  ↑↓键选择补全选项")
    print("   ✅ Enter/Tab确认补全")
    print("   🚪 Esc最小化窗口")
    print("   🖱️  右键显示菜单")
    print("   ⚙️  Ctrl+Shift+T 切换置顶")
    print("   🔄 Ctrl+Shift+H 切换显示")
    print("")
    print("🛑 按 Ctrl+C 退出应用")
    
    try:
        # 启动webview应用
        webview.start(
            window,
            api,  # 传入API对象
            debug=False,
            http_server=False,
        )
    except KeyboardInterrupt:
        print("\n👋 感谢使用 AIword！")
        sys.exit(0)
    except Exception as e:
        print(f"❌ 应用启动失败: {e}")
        print("请检查是否安装了所需依赖：pip install -r requirements.txt")
        sys.exit(1)

if __name__ == '__main__':
    main() 