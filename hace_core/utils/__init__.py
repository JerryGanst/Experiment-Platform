"""
实用工具包

注意：UnifiedMonitor已移动到src.monitoring模块
"""

# 为了向后兼容，提供重定向导入
try:
    from src.monitoring import UnifiedMonitor
    print("⚠️ UnifiedMonitor已移动到src.monitoring模块，请更新导入语句")
except ImportError:
    print("❌ 无法从src.monitoring导入UnifiedMonitor，请检查模块路径") 