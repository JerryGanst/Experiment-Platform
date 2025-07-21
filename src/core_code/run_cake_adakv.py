#!/usr/bin/env python3
"""
CAKE-AdaKV 统一运行脚本

这是一个简化的入口脚本，提供最常用的运行场景。
对于更多高级选项，请使用 launcher.py
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd: str) -> int:
    """运行命令并返回退出码"""
    print(f"执行命令: {cmd}")
    return subprocess.call(cmd, shell=True)


def main():
    """主函数"""
    print("=" * 60)
    print("CAKE-AdaKV 统一集成框架")
    print("=" * 60)
    
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    launcher_path = script_dir / "launcher.py"
    
    if not launcher_path.exists():
        print(f"❌ 找不到 launcher.py: {launcher_path}")
        return 1
    
    # 显示菜单
    print("\n请选择运行模式:")
    print("1. 快速测试 (合成数据, BL=256)")
    print("2. 开发模式 (启用所有监控)")
    print("3. 生产模式 (优化性能)")
    print("4. 基准测试 (BL=128/256/512/1024)")
    print("5. 自定义运行")
    print("0. 退出")
    
    choice = input("\n请输入选项 (0-5): ").strip()
    
    if choice == "0":
        print("退出程序")
        return 0
    
    elif choice == "1":
        # 快速测试
        cmd = f"python {launcher_path} --synthetic --bl 256 --detailed"
        return run_command(cmd)
    
    elif choice == "2":
        # 开发模式
        bl = input("请输入 Budget Limit (默认 512): ").strip() or "512"
        cmd = f"python {launcher_path} --env dev --synthetic --bl {bl} --detailed"
        return run_command(cmd)
    
    elif choice == "3":
        # 生产模式
        input_file = input("请输入注意力权重文件路径 (留空使用合成数据): ").strip()
        bl = input("请输入 Budget Limit (默认 4096): ").strip() or "4096"
        output_file = input("请输入输出文件路径 (留空打印到屏幕): ").strip()
        
        cmd = f"python {launcher_path} --env prod --bl {bl}"
        if input_file:
            cmd += f" -i {input_file}"
        else:
            cmd += " --synthetic"
        if output_file:
            cmd += f" -o {output_file}"
            
        return run_command(cmd)
    
    elif choice == "4":
        # 基准测试
        print("\n开始基准测试...")
        bl_values = [128, 256, 512, 1024]
        
        results_dir = Path("benchmark_results")
        results_dir.mkdir(exist_ok=True)
        
        for bl in bl_values:
            print(f"\n测试 BL={bl}...")
            output_file = results_dir / f"result_bl_{bl}.json"
            cmd = f"python {launcher_path} --synthetic --bl {bl} --monitor --detailed -o {output_file}"
            
            ret = run_command(cmd)
            if ret != 0:
                print(f"❌ BL={bl} 测试失败")
            else:
                print(f"✅ BL={bl} 结果保存到: {output_file}")
        
        print(f"\n所有结果保存在: {results_dir}")
        return 0
    
    elif choice == "5":
        # 自定义运行
        print("\n构建自定义命令...")
        
        # 基本选项
        use_config = input("使用配置文件? (y/n, 默认 n): ").strip().lower() == 'y'
        if use_config:
            config_file = input("配置文件路径: ").strip()
            cmd = f"python {launcher_path} --config {config_file}"
        else:
            cmd = f"python {launcher_path}"
            
            # 输入输出
            input_file = input("注意力权重文件 (留空使用合成数据): ").strip()
            if input_file:
                cmd += f" -i {input_file}"
            else:
                cmd += " --synthetic"
                
            # 缓存大小
            bl = input("Budget Limit (留空使用默认): ").strip()
            if bl:
                cmd += f" --bl {bl}"
                
            # 监控选项
            if input("启用监控? (y/n): ").strip().lower() == 'y':
                cmd += " --monitor"
            if input("启用自动调优? (y/n): ").strip().lower() == 'y':
                cmd += " --auto-tune"
            if input("输出详细结果? (y/n): ").strip().lower() == 'y':
                cmd += " --detailed"
                
            # 输出文件
            output_file = input("输出文件路径 (留空打印到屏幕): ").strip()
            if output_file:
                cmd += f" -o {output_file}"
        
        print(f"\n最终命令: {cmd}")
        if input("执行该命令? (y/n): ").strip().lower() == 'y':
            return run_command(cmd)
        else:
            print("取消执行")
            return 0
    
    else:
        print("❌ 无效的选项")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        sys.exit(1) 