"""
Seed Analysis System - Project Runner
This script starts the project from any location, automatically setting correct paths and environment variables
"""

import os
import sys
import subprocess

# Get absolute path to project root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add project root directory to Python path
sys.path.insert(0, ROOT_DIR)

# Set environment variables
os.environ["PYTHONPATH"] = ROOT_DIR
os.environ["PYTHONHASHSEED"] = "123"

def check_basic_dependencies():
    """快速检查基本依赖是否可导入"""
    critical_deps = ["torch", "fastapi", "uvicorn"]
    missing_deps = []

    for dep in critical_deps:
        try:
            __import__(dep)
        except ImportError:
            missing_deps.append(dep)

    if missing_deps:
        print("=" * 50)
        print("关键依赖缺失")
        print("=" * 50)
        print("以下关键包无法导入:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\n请使用以下命令安装:")
        print(f"uv add {' '.join(missing_deps)}")
        print("=" * 50)
        return False

    return True

def main():
    """Main function to start the project"""
    # 快速检查关键依赖
    if not check_basic_dependencies():
        return 1

    # 环境初始化将在backend/main.py中统一处理
    # 这里只做基本的导入检查
    try:
        import torch
        print("PyTorch导入成功")
    except Exception as e:
        print(f"PyTorch导入失败: {str(e)}")
        return 1

    # 设置后端目录和主脚本路径
    backend_dir = os.path.join(ROOT_DIR, "backend")
    main_script = os.path.join(backend_dir, "main.py")

    # 切换到后端目录
    os.chdir(backend_dir)

    # 简化的启动信息
    print("=" * 50)
    print("油菜籽成分分析系统")
    print("=" * 50)
    print(f"项目根目录: {ROOT_DIR}")
    print(f"Python: {sys.executable}")
    print(f"启动后端服务...")
    print("=" * 50)

    # 运行后端
    try:
        subprocess.run([sys.executable, main_script])
        return 0
    except KeyboardInterrupt:
        print("\n服务已停止")
        return 0
    except Exception as e:
        print(f"后端运行错误: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
