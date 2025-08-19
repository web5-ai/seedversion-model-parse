"""
统一的环境管理模块
整合环境检查、依赖验证、目录设置和确定性计算配置

这个模块将原来分散在 utils/environment.py、utils/force_env.py 和 utils/env_manager.py
中的功能合并到一个统一的接口中，避免重复代码和重复调用。
"""

import os
import sys
import random
import numpy as np
import torch
import ctypes
import platform
import pkg_resources
from typing import Dict, Any, List
from pathlib import Path

# 导入配置和日志
from config import MODEL_CONFIG
from utils.logging_config import get_logger

# 获取日志记录器
logger = get_logger("Environment")

# 全局状态标记，防止重复执行
_environment_initialized = False
_initialization_result = None

def initialize_environment(seed: int = 123, verbose: bool = False) -> Dict[str, Any]:
    """
    统一的环境初始化入口

    Args:
        seed: 随机种子，默认为123
        verbose: 是否输出详细信息

    Returns:
        Dict: 初始化结果信息
    """
    global _environment_initialized, _initialization_result

    # 如果已经初始化过，直接返回结果
    if _environment_initialized:
        if verbose:
            logger.info("环境已初始化，跳过重复设置")
        return _initialization_result

    # 如果之前初始化失败，也要重新尝试
    if _initialization_result is not None and not _initialization_result.get("success", False):
        if verbose:
            logger.info("上次初始化失败，重新尝试...")

    logger.info("开始环境初始化...")

    # 显示启动横幅
    _show_banner()

    try:
        # 1. 检查依赖
        logger.info("检查依赖包...")
        if not _check_dependencies():
            _initialization_result = {"success": False, "error": "依赖检查失败"}
            return _initialization_result

        # 2. 设置基础环境
        logger.info("设置基础环境...")
        if not _setup_directories():
            _initialization_result = {"success": False, "error": "目录设置失败"}
            return _initialization_result

        # 3. 配置CUDA和设备
        logger.info("配置计算设备...")
        _configure_device()

        # 4. 强制环境设置（确定性计算）
        logger.info("配置确定性计算环境...")
        env_info = _force_deterministic_environment(seed)

        # 5. 显示总结信息
        _show_summary(env_info, verbose)

        # 标记初始化完成
        _environment_initialized = True
        _initialization_result = {
            "success": True,
            "seed": seed,
            "environment_info": env_info
        }

        logger.info("环境初始化完成")
        return _initialization_result

    except Exception as e:
        logger.error(f"环境初始化失败: {str(e)}")
        _initialization_result = {"success": False, "error": str(e)}
        return _initialization_result


def _show_banner():
    """显示启动横幅"""
    banner_lines = [
        "=" * 60,
        "油菜籽成分分析系统",
        "基于深度学习的图像识别技术",
        "=" * 60
    ]

    for line in banner_lines:
        logger.info(line)


def _check_dependencies() -> bool:
    """检查必要的依赖是否已安装"""
    required_packages = [
        'torch>=1.7.0',
        'torchvision>=0.8.0',
        'numpy>=1.19.0',
        'matplotlib>=3.3.0',
        'Pillow>=8.0.0'
    ]

    missing = []
    installed = []

    for package in required_packages:
        package_name = package.split('>=')[0]
        try:
            dist = pkg_resources.get_distribution(package_name)
            installed.append(f"{package_name}=={dist.version}")
        except pkg_resources.DistributionNotFound:
            missing.append(package)

    if missing:
        logger.error("缺少以下依赖包:")
        for package in missing:
            logger.error(f"   - {package}")
        logger.info("请使用以下命令安装缺少的依赖:")
        logger.info(f"   uv add {' '.join(missing)}")
        return False

    logger.info(f"依赖检查通过 ({len(installed)} 个包)")
    return True


def _setup_directories() -> bool:
    """设置必要的目录结构"""
    required_dirs = [
        'weights',
        'results',
        'tests/test_images'
    ]

    created_dirs = []
    project_root = Path(__file__).parent.parent

    for directory in required_dirs:
        dir_path = project_root / directory
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                created_dirs.append(directory)
            except Exception as e:
                logger.error(f"创建目录失败: {dir_path}, 错误: {str(e)}")
                return False

    if created_dirs:
        logger.info(f"已创建目录: {', '.join(created_dirs)}")

    return True


def _configure_device():
    """配置计算设备"""
    try:
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA可用: {device_name}")

            # 使用专门的CUDA兼容性检查
            try:
                from utils.cuda_compatibility import test_cuda_operations
                test_results = test_cuda_operations()

                # 根据测试结果决定使用的设备
                if test_results["basic_cuda"] and test_results["tensor_operations"]:
                    MODEL_CONFIG["device"] = "cuda"
                    logger.info("CUDA功能测试通过，将使用CUDA进行计算")

                    # 如果NMS失败，给出警告但仍使用CUDA
                    if not test_results["torchvision_nms"]:
                        logger.warning("torchvision NMS操作在CUDA上失败，检测功能可能受影响")
                        logger.warning("建议运行: python -m utils.cuda_compatibility 查看详细修复建议")
                else:
                    logger.warning("CUDA基本操作测试失败，将使用CPU进行计算")
                    logger.warning("建议运行: python -m utils.cuda_compatibility 查看详细修复建议")
                    MODEL_CONFIG["device"] = "cpu"

            except ImportError:
                # 如果无法导入兼容性检查模块，使用简单测试
                logger.warning("无法导入CUDA兼容性检查模块，使用简单测试")
                try:
                    test_tensor = torch.randn(100, 100).cuda()
                    test_result = test_tensor.sum()
                    test_result.cpu()
                    MODEL_CONFIG["device"] = "cuda"
                    logger.info("简单CUDA测试通过，将使用CUDA进行计算")
                except Exception as simple_test_error:
                    logger.warning(f"简单CUDA测试失败: {str(simple_test_error)[:100]}...")
                    MODEL_CONFIG["device"] = "cpu"

            except Exception as cuda_test_error:
                logger.warning(f"CUDA兼容性检查失败: {str(cuda_test_error)[:100]}...")
                logger.warning("检测到CUDA兼容性问题，将使用CPU进行计算")
                MODEL_CONFIG["device"] = "cpu"
        else:
            logger.info("CUDA不可用，将使用CPU进行计算")
            MODEL_CONFIG["device"] = "cpu"
    except Exception as e:
        logger.warning(f"检查CUDA状态时出错: {str(e)}")
        MODEL_CONFIG["device"] = "cpu"


def _force_deterministic_environment(seed: int) -> Dict[str, Any]:
    """强制设置确定性计算环境"""
    # 收集原始环境信息
    env_info = {
        "Python版本": sys.version.split('\n')[0],
        "Python解释器": sys.executable,
        "工作目录": os.getcwd(),
        "平台": platform.platform()
    }

    # 输出系统环境信息
    logger.info(f"系统环境: Python {platform.python_version()} | {platform.system()}")

    # 1. 设置Python相关环境变量
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. 设置Python随机种子
    random.seed(seed)

    # 3. 设置NumPy随机种子
    np.random.seed(seed)

    # 4. 设置PyTorch随机种子
    torch.manual_seed(seed)

    # 5. 设置PyTorch CUDA随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # 设置当前CUDA设备
        torch.cuda.set_device(0)

        # 同步CUDA操作
        torch.cuda.synchronize()

        # 设置CUDA确定性操作
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        # 对于PyTorch 1.8+，启用确定性算法
        try:
            torch.use_deterministic_algorithms(True)
        except:
            logger.info("PyTorch版本不支持use_deterministic_algorithms")

    # 6. 设置PyTorch后端参数
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 7. 设置PyTorch线程数
    torch.set_num_threads(1)

    # 8. 设置Windows特定参数
    if platform.system() == 'Windows':
        try:
            process_handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetPriorityClass(process_handle, 0x00000080)
        except:
            logger.warning("设置Windows进程优先级失败")

        try:
            mask = 1  # 使用第一个CPU核心
            ctypes.windll.kernel32.SetProcessAffinityMask(process_handle, mask)
        except:
            logger.warning("设置Windows处理器亲和性失败")

    # 9. 禁用NumPy多线程
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass

    # 10. 设置浮点数精度
    torch.set_default_dtype(torch.float32)

    # 输出总结日志
    thread_info = f"线程数={torch.get_num_threads()}"
    deterministic_info = "确定性=是" if torch.backends.cudnn.deterministic else "确定性=否"
    logger.info(f"环境初始化完成: 种子={seed} | {thread_info} | {deterministic_info}")

    return _get_environment_info(seed, env_info)


def _get_environment_info(seed: int, env_info: Dict[str, Any] = None) -> Dict[str, Any]:
    """获取环境信息的辅助函数"""
    env_settings = {
        "seed": seed,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "torch_threads": torch.get_num_threads(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_available": torch.cuda.is_available(),
        "platform": platform.system(),
        "python_version": platform.python_version()
    }

    if env_info:
        env_settings.update(env_info)

    return env_settings


def _show_summary(env_info: Dict[str, Any], verbose: bool = False):
    """显示环境配置总结"""
    summary_lines = [
        "环境配置总结:",
        f"   Python: {env_info.get('python_version', 'Unknown')}",
        f"   平台: {env_info.get('platform', 'Unknown')}",
        f"   随机种子: {env_info.get('seed', 'Unknown')}",
        f"   PyTorch线程: {env_info.get('torch_threads', 'Unknown')}",
    ]

    # CUDA信息
    if env_info.get('cuda_available', False):
        try:
            device_name = torch.cuda.get_device_name(0)
            summary_lines.append(f"   CUDA: {device_name}")
        except:
            summary_lines.append("   CUDA: 可用")
    else:
        summary_lines.append("   计算设备: CPU")

    # 确定性设置
    deterministic = env_info.get('cudnn_deterministic', False)
    summary_lines.append(f"   确定性计算: {'是' if deterministic else '否'}")

    for line in summary_lines:
        logger.info(line)

    # 详细信息（仅在verbose模式下显示）
    if verbose:
        logger.info("详细配置:")
        for key, value in env_info.items():
            if key not in ['seed', 'python_version', 'platform', 'torch_threads', 'cuda_available', 'cudnn_deterministic']:
                logger.info(f"   {key}: {value}")


def get_environment_status() -> Dict[str, Any]:
    """获取当前环境状态"""
    if _environment_initialized:
        return _initialization_result
    else:
        return {"success": False, "error": "环境未初始化"}


def reset_environment():
    """重置环境状态（用于测试）"""
    global _environment_initialized, _initialization_result
    _environment_initialized = False
    _initialization_result = None
    logger.info("环境状态已重置")


# 为了向后兼容，保留原有的函数名
def force_environment(seed: int = 123) -> Dict[str, Any]:
    """向后兼容的函数，调用新的统一初始化函数"""
    result = initialize_environment(seed=seed, verbose=False)
    return result.get("environment_info", {})


# 为了向后兼容，提供原有的函数接口
def check_dependencies() -> bool:
    """向后兼容的依赖检查函数"""
    return _check_dependencies()


def setup_environment() -> bool:
    """向后兼容的环境设置函数"""
    _configure_device()
    return True
