"""
强制设置环境变量和计算参数，确保在不同启动方式下获得一致的结果
"""

import os
import sys
import random
import numpy as np
import torch
import ctypes
import platform

# 导入集中的日志配置
from utils.logging_config import get_logger

# 获取日志记录器
logger = get_logger("ForceEnv")

# 设置是否输出详细日志
VERBOSE_LOGGING = False

def force_environment(seed=123):
    """
    强制设置环境变量和计算参数

    Args:
        seed: 随机种子，默认为123
    """
    if VERBOSE_LOGGING:
        logger.info("开始强制设置环境...")

    # 收集原始环境信息（但不输出详细日志）
    env_info = {
        "Python版本": sys.version.split('\n')[0],
        "Python解释器": sys.executable,
        "工作目录": os.getcwd(),
        "平台": platform.platform()
    }

    # 只输出一条简洁的日志
    logger.info(f"系统环境: Python {platform.python_version()}, {platform.system()}, {'CUDA可用' if torch.cuda.is_available() else 'CUDA不可用'}")

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

        # 获取CUDA设备信息
        device_count = torch.cuda.device_count()
        if VERBOSE_LOGGING:
            logger.info(f"CUDA设备数量: {device_count}")
            for i in range(device_count):
                logger.info(f"CUDA设备{i}: {torch.cuda.get_device_name(i)}")

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
            if VERBOSE_LOGGING:
                logger.info("PyTorch版本不支持use_deterministic_algorithms")

    # 6. 设置PyTorch后端参数
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 7. 设置PyTorch线程数
    torch.set_num_threads(1)

    # 8. 设置Windows特定参数
    if platform.system() == 'Windows':
        # 设置进程优先级
        try:
            process_handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetPriorityClass(process_handle, 0x00000080)  # NORMAL_PRIORITY_CLASS
        except:
            if VERBOSE_LOGGING:
                logger.warning("设置Windows进程优先级失败")

        # 设置处理器亲和性（将进程限制在特定CPU核心上）
        try:
            # 将进程限制在第一个CPU核心上
            mask = 1  # 使用第一个CPU核心
            ctypes.windll.kernel32.SetProcessAffinityMask(process_handle, mask)
        except:
            if VERBOSE_LOGGING:
                logger.warning("设置Windows处理器亲和性失败")

    # 9. 禁用NumPy多线程
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass

    # 10. 设置浮点数精度
    torch.set_default_dtype(torch.float32)

    # 输出一条总结日志
    logger.info(f"环境已设置: 随机种子={seed}, PyTorch线程数={torch.get_num_threads()}, CUDA确定性={torch.backends.cudnn.deterministic}")

    # 收集简化的环境设置信息
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

    # 如果需要详细日志，则输出完整配置
    if VERBOSE_LOGGING:
        logger.info("环境强制设置完成，详细配置如下:")
        for key, value in env_settings.items():
            logger.info(f"  {key}: {value}")

    # 使用原始环境信息
    env_settings.update(env_info)

    return env_settings

# 不再自动执行，避免重复调用
# 如果需要自动执行，请取消下面的注释
# force_environment()
