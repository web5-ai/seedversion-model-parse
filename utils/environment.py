"""
环境检查和依赖管理
"""
import os
import sys
import pkg_resources
import logging

logger = logging.getLogger("Environment")

def check_dependencies():
    """
    检查必要的依赖是否已安装
    
    Returns:
        bool: 是否所有依赖都已安装
    """
    required_packages = [
        'torch>=1.7.0',
        'torchvision>=0.8.0',
        'numpy>=1.19.0',
        'matplotlib>=3.3.0',
        'Pillow>=8.0.0'
    ]
    
    missing = []
    
    for package in required_packages:
        package_name = package.split('>=')[0]
        try:
            pkg_resources.get_distribution(package_name)
        except pkg_resources.DistributionNotFound:
            missing.append(package)
    
    if missing:
        logger.error("缺少以下依赖包:")
        for package in missing:
            logger.error(f"  - {package}")
        logger.info("请使用以下命令安装缺少的依赖:")
        logger.info(f"  pip install {' '.join(missing)}")
        return False
    
    return True

def setup_environment():
    """
    设置运行环境
    
    Returns:
        bool: 环境设置是否成功
    """
    # 检查必要的目录是否存在，不存在则创建
    required_dirs = [
        'weights',
        'results',
        'tests/test_images'
    ]
    
    for directory in required_dirs:
        dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), directory)
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)
                logger.info(f"已创建目录: {dir_path}")
            except Exception as e:
                logger.error(f"创建目录失败: {dir_path}, 错误: {str(e)}")
                return False
    
    # 检查CUDA是否可用
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"CUDA可用: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("CUDA不可用，将使用CPU进行计算")
    except Exception as e:
        logger.warning(f"检查CUDA状态时出错: {str(e)}")
    
    return True