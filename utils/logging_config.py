"""
集中的日志配置模块，避免多处配置导致的冲突
"""

import logging
import sys
from config import SYSTEM_CONFIG

# 标记是否已经配置过根日志器
_root_logger_configured = False

def get_logger(name, level=None):
    """
    获取配置好的日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别，默认使用config.py中的配置
        
    Returns:
        配置好的日志记录器
    """
    global _root_logger_configured
    
    # 如果未指定级别，使用配置中的默认级别
    if level is None:
        level = SYSTEM_CONFIG.get("log_level", "INFO")
    
    # 将字符串级别转换为logging级别
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    numeric_level = level_map.get(level, logging.INFO)
    
    # 获取日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    
    # 如果日志记录器已经有处理器，直接返回
    if logger.handlers:
        return logger
    
    # 如果根日志器尚未配置，配置根日志器
    if not _root_logger_configured:
        # 配置根日志器
        root_logger = logging.getLogger()
        
        # 清除现有的处理器
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        
        # 创建格式化器
        formatter = logging.Formatter(SYSTEM_CONFIG.get("log_format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        console_handler.setFormatter(formatter)
        
        # 添加处理器到根日志器
        root_logger.addHandler(console_handler)
        
        # 标记根日志器已配置
        _root_logger_configured = True
    
    return logger

# 预配置一些常用的日志记录器
model_logger = get_logger("Model")
utils_logger = get_logger("Utils")
api_logger = get_logger("API")
