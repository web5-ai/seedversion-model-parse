"""
配置文件，存储所有可配置参数
"""

# 模型相关配置
MODEL_CONFIG = {
    "model_path": "weights/",
    "default_model": "FasterNet",
    "expected_components": 2,
    "component_names": ["蛋白质", "油脂"],
    "device": "cuda", # 在setup_environment函数中初始化设置
    # 检测模型配置
    "detect_model": "EfficientNetB0Classifier",
    "detect_model_path": "weights/EfficientNetB0Classifier.pth",
    "detect_conf_threshold": 0.9,
    "detect_iou_threshold": 0.5
    # "quality_thresholds": {
    #     "油酸": 0.5,  # 油酸含量高于0.5为优质
    #     "亚油酸": 0.3  # 亚油酸含量高于0.3为营养价值高
    # }
}

# 图像处理配置
IMAGE_CONFIG = {
    "default_image_path": "tests/test_images/image_custom.jpg",
    "default_images_dir": "tests/images",
    "resize_dimensions": (224, 224),
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225]
}

# 输出配置 这里主要在test_model用到了，就没改动了，搞了个适配接口的存储相关配置
OUTPUT_CONFIG = {
    "default_output_path": "results/prediction_result.png",
    "chart_title": "油菜籽成分含量预测",
    "chart_size": (10, 6)
}

# 系统配置
SYSTEM_CONFIG = {
    "default_seed": 123,
    "log_level": "INFO",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}

# 后台配置
BACKEND_CONFIG = {
    # 日志配置
    "log_dir": r"E:\Proj\seedversion-model-parse\backend\logs",  # 日志目录
    "images_dir": "",  # 图像目录，每次会在runpy重新设置
    "log_format": "%(asctime)s - %(levelname)s - %(message)s",
    "log_date_format": "%Y-%m-%d %H:%M:%S",
    "log_encoding": "utf-8",  # 日志文件编码

    # 服务器配置
    "host": "0.0.0.0",
    "port": 8123,
    "reload": True,  # 禁用自动重载，避免重复初始化

    # 内存监控配置
    "memory_limit": 1024 * 1024 * 10000,  # 10000MB
    "memory_check_interval": 5,  # 内存检查间隔（秒）

    # 邮件配置
    "smtp_server": "smtp.qq.com",
    "smtp_port": 587,
    "smtp_username": "851680026@qq.com",
    "smtp_password": "krrlmqusmxkdbcdg",
    "recipient_email": "851680026@qq.com",

    # 主线程检查间隔
    "main_thread_check_interval": 60  # 主线程检查间隔（秒）
}