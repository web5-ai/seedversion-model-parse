"""
配置文件，存储所有可配置参数
"""

# 模型相关配置
MODEL_CONFIG = {
    "model_path": "weights/fasternet_model.pt",
    "expected_components": 5,
    "component_names": ["油酸", "亚油酸", "亚麻酸", "棕榈酸", "硬脂酸"],
    "quality_thresholds": {
        "油酸": 0.5,  # 油酸含量高于0.5为优质
        "亚油酸": 0.3  # 亚油酸含量高于0.3为营养价值高
    }
}

# 图像处理配置
IMAGE_CONFIG = {
    "default_image_path": "tests/test_images/image_custom.png",
    "resize_dimensions": (224, 224),
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225]
}

# 输出配置
OUTPUT_CONFIG = {
    "default_output_path": "results/prediction_result.png",
    "chart_title": "油菜籽成分含量预测",
    "chart_size": (10, 6)
}

# 系统配置
SYSTEM_CONFIG = {
    "default_seed": 42,
    "log_level": "INFO",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}