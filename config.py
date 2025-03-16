import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 模型配置
MODEL_CONFIG = {
    'model_path': os.path.join(BASE_DIR, 'res', 'vanillanet.pt'),
    'input_size': (224, 224),
    'batch_size': 1,
}

# 图像处理配置
IMAGE_CONFIG = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'resize': (224, 224)
}

# 日志配置
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_dir': os.path.join(BASE_DIR, 'logs')
}