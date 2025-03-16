import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 模型配置
MODEL_CONFIG = {
    'model_path': 'res/vanillanet.pt',  # 模型路径
    'image_size': 224,  # 图像大小
    'num_classes': 5,  # 输出类别数量
    'component_names': ["油酸", "亚油酸", "亚麻酸", "棕榈酸", "硬脂酸"]  # 成分名称
}

# 日志配置
LOG_CONFIG = {
    'log_dir': 'logs',  # 日志目录
    'log_level': 'INFO',  # 日志级别
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # 日志格式
}

# 输出配置
OUTPUT_CONFIG = {
    'result_dir': 'results',  # 结果目录
    'csv_file': 'results/prediction_results.csv',  # CSV文件
    'image_file': 'results/prediction_result.png'  # 图像文件
}