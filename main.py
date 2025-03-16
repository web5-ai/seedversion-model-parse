import logging
import os
from models.model_loader import ModelLoader
from utils.image_processor import ImageProcessor
from config import MODEL_CONFIG, LOG_CONFIG

def setup_logging():
    """
    设置日志配置
    """
    if not os.path.exists(LOG_CONFIG['log_dir']):
        os.makedirs(LOG_CONFIG['log_dir'])
    
    logging.basicConfig(
        level=getattr(logging, LOG_CONFIG['level']),
        format=LOG_CONFIG['format'],
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(LOG_CONFIG['log_dir'], 'app.log'))
        ]
    )

def init_model(model_path):
    """
    初始化模型
    """
    try:
        model_loader = ModelLoader(model_path)
        return model_loader
    except Exception as e:
        logging.error(f"模型初始化失败: {str(e)}")
        return None

def main():
    # 初始化日志
    setup_logging()
    
    # 初始化模型
    model_loader = init_model(MODEL_CONFIG['model_path'])
    if model_loader is None:
        return
    
    # 初始化图像处理器
    image_processor = ImageProcessor()
    
    logging.info("系统初始化完成，ready for inference")

if __name__ == "__main__":
    main()