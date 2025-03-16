import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from torchvision import transforms  # 添加缺失的导入
from models.model_loader import ModelLoader

def test_model_inference(model_path, image_path):
    """测试模型推理功能"""
    try:
        # 初始化模型
        model_loader = ModelLoader(model_path)
        device = model_loader.device  # 获取模型的设备
        
        # 使用模型加载器中的标准预处理
        transform = model_loader.get_transform()
        
        # 加载并检查图像
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            raise ValueError(f"图像加载失败: {str(e)}")
            
        # 图像预处理
        try:
            image_tensor = transform(image)
            image_tensor = image_tensor.to(device)  # 将张量移到正确的设备上
        except Exception as e:
            raise ValueError(f"图像预处理失败: {str(e)}")
            
        # 模型推理
        try:
            result = model_loader.predict(image_tensor)
            logging.info(f"成功处理图像 {image_path}")
            return result
        except RuntimeError as e:
            raise RuntimeError(f"模型推理失败: {str(e)}")
        
    except Exception as e:
        logging.error(f"推理过程发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    MODEL_PATH = "/Users/zhangandy/Documents/work/code/party/zhaojie/mendianyunying/pythonVesion/res/vanillanet.pt"
    IMAGE_PATH = "/Users/zhangandy/Documents/work/code/party/zhaojie/mendianyunying/pythonVesion/tests/test_images/image_custom.png"
    
    try:
        result = test_model_inference(MODEL_PATH, IMAGE_PATH)
        print("\n测试结果:")
        print(f"预测类别: {result['predicted_class']}")
        print(f"预测置信度: {result['confidence']:.2%}")
        print(f"原始输出: {result['raw_output']}")
    except Exception as e:
        print(f"测试失败: {str(e)}")