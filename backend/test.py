'''
用于测试模型
'''

from model_api import ModelAPI
from utils.model_loader import ModelLoader
from PIL import Image

def get_detection_counts(results):
    """
    获取检测结果中各类别的数量统计
    
    Args:
        results: ultralytics.engine.results.Results对象
        
    Returns:
        dict: 各类别及其数量的统计字典
    """
    # 初始化类别计数字典
    counts = {}
    
    # 确保results是Results对象
    if hasattr(results, 'boxes') and results.boxes is not None:
        # 获取类别ID和置信度
        boxes = results.boxes
        class_ids = boxes.cls.cpu().numpy()  # 将类别ID移至CPU并转换为numpy数组
        
        # 获取类别名称映射
        class_names = results.names if hasattr(results, 'names') else {}
        
        # 统计各类别数量
        for class_id in class_ids:
            # 获取类别名称，如果没有则使用ID
            class_name = class_names.get(int(class_id), f'class_{int(class_id)}')
            # 更新计数
            counts[class_name] = counts.get(class_name, 0) + 1
    
    return counts


def test_model(image_path):
    # 加载图像
    image = Image.open(image_path)
    api = ModelAPI()
    results = api.detect_and_eval(image)
    print(results)

def main():
    # 测试图像路径
    image_path = "example/flaxseed1.jpg"
    test_model(image_path)

if __name__ == "__main__":
    main()
