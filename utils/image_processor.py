import os
import torch
import torchvision.transforms as transforms
from PIL import Image

class ImageProcessor:
    """
    图像处理器，用于预处理图像
    """
    def __init__(self, image_size=224):
        """
        初始化图像处理器
        
        Args:
            image_size: 图像大小
        """
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess_image(self, image_path):
        """
        预处理图像
        
        Args:
            image_path: 图像路径或PIL图像对象
        
        Returns:
            预处理后的图像张量
        """
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor
    
    def batch_preprocess(self, image_paths):
        """
        批量预处理图像
        
        Args:
            image_paths: 图像路径列表
        
        Returns:
            预处理后的图像张量列表
        """
        tensors = []
        for path in image_paths:
            tensor = self.preprocess_image(path)
            tensors.append(tensor)
        
        return tensors