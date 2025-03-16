import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class ImageProcessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image_path):
        """
        图像预处理
        Args:
            image_path: 图像路径
        Returns:
            处理后的张量
        """
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)
        return image_tensor.unsqueeze(0)