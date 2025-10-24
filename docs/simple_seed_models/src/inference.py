#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
种子识别推理模块

简洁的推理接口，只需要模型结构和权重文件
"""

import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import time
from torchvision import transforms

from models import create_model


class SeedInference:
    """
    种子识别推理器
    
    简洁版本，只需要模型名称和权重文件路径
    """
    
    def __init__(self, model_name: str, weights_path: str, device: str = 'auto'):
        """
        初始化推理器
        
        Args:
            model_name: 模型名称 ('efficientnet_b0', 'resnet18', 'custom_cnn')
            weights_path: 权重文件路径 (.pth文件)
            device: 设备 ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.class_names = ['background', 'rapeseed']
        
        # 设备选择
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 创建模型
        self.model = create_model(model_name, num_classes=2)
        
        # 加载权重
        self._load_weights(weights_path)
        
        # 设置为评估模式
        self.model.to(self.device)
        self.model.eval()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ 模型加载成功: {model_name} on {self.device}")
    
    def _load_weights(self, weights_path: str):
        """加载模型权重"""
        weights_path = Path(weights_path)
        
        if not weights_path.exists():
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")
        
        try:
            # 尝试加载权重
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
        except Exception as e:
            # 如果失败，尝试从checkpoint中提取
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
    
    def predict(self, image_path: str):
        """
        预测单张图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            dict: 预测结果
        """
        # 加载和预处理图片
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 推理
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        inference_time = time.time() - start_time
        
        return {
            'predicted_class': self.class_names[predicted_class],
            'confidence': confidence,
            'probabilities': {
                'background': probabilities[0][0].item(),
                'rapeseed': probabilities[0][1].item()
            },
            'inference_time_ms': inference_time * 1000
        }
    
    def predict_batch(self, image_paths: list, batch_size: int = 8):
        """
        批量预测
        
        Args:
            image_paths: 图片路径列表
            batch_size: 批次大小
            
        Returns:
            list: 预测结果列表
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []
            
            # 预处理批次
            for path in batch_paths:
                image = Image.open(path).convert('RGB')
                tensor = self.transform(image)
                batch_tensors.append(tensor)
            
            # 批次推理
            batch_input = torch.stack(batch_tensors).to(self.device)
            
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(batch_input)
                probabilities = F.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1)
            
            batch_time = time.time() - start_time
            
            # 处理结果
            for j, path in enumerate(batch_paths):
                pred_class = predicted_classes[j].item()
                confidence = probabilities[j][pred_class].item()
                
                results.append({
                    'image_path': path,
                    'predicted_class': self.class_names[pred_class],
                    'confidence': confidence,
                    'probabilities': {
                        'background': probabilities[j][0].item(),
                        'rapeseed': probabilities[j][1].item()
                    },
                    'inference_time_ms': batch_time * 1000 / len(batch_paths)
                })
        
        return results
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),
            'device': str(self.device),
            'class_names': self.class_names
        }


# 便捷函数
def load_model(model_name: str, weights_path: str, device: str = 'auto'):
    """
    便捷函数：快速加载模型
    
    Args:
        model_name: 模型名称 ('efficientnet_b0', 'resnet18', 'custom_cnn')
        weights_path: 权重文件路径
        device: 设备选择
        
    Returns:
        SeedInference: 推理器实例
    """
    return SeedInference(model_name, weights_path, device)


if __name__ == "__main__":
    # 测试推理器
    print("测试推理器...")
    
    # 这里需要实际的权重文件路径
    # inference = load_model('efficientnet_b0', 'path/to/weights.pth')
    # result = inference.predict('path/to/image.jpg')
    # print(result)
    
    print("推理器模块加载完成！")
