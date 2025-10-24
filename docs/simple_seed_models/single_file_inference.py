#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
种子识别 - 单文件版本

包含模型定义和推理功能的完整单文件解决方案
只需要这个文件 + 权重文件即可使用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from pathlib import Path
import time
from torchvision import transforms


# ==================== 模型定义 ====================

class EfficientNetB0Classifier(nn.Module):
    """EfficientNet-B0 种子分类器"""
    
    def __init__(self, num_classes=2, dropout=0.0):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class ResNet18Classifier(nn.Module):
    """ResNet18 种子分类器"""
    
    def __init__(self, num_classes=2, dropout=0.0):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


# ==================== 推理类 ====================

class SeedClassifier:
    """种子识别分类器 - 单文件版本"""
    
    def __init__(self, model_name: str, weights_path: str, device: str = 'auto'):
        """
        初始化分类器
        
        Args:
            model_name: 模型名称 ('efficientnet_b0' 或 'resnet18')
            weights_path: 权重文件路径
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
        if model_name == 'efficientnet_b0':
            self.model = EfficientNetB0Classifier(num_classes=2)
        elif model_name == 'resnet18':
            self.model = ResNet18Classifier(num_classes=2)
        else:
            raise ValueError(f"不支持的模型: {model_name}. 可选: efficientnet_b0, resnet18")
        
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
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
        except Exception as e:
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
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
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
        """批量预测"""
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []
            
            for path in batch_paths:
                image = Image.open(path).convert('RGB')
                tensor = self.transform(image)
                batch_tensors.append(tensor)
            
            batch_input = torch.stack(batch_tensors).to(self.device)
            
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(batch_input)
                probabilities = F.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1)
            
            batch_time = time.time() - start_time
            
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


# ==================== 使用示例 ====================

def main():
    """使用示例"""
    
    print("🌱 种子识别 - 单文件版本")
    print("=" * 40)
    
    # 使用EfficientNet-B0 (推荐)
    classifier = SeedClassifier(
        model_name="efficientnet_b0",
        weights_path="models/efficientnet_b0_weights.pth"
    )
    
    # 测试图片
    test_image = "../example/test.jpg"
    if Path(test_image).exists():
        result = classifier.predict(test_image)
        
        print(f"预测结果: {result['predicted_class']}")
        print(f"置信度: {result['confidence']:.3f}")
        print(f"推理时间: {result['inference_time_ms']:.1f}ms")
        print(f"详细概率:")
        for class_name, prob in result['probabilities'].items():
            print(f"  {class_name}: {prob:.3f}")
    else:
        print(f"⚠️ 测试图片不存在: {test_image}")


if __name__ == "__main__":
    main()
