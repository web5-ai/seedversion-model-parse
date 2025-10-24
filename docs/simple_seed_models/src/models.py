#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
种子识别模型定义

包含EfficientNet-B0和ResNet18的模型结构定义
"""

import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNetB0Classifier(nn.Module):
    """EfficientNet-B0 种子分类器"""
    
    def __init__(self, num_classes=2, dropout=0.0):
        super().__init__()
        
        # 加载预训练的EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights=None)
        
        # 替换分类头
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
        
        # 加载预训练的ResNet18
        self.backbone = models.resnet18(weights=None)
        
        # 替换分类头 (保持与训练时一致的结构)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class CustomCNN(nn.Module):
    """自定义CNN 种子分类器"""
    
    def __init__(self, num_classes=2, dropout=0.0):
        super().__init__()
        
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 第三个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 第四个卷积块
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # 自适应平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def create_model(model_name: str, num_classes: int = 2, dropout: float = 0.0):
    """
    创建模型实例
    
    Args:
        model_name: 模型名称 ('efficientnet_b0', 'resnet18', 'custom_cnn')
        num_classes: 类别数量
        dropout: Dropout率
        
    Returns:
        torch.nn.Module: 模型实例
    """
    model_name = model_name.lower()
    
    if model_name == 'efficientnet_b0':
        return EfficientNetB0Classifier(num_classes, dropout)
    elif model_name == 'resnet18':
        return ResNet18Classifier(num_classes, dropout)
    elif model_name == 'custom_cnn':
        return CustomCNN(num_classes, dropout)
    else:
        raise ValueError(f"不支持的模型: {model_name}. 可选: efficientnet_b0, resnet18, custom_cnn")


if __name__ == "__main__":
    # 测试模型创建
    print("测试模型创建...")
    
    models_to_test = ['efficientnet_b0', 'resnet18', 'custom_cnn']
    
    for model_name in models_to_test:
        model = create_model(model_name)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{model_name}: {total_params:,} 参数")
    
    print("模型创建测试完成！")
