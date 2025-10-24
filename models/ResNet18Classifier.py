"""
ResNet18 种子分类器

基于ResNet18的种子识别模型，用于二分类任务（背景/种子）
适配项目的模型加载和推理框架
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pathlib import Path


class ResNet18Classifier(nn.Module):
    """
    ResNet18 种子分类器
    
    用于种子识别的二分类模型，输出背景/种子的概率
    """
    
    def __init__(self, num_classes=2, dropout=0.0, device='cpu'):
        """
        初始化模型
        
        Args:
            num_classes: 分类数量，默认2（背景/种子）
            dropout: Dropout率
            device: 运行设备
        """
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.class_names = ['background', 'rapeseed']
        
        # 加载预训练的ResNet18
        self.backbone = models.resnet18(weights=None)
        
        # 替换分类头（保持与训练时一致的结构）
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        """前向传播"""
        return self.backbone(x)
    
    def predict_proba(self, x):
        """
        预测概率
        
        Args:
            x: 输入张量
            
        Returns:
            torch.Tensor: 类别概率
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            return probabilities
    
    def predict(self, x):
        """
        预测类别
        
        Args:
            x: 输入张量
            
        Returns:
            dict: 预测结果
        """
        probabilities = self.predict_proba(x)
        predicted_class = torch.argmax(probabilities, dim=1)
        confidence = torch.max(probabilities, dim=1)[0]
        
        results = []
        for i in range(x.size(0)):
            pred_idx = predicted_class[i].item()
            conf = confidence[i].item()
            probs = probabilities[i]
            
            results.append({
                'predicted_class': self.class_names[pred_idx],
                'predicted_index': pred_idx,
                'confidence': conf,
                'probabilities': {
                    'background': probs[0].item(),
                    'rapeseed': probs[1].item()
                }
            })
        
        return results[0] if len(results) == 1 else results
    
    def load_model_weight(self, weight_path):
        """
        加载模型权重
        
        Args:
            weight_path: 权重文件路径
            
        Returns:
            dict: 模型状态字典
        """
        weight_path = Path(weight_path)
        
        if not weight_path.exists():
            raise FileNotFoundError(f"权重文件不存在: {weight_path}")
        
        try:
            # 尝试直接加载状态字典
            state_dict = torch.load(weight_path, map_location=torch.device(self.device), weights_only=True)
            self.load_state_dict(state_dict, strict=True)
        except Exception as e:
            try:
                # 尝试从checkpoint中提取
                checkpoint = torch.load(weight_path, map_location=torch.device(self.device), weights_only=False)
                if 'model_state_dict' in checkpoint:
                    self.load_state_dict(checkpoint['model_state_dict'], strict=True)
                elif 'state_dict' in checkpoint:
                    self.load_state_dict(checkpoint['state_dict'], strict=True)
                else:
                    self.load_state_dict(checkpoint, strict=True)
            except Exception as e2:
                raise RuntimeError(f"无法加载权重文件 {weight_path}: {e2}")
        
        return self.state_dict()
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'ResNet18Classifier',
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),
            'device': str(self.device)
        }


if __name__ == '__main__':
    # 测试模型创建
    print("测试 ResNet18Classifier 模型...")
    
    model = ResNet18Classifier(num_classes=2, device='cpu')
    
    # 打印模型信息
    info = model.get_model_info()
    print(f"模型名称: {info['model_name']}")
    print(f"参数数量: {info['total_parameters']:,}")
    print(f"模型大小: {info['model_size_mb']:.1f}MB")
    print(f"类别名称: {info['class_names']}")
    
    # 测试前向传播
    test_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(test_input)
        print(f"输出形状: {output.shape}")
        
        # 测试预测
        result = model.predict(test_input)
        print(f"预测结果: {result}")
    
    print("ResNet18Classifier 测试完成！")
