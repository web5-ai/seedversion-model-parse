import torch
import torch.nn as nn
from torchvision import models

class RipenessClassifier:
    """
    油菜籽成熟度分类器
    分类类别：绿熟、黄熟、完熟
    """
    def __init__(self, num_classes=3, device='cpu', model_path='fruit_ripeness_model.pth'):
        """
        初始化成熟度分类器
        
        Args:
            num_classes: 分类类别数量（默认为3：绿熟、黄熟、完熟）
            device: 运行设备（'cpu' 或 'cuda'）
            model_path: 模型权重文件路径
        """
        self.num_classes = num_classes
        self.device = device
        self.model_path = model_path
        self.classes = ["绿熟", "黄熟", "完熟"]
        
        # 创建 ResNet18 模型
        self.model = models.resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        
        # 修改全连接层以适应3分类任务
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # 加载模型权重
        self.load_model_weight(model_path)
        
        # 移动到指定设备
        self.model.to(device)
        
        # 设置为评估模式
        self.model.eval()
    
    def load_model_weight(self, model_path):
        """
        加载模型权重
        
        Args:
            model_path: 模型权重文件路径
        """
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"成功加载成熟度分类器模型: {model_path}")
        except Exception as e:
            print(f"加载成熟度分类器模型失败: {str(e)}")
            raise
    
    def predict(self, image_tensor):
        """
        预测图像的成熟度
        
        Args:
            image_tensor: 预处理后的图像张量
        
        Returns:
            dict: 预测结果，包含预测类别、置信度和各类别概率
        """
        with torch.no_grad():
            # 前向传播
            outputs = self.model(image_tensor)
            
            # 获取预测类别
            _, predicted = torch.max(outputs, 1)
            predicted_idx = predicted.item()
            
            # 计算各类别的概率
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence = torch.max(probabilities, dim=1)[0].item()
            
            # 构建结果字典
            result = {
                'predicted_class': self.classes[predicted_idx],
                'predicted_index': predicted_idx,
                'confidence': confidence,
                'probabilities': {
                    self.classes[i]: probabilities[0][i].item() 
                    for i in range(self.num_classes)
                }
            }
            
            return result
    
    def forward(self, x):
        """
        前向传播方法
        
        Args:
            x: 输入张量
        
        Returns:
            模型输出
        """
        return self.model(x)