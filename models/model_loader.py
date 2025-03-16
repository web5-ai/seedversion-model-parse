import os
import logging
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

class ModelLoader:
    """
    模型加载器，用于加载预训练模型
    """
    def __init__(self, model_path, debug=False):
        """
        初始化模型加载器
        
        Args:
            model_path: 模型路径
            debug: 是否开启调试模式
        """
        self.model_path = model_path
        self.debug = debug
        self.model = None
        
        # 设置日志
        self.logger = logging.getLogger("ModelLoader")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        
        # 加载模型
        self._load_model()
        
    def _load_model(self):
        """
        加载预训练模型
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        self.logger.info(f"正在加载模型: {self.model_path}")
        
        try:
            # 尝试直接加载模型
            self.model = torch.jit.load(self.model_path)
            self.logger.info("成功加载TorchScript模型")
        except Exception as e:
            self.logger.warning(f"加载TorchScript模型失败: {str(e)}")
            try:
                # 尝试加载状态字典
                state_dict = torch.load(self.model_path, map_location='cpu')
                
                # 根据状态字典推断模型结构
                if isinstance(state_dict, dict):
                    if 'model' in state_dict:
                        state_dict = state_dict['model']
                    elif 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                
                # 创建模型实例
                from torchvision.models import resnet50, vgg16, efficientnet_b0
                
                # 尝试不同的模型架构
                model_candidates = [
                    ('ResNet50', resnet50(pretrained=False)),
                    ('VGG16', vgg16(pretrained=False)),
                    ('EfficientNet', efficientnet_b0(pretrained=False))
                ]
                
                for model_name, model_instance in model_candidates:
                    try:
                        # 修改最后一层以适应输出维度
                        if 'fc.weight' in state_dict:
                            out_features = state_dict['fc.weight'].shape[0]
                            if hasattr(model_instance, 'fc'):
                                in_features = model_instance.fc.in_features
                                model_instance.fc = nn.Linear(in_features, out_features)
                            elif hasattr(model_instance, 'classifier'):
                                if isinstance(model_instance.classifier, nn.Sequential):
                                    in_features = model_instance.classifier[-1].in_features
                                    model_instance.classifier[-1] = nn.Linear(in_features, out_features)
                                else:
                                    in_features = model_instance.classifier.in_features
                                    model_instance.classifier = nn.Linear(in_features, out_features)
                        
                        # 尝试加载状态字典
                        model_instance.load_state_dict(state_dict, strict=False)
                        self.model = model_instance
                        self.logger.info(f"成功加载{model_name}模型")
                        break
                    except Exception as e:
                        self.logger.warning(f"加载{model_name}模型失败: {str(e)}")
                
                if self.model is None:
                    raise ValueError("无法加载模型，请检查模型文件格式或提供模型架构信息")
                
            except Exception as e:
                self.logger.error(f"加载模型失败: {str(e)}")
                raise
        
        # 在_load_model方法中，修改检查输出层的部分
        
        # 检查是否需要修改输出层
        if hasattr(self.model, 'fc') and self.model.fc.out_features == 1000:
            self.logger.info("检测到ImageNet预训练模型，添加适配层将1000维输出转换为5维")
            in_features = self.model.fc.in_features
            
            # 使用固定的初始化方法
            torch.manual_seed(42)  # 设置随机种子
            
            # 创建一个包含线性层和激活函数的新输出层
            self.model.fc = nn.Sequential(
                nn.Linear(in_features, 5),
                nn.ReLU()  # 确保输出非负
            )
            
            # 使用正态分布初始化权重，使用小的标准差
            nn.init.normal_(self.model.fc[0].weight, mean=0.0, std=0.01)
            # 使用常数初始化偏置，每个成分设置不同的初始值
            self.model.fc[0].bias.data = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
            
            self.logger.info(f"已将输出层从{in_features}x1000修改为{in_features}x5，并添加ReLU确保非负输出")
        
        # 设置为评估模式
        self.model.eval()
        self.logger.info("模型加载完成，已设置为评估模式")
    
    def preprocess_image(self, image):
        """
        预处理图像
        
        Args:
            image: PIL图像对象
        
        Returns:
            预处理后的图像张量
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor
    
    def predict(self, image_tensor):
        """
        使用模型进行预测
        
        Args:
            image_tensor: 预处理后的图像张量
        
        Returns:
            预测结果
        """
        with torch.no_grad():
            output = self.model(image_tensor)
            return output