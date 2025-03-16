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
        
    def _analyze_model(self):
        """
        分析模型结构并生成详细报告
        """
        model_info = []
        model_info.append("=== 模型详细信息报告 ===\n")
        
        # 基础信息
        model_info.append("1. 基础信息:")
        model_info.append(f"- 模型文件: {self.model_path}")
        model_info.append(f"- 模型类型: {self.model.__class__.__name__}")
        model_info.append(f"- 模型大小: {os.path.getsize(self.model_path) / (1024*1024):.2f} MB")
        
        # 结构信息
        model_info.append("\n2. 结构信息:")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        model_info.append(f"- 总参数量: {total_params:,}")
        model_info.append(f"- 可训练参数量: {trainable_params:,}")
        
        # 层信息
        model_info.append("\n3. 主要层信息:")
        for name, module in self.model.named_children():
            params = sum(p.numel() for p in module.parameters())
            model_info.append(f"- {name}: {module.__class__.__name__}")
            model_info.append(f"  参数量: {params:,}")
        
        # 输入输出信息
        model_info.append("\n4. 输入输出信息:")
        model_info.append("- 输入尺寸: (224, 224)")
        model_info.append("- 输入通道: 3 (RGB)")
        if hasattr(self.model, 'fc'):
            if isinstance(self.model.fc, nn.Sequential):
                model_info.append(f"- 输出维度: {self.model.fc[0].out_features}")
            else:
                model_info.append(f"- 输出维度: {self.model.fc.out_features}")
        
        # 预处理信息
        model_info.append("\n5. 预处理信息:")
        model_info.append("- 图像缩放: 224x224")
        model_info.append("- 归一化参数:")
        model_info.append("  均值: [0.485, 0.456, 0.406]")
        model_info.append("  标准差: [0.229, 0.224, 0.225]")
        
        # 保存信息到文件
        model_dir = os.path.dirname(self.model_path)
        info_path = os.path.join(model_dir, "model_info.txt")
        try:
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(model_info))
            self.logger.info(f"模型详细信息已保存至: {info_path}")
        except Exception as e:
            self.logger.error(f"保存模型信息失败: {str(e)}")
        
        return '\n'.join(model_info)
    
    def _load_model(self):
        """
        加载预训练模型
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        self.logger.info(f"正在加载模型: {self.model_path}")
        
        try:
            # 首先尝试直接加载PyTorch模型
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
            
            # 设置为评估模式
            self.model.eval()
            self.logger.info("模型加载完成，已设置为评估模式")
            
            # 分析并保存模型信息
            model_info = self._analyze_model()
            self.logger.info("\n" + model_info)
            
        except Exception as e:
            self.logger.error(f"加载模型失败: {str(e)}")
            raise
    
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