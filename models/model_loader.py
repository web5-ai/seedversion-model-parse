import torch
import torch.nn as nn
import logging
from collections import OrderedDict

class StageBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        self.act = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=7, padding=3),  # 修正：输出通道数改为 out_channels
            nn.BatchNorm2d(out_channels)  # BatchNorm 通道数与输出通道数相同
        )

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.act(x)
        return x + identity  # 添加残差连接

class FasterNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.stem1 = nn.Sequential(
            nn.Conv2d(3, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512)
        )
        
        self.stem2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=7, stride=1, padding=3)  # 修正：输出通道数改为 512
        )
        self.stem2[2].bn = nn.BatchNorm2d(512)  # BatchNorm 通道数改为 512

        # 修正通道数配置
        channels = [(512, 512), (512, 512), (512, 1024), (1024, 1024),
                   (1024, 2048), (2048, 2048), (2048, 2048), (2048, 2048),
                   (2048, 2048), (2048, 2048), (2048, 4096)]
        
        self.stages = nn.ModuleList([
            StageBlock(in_ch, out_ch) for in_ch, out_ch in channels
        ])

        # 修改分类器结构
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls1 = nn.Sequential(
            nn.Conv2d(4096, num_classes, kernel_size=1)
        )
        self.cls2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.stem1(x)
        x = self.stem2(x)
        
        for stage in self.stages:
            x = stage(x)
        
        x = self.avgpool(x)
        x = self.cls1(x)
        x = self.cls2(x)
        return x.flatten(1)  # 最后展平输出

class ModelLoader:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            
            if isinstance(state_dict, OrderedDict):
                # 先分析模型结构
                model_config = self._analyze_model_config(state_dict)
                logging.info(f"分析得到的模型配置: {model_config}")
                
                # 根据分析结果创建模型
                self.model = self.create_model(**model_config)
                
                # 加载权重
                try:
                    self.model.load_state_dict(state_dict)
                except:
                    # 如果直接加载失败，尝试调整权重键名
                    adjusted_state_dict = self._adjust_state_dict(state_dict)
                    self.model.load_state_dict(adjusted_state_dict)
                
                self.model.eval()
                logging.info("模型加载成功")
            else:
                self.model = state_dict
                self.model.eval()
                
        except Exception as e:
            logging.error(f"模型加载失败: {str(e)}")
            raise e

    def _analyze_model_config(self, state_dict):
        """分析模型配置"""
        config = {
            'num_classes': 2  # 默认值
        }
        
        # 分析并打印关键层的形状
        key_layers = {}
        for key, value in state_dict.items():
            if 'stem' in key or 'stages' in key or 'cls' in key:
                key_layers[key] = value.shape
                logging.info(f"关键层 - {key}: {value.shape}")
        
        # 分析输出类别数
        for key, value in state_dict.items():
            if 'cls2.weight' in key:
                config['num_classes'] = value.size(0)
                break
        
        # 分析网络结构特点
        stem2_shape = None
        for key in key_layers:
            if 'stem2.2' in key:
                stem2_shape = key_layers[key]
                break
        
        if stem2_shape:
            logging.info(f"stem2 最后一层形状: {stem2_shape}")
        
        logging.info(f"模型结构分析完成: {config}")
        return config

    def create_model(self, **kwargs):
        """根据配置创建模型"""
        return FasterNet(**kwargs)

    # 添加标准化预处理方法
    def get_transform(self, image_size=224):
        """获取标准图像预处理流程"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict(self, input_tensor):
        """改进预测方法，添加输入检查"""
        if self.model is None:
            raise ValueError("模型未正确加载")
            
        # 确保输入是正确的维度
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        if input_tensor.shape[1] != 3:
            raise ValueError(f"输入图像必须是3通道，当前是{input_tensor.shape[1]}通道")
            
        self.model.eval()  # 确保模型在评估模式
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            
        return {
            'raw_output': output.cpu().numpy(),
            'probabilities': probabilities.cpu().numpy(),
            'predicted_class': output.argmax(dim=1).item(),
            'confidence': probabilities.max().item()
        }

    def _adjust_state_dict(self, state_dict):
        """调整权重键名以匹配模型结构"""
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # 移除 'model.' 前缀
            if k.startswith('model.'):
                name = k[6:]  # 去掉 'model.'
            else:
                name = k
        
            # 处理 stem2 的特殊命名
            if 'stem2.2.bn.' in name:
                name = name.replace('stem2.2.bn.', 'stem2.2.bn.')
            elif 'stem2.' in name:
                idx = int(name.split('.')[1])
                if idx < 3:  # 前三层保持原样
                    name = name
        
            # 处理 act 层的特殊命名
            if '.act.weight' in name:
                name = name.replace('.act.weight', '.act.0.weight')
            elif '.act.bn.' in name:
                name = name.replace('.act.bn.', '.act.1.')
        
            # 处理分类器层的命名
            if 'cls1.2.' in name:
                name = name.replace('cls1.2.', 'cls1.1.')
            elif 'cls2.0.' in name:
                name = name.replace('cls2.0.', 'cls2.')
        
            new_state_dict[name] = v
        return new_state_dict