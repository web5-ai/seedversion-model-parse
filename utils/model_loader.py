import os
import logging
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import datetime
import random
import numpy as np
from PIL import Image
from typing import Literal
from models import MPViT, ResNet, FasterNet, EfficientNet, Swin, VanillaNet, MODEL_OPTIONS
from config import MODEL_CONFIG
import traceback

# 设置日志
logger = logging.getLogger("ModelLoader")
logger.propagate = False
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class ModelLoader:
    """
    模型加载器，用于加载预训练模型
    """
    def __init__(self, model_path=None, debug=False, device='cpu'):
        """
        初始化模型加载器

        Args:
            model_path: 模型路径，由调用函数提供，一般为config的默认值
            debug: 是否开启调试模式
        """
        self.model_path = model_path
        self.debug = debug
        self.model = None
        self.device = device

        
    def _analyze_state_dict(self):
        """
        分析模型状态字典的结构
        将状态字典结构进行分析，不保留张量数据，只保留张量形状
        """
        state_dict = self.state_dict
        state_dict_info = {}
        for key, value in state_dict.items():
            state_dict_info[key] = value.shape
        # 提取模型名
        model_name = os.path.basename(self.model_path)
        # 保存到文件中，文件名为model_name_state_dict_info.txt
        model_dir = os.path.dirname(self.model_path)
        info_path = os.path.join(model_dir, f"{model_name.split('.')[0]}_state_dict_info.txt")
        with open(info_path, 'w') as f:
            for key, shape in state_dict_info.items():
                f.write(f"{key}: {shape}\n")
        return state_dict_info
    
    def _analyze_model(self):
        """
        分析模型结构并生成详细报告
        """
        model_info = []
        model_info.append("=== 模型详细信息报告 ===\n")
        model_info.append(f"加载时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
        
        # 模型结构与状态字典差异信息
        model_info.append("\n4. 模型结构与状态字典差异信息:")
        # model_info.append(f"- 状态字典相对模型结构缺少的参数量: {len(self.missing_keys)}")
        # model_info.append(f"- 状态字典相对模型结构多出的参数量: {len(self.unexpected_keys)}")
        # 输入输出信息
        model_info.append("\n5. 输入输出信息:")
        model_info.append("- 输入尺寸: (224, 224)")
        model_info.append("- 输入通道: 3 (RGB)")
        if hasattr(self.model, 'fc'):
            if isinstance(self.model.fc, nn.Sequential):
                model_info.append(f"- 输出维度: {self.model.fc[0].out_features}")
            else:
                model_info.append(f"- 输出维度: {self.model.fc.out_features}")
        
        # 预处理信息
        model_info.append("\n6. 预处理信息:")
        model_info.append("- 图像缩放: 224x224")
        model_info.append("- 归一化参数:")
        model_info.append("  均值: [0.485, 0.456, 0.406]")
        model_info.append("  标准差: [0.229, 0.224, 0.225]")

        # 保存信息到文件，不同的模型会进行标识
        model_dir = os.path.dirname(self.model_path)
        # info_path = os.path.join(model_dir, "model_info.txt")
        modelfile_name = os.path.splitext(os.path.basename(self.model_path))[0]
        info_path = os.path.join(model_dir, f"{modelfile_name}_{self.model_name}_info.txt")
        try:
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(model_info))
            logger.info(f"模型详细信息已保存至: {info_path}")
        except Exception as e:
            logger.error(f"保存模型信息失败: {str(e)}")
        
        return '\n'.join(model_info)
    
    def set_seed(self,seed=None):
        """
        设置随机种子以确保结果可重复
        
        Args:
            seed: 随机种子在外面调用时需要传入config中的seed，默认为None
        """
        if seed is None:
            seed = int(datetime.now().timestamp())#如果为none就用时间戳代替
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True # 固定卷积算法以提高性能
        torch.backends.cudnn.benchmark = False # 关闭动态卷积算法
        logger.info(f"已设置随机种子: {seed}")

    def load_model(self, model_name:MODEL_OPTIONS):
        """
        加载预训练模型

        Args:
            model: 用来选择加载状态字典的模型结构，默认为ResNet
        """
        
        try:

            self.model_name = model_name
            # if 生成类
            # if model_name == "MPViT":
            #     model_class = MPViT
            # elif model_name == "ResNet":
            #     model_class = ResNet
            # elif model_name == "FasterNet":
            #     model_class = FasterNet
            # elif model_name == "EfficientNet":
            #     model_class = EfficientNet
            # elif model_name == "Swin":
            #     model_class = Swin
            # elif model_name == "VanillaNet":
            #     model_class = VanillaNet
            # else:
            #     raise ValueError(f"不支持的模型: {model_name}")
            # # 加载模型
            # self.model = model_class()  # 创建模型实例
            try:
                model_class = globals()[model_name]  # 尝试从全局变量中获取模型类
                self.model = model_class(device = self.device)  # 创建模型实例
                self.model.to(self.device)
                param = next(self.model.parameters())
                logger.info(f'模型加载到{param.device}设备上')
            except KeyError:
                raise ValueError(f"不支持的模型: {model_name}")
            model_path = os.path.join(MODEL_CONFIG['model_path'], f'{model_name}.pt')
            if self.debug: # 如果是debug模式才存储状态字典
                self.state_dict = self.model.load_model_weight(model_path) # 这里是模型结构的状态字典，不是加载的状态字典
            else:
                self.model.load_model_weight(model_path)
            # # 状态字典加载用模型封装的加载方法
            # self.state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            # # 加载和model_name同名的模型
            # self.model.load_state_dict(self.state_dict, strict=True)  # 加载状态字典，允许部分参数不匹配

            logger.info(f"成功加载{model_name}模型")
        except Exception as e:
            logger.warning(f"加载{model_name}模型失败: {str(e)}")
            tb = traceback.format_exc()  # 获取完整的异常信息
            logger.warning(f"加载模型错误信息: {tb}")
            # raise ValueError("无法加载模型，请检查模型文件格式或提供模型架构信息")
            
        # 设置为评估模式
        self.model.eval()
        logger.info("模型加载完成，已设置为评估模式")
    
    def unload_model(self):
        """
        卸载模型
        """
        if self.model is not None:
            del self.model  # 删除模型实例
            self.model = None  # 将模型设置为None
            self.model_name = None  # 将模型名称设置为None
            logger.info("模型已卸载")

    def preprocess_image(self, image, size=224):
        """
        预处理图像
        
        Args:
            image: PIL图像对象
        
        Returns:
            预处理后的图像张量
        """
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        image_tensor = transform(image).unsqueeze(0)

        image_tensor = image_tensor.to(self.device)
        logger.info(f"图像张量处于{image_tensor.device}设备上")

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