'''
模型对外提供的API接口
目前主要是推理功能，封装之后再给fastapi调用
'''

import os
import sys

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path) 

import torch
import numpy as np
from PIL import Image
import logging
from typing import Literal
from utils.model_loader import ModelLoader
from config import MODEL_CONFIG, IMAGE_CONFIG, OUTPUT_CONFIG, SYSTEM_CONFIG
from utils.environment import check_dependencies, setup_environment
def setup_logger(name="ModelAPI", level=SYSTEM_CONFIG["log_level"]):
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别，默认使用config.py中的配置
    
    Returns:
        配置好的日志记录器
    """
    
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    logging.basicConfig(
        level=level_map.get(level, logging.INFO),
        format=SYSTEM_CONFIG["log_format"],
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(name)

logger = setup_logger()

class ModelAPI:
    """
    模型API类，提供模型推理功能
    运行时只保存模型信息
    设计思路，所有可以由config配置的地方，全部交给config，尽量不在代码里写死
    成员变量:
        model_path: 模型文件路径，默认用config的，但是因为可能要适配其他模型，所以保留接口
        model_name: 模型名称
        model: 模型实例
        logger: 日志记录器
        device: 设备，默认为cuda，如果cuda不可用，则使用cpu
        loader: 模型加载器，用于加载模型
    """
    def __init__(self, device:Literal['cuda','cpu']='cuda'):
        """
        初始化模型，提前加载状态字典，随时可以转化成模型

        Args:
            model_path: 模型文件路径，默认用config的，但是因为可能要适配其他模型，所以保留接口
            model_name: 模型名称
            device: 设备，默认为cuda，如果cuda不可用，则使用cpu 
        """
        check_dependencies() # 检查依赖
        setup_environment() # 设置环境变量
                
        if device == 'cpu':
            self.device = torch.device('cpu') # 使用cpu
            logger.info("使用CPU进行推理")
        elif device == 'cuda':
            if torch.cuda.is_available(): # 检查cuda是否可用
                self.device = torch.device('cuda') # 使用cuda
                logger.info("使用GPU进行推理")
            else: # 如果cuda不可用，使用cpu
                self.device = torch.device('cpu') # 使用cpu
                logger.info("GPU不可用，使用CPU进行推理")
        self.loader = ModelLoader() # 初始化加载器

    def generate_text_evals(self,output, components)->dict:
        """
        生成文本报告
        这个函数私有化
        Args:
            output_np: 模型输出的numpy数组
            components: 成分名称列表

        Returns:
            返回文本
        """
        # 将输出转换为numpy数组
        output_np = output.cpu().numpy().flatten()
        
        # 如果输出维度大于预期的成分数量，只取前几个值
        expected_components = MODEL_CONFIG["expected_components"]
        if len(output_np) > expected_components:
            logger.warning(f"模型输出维度({len(output_np)})大于预期成分数量({expected_components})，只取前{expected_components}个值")
            output_np = output_np[:expected_components]
    
        # 找出含量最高的成分
        protein = output_np[0] # 蛋白质含量

        oil = output_np[1] # 油含量
        
        # 找出含量最低的成分
        min_index = np.argmin(output_np)
        min_component = components[min_index]
        min_value = output_np[min_index]

        # 打印数值结果
        logger.info("\n=== 油菜籽成分含量预测报告 ===")
        logger.info("\n成分含量预测结果:")
        for i, comp in enumerate(components):
            logger.info(f"{comp}: {output_np[i]:.4f}")
        
        # 添加详细的文字结论
        logger.info("\n预测结论:")
        logger.info(f'\n预测模型: {self.model_name if self.model_name else "未知模型"}')

        return  {
            "protein": float(protein),
            "oil": float(oil),
        }
    
    def eval_image(self, image, model_name:Literal['MPViT', 'ResNet', 'FasterNet', 'EfficientNet', 'Swin', 'VanillaNet'])->dict:
        """
        对单张图像进行预测，返回预测结果的字典

        Args:
            image: 图像文件
            model_name: 模型名称
            model_path: 模型文件路径，默认使用config的，但是因为可能要适配其他模型，所以保留接口

        Returns:
            预测结果的字典
        """

        try:
            self.loader._load_model(model_name) # 加载模型
            self.model_name = model_name # 设置模型名称
            preprocessed_image = self.loader.preprocess_image(image) # 预处理图像
            # 进行预测
            output = self.loader.predict(preprocessed_image) # 进行预测

            # 生成文本报告
            component_names = MODEL_CONFIG["component_names"]

            evals = self.generate_text_evals(output,component_names) # 生成文本报告
            logger.info(f"图像 {image} 预测完成")
        except Exception as e:
            logger.error(f"预测 {image} 失败: {str(e)}")
            return None
        
        return evals
