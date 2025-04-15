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
import psutil
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

        self.loader = ModelLoader(device=MODEL_CONFIG['device']) # 初始化加载器

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
    
        oil = output_np[0] # 蛋白质含量

        protein = output_np[1] # 油含量


        # 打印数值结果
        logger.info("=== 油菜籽成分含量预测报告 ===")
        logger.info(f'预测模型: {self.model_name if self.model_name else "未知模型"}')
        logger.info("成分含量预测结果:")
        for i, comp in enumerate(components):
            logger.info(f"{comp}: {output_np[i]:.4f}")
        logger.info("===       报告结束       ===")
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
            # 检查是否有可用的CUDA设备
            if MODEL_CONFIG['device'] == 'cuda':
                # 获取初始显存使用情况
                initial_memory = torch.cuda.memory_allocated()
            else:
                logger.warning("未使用CUDA，无法监控显存消耗。")
                initial_memory = 0

            self.loader.load_model(model_name) # 加载模型
            self.model_name = model_name # 设置模型名称
            size = 256 if model_name == 'Swin' else 224 # 设置图像大小，Swin需要256，其他模型需要224
            preprocessed_image = self.loader.preprocess_image(image, size) # 预处理图像
            # 进行预测
            seed = self.loader.set_seed(50)
            output = self.loader.predict(preprocessed_image) # 进行预测
            self.loader.unload_model() # 卸载模型
            # 生成文本报告
            component_names = MODEL_CONFIG["component_names"]
            evals = self.generate_text_evals(output,component_names) # 生成文本报告

            if torch.cuda.is_available():
                # 获取最终显存使用情况
                final_memory = torch.cuda.memory_allocated()
                # 计算显存消耗
                memory_consumed = final_memory - initial_memory
                # 转化为MB
                memory_consumed = memory_consumed / (1024 * 1024)
                logger.info(f"图像预测过程中显存消耗: {memory_consumed} MB")
            else:
                memory_consumed = 0

            evals['memory_cost'] = memory_consumed
            evals['seed'] = seed
        except Exception as e:
            logger.error(f"图像预测失败: {str(e)}")
            raise e
        
        return evals
