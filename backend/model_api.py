'''
模型对外提供的API接口
目前主要是推理功能，封装之后再给fastapi调用
'''

import os
import sys

# # 添加当前目录到Python路径，确保可以导入本地模块
# current_dir = os.path.dirname(os.path.abspath(__file__))
# if current_dir not in sys.path:
#     sys.path.append(current_dir)
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将根目录添加到系统路径
sys.path.append(root_path) 
import torch
import numpy as np
from PIL import Image
import logging
from typing import Literal
from models.model_loader import ModelLoader
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
    def __init__(self, model_path=MODEL_CONFIG['model_path'], device:Literal['cuda','cpu']='cuda'):
        """
        初始化模型，提前加载状态字典，随时可以转化成模型

        Args:
            model_path: 模型文件路径，默认用config的，但是因为可能要适配其他模型，所以保留接口
            model_name: 模型名称
            device: 设备，默认为cuda，如果cuda不可用，则使用cpu 
        """
        check_dependencies() # 检查依赖
        setup_environment() # 设置环境变量
        
        self.model_path = model_path
        self.model_paths = [os.path.join('weights', f) for f in os.listdir('weights') if f.endswith('.pt')] # 读取所有路径
        
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
        self.loader = ModelLoader(model_path=self.model_path) # 初始化加载器

    def _generate_text_report(self,output, components)->dict:
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
        max_index = np.argmax(output_np)
        max_component = components[max_index]
        max_value = output_np[max_index]
        
        # 找出含量最低的成分
        min_index = np.argmin(output_np)
        min_component = components[min_index]
        min_value = output_np[min_index]
        
        # 计算平均含量
        avg_value = np.mean(output_np)
        
        # 根据含量高低对成分进行排序
        sorted_indices = np.argsort(output_np)[::-1]  # 从高到低排序

        # 打印数值结果
        logger.info("\n=== 油菜籽成分含量预测报告 ===")
        logger.info("\n成分含量预测结果:")
        for i, comp in enumerate(components):
            logger.info(f"{comp}: {output_np[i]:.4f}")
        
        # 添加详细的文字结论
        logger.info("\n预测结论:")
        logger.info(f'\n预测模型: {self.model_name if self.model_name else "未知模型"}')
        # 输出结论
        logger.info(f"1. 该油菜籽样本中含量最高的成分是 {max_component}，含量为 {max_value:.4f}")
        logger.info(f"2. 含量最低的成分是 {min_component}，含量为 {min_value:.4f}")
        logger.info(f"3. 所有成分的平均含量为 {avg_value:.4f}")
        
        logger.info("4. 各成分含量从高到低排序:")
        for i, idx in enumerate(sorted_indices):
            logger.info(f"   {i+1}. {components[idx]}: {output_np[idx]:.4f}")
        
        # 添加一些简单的品质评估（使用配置中的阈值）
        logger.info("5. 品质评估:")
        thresholds = MODEL_CONFIG["quality_thresholds"]
        
        # 检查油酸含量
        if "油酸" in components and "油酸" in thresholds:
            oil_index = components.index("油酸")
            oil_value = output_np[oil_index]
            oil_threshold = thresholds["油酸"]
            
            if oil_value > oil_threshold:
                logger.info(f"   油酸含量较高 ({oil_value:.4f})，品质较好")
            else:
                logger.info(f"   油酸含量较低 ({oil_value:.4f})，品质一般")
        
        # 检查亚油酸含量
        if "亚油酸" in components and "亚油酸" in thresholds:
            linoleic_index = components.index("亚油酸")
            linoleic_value = output_np[linoleic_index]
            linoleic_threshold = thresholds["亚油酸"]
            
            if linoleic_value > linoleic_threshold:
                logger.info(f"   亚油酸含量较高 ({linoleic_value:.4f})，营养价值较高")
            else:
                logger.info(f"   亚油酸含量较低 ({linoleic_value:.4f})，营养价值一般")
        
        # # 将结果保存到文本文件
        # if save_path:
        #     # 添加时间戳到文件名，避免覆盖现有报告
        #     timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        #     txt_path = os.path.splitext(save_path)[0] + f"_report_{timestamp}.txt"
        #     try:
        #         with open(txt_path, 'w', encoding='utf-8') as f:
        #             f.write("=== 油菜籽成分含量预测报告 ===\n\n")
        #             f.write(f"预测模型: {model_name if model_name else '未知模型'}\n")
        #             f.write("成分含量预测结果:\n")
        #             for i, comp in enumerate(components):
        #                 f.write(f"{comp}: {output_np[i]:.4f}\n")
                    
        #             f.write("\n预测结论:\n")
        #             f.write(f"1. 该油菜籽样本中含量最高的成分是 {max_component}，含量为 {max_value:.4f}\n")
        #             f.write(f"2. 含量最低的成分是 {min_component}，含量为 {min_value:.4f}\n")
        #             f.write(f"3. 所有成分的平均含量为 {avg_value:.4f}\n")
                    
        #             f.write("\n4. 各成分含量从高到低排序:\n")
        #             for i, idx in enumerate(sorted_indices):
        #                 f.write(f"   {i+1}. {components[idx]}: {output_np[idx]:.4f}\n")
                    
        #             f.write("\n5. 品质评估:\n")
        #             if output_np[0] > 0.5:
        #                 f.write(f"   油酸含量较高 ({output_np[0]:.4f})，品质较好\n")
        #             else:
        #                 f.write(f"   油酸含量较低 ({output_np[0]:.4f})，品质一般\n")
                    
        #             if output_np[1] > 0.3:
        #                 f.write(f"   亚油酸含量较高 ({output_np[1]:.4f})，营养价值较高\n")
        #             else:
        #                 f.write(f"   亚油酸含量较低 ({output_np[1]:.4f})，营养价值一般\n")
                    
        #             f.write(f"\n报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
        #         logger.info(f"\n文本报告已保存至: {txt_path}")
        #     except Exception as e:
        #         logger.error(f"保存报告失败: {str(e)}")
        
        # logger.info("\n=== 报告结束 ===")
        report = {
            "max_component": max_component,
            "max_value": max_value,
            "min_component": min_component,
            "min_value": min_value,
            "avg_value": avg_value,
            "sorted_indices": sorted_indices
        }
        # 把report里的数字转化为纯数字
        for key, value in report.items():
            if isinstance(value, np.float32):
                report[key] = float(value)
        return report
    
    def eval_image(self, image, model_name:Literal['ResNet','VGG','FasterNet'], model_path=None)->dict:
        """
        对单张图像进行预测

        Args:
            image_path: 图像文件路径 这里还不知道拿到的是URL还是文件路径
            model_name: 模型名称
            model_path: 模型文件路径，默认使用config的，但是因为可能要适配其他模型，所以保留接口

        Returns:
            预测结果的字典
        """
        if model_path is not None:
            self.model_path = model_path # 如果传入了模型路径，就使用传入的模型路径
        self.model_name = model_name # 保存模型名称
        self.loader.set_seed(SYSTEM_CONFIG['default_seed']) # 设置随机种子
        self.loader._load_model(model_name=model_name, model_path=self.model_path) # 加载模型 这里还没添加GPU选项
        logger.info(f"开始对图像 {image} 进行预测")
        # 加载图像
        if image.startswith('http'): # 如果是URL
            import requests
            from io import BytesIO
            try:
                response = requests.get(image) # 下载图像
                response.raise_for_status() # 检查是否下载成功
                image = Image.open(BytesIO(response.content)).convert('RGB') # 打开图像并转换为RGB模式
                logger.info(f"图像 {image} 下载成功，大小为 {image.size}")
            except Exception as e:
                logger.error(f"下载图像 {image} 失败: {str(e)}")
                return None
        else: # 如果是本地路径
            if not os.path.exists(image): # 如果文件不存在
                logger.error(f"图像 {image} 不存在")
                return None
            # 检查文件是否为图像文件
            if not image.lower().endswith(('.png', '.jpg', '.jpeg')): # 如果不是图像文件
                logger.error(f"文件 {image} 不是图像文件")
                return None
            image = Image.open(image).convert('RGB') # 打开图像并转换为RGB模式
            logger.info(f"图像 {image} 加载成功，大小为 {image.size}")
        # 都不是的话默认是图像文件
        try:
           
            preprocessed_image = self.loader.preprocess_image(image) # 预处理图像
            logger.info(f"图像 {image} 预处理成功")
            # 进行预测
            output = self.loader.predict(preprocessed_image) # 进行预测
            logger.info(f"图像 {image} 预测成功")

            # 生成文本报告
            component_names = MODEL_CONFIG["component_names"][:output.shape[1]]

            report = self._generate_text_report(output,component_names) # 生成文本报告
            logger.info(f"图像 {image} 文本报告生成成功")
        except Exception as e:
            logger.error(f"预测 {image} 失败: {str(e)}")
            return None
        
        return report