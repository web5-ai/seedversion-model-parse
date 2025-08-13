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
import os
from typing import Literal
from utils.model_loader import ModelLoader
from config import MODEL_CONFIG, SYSTEM_CONFIG
from utils.environment import check_dependencies, setup_environment
from utils.logging_config import get_logger

# 获取日志记录器
logger = get_logger("ModelAPI")

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
    def __init__(self, device:Literal['cuda','cpu', None]=None):
        """
        初始化模型，提前加载状态字典，随时可以转化成模型

        Args:
            model_path: 模型文件路径，默认用config的，但是因为可能要适配其他模型，所以保留接口
            model_name: 模型名称
            device: 设备，默认为cuda，如果cuda不可用，则使用cpu
        """
        check_dependencies() # 检查依赖
        setup_environment() # 设置环境变量
        if device is None: # 如果没有指定设备，则使用config中的默认设备
            self.loader = ModelLoader(device=MODEL_CONFIG['device']) # 初始化加载器
        else: # 如果指定了设备，则使用指定的设备
            self.loader = ModelLoader(device=device) # 初始化加载器
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

        # 标准化处理：控制精度以确保一致性
        # 将数值四舍五入到固定小数位，以减少浮点误差的影响
        decimal_places = 6  # 保留6位小数
        output_np = np.round(output_np, decimal_places)

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

    def set_seed(self, seed:int = None):
        '''
        封装model_loader的set_seed方法
        '''
        return self.loader.set_seed(seed)

    def get_seed_info(self):
        '''
        封装model_loader的get_seed_info方法
        '''
        return self.loader.get_seed_info()

    def run_env_test(self, model_name='FasterNet', test_image_path=None, seed=123):
        '''
        运行环境测试，测试当前环境下的各种变量情况

        Args:
            model_name: 模型名称，默认为FasterNet
            test_image_path: 测试图像路径，默认使用config中的默认图像
            seed: 随机种子，默认使用config中的默认种子

        Returns:
            包含环境测试结果的字典
        '''
        from config import IMAGE_CONFIG

        # 如果未指定测试图像路径，使用配置中的默认路径
        if test_image_path is None:
            test_image_path = IMAGE_CONFIG["default_image_path"]

        # 确保使用绝对路径
        if not os.path.isabs(test_image_path):
            # 获取项目根目录
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            test_image_path = os.path.join(root_dir, test_image_path)

        logger.info(f"开始运行环境测试... 模型: {model_name}, 图像: {test_image_path}")
        result = self.loader.env_test(model_name, test_image_path, seed)
        return result
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
        import datetime

        try:
            # 记录开始时间
            start_time = datetime.datetime.now()

            # 检查是否有可用的CUDA设备
            if MODEL_CONFIG['device'] == 'cuda':
                # 获取初始显存使用情况
                initial_memory = torch.cuda.memory_allocated()
            else:
                logger.warning("未使用CUDA，无法监控显存消耗。")
                initial_memory = 0

            # 强制设置随机种子，确保结果可重复
            seed = self.loader.set_seed(SYSTEM_CONFIG['default_seed'])

            # 获取随机种子信息
            seed_info = self.loader.get_seed_info()

            # 加载模型
            self.loader.load_model(model_name) # 加载模型
            self.model_name = model_name # 设置模型名称

            # 预处理图像
            size = 256 if model_name == 'Swin' else 224 # 设置图像大小，Swin需要256，其他模型需要224

            # 确保在预处理前同步CUDA操作
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            preprocessed_image = self.loader.preprocess_image(image, size) # 预处理图像

            # 确保在预处理后同步CUDA操作
            if torch.cuda.is_available():
                torch.cuda.synchronize()

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

            # 记录结束时间并计算耗时
            end_time = datetime.datetime.now()
            time_delta = (end_time - start_time).total_seconds()

            evals['memory_cost'] = memory_consumed
            evals['time_delta'] = time_delta
            evals['seed'] = seed
            evals['seed_info'] = seed_info
        except Exception as e:
            logger.error(f"图像预测失败: {str(e)}")
            raise e

        return evals

    def detect_objects(self, image, conf_threshold: float = None, iou_threshold: float = None) -> dict:
        """
        对图像进行目标检测

        Args:
            image: 输入图像
            conf_threshold: 置信度阈值，默认使用配置中的值
            iou_threshold: IoU阈值，默认使用配置中的值

        Returns:
            检测结果字典，包含是否检测到对象和检测详情
        """
        import datetime

        try:
            # 记录开始时间
            start_time = datetime.datetime.now()

            # 使用配置中的默认阈值
            if conf_threshold is None:
                conf_threshold = MODEL_CONFIG.get("detect_conf_threshold", 0.25)
            if iou_threshold is None:
                iou_threshold = MODEL_CONFIG.get("detect_iou_threshold", 0.45)

            # 加载检测模型
            detect_model = MODEL_CONFIG.get("detect_model", "YOLO")
            success = self.loader.load_detect_model(detect_model)

            if not success:
                return {
                    "success": False,
                    "error": "检测模型加载失败",
                    "detected": False,
                    "objects": []
                }

            # 进行目标检测
            results = self.loader.detect(image, conf_threshold, iou_threshold)

            # 卸载检测模型
            self.loader.unload_model()

            # 分析检测结果
            detected_objects = []
            has_detection = False

            if results:
                for result in results:
                    if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                        has_detection = True
                        # 提取检测框信息
                        boxes = result.boxes
                        for i in range(len(boxes)):
                            box_info = {
                                "confidence": float(boxes.conf[i]) if hasattr(boxes, 'conf') else 0.0,
                                "class_id": int(boxes.cls[i]) if hasattr(boxes, 'cls') else -1,
                                "bbox": boxes.xyxy[i].tolist() if hasattr(boxes, 'xyxy') else []
                            }
                            # 添加类别名称
                            if hasattr(result, 'names') and box_info["class_id"] in result.names:
                                box_info["class_name"] = result.names[box_info["class_id"]]
                            detected_objects.append(box_info)

            # 记录结束时间
            end_time = datetime.datetime.now()
            time_delta = (end_time - start_time).total_seconds()

            return {
                "success": True,
                "detected": has_detection,
                "objects": detected_objects,
                "detection_count": len(detected_objects),
                "time_delta": time_delta,
                "conf_threshold": conf_threshold,
                "iou_threshold": iou_threshold
            }

        except Exception as e:
            logger.error(f"目标检测失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "detected": False,
                "objects": []
            }

    def detect_and_eval(self, image, model_name: Literal['MPViT', 'ResNet', 'FasterNet', 'EfficientNet', 'Swin', 'VanillaNet'] = 'FasterNet',
                       conf_threshold: float = None, iou_threshold: float = None) -> dict:
        """
        先进行目标检测，如果检测到对象则进行成分分析，否则返回未检测到种子对象

        Args:
            image: 输入图像
            model_name: 用于成分分析的模型名称
            conf_threshold: 检测置信度阈值
            iou_threshold: 检测IoU阈值

        Returns:
            包含检测和分析结果的字典
        """
        import datetime

        # 记录总开始时间
        total_start_time = datetime.datetime.now()

        # 第一步：目标检测
        logger.info("开始目标检测...")
        detection_result = self.detect_objects(image, conf_threshold, iou_threshold)

        # 检查检测是否成功
        if not detection_result["success"]:
            return {
                "success": False,
                "stage": "detection",
                "error": detection_result.get("error", "检测失败"),
                "detection_result": detection_result
            }

        # 检查是否检测到对象
        if not detection_result["detected"]:
            logger.info("未检测到种子对象，跳过成分分析")
            total_end_time = datetime.datetime.now()
            total_time_delta = (total_end_time - total_start_time).total_seconds()

            return {
                "success": True,
                "stage": "detection_only",
                "message": "未检测到种子对象",
                "detected": False,
                "detection_result": detection_result,
                "evaluation_result": None,
                "total_time_delta": total_time_delta
            }

        # 第二步：成分分析
        logger.info(f"检测到 {detection_result['detection_count']} 个对象，开始成分分析...")
        try:
            evaluation_result = self.eval_image(image, model_name)

            # 计算总耗时
            total_end_time = datetime.datetime.now()
            total_time_delta = (total_end_time - total_start_time).total_seconds()

            return {
                "success": True,
                "stage": "complete",
                "message": "检测和分析完成",
                "detected": True,
                "detection_result": detection_result,
                "evaluation_result": evaluation_result,
                "total_time_delta": total_time_delta
            }

        except Exception as e:
            logger.error(f"成分分析失败: {str(e)}")
            total_end_time = datetime.datetime.now()
            total_time_delta = (total_end_time - total_start_time).total_seconds()

            return {
                "success": False,
                "stage": "evaluation",
                "error": str(e),
                "detected": True,
                "detection_result": detection_result,
                "evaluation_result": None,
                "total_time_delta": total_time_delta
            }
