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
from backend.tools import get_detection_counts
# 获取日志记录器
logger = get_logger("ModelAPI")

# 全局计数器，用于跟踪ModelAPI实例
_model_api_instance_count = 0

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
        # 环境初始化已在main.py中统一处理，这里不再重复调用
        # check_dependencies() # 检查依赖
        # setup_environment() # 设置环境变量

        # 获取日志记录器
        from utils.logging_config import get_logger
        self.logger = get_logger("ModelAPI")

        # 使用全局计数器控制日志输出
        global _model_api_instance_count
        _model_api_instance_count += 1

        # 简化日志输出，使用INFO风格
        if _model_api_instance_count == 1:
            self.logger.info("Started ModelAPI initialization")
        else:
            self.logger.debug(f"Creating ModelAPI instance #{_model_api_instance_count}")

        if device is None: # 如果没有指定设备，则使用config中的默认设备
            self.loader = ModelLoader(device=MODEL_CONFIG['device']) # 初始化加载器
        else: # 如果指定了设备，则使用指定的设备
            self.loader = ModelLoader(device=device) # 初始化加载器
        
        # 创建V3专用的加载器实例
        self.v3_loader = ModelLoader(device=MODEL_CONFIG['device'], v3_mode=True) # 初始化V3加载器

        # 简化完成日志
        if _model_api_instance_count == 1:
            self.logger.info("ModelAPI initialization complete")
        else:
            self.logger.debug(f"ModelAPI instance #{_model_api_instance_count} ready")
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

        # 初始化返回字典
        result = {
            "protein": 0.0,
            "oil": 0.0,
            "water": 0.0,
            "cho": 0.0
        }

        # 处理不同维度的输出
        if len(output_np) >= 2:
            # 基础成分：油脂和蛋白质
            oil = output_np[0] # 油含量
            protein = output_np[1] # 蛋白质含量
            result["oil"] = float(oil)
            result["protein"] = float(protein)
        
        if len(output_np) >= 4:
            # 额外成分：水分和CHO
            water = output_np[2] # 水分含量
            cho = output_np[3] # CHO含量
            result["water"] = float(water)
            result["cho"] = float(cho)

        # 打印数值结果
        logger.info("=== 油菜籽成分含量预测报告 ===")
        logger.info(f'预测模型: {self.model_name if self.model_name else "未知模型"}')
        logger.info("成分含量预测结果:")
        
        # 输出所有成分
        if len(output_np) >= 1:
            logger.info(f"油脂: {result['oil']:.4f}")
        if len(output_np) >= 2:
            logger.info(f"蛋白质: {result['protein']:.4f}")
        if len(output_np) >= 3:
            logger.info(f"水分: {result['water']:.4f}")
        if len(output_np) >= 4:
            logger.info(f"CHO: {result['cho']:.4f}")
            
        logger.info("===       报告结束       ===")
        return result

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
        except Exception as e:
            logger.error(f"图像预测失败: {str(e)}")
            raise e

        return evals
    
    def eval_image_v3(self, image, model_name:Literal['MPViT', 'ResNet', 'FasterNet', 'EfficientNet', 'Swin', 'VanillaNet'])->dict:
        """
        V3版本的图像预测方法，使用v3_loader加载模型

        Args:
            image: 图像文件
            model_name: 模型名称

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
            seed = self.v3_loader.set_seed(SYSTEM_CONFIG['default_seed'])

            # 获取随机种子信息
            seed_info = self.v3_loader.get_seed_info()

            # 使用v3_loader加载模型
            self.v3_loader.load_model(model_name)
            self.model_name = model_name

            # 预处理图像
            size = 256 if model_name == 'Swin' else 224

            # 确保在预处理前同步CUDA操作
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            preprocessed_image = self.v3_loader.preprocess_image(image, size)

            # 确保在预处理后同步CUDA操作
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            output = self.v3_loader.predict(preprocessed_image)
            # 打印模型返回的原始数据
            logger.info("=== V3模型原始输出数据 ===")
            logger.info(f"原始输出类型: {type(output)}")
            if isinstance(output, torch.Tensor):
                logger.info(f"原始输出形状: {output.shape}")
                logger.info(f"原始输出值: {output}")
                logger.info(f"原始输出值(分离): {output.detach().cpu().numpy()}")
            elif isinstance(output, (list, tuple)):
                logger.info(f"原始输出长度: {len(output)}")
                for i, item in enumerate(output):
                    if isinstance(item, torch.Tensor):
                        logger.info(f"输出[{i}] 形状: {item.shape}, 值: {item}")
                    else:
                        logger.info(f"输出[{i}] 类型: {type(item)}, 值: {item}")
            else:
                logger.info(f"原始输出: {output}")
            logger.info("=== V3原始输出数据结束 ===")
            self.v3_loader.unload_model()
            # 生成文本报告
            component_names = MODEL_CONFIG["component_names"]
            evals = self.generate_text_evals(output,component_names)

            if torch.cuda.is_available():
                # 获取最终显存使用情况
                final_memory = torch.cuda.memory_allocated()
                # 计算显存消耗
                memory_consumed = final_memory - initial_memory
                # 转化为MB
                memory_consumed = memory_consumed / (1024 * 1024)
                logger.info(f"V3图像预测过程中显存消耗: {memory_consumed} MB")
            else:
                memory_consumed = 0

            # 记录结束时间并计算耗时
            end_time = datetime.datetime.now()
            time_delta = (end_time - start_time).total_seconds()

            evals['memory_cost'] = memory_consumed
            evals['time_delta'] = time_delta
        except Exception as e:
            logger.error(f"V3图像预测失败: {str(e)}")
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
                conf_threshold = MODEL_CONFIG.get("detect_conf_threshold", 0.9)
            if iou_threshold is None:
                iou_threshold = MODEL_CONFIG.get("detect_iou_threshold", 0.5)

            # 输出检测参数
            logger.info(f"🎯 目标检测参数 - 置信度阈值: {conf_threshold}, IoU阈值: {iou_threshold}")

            # 加载检测模型
            detect_model = MODEL_CONFIG.get("detect_model", "YOLO")
            logger.info(f"🔍 加载检测模型: {detect_model}")

            # 判断是分类器还是检测器
            if detect_model in ["EfficientNetB0Classifier", "ResNet18Classifier", "CustomCNNClassifier"]:
                # 使用分类器模型
                success = self.loader.load_classifier_model(detect_model)
                if not success:
                    return {
                        "success": False,
                        "error": "分类器模型加载失败",
                        "object_classes_counts": False,
                        "objects": []
                    }

                # 进行分类预测
                from PIL import Image as PILImage
                if not isinstance(image, PILImage.Image):
                    # 如果不是PIL图像，先转换
                    if hasattr(image, 'shape'):  # numpy array
                        image = PILImage.fromarray(image)
                    else:
                        raise ValueError("不支持的图像格式")

                image_tensor = self.loader.preprocess_image(image, size=224)
                results = self.loader.classify(image_tensor)
            else:
                # 使用传统检测模型
                success = self.loader.load_detect_model(detect_model)
                if not success:
                    return {
                        "success": False,
                        "error": "检测模型加载失败",
                        "object_classes_counts": [],
                        "objects": []
                    }

                # 进行目标检测
                results = self.loader.detect(image, conf_threshold, iou_threshold)

            # 卸载检测模型
            self.loader.unload_model()

            # 分析检测结果
            object_classes_objects = [] # 检测到的种子
            object_classes_counts = {} # 检测到种子类别的统计

            # 这个是训练的其他的分类器，原来只有一个YOLO，YOLO也够用
            if detect_model in ["EfficientNetB0Classifier", "ResNet18Classifier", "CustomCNNClassifier"]:
                # 处理分类器结果
                if results and results.get('predicted_class') == 'rapeseed':

                    # 为分类器创建一个虚拟的检测框（覆盖整个图像）
                    if hasattr(image, 'size'):
                        width, height = image.size
                    else:
                        # 如果是numpy数组
                        height, width = image.shape[:2]

                    object_classes_objects.append({
                        "confidence": results.get('confidence', 0.0),
                        "class_id": 1,  # rapeseed
                        "class_name": "rapeseed",
                        "bbox": [0, 0, width, height],  # 整个图像
                        "classification_result": results
                    })
            else: # 不是自定义的分类器的话，就会默认使用YOLO
                # 处理YOLO检测器结果
                if results:
                    for result in results:
                        if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                            object_classes_counts = get_detection_counts(result)
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
                                object_classes_objects.append(box_info)

            # 记录结束时间
            end_time = datetime.datetime.now()
            time_delta = (end_time - start_time).total_seconds()

            return {
                "success": True,
                "object_classes_counts": object_classes_counts,
                "objects": object_classes_objects,
                "detection_count": len(object_classes_objects),
                "time_delta": time_delta,
                "conf_threshold": conf_threshold,
                "iou_threshold": iou_threshold
            }

        except Exception as e:
            logger.error(f"目标检测失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "object_classes_counts": {},
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

        # 获取实际使用的阈值参数
        actual_conf = conf_threshold if conf_threshold is not None else MODEL_CONFIG["detect_conf_threshold"]
        actual_iou = iou_threshold if iou_threshold is not None else MODEL_CONFIG["detect_iou_threshold"]

        # 特殊处理：如果conf_threshold是0.5，自动调整为0.9
        if conf_threshold == 0.5:
            logger.info(f"⚠️ 检测到置信度阈值为0.5，自动调整为0.9")
            conf_threshold = 0.9
            actual_conf = 0.9

        # 输出预测请求参数
        logger.info(f"🔍 预测请求参数 - 模型: {model_name}, 置信度阈值: {actual_conf}, IoU阈值: {actual_iou}")

        # 记录总开始时间
        total_start_time = datetime.datetime.now()

        # 第一步：目标检测
        logger.info("开始目标检测...")
        detection_result = self.detect_objects(image, conf_threshold, iou_threshold)

        # 检查检测是否成功
        if not detection_result["success"]:
            return {
                "success": False,
                "object_classes_counts": {},
                "message": f"检测失败: {detection_result.get('error', '未知错误')}",
                "objects": [],
                "protein": 0.0,
                "oil": 0.0,
                "time_delta": 0.0
            }

        # 检查是否检测到对象
        if detection_result["object_classes_counts"] == {}:
            logger.info("未检测到种子对象，跳过成分分析")
            total_end_time = datetime.datetime.now()
            total_time_delta = (total_end_time - total_start_time).total_seconds()

            return {
                "success": True,
                "object_classes_counts": {},
                "message": "未检测到种子对象",
                "objects": detection_result.get("objects", []),
                "protein": 0.0,
                "oil": 0.0,
                "time_delta": total_time_delta
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
                "object_classes_counts": detection_result["object_classes_counts"],
                "message": "检测和分析完成",
                "objects": detection_result.get("objects", []),
                "protein": evaluation_result.get("protein", 0.0),
                "oil": evaluation_result.get("oil", 0.0),
                "time_delta": total_time_delta
            }

        except Exception as e:
            logger.error(f"成分分析失败: {str(e)}")
            total_end_time = datetime.datetime.now()
            total_time_delta = (total_end_time - total_start_time).total_seconds()

            return {
                "success": False,
                "object_classes_counts": True,
                "message": f"分析失败: {str(e)}",
                "objects": detection_result.get("objects", []),
                "protein": 0.0,
                "oil": 0.0,
                "time_delta": total_time_delta
            }

    def predict_v1(self, image, model_name: Literal['MPViT', 'ResNet', 'FasterNet', 'EfficientNet', 'Swin', 'VanillaNet'] = 'FasterNet',
                   conf_threshold: float = None, iou_threshold: float = None) -> dict:
        """
        V1预测接口：返回简化结果，只包含是否检测到和数值预测数据

        Args:
            image: 输入图像
            model_name: 用于成分分析的模型名称
            conf_threshold: 检测置信度阈值
            iou_threshold: 检测IoU阈值

        Returns:
            {
                "object_classes_counts": list,      # 检测到的种子对象类别
                "protein": float,      # 蛋白质含量（如果检测到）
                "oil": float,          # 油脂含量（如果检测到）
                "message": str,        # 状态消息
                "time_delta": float    # 总耗时（秒）
            }
        """
        # 获取实际使用的阈值参数
        actual_conf = conf_threshold if conf_threshold is not None else MODEL_CONFIG["detect_conf_threshold"]
        actual_iou = iou_threshold if iou_threshold is not None else MODEL_CONFIG["detect_iou_threshold"]

        # 输出预测请求参数
        logger.info(f"📊 V1预测请求 - 模型: {model_name}, 置信度阈值: {actual_conf}, IoU阈值: {actual_iou}")

        # 调用完整的检测和评估方法
        full_result = self.detect_and_eval(image, model_name, conf_threshold, iou_threshold)

        # 转换为v1格式的简化结果
        v1_result = {
            "object_classes_counts": full_result.get("object_classes_counts", False),
            "protein": 0.0,
            "oil": 0.0,
            "message": full_result.get("message", ""),
            "time_delta": full_result.get("total_time_delta", 0.0)
        }

        # 如果检测到对象且有评估结果，提取数值
        if full_result.get("object_classes_counts") and full_result.get("evaluation_result"):
            eval_result = full_result["evaluation_result"]
            v1_result["protein"] = eval_result.get("protein", 0.0)
            v1_result["oil"] = eval_result.get("oil", 0.0)

        return v1_result

    def detect_and_eval_v3(self, image, conf_threshold: float = None, iou_threshold: float = None) -> dict:
        """
        V3版本的检测和评估方法，使用不同的模型加载方式

        Args:
            image: 输入图像
            conf_threshold: 检测置信度阈值
            iou_threshold: 检测IoU阈值

        Returns:
            包含检测和分析结果的字典
        """
        import datetime

        # V3固定使用FasterNet模型
        model_name = "FasterNet"

        # 获取实际使用的阈值参数
        actual_conf = conf_threshold if conf_threshold is not None else MODEL_CONFIG["detect_conf_threshold"]
        actual_iou = iou_threshold if iou_threshold is not None else MODEL_CONFIG["detect_iou_threshold"]

        # 特殊处理：如果conf_threshold是0.5，自动调整为0.9
        if conf_threshold == 0.5:
            logger.info(f"⚠️ 检测到置信度阈值为0.5，自动调整为0.9")
            conf_threshold = 0.9
            actual_conf = 0.9

        # 输出预测请求参数
        logger.info(f"🔍 V3预测请求参数 : 置信度阈值: {actual_conf}, IoU阈值: {actual_iou}")

        # 记录总开始时间
        total_start_time = datetime.datetime.now()

        # 第一步：目标检测
        logger.info("开始目标检测...")
        detection_result = self.detect_objects(image, conf_threshold, iou_threshold)

        # 检查检测是否成功
        if not detection_result["success"]:
            return {
                "success": False,
                "object_classes_counts": {},
                "message": f"检测失败: {detection_result.get('error', '未知错误')}",
                "objects": [],
                "protein": 0.0,
                "oil": 0.0,
                "water": 0.0,
                "cho": 0.0,
                "time_delta": 0.0
            }

        # 检查是否检测到对象
        if detection_result["object_classes_counts"] == {}:
            logger.info("未检测到种子对象，跳过成分分析")
            total_end_time = datetime.datetime.now()
            total_time_delta = (total_end_time - total_start_time).total_seconds()

            return {
                "success": True,
                "object_classes_counts": {},
                "message": "未检测到种子对象",
                "objects": detection_result.get("objects", []),
                "protein": 0.0,
                "oil": 0.0,
                "water": 0.0,
                "cho": 0.0,
                "time_delta": total_time_delta
            }

        # 第二步：成分分析（V3特殊模型加载）
        logger.info(f"检测到 {detection_result['detection_count']} 个对象，开始V3成分分析...")
        try:
        
            # 使用v3_loader进行评估
            evaluation_result = self.eval_image_v3(image, model_name)

            # 计算总耗时
            total_end_time = datetime.datetime.now()
            total_time_delta = (total_end_time - total_start_time).total_seconds()

            # 构建返回结果，包含所有成分指标
            result = {
                "success": True,
                "object_classes_counts": detection_result["object_classes_counts"],
                "message": "V3检测和分析完成",
                "objects": detection_result.get("objects", []),
                "protein": evaluation_result.get("protein", 0.0),
                "oil": evaluation_result.get("oil", 0.0),
                "water": evaluation_result.get("water", 0.0),
                "cho": evaluation_result.get("cho", 0.0),
                "time_delta": total_time_delta
            }

            # 输出完整结果日志
            logger.info("=== V3模型预测完整结果 ===")
            logger.info(f"✅ V3检测和分析完成")
            logger.info(f"🎯 检测到的对象: {result['object_classes_counts']}")
            logger.info(f"🔬 V3成分分析结果:")
            logger.info(f"   蛋白质: {result['protein']:.2f}%")
            logger.info(f"   油脂: {result['oil']:.2f}%")
            logger.info(f"   水分: {result['water']:.2f}%")
            logger.info(f"   CHO: {result['cho']:.2f}%")
            logger.info(f"⏱️  总耗时: {result['time_delta']:.2f}秒")
            logger.info("=== V3结果展示结束 ===")

            return result

        except Exception as e:
            logger.error(f"V3成分分析失败: {str(e)}")
            total_end_time = datetime.datetime.now()
            total_time_delta = (total_end_time - total_start_time).total_seconds()

            return {
                "success": False,
                "object_classes_counts": True,
                "message": f"V3分析失败: {str(e)}",
                "objects": detection_result.get("objects", []),
                "protein": 0.0,
                "oil": 0.0,
                "water": 0.0,
                "cho": 0.0,
                "time_delta": total_time_delta
            }

    def predict_v2(self, image, model_name: Literal['MPViT', 'ResNet', 'FasterNet', 'EfficientNet', 'Swin', 'VanillaNet'] = 'FasterNet',
                   conf_threshold: float = None, iou_threshold: float = None) -> dict:
        """
        V2预测接口：返回简洁的完整结果

        Args:
            image: 输入图像
            model_name: 用于成分分析的模型名称
            conf_threshold: 检测置信度阈值
            iou_threshold: 检测IoU阈值

        Returns:
            {
                "success": bool,      # 操作是否成功
                "object_classes_counts": {},     # 检测到的对象类别统计
                "message": str,       # 状态消息
                "objects": list,      # 检测到的对象列表
                "protein": float,     # 蛋白质含量
                "oil": float,         # 油脂含量
                "time_delta": float   # 总耗时（秒）
            }
        """
        # 获取实际使用的阈值参数
        actual_conf = conf_threshold if conf_threshold is not None else MODEL_CONFIG["detect_conf_threshold"]
        actual_iou = iou_threshold if iou_threshold is not None else MODEL_CONFIG["detect_iou_threshold"]

        # 输出预测请求参数
        logger.info(f"V2预测请求 - 模型: {model_name}, 置信度阈值: {actual_conf}, IoU阈值: {actual_iou}")

        # 直接返回完整结果
        return self.detect_and_eval(image, model_name, conf_threshold, iou_threshold)


    def predict_ripeness(self, image) -> dict:
        """
        预测油菜籽的成熟度（绿熟、黄熟、完熟）
        先判断是否为油菜籽，再预测成熟度

        Args:
            image: 输入图像

        Returns:
            {
                "success": bool,           # 操作是否成功
                "is_rapeseed": bool,        # 是否为油菜籽
                "ripeness_class": str,       # 预测的成熟度类别（绿熟、黄熟、完熟）
                "confidence": float,        # 预测置信度
                "probabilities": dict,       # 各类别的概率
                "message": str,            # 状态消息
                "time_delta": float        # 总耗时（秒）
            }
        """
        import datetime

        try:
            # 记录开始时间
            start_time = datetime.datetime.now()

            logger.info("开始成熟度分类...")

            # 加载成熟度分类模型
            ripeness_model = MODEL_CONFIG.get("ripeness_model", "RipenessClassifier")
            success = self.loader.load_ripeness_model(ripeness_model)
            
            if not success:
                return {
                    "success": False,
                    "is_rapeseed": False,
                    "ripeness_class": "",
                    "confidence": 0.0,
                    "probabilities": {},
                    "message": "成熟度分类模型加载失败",
                    "time_delta": 0.0
                }

            # 预处理图像（与 demo2 保持一致：Resize(256) + CenterCrop(224)）
            from torchvision import transforms
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
            preprocessed_image = transform(image).unsqueeze(0).to(self.loader.device)

            # === 构建特征提取器（去掉最后的分类头）===
            feature_extractor = torch.nn.Sequential(*list(self.loader.model.model.children())[:-1])
            feature_extractor = feature_extractor.to(self.loader.device).eval()

            # === 加载油菜籽平均特征 ===
            try:
                root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                mean_feat_path = os.path.join(root_dir, 'weights/rapeseed_global_mean_feature.npy')
                mean_feat = np.load(mean_feat_path)
            except FileNotFoundError:
                logger.warning("未找到 'rapeseed_global_mean_feature.npy'，跳过油菜籽判断")
                mean_feat = None

            # === 提取特征并计算与油菜籽平均特征的相似度 ===
            similarity = 0.0
            is_rapeseed = True
            SIMILARITY_THRESHOLD = 0.65

            if mean_feat is not None:
                with torch.no_grad():
                    # preprocessed_image 已经是 [1, C, H, W]，不需要再 unsqueeze
                    feat = feature_extractor(preprocessed_image)
                    feat = torch.flatten(feat, 1).cpu().numpy()
                
                # 调试信息
                logger.info(f"特征形状: {feat.shape}")
                logger.info(f"特征前10维: {feat[0, :10]}")
                logger.info(f"特征均值: {feat.mean():.6f}")
                logger.info(f"特征标准差: {feat.std():.6f}")
                
                # 计算余弦相似度（使用numpy实现，避免sklearn依赖）
                def cosine_similarity_np(a, b):
                    """计算两个向量之间的余弦相似度"""
                    norm_a = np.linalg.norm(a)
                    norm_b = np.linalg.norm(b)
                    if norm_a == 0 or norm_b == 0:
                        return 0.0
                    return np.dot(a, b) / (norm_a * norm_b)
                
                similarity = cosine_similarity_np(feat[0], mean_feat)
                logger.info(f"计算得到的相似度: {similarity:.6f}")

                # 判断是否为油菜籽
                if similarity < SIMILARITY_THRESHOLD:
                    self.loader.unload_model()
                    end_time = datetime.datetime.now()
                    time_delta = (end_time - start_time).total_seconds()
                    
                    logger.info(f"输入不是油菜籽（相似度={similarity:.3f} < {SIMILARITY_THRESHOLD}）")
                    return {
                        "success": True,
                        "is_rapeseed": False,
                        "ripeness_class": "",
                        "confidence": 0.0,
                        "probabilities": {},
                        "message": f"输入不是油菜籽（相似度={similarity:.3f} < {SIMILARITY_THRESHOLD}）",
                        "time_delta": time_delta
                    }

            # 进行成熟度分类
            result = self.loader.classify(preprocessed_image)

            # 卸载模型
            self.loader.unload_model()

            # 记录结束时间并计算耗时
            end_time = datetime.datetime.now()
            time_delta = (end_time - start_time).total_seconds()

            logger.info(f"成熟度分类完成: 类别={result.get('predicted_class', '')}, 置信度={result.get('confidence', 0):.4f}, 相似度={similarity:.3f}, 耗时={time_delta:.3f}秒")

            return {
                "success": True,
                "is_rapeseed": True,
                "ripeness_class": result.get("predicted_class", ""),
                "confidence": result.get("confidence", 0.0),
                "probabilities": result.get("probabilities", {}),
                "message": "成熟度分类完成",
                "time_delta": time_delta
            }

        except Exception as e:
            logger.error(f"成熟度分类失败: {str(e)}")
            return {
                "success": False,
                "is_rapeseed": False,
                "ripeness_class": "",
                "confidence": 0.0,
                "probabilities": {},
                "message": f"成熟度分类失败: {str(e)}",
                "time_delta": 0.0
            }

    def predict_ripeness_v3(self, image) -> dict:
        """
        V3版本成熟度分类，参考demo3实现
        使用KNN（K=5）判断是否为油菜籽，使用ResNet18进行成熟度分类

        Args:
            image: 输入图像（PIL Image）

        Returns:
            {
                "success": bool,           # 操作是否成功
                "is_rapeseed": bool,       # 是否为油菜籽
                "ripeness_class": str,     # 成熟度类别（未熟、半熟、全熟）
                "confidence": float,       # 预测置信度
                "similarity": float,       # KNN平均相似度
                "probabilities": dict,     # 各类别的概率
                "message": str,            # 状态消息
                "time_delta": float        # 总耗时（秒）
            }
        """
        import datetime
        import numpy as np
        import torch
        import torch.nn as nn
        from torchvision import transforms, models
        import os

        # 自定义余弦相似度计算函数，避免依赖 scikit-learn
        def cosine_similarity(a, b):
            """计算两个向量之间的余弦相似度"""
            a = np.array(a)
            b = np.array(b)
            
            # 确保输入是二维数组
            if a.ndim == 1:
                a = a.reshape(1, -1)
            if b.ndim == 1:
                b = b.reshape(1, -1)
            
            # 计算点积
            dot_product = np.dot(a, b.T)
            
            # 计算范数
            norm_a = np.linalg.norm(a, axis=1, keepdims=True)
            norm_b = np.linalg.norm(b, axis=1, keepdims=True)
            
            # 计算余弦相似度
            similarity = dot_product / (norm_a * norm_b.T + 1e-10)  # 添加小值避免除零
            
            return similarity

        # ==================== 配置 ====================
        NUM_CLASSES = 3
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        KNN_K = 5
        KNN_THRESHOLD = 0.7  # KNN判断阈值
        classes = ["绿熟", "黄熟", "完熟"]

        try:
            # 记录开始时间
            start_time = datetime.datetime.now()
            logger.info("开始V3成熟度分类（参考demo3）...")

            # ==================== 加载模型 ====================
            model = models.resnet18(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, NUM_CLASSES)
            )

            # 获取模型路径
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(root_dir, 'weights/fruit_ripeness_model.pth')

            if not os.path.exists(model_path):
                return {
                    "success": False,
                    "is_rapeseed": False,
                    "ripeness_class": "",
                    "confidence": 0.0,
                    "similarity": 0.0,
                    "probabilities": {},
                    "message": f"模型文件不存在: {model_path}",
                    "time_delta": 0.0
                }

            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model = model.to(DEVICE).eval()

            # ==================== 特征提取器 ====================
            feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])  # 移除 fc 层
            feature_extractor = feature_extractor.to(DEVICE).eval()

            # ==================== 加载特征库 ====================
            features_path = os.path.join(root_dir, 'weights/all_rapeseed_features.npy')
            try:
                all_rapeseed_features = np.load(features_path)  # shape: (N, 512)
                if all_rapeseed_features.ndim != 2 or all_rapeseed_features.shape[1] != 512:
                    return {
                        "success": False,
                        "is_rapeseed": False,
                        "ripeness_class": "",
                        "confidence": 0.0,
                        "similarity": 0.0,
                        "probabilities": {},
                        "message": f"特征库格式错误: {features_path}",
                        "time_delta": 0.0
                    }
                logger.info(f"成功加载油菜籽特征库: {all_rapeseed_features.shape[0]} 个样本")
            except FileNotFoundError:
                return {
                    "success": False,
                    "is_rapeseed": False,
                    "ripeness_class": "",
                    "confidence": 0.0,
                    "similarity": 0.0,
                    "probabilities": {},
                    "message": f"特征库文件不存在: {features_path}",
                    "time_delta": 0.0
                }

            # ==================== 图像预处理 ====================
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])

            img_tensor = transform(image).unsqueeze(0).to(DEVICE)  # [1, 3, 224, 224]

            # ==================== KNN判断是否为油菜籽 ====================
            with torch.no_grad():
                feat = feature_extractor(img_tensor)          # [1, 512, 1, 1]
                feat = torch.flatten(feat, 1).cpu().numpy()   # [1, 512]

            # 计算余弦相似度
            sims = cosine_similarity(feat, all_rapeseed_features)[0]  # shape: (N,)
            top_k_sims = np.sort(sims)[-KNN_K:]                    # 最大的 k 个相似度
            avg_top_k_sim = np.mean(top_k_sims)

            logger.info(f"KNN平均相似度: {avg_top_k_sim:.3f}, 阈值: {KNN_THRESHOLD}")

            # 判断是否为油菜籽
            # if avg_top_k_sim < KNN_THRESHOLD:
            #     end_time = datetime.datetime.now()
            #     time_delta = (end_time - start_time).total_seconds()

            #     logger.info(f"输入不是油菜籽（Top-{KNN_K} 平均相似度 = {avg_top_k_sim:.3f} < {KNN_THRESHOLD}）")
            #     return {
            #         "success": True,
            #         "is_rapeseed": False,
            #         "ripeness_class": "",
            #         "confidence": 0.0,
            #         "similarity": float(avg_top_k_sim),
            #         "probabilities": {},
            #         "message": f"输入不是油菜籽（Top-{KNN_K} 平均相似度 = {avg_top_k_sim:.3f} < {KNN_THRESHOLD}）",
            #         "time_delta": time_delta
            #     }

            # ==================== 是油菜籽，预测成熟度 ====================
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))

            # 构建概率字典
            probabilities = {classes[i]: float(probs[i]) for i in range(len(classes))}

            # 记录结束时间
            end_time = datetime.datetime.now()
            time_delta = (end_time - start_time).total_seconds()

            logger.info(f"V3成熟度分类完成: 类别={classes[pred_idx]}, "
                       f"置信度={probs[pred_idx]:.4f}, 相似度={avg_top_k_sim:.3f}, 耗时={time_delta:.3f}秒")

            return {
                "success": True,
                "is_rapeseed": True,
                "ripeness_class": classes[pred_idx],
                "confidence": float(probs[pred_idx]),
                "similarity": float(avg_top_k_sim),
                "probabilities": probabilities,
                "message": "成熟度分类完成",
                "time_delta": time_delta
            }

        except Exception as e:
            logger.error(f"V3成熟度分类失败: {str(e)}")
            return {
                "success": False,
                "is_rapeseed": False,
                "ripeness_class": "",
                "confidence": 0.0,
                "similarity": 0.0,
                "probabilities": {},
                "message": f"V3成熟度分类失败: {str(e)}",
                "time_delta": 0.0
            }