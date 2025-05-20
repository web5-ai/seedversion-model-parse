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
        # random.seed(seed)
        # np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
        logger.info(f"已设置随机种子: {seed}")
        return seed

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

    def get_seed_info(self):
        """
        获取当前随机种子和相关配置信息

        Returns:
            包含各种随机种子信息的字典
        """
        seed_info = {}

        # 获取PyTorch CPU随机种子
        # 注意：PyTorch没有直接API获取当前CPU种子，只能获取CUDA种子
        # 我们可以通过生成一个随机数并重新设置种子来间接获取

        # 获取CUDA随机种子
        if torch.cuda.is_available():
            try:
                cuda_seed = torch.cuda.initial_seed()
                seed_info["torch_cuda_seed"] = cuda_seed

                # 获取所有GPU设备的随机种子
                device_count = torch.cuda.device_count()
                cuda_seeds = {}
                for i in range(device_count):
                    with torch.cuda.device(i):
                        cuda_seeds[f"device_{i}"] = torch.cuda.initial_seed()
                seed_info["torch_cuda_seeds_by_device"] = cuda_seeds
            except Exception as e:
                seed_info["torch_cuda_seed_error"] = str(e)
        else:
            seed_info["torch_cuda_available"] = False

        # 获取环境变量中的随机种子
        seed_info["env_pythonhashseed"] = os.environ.get("PYTHONHASHSEED", "Not set")

        # 获取PyTorch配置信息
        seed_info["torch_deterministic"] = torch.backends.cudnn.deterministic
        seed_info["torch_benchmark"] = torch.backends.cudnn.benchmark

        logger.info(f"获取到随机种子信息: {seed_info}")
        return seed_info

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

    def env_test(self, model_name:MODEL_OPTIONS="ResNet", test_image_path=None, seed=None):
        """
        环境测试函数，用于测试当前环境下的各种变量情况

        Args:
            model_name: 模型名称，默认为ResNet
            test_image_path: 测试图像路径，默认使用config中的默认图像
            seed: 随机种子，默认使用config中的默认种子

        Returns:
            包含环境测试结果的字典
        """
        from PIL import Image
        from hashlib import sha256
        from io import BytesIO
        import os
        import sys
        from config import IMAGE_CONFIG, SYSTEM_CONFIG

        # 初始化结果字典
        result = {
            "environment": {},
            "model": {},
            "image": {},
            "prediction": {},
            "seed_info": {}
        }

        # 记录环境信息
        result["environment"]["python_version"] = sys.version
        result["environment"]["torch_version"] = torch.__version__
        result["environment"]["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            result["environment"]["cuda_version"] = torch.version.cuda
            result["environment"]["gpu_name"] = torch.cuda.get_device_name(0)

        # 设置随机种子
        if seed is None:
            seed = SYSTEM_CONFIG["default_seed"]
        actual_seed = self.set_seed(seed)
        result["seed_info"]["set_seed"] = actual_seed

        # 获取随机种子信息
        seed_info = self.get_seed_info()
        result["seed_info"].update(seed_info)

        # 使用默认测试图像
        if test_image_path is None:
            test_image_path = IMAGE_CONFIG["default_image_path"]

        # 检查测试图像是否存在
        if not os.path.exists(test_image_path):
            logger.error(f"测试图像不存在: {test_image_path}")
            result["image"]["error"] = f"测试图像不存在: {test_image_path}"
            return result

        # 加载图像并计算原始哈希值
        try:
            original_image = Image.open(test_image_path).convert("RGB")
            # 计算原始图像哈希值
            image_bytes = BytesIO()
            original_image.save(image_bytes, "PNG")
            original_hash = sha256(image_bytes.getvalue()).hexdigest()
            result["image"]["original_path"] = test_image_path
            result["image"]["original_size"] = original_image.size
            result["image"]["original_hash"] = original_hash
            result["image"]["original_mode"] = original_image.mode
        except Exception as e:
            logger.error(f"加载测试图像失败: {str(e)}")
            result["image"]["error"] = f"加载测试图像失败: {str(e)}"
            return result

        # 加载模型
        try:
            self.load_model(model_name)
            result["model"]["name"] = model_name
            result["model"]["device"] = self.device
            # 获取模型参数数量
            total_params = sum(p.numel() for p in self.model.parameters())
            result["model"]["total_params"] = total_params
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            result["model"]["error"] = f"加载模型失败: {str(e)}"
            return result

        # 预处理图像并计算处理后的哈希值
        try:
            size = 256 if model_name == 'Swin' else 224
            preprocessed_tensor = self.preprocess_image(original_image, size)

            # 将预处理后的张量转换回图像以计算哈希值
            # 注意：这只是一个近似，因为归一化后的张量不能完全还原为原始图像
            processed_img = transforms.ToPILImage()(preprocessed_tensor.squeeze(0).cpu())

            # 计算处理后图像哈希值
            processed_bytes = BytesIO()
            processed_img.save(processed_bytes, "PNG")
            processed_hash = sha256(processed_bytes.getvalue()).hexdigest()

            result["image"]["processed_size"] = (size, size)
            result["image"]["processed_hash"] = processed_hash
            result["image"]["hash_changed"] = original_hash != processed_hash
        except Exception as e:
            logger.error(f"预处理图像失败: {str(e)}")
            result["image"]["preprocessing_error"] = f"预处理图像失败: {str(e)}"
            return result

        # 进行预测
        try:
            with torch.no_grad():
                output = self.model(preprocessed_tensor)

            # 记录预测结果
            output_np = output.cpu().numpy().flatten()
            result["prediction"]["raw_output"] = output_np.tolist()

            # 如果是二维输出（蛋白质和油脂），则记录具体值
            if len(output_np) >= 2:
                result["prediction"]["protein"] = float(output_np[0])
                result["prediction"]["oil"] = float(output_np[1])

            # 计算预测结果的哈希值（用于验证结果的一致性）
            prediction_hash = sha256(output_np.tobytes()).hexdigest()
            result["prediction"]["hash"] = prediction_hash
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            result["prediction"]["error"] = f"预测失败: {str(e)}"

        # 卸载模型
        self.unload_model()

        logger.info(f"环境测试完成，结果: {result}")
        return result