'''
客户给的文件比较杂，不敢乱动
通过init提供模型调用接口
'''
from typing import Literal
# 导入模型全部改成options一致的名字，方便调用
from .build_mpvit import MPViT # 导入MPViT模型
from .Swin import model as Swin # 导入Swin模型 多封装一层以适配模型
# from .build_swinv2 import SwinTransformerV2 as Swin # 导入SwinTransformerV2模型
from .VanillaNet import VanillaNet
# from .build_vanillanet import VanillaNet # 导入VanillaNet模型
from .model_zoo import FasterNet # 导入FasterNet模型
from .ResNet18 import ResNet18 as ResNet # 导入ResNet18模型
from .model_zoo import Efficient_net as EfficientNet # 导入EfficientNet模型

MODEL_OPTIONS = Literal['MPViT', 'ResNet', 'FasterNet', 'EfficientNet', 'Swin', 'VanillaNet'] # 模型选项